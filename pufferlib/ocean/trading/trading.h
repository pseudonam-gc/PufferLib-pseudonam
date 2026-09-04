/* Trading: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "raylib.h"
#include "utils.h"

// Environment duration
const int MAX_TICKS = 2000;

// Variable upper bounds / lower bounds
const int MAX_OPTIONS = 32;
const int GLOBAL_FEATURES = 5;
const int INIT_FUNDS = 10000;

// Owned contracts live in a heap, not env->observations, since holdings can
// grow past MAX_OPTIONS over an episode. This just caps that heap's size.
const int OWNED_CAPACITY = 8192;

// Required struct. Only use floats!
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n; // Required as the last field
} Log;

typedef struct {
    Log log;
    float* observations; // flat float32 buffer, must match trading.py's Box dtype
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int tick; // number of time steps 
    int step; // number of env steps (can be multiple per tick)
    OwnedOptionHeap owned; // currently-held contracts, sorted by expiry
    Underlying underlying;
    float episode_reward; // sum of every tick's reward this episode: reset in c_reset
    Option pending_market[MAX_OPTIONS]; // tick's closing quotes, bought against at the start of the next c_step
    float tick_spend; // how much was spent on options this tick, for min_tick_spend reward
    float min_tick_spend; // forced-investment param: reward -= max(0, this - tick_spend)
    // random noise added to the price of the option, the range being [market_noise_lower, market_noise_upper
    float market_noise_lower;
    float market_noise_upper;
} Trading;

void add_log(Trading* env) {
    env->log.perf += (env->episode_reward > 0) ? 1 : 0;
    env->log.score += env->episode_reward;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_reward;
    env->log.n++;
}

void allocate_all(Trading* env) {
    int option_info_features = MAX_OPTIONS * OPTION_FEATURES * 2;
    int memsize = (option_info_features + GLOBAL_FEATURES) * sizeof(float);
    env->observations = (float*)calloc(memsize, 1);
    env->actions = (int*)calloc(1, sizeof(int)); // single Discrete(MAX_OPTIONS+1) choice
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->owned.data = NULL; // allocated on first reset
    // Demo defaults; the Python side sets these via my_init/kwargs.
    env->tick_spend = 0.0f;
    env->min_tick_spend = 10.0f;
    env->market_noise_lower = -0.06f;
    env->market_noise_upper = 0.1f;
}

void free_all(Trading* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->owned.data);
}

// Required function
void c_reset(Trading* env) {
    int option_info_features = MAX_OPTIONS * OPTION_FEATURES * 2;
    int memsize = (option_info_features + GLOBAL_FEATURES) * sizeof(float);
    memset(env->observations, 0, memsize);
    env->observations[option_info_features + 0] = INIT_FUNDS;
    env->tick = 0;
    env->step = 0;
    env->tick_spend = 0.0f;
    env->episode_reward = 0.0f;
    init_underlying(&env->underlying);
    memset(env->pending_market, 0, sizeof(env->pending_market));

    if (env->owned.data == NULL) {
        env->owned.data = (Option*)calloc(OWNED_CAPACITY, sizeof(Option));
        env->owned.capacity = OWNED_CAPACITY;
    }
    env->owned.count = 0;
}

// Black-Scholes price of a European call option.
float normal_cdf(float x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

float black_scholes(
    float S, // current stock price
    float K, // strike price
    float T, // time to expiration in years
    float r, // risk-free interest rate
    float sigma // underlying volatility
) {
    float d1 = (log(S / K) + (r + 0.5 * pow(sigma, 2)) * T) / (sigma * sqrt(T));
    float d2 = d1 - sigma * sqrt(T);
    return S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2);
}

Option* gen_test_options(Underlying* underlying, int tick, float noise_lower, float noise_upper) {
    static Option options[MAX_OPTIONS];
    for (int i = 0; i < MAX_OPTIONS; i++) {
        int ticks = 1 + (rand() % 100);
        options[i].time_to_expiry = ticks / TICKS_PER_YEAR;
        options[i].expiry_tick = tick + ticks; // set once, here, at true generation time
        options[i].strike_price = underlying->price * (0.8f + 0.4f * ((float)rand() / RAND_MAX));
        options[i].theoretical_price = fmaxf(0.0f, black_scholes(underlying->price, options[i].strike_price,
            options[i].time_to_expiry, underlying->drift, underlying->volatility));
        options[i].type = CALL; // TODO: generate puts too
        options[i].mask = 1.0f;

        float noise = noise_lower + ((float)rand() / RAND_MAX) * (noise_upper - noise_lower);
        options[i].price = fmaxf(0.0f, options[i].theoretical_price * (1.0f + noise));
        options[i].theoretical_price -= options[i].price; // DEBUG: store the edge in the theoretical_price field for reward shaping
    }
    return options;
}

// Required function
void c_step(Trading* env) {
    env->terminals[0] = 0;
    int option_info_size = 2*MAX_OPTIONS*OPTION_FEATURES;
    float cash = env->observations[option_info_size + 0];

    // one option purchase per env step, but time doesn't advance until no-op is selected.
    int choice = env->actions[0];
    float total_reward = 0.0f;
    if (choice >= 0 && choice < MAX_OPTIONS) {
        bool buyable = env->pending_market[choice].mask && env->pending_market[choice].expiry_tick <= MAX_TICKS;
        if (buyable) {
            float edge = env->pending_market[choice].theoretical_price; 
            cash += edge;
            // theoretical_price is currently hardcoded into the model and gives reward instantly, instead of at expiry.
            // this is just for testing and is obviously unrealistic. delayed rewards problem go brr.
            total_reward += edge;
            env->tick_spend += env->pending_market[choice].price;
            env->pending_market[choice].mask = 0; // no longer buyable
            // zero everything
            env->pending_market[choice].time_to_expiry = 0.0f;
            env->pending_market[choice].strike_price = 0.0f;
            env->pending_market[choice].price = 0.0f;
            env->pending_market[choice].theoretical_price = 0.0f;
            env->pending_market[choice].type = CALL;
            env->pending_market[choice].expiry_tick = 0;
        } else {
            /*if (!env->pending_market[choice].mask) {
                total_reward -= 2.0f;
            }*/
            // end the trading tick early, decrease reward by 5 to penalize 
            choice = MAX_OPTIONS;
        }
    }

    //bool advance_market = (choice == MAX_OPTIONS);
    bool advance_market = true;

    // advance tick if choice is the final no-op action, moving to the next time interval
    if (advance_market) {
        total_reward -= fmaxf(0.0f, env->min_tick_spend - env->tick_spend);
        env->tick += 1;

        env->underlying.price = gbm_step(env->underlying.price,
            env->underlying.drift, env->underlying.volatility, 1.0f/TICKS_PER_YEAR);

        env->rewards[0] = total_reward;
        env->episode_reward += total_reward;

        if (env->tick >= MAX_TICKS) {
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
            cash = env->observations[option_info_size + 0];
        }

        // amount spent on tick is reset
        env->tick_spend = 0.0f;
        
        // generate the next market 
        memcpy(env->pending_market, gen_test_options(&env->underlying, env->tick,
            env->market_noise_lower, env->market_noise_upper), MAX_OPTIONS * sizeof(Option));
            
    }

    env->step += 1;

    store_options(env->observations + MAX_OPTIONS*OPTION_FEATURES, env->pending_market, MAX_OPTIONS);

    Option top_k[MAX_OPTIONS] = {0};
    heap_top_k(&env->owned, MAX_OPTIONS, top_k);
    store_options(env->observations, top_k, MAX_OPTIONS);

    env->observations[option_info_size + 0] = cash;
    env->observations[option_info_size + 2] = (float)env->tick;
    env->observations[option_info_size + 4] = (float)env->owned.count;
}

// Required function. Should handle creating the client on first call
void c_render(Trading* env) {
    
    /*
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "PufferLib Trading");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int px = 64;
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int tex = env->observations[i*env->size + j];
            if (tex == EMPTY) {
                continue;
            }
            Color color = (tex == AGENT) ? (Color){0, 187, 187, 255} : (Color){187, 0, 0, 255};
            DrawRectangle(j*px, i*px, px, px, color);
        }
    }

    EndDrawing();*/
    #warning "Trading env render not implemented yet"
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Trading* env) {
    free(env->owned.data);
    env->owned.data = NULL;

    if (IsWindowReady()) {
        CloseWindow();
    }
}
