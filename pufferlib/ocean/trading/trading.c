// Local smoke test: random buy actions, no NN, no human control yet (render
// isn't implemented -- see the #warning in c_render).

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "trading.h"

#define DEMO_STEPS 1999
#define LOG_STEPS 100
#define BUY_PROBABILITY 0.05f

float demo() {
    Trading env = {0};
    allocate_all(&env);
    c_reset(&env);

    for (int step = 0; step < DEMO_STEPS; step++) {
        // c_step already ignores actions on masked (invalid) market slots.
        for (int i = 0; i < MAX_OPTIONS; i++) {
            env.actions[i] = ((float)rand() / RAND_MAX) < BUY_PROBABILITY;
        }

        c_step(&env);

        float cash = env.observations[2*MAX_OPTIONS*OPTION_FEATURES + 0];
        
        /*if (step % LOG_STEPS == 0 || step == DEMO_STEPS - 1) {
            printf("step=%d tick=%d cash=%.2f owned=%d underlying=%.2f\n", step, env.tick, cash, env.owned.count, env.underlying.price);
        }*/

        if (env.terminals[0]) {
            c_reset(&env);
        }
    }
    printf("Demo finished. PnL: %.2f\n", env.observations[2*MAX_OPTIONS*OPTION_FEATURES + 0] - 10000);
    free_all(&env);
    // returns pnl
    return env.observations[2*MAX_OPTIONS*OPTION_FEATURES + 0] - 10000;
}

int main() {
    float pnl_sum = 0;
    float pnl_sq_sum = 0;
    int tries = 100;
    for (int i = 0; i < tries; i++) {
        srand(clock());
        float pnl = demo();
        pnl_sum += pnl;
        pnl_sq_sum += pnl * pnl;
    }

    float mean = pnl_sum / tries;
    // Sample variance (n-1) of individual trial PnLs. The std dev of the
    // *average* (standard error of the mean) shrinks as 1/sqrt(n) -- it is
    // not the same number as the spread of a single trial's PnL.
    float sample_var = (pnl_sq_sum - tries * mean * mean) / (tries - 1);
    float sem = sqrtf(sample_var / tries);

    printf("Average PnL over %d demos: %.2f\n", tries, mean);
    printf("Std dev of a single trial's PnL: %.2f\n", sqrtf(sample_var));
    printf("Std error of the mean PnL: %.2f\n", sem);
    return 0;
    /*
    Average PnL over 10000 demos: 2.88
    Std dev of a single trial's PnL: 2118.54
    Std error of the mean PnL: 21.19
    */
}
