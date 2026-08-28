#include "pufferlib/ocean/g2048/g2048.h"
#include <sys/time.h>

long long time_in_micros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main() {
    Game env = {0};
    unsigned char obs[289];
    int actions[1] = {0};
    float rewards[1] = {0};
    unsigned char terms[1] = {0};
    
    env.observations = obs;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terms;
    env.can_go_over_65536 = true;

    init(&env);
    c_reset(&env);

    long long time_move = 0;
    long long time_stats = 0;
    long long time_obs = 0;
    long long time_is_over = 0;
    long long time_other = 0;

    for (int i=0; i<10000000; i++) {
        env.actions[0] = rand() % 4;
        
        long long t0 = time_in_micros();
        float reward = 0.0f;
        float score_add = 0.0f;
        unsigned char prev_max_tile = env.max_tile;
        bool did_move = move(&env, env.actions[0] + 1, &reward, &score_add);
        env.tick++;
        long long t1 = time_in_micros();
        time_move += (t1 - t0);

        if (did_move) {
            env.moves_made++;
            place_tile_at_random_cell(&env, get_new_tile());
            env.score += score_add;

            long long t2 = time_in_micros();
            reward += update_stats_and_get_heuristic_rewards(&env);
            long long t3 = time_in_micros();
            time_stats += (t3 - t2);

            reward *= env.reward_scaler;

            long long t4 = time_in_micros();
            update_observations(&env);
            long long t5 = time_in_micros();
            time_obs += (t5 - t4);
            
            int tick_multiplier = max(1, env.lifetime_max_tile - 8);
            env.max_episode_ticks = max(BASE_MAX_TICKS * tick_multiplier, env.score / 4);
        } else {
            reward = INVALID_MOVE_PENALTY;
        }

        long long t6 = time_in_micros();
        bool game_over = is_game_over(&env);
        long long t7 = time_in_micros();
        time_is_over += (t7 - t6);

        bool max_ticks_reached = env.tick >= env.max_episode_ticks;
        bool max_level_reached = env.stop_at_65536 && env.max_tile >= 16;
        env.terminals[0] = (game_over || max_ticks_reached || max_level_reached) ? 1 : 0;

        if (game_over) reward = GAME_OVER_PENALTY;
        env.rewards[0] = reward;
        env.episode_reward += reward;

        long long t8 = time_in_micros();
        if (env.terminals[0]) {
            add_log(&env);
            c_reset(&env);
        }
        long long t9 = time_in_micros();
        time_other += (t9 - t8);
    }
    
    printf("Move: %lld us\n", time_move);
    printf("Stats: %lld us\n", time_stats);
    printf("Obs: %lld us\n", time_obs);
    printf("IsOver: %lld us\n", time_is_over);
    printf("Other (reset etc): %lld us\n", time_other);
    
    return 0;
}
