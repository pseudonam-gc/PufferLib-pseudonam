#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }

#define SIZE 4
#define EMPTY 0
#define UP 1
#define DOWN 2
#define LEFT 3
#define RIGHT 4
#define BASE_MAX_TICKS 1000

// These work well
#define MERGE_REWARD_WEIGHT 0.0625f
#define INVALID_MOVE_PENALTY -0.05f
#define GAME_OVER_PENALTY -1.0f

// These may need experimenting, but work for now
#define STATE_REWARD_WEIGHT 0.01f // Fixed, small reward for maintaining "desirable" states
#define MONOTONICITY_REWARD_WEIGHT 0.00003f

// Features: 18 per cell
// 1. Normalized tile value (current_val / max_val)
// 2. One-hot for empty (1 if empty, 0 if occupied)
// 3-18. One-hot for tile values 2^1 to 2^16 (16 features)
#define NUM_FEATURES 18

typedef struct {
    float perf;
    float score;
    float merge_score;
    float episode_return;
    float episode_length;
    float lifetime_max_tile;
    float reached_32768;
    float reached_65536;
    float snake_state;
    float monotonicity_reward;
    float snake_reward;
    float n;
} Log;

typedef struct {
    Log log;                        // Required
    unsigned char* observations;    // Cheaper in memory if encoded in uint_8
    int* actions;                   // Required
    float* rewards;                 // Required
    unsigned char* terminals;       // Required

    bool can_go_over_65536;         // Set false for training, true for eval
    float reward_scaler;            // Pufferlib clips rew from -1 to 1, adjust the resulting rew accordingly

    float endgame_env_prob;         // The prob of env being initialized as an endgame-only env
    bool is_endgame_env;
    float scaffolding_ratio;        // The ratio for "scaffolding" runs, in which higher blocks are spawned
    bool is_scaffolding_episode;
    bool use_heuristic_rewards;
    float snake_reward_weight;
    bool use_sparse_reward;         // Ignore all rewards and provide 1 for reaching 16k, 32k, 65k

    int score;
    int tick;
    int bonus;
    unsigned char grid[SIZE][SIZE];
    unsigned char lifetime_max_tile;
    unsigned char max_tile;         // Episode max tile
    float episode_reward;           // Accumulate episode reward
    float monotonicity_reward;
    float snake_reward;
    int moves_made;
    int max_episode_ticks;          // Dynamic max_ticks based on score
    bool is_snake_state;
    int snake_state_tick;
    bool stop_at_65536;

    unsigned char bag[3];
    unsigned char next_box;
    unsigned char next_triplet;
    unsigned char valid_spawn_positions[16]; // Positions where new tiles can spawn

    // Cached values to avoid recomputation
    int empty_count;
    bool game_over_cached;
    bool grid_changed;
} Game;

// Precomputed color table for rendering optimization
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_BLACK = (Color){0, 0, 0, 255};
const Color PUFF_PURPLE = (Color){128, 0, 128, 255};

static Color tile_colors[4] = {
    {6, 24, 24, 255}, // Empty/background
    {163, 206, 220, 255}, // 1
    {255, 128, 128, 255}, // 2
    {220, 220, 220, 255}, // 3+
};

// Precomputed pow(x, 1.5) lookup table for x in [0, 19] to avoid expensive pow() calls.
static const unsigned char pow_1_5_lookup[20] = {
    0, 1, 2, 5, 8, 11, 14, 18, 22, 27, 31, 36, 41, 46, 52, 57, 64, 69, 75, 81
};

static const float piece_vals[20] = {
    0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304, 196608
};

static const float piece_scores[20] = {
    0, 0, 0, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163
};

static inline float calculate_perf(unsigned char max_tile) {
    // Goal is to reach 6144
    float perf = piece_vals[max_tile] / 6144.0f;
    if (perf > 1.0f) perf = 1.0f;
    return perf;
}

// --- Logging ---
void add_log(Game* game);

// --- Required functions for env_binding.h ---
void c_reset(Game* game);
void c_step(Game* game);
void c_render(Game* game);
void c_close(Game* game);

void init(Game* game) {
    game->lifetime_max_tile = 0;
    game->is_endgame_env = (rand() / (float)RAND_MAX) < game->endgame_env_prob;
}

void update_observations(Game* game) {
    // Observation layout: 24 bytes total
    // - 16 bytes: raw tile indices (0-16) for 4x4 grid
    // - 1 byte: next_box one-hot (0-16)
    // - 4 bytes: bag counts [bag[0], bag[1], bag[2], total]

    int num_cell = SIZE * SIZE;
    memset(game->observations, 0, (num_cell + 5) * sizeof(unsigned char));

    // Raw tile indices (0-16), capped at 16
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int idx = i * SIZE + j;
            game->observations[idx] = min(game->grid[i][j], 16);
        }
    }

    // Additional obs at offset 16
    int offset = num_cell;
    
    if (game->next_box >= 1 && game->next_box <= 3) {
        game->observations[offset] = game->next_box;
    } else {
        game->observations[offset] = min(game->next_triplet, 16);
    }
    // Bag counts
    game->observations[offset + 1] = game->bag[0];
    game->observations[offset + 2] = game->bag[1];
    game->observations[offset + 3] = game->bag[2];
    game->observations[offset + 4] = game->bag[0] + game->bag[1] + game->bag[2];

    // Next box one-hot
    // if next box is 1-3, set observation to that
    // if next box is a bonus, reveal next_triplet index instead
}

void add_log(Game* game) {
    // Scaffolding runs will distort stats, so skip logging
    if (game->is_endgame_env || game->is_scaffolding_episode) return;

    // Update the lifetime best
    if (game->max_tile > game->lifetime_max_tile) {
        game->lifetime_max_tile = game->max_tile;
    }
    
    game->log.score += piece_scores[game->max_tile];
    game->log.perf += calculate_perf(game->max_tile);
    game->log.merge_score += (float)game->score;
    game->log.episode_length += game->tick;
    game->log.episode_return += game->episode_reward;
    game->log.lifetime_max_tile += piece_vals[game->lifetime_max_tile];
    game->log.reached_32768 += (game->max_tile >= 15);
    game->log.reached_65536 += (game->max_tile >= 16);
    game->log.snake_state += (float)game->snake_state_tick / (float)game->tick;
    game->log.monotonicity_reward += game->monotonicity_reward * MONOTONICITY_REWARD_WEIGHT * game->reward_scaler;
    game->log.snake_reward += game->snake_reward * game->snake_reward_weight * game->reward_scaler;
    game->log.n += 1;
}

static inline unsigned char draw_bonus_tile(Game* game) {
    // Max bonus is max_tile / 8, which in index terms is max_tile - 3
    // Min bonus is always index 4 (value 6)
    unsigned char max_bonus_idx = game->max_tile - 3;
    unsigned char min_bonus_idx = 4;
    int num_bonuses = max_bonus_idx - min_bonus_idx + 1;

    if (num_bonuses <= 1) {
        // Only one possible bonus (6)
        game -> next_triplet = min_bonus_idx;
        return min_bonus_idx;
    } else if (num_bonuses == 2) {
        // Two possible bonuses (6, 12), uniform
        game -> next_triplet = min_bonus_idx;
        return min_bonus_idx + (rand() % 2);
    } else {
        // Four or more: use triplet system
        // Pick a random triplet, then pick uniformly within it
        int num_triplets = num_bonuses - 2;
        int triplet = rand() % num_triplets;
        int pos_in_triplet = rand() % 3;
        // Store the triplet for observation
        game->next_triplet = triplet + min_bonus_idx;
        return min_bonus_idx + triplet + pos_in_triplet;
    }
}

static inline void draw_from_bag(Game* game) {
    if ((game->moves_made % 21) == game->bonus && game->max_tile > 6) {
        // draw a bonus tile using proper probability weighting
        game->next_box = draw_bonus_tile(game);
    } else {
        // draw tile from bag, put in next_box
        int bag_size = game->bag[0] + game->bag[1] + game->bag[2];
        int r = rand() % bag_size;
        if (r < game->bag[0]) {
            game->next_box = 1;
            game->bag[0]--;
        } else if (r < game->bag[0] + game->bag[1]) {
            game->next_box = 2;
            game->bag[1]--;
        } else {
            game->next_box = 3;
            game->bag[2]--;
        }
        if (bag_size == 1) {
            // refill bag
            game->bag[0] = 4;
            game->bag[1] = 4;
            game->bag[2] = 4;
        }
    }
}

static inline void place_tile_at_random_cell(Game* game, unsigned char tile) {
    if (game->empty_count == 0) return;

    int target = rand() % game->empty_count;
    int pos = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->grid[i][j] == EMPTY) {
                if (pos == target) {
                    game->grid[i][j] = tile;
                    game->empty_count--;
                    return;
                }
                pos++;
            }
        }
    }
}

static inline void place_tile(Game* game, unsigned char tile) {
    // randomly place tiles on one of the cells such that:
    // 1) it appears on the opposite side of the last move 
    // 2) it appears on a line that has been moved
    
    int valid_spawn_positions_count = 0;
    for (int i = 0; i < SIZE * SIZE; i++) {
        if (game->valid_spawn_positions[i] == 1) {
            valid_spawn_positions_count++;
        }
    }
    
    int target = rand() % valid_spawn_positions_count;
    int pos = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (game->valid_spawn_positions[SIZE*i+j] == 0) continue;
            if (pos == target) {
                game->grid[i][j] = tile;
                game->empty_count--;
                return;
            }
            pos++;
        }
    }
}

void set_scaffolding_curriculum(Game* game) {
    game->stop_at_65536 = true;

    int preplaced_tiles = rand() % 4 + 3; // Preplace between 3 to 6 tiles
    // None of them can be ones or twos.

    int min_tile = max(3, game->lifetime_max_tile - 2);
    int max_tile = min(16, game->lifetime_max_tile);

    // Add to tiles array
    for (int i = 0; i < preplaced_tiles; i++) {
        unsigned char tile = (rand() % (max_tile - min_tile + 1)) + min_tile;
        place_tile_at_random_cell(game, tile);
        game->empty_count--;
    }
}

void set_endgame_curriculum(Game* game) {
    game->stop_at_65536 = true;
    int curriculum = rand() % 4;

    // Place the tiles in the second-third rows, so that they can be moved up in the first move
    unsigned char tiles[] = {15, 14, 13, 12};
    memcpy(game->grid[1], tiles, 4);
    game->empty_count -= 4;

    if (curriculum >= 1) { game->grid[2][3] = 11; game->empty_count--; }
    if (curriculum >= 2) { 
        game->grid[2][2] = 10;
        game->grid[2][1] = 9;
        game->grid[2][0] = 8;
        game->empty_count -= 3;
    }
}

void c_reset(Game* game) {
    memset(game->grid, EMPTY, SIZE * SIZE);
    game->score = 0;
    game->tick = 0;
    game->episode_reward = 0;
    game->empty_count = SIZE * SIZE;
    game->game_over_cached = false;
    game->grid_changed = true;
    game->moves_made = 0;
    game->max_episode_ticks = BASE_MAX_TICKS;
    game->max_tile = 0;
    game->snake_state_tick = 0;
    game->monotonicity_reward = 0;
    game->snake_reward = 0;
    game->is_snake_state = false;
    game->stop_at_65536 = game->can_go_over_65536;
    game->bag[0] = 4;
    game->bag[1] = 4;
    game->bag[2] = 4;
    game->next_triplet = 0; // Reset

    memset(game->valid_spawn_positions, 0, SIZE * SIZE * sizeof(unsigned char));
    // End game envs only do endgame curriculum
    if (game->is_endgame_env) {
        set_endgame_curriculum(game);
        
    } else {
        // Higher tiles are spawned in scaffolding episodes
        // Having high tiles saves moves to get there, allowing agents to experience it faster
        
        // disable scaffolding for now
        game->is_scaffolding_episode = (rand() / (float)RAND_MAX) < game->scaffolding_ratio;
        if (game->is_scaffolding_episode && game->lifetime_max_tile >= 6) {
            set_scaffolding_curriculum(game);
        } else {
            // Add nine random tiles at the start
            draw_from_bag(game);
            for (int i = 0; i < 9; i++) {
                place_tile_at_random_cell(game, game->next_box);
                game->score += (float)(piece_scores[game->next_box]);
                draw_from_bag(game);
            }
        }
    }

    update_observations(game);
}

// Optimized slide and merge with fewer memory operations
static inline bool slide_and_merge(Game* game, unsigned char* row, float* reward, float* score_increase) {
    bool moved = false;

    for (int i = 1; i < SIZE; i++) {
        if (row[i] != EMPTY) {
            // merge if the same and not 1+2
            if (row[i] == row[i-1] && row[i] >= 3) {
                *reward += ((float)row[i]) * MERGE_REWARD_WEIGHT;
                *score_increase += (float)(piece_scores[row[i]]);
                row[i-1]++;
                row[i] = EMPTY;
                moved = true;
                game->max_tile = max(game->max_tile, row[i-1]);
            }
            // 1+2 condition
            if ((row[i] == 1 && row[i-1] == 2) || (row[i] == 2 && row[i-1] == 1)) {
                *reward += 3.0f * MERGE_REWARD_WEIGHT;
                *score_increase += 3.0f;
                row[i-1] = 3;
                row[i] = EMPTY;
                moved = true;
            }
        }
        // move one tile left if possible 
        if (i > 0 && row[i-1] == EMPTY && row[i] != EMPTY) {  
            row[i-1] = row[i];
            row[i] = EMPTY;
            moved = true;
        }
    }
    
    return moved;
}

bool move(Game* game, int direction, float* reward, float* score_increase) {
    bool moved = false;
    unsigned char temp[SIZE];
    // reset valid_spawn_positions
    for (int i = 0; i < SIZE * SIZE; i++) {
        game->valid_spawn_positions[i] = 0;
    }

    if (direction == UP || direction == DOWN) {
        for (int col = 0; col < SIZE; col++) {
            // Extract column
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == UP) ? i : SIZE - 1 - i;
                temp[i] = game->grid[idx][col];
            }
            
            if (slide_and_merge(game, temp, reward, score_increase)) {
                moved = true;
                // Write back column
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == UP) ? i : SIZE - 1 - i;
                    game->grid[idx][col] = temp[i];
                }
                // Update valid_spawn_positions 
                if (direction == UP) {
                    game->valid_spawn_positions[SIZE*(SIZE-1)+col] = 1; // Bottom row
                } else {
                    game->valid_spawn_positions[col] = 1; // Top row
                }
            }
        }
    } else {
        for (int row = 0; row < SIZE; row++) {
            // Extract row
            for (int i = 0; i < SIZE; i++) {
                int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                temp[i] = game->grid[row][idx];
            }
            
            if (slide_and_merge(game, temp, reward, score_increase)) {
                moved = true;
                // Write back row
                for (int i = 0; i < SIZE; i++) {
                    int idx = (direction == LEFT) ? i : SIZE - 1 - i;
                    game->grid[row][idx] = temp[i];
                }
                // Update valid_spawn_positions
                if (direction == LEFT) {
                    game->valid_spawn_positions[SIZE*row+(SIZE-1)] = 1; // Rightmost column
                } else {
                    game->valid_spawn_positions[SIZE*row+0] = 1; // Leftmost column
                }
            }
        }
    }

    if (moved) {
        game->grid_changed = true;
        game->game_over_cached = false; // Invalidate cache
    }

    return moved;
}

bool is_game_over(Game* game) {
    // Use cached result if grid hasn't changed
    if (!game->grid_changed && game->game_over_cached) {
        return game->game_over_cached;
    }
    
    // Quick check: if there are empty cells, game is not over
    if (game->empty_count > 0) {
        game->game_over_cached = false;
        game->grid_changed = false;
        return false;
    }
    
    // Check for possible merges
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            unsigned char current = game->grid[i][j];
            if (i < SIZE - 1) {
                unsigned char below = game->grid[i + 1][j];
                // Same value merge (tiles >= 2) or 1+2 merge (sum == 3, no empties at this point)
                if ((current == below && current >= 3) || (current + below == 3)) {
                    game->game_over_cached = false;
                    game->grid_changed = false;
                    return false;
                }
            }
            if (j < SIZE - 1) {
                unsigned char right = game->grid[i][j + 1];
                if ((current == right && current >= 3) || (current + right == 3)) {
                    game->game_over_cached = false;
                    game->grid_changed = false;
                    return false;
                }
            }
        }
    }
    
    game->game_over_cached = true;
    game->grid_changed = false;
    return true;
}

// Combined grid stats and heuristic calculation for performance
float update_stats_and_get_heuristic_rewards(Game* game) {
    int empty_count = 0;
    int top_row_count = 0;
    unsigned char max_tile = 0;
    unsigned char second_max_tile = 0;
    unsigned char max_tile_in_row234 = 0;
    float heuristic_state_reward = 0.0f;
    float monotonicity_reward = 0.0f;
    float snake_reward = 0.0f;
    game->is_snake_state = false;
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            unsigned char val = game->grid[i][j];
            
            // Update empty count and max tile
            if (val == EMPTY) empty_count++;

            // Count filled cells in the top row
            if (i == 0 && val != EMPTY) top_row_count++;
            
            // Allow max and the second max tile to be the same
            if (val >= max_tile) {
                second_max_tile = max_tile;
                max_tile = val;
            } else if (val > second_max_tile && val < max_tile) {
                second_max_tile = val;
            }

            // Get the max tile in the 2nd, 3rd, 4th row
            if (i > 0 && val > max_tile_in_row234) max_tile_in_row234 = val;
        }
    }

    game->empty_count = empty_count;
    game->max_tile = max_tile;

    /* Heuristic rewards */

    // Filled top row reward: A simple nudge to keep the top row filled
    if (top_row_count == SIZE) heuristic_state_reward += STATE_REWARD_WEIGHT;

    bool max_in_top_left = (game->grid[0][0] == max_tile);

    // Corner reward: A simple nudge to keep the max tiles horizontally in the top row, left corner.
    // When agents learn to put the max tile on the other corners, or put max tiles vertically
    // they miss out snake rew, and this does happen sometimes.
    if (max_in_top_left && game->grid[0][1] == second_max_tile && max_tile > 4) {
        heuristic_state_reward += STATE_REWARD_WEIGHT;
    }

    // Snake reward: look for the snake pattern, only when the max tile is at top left
    if (max_in_top_left) {
        monotonicity_reward += pow_1_5_lookup[max_tile];
        int evidence_for_snake = 0;

        for (int i = 0; i < 2; i++) {
            unsigned char row_min = 32;
            unsigned char next_row_max = 0;
            for (int j = 0; j < SIZE; j++) {
                unsigned char val = game->grid[i][j];

                // Check horizontal monotonicity (snake pattern) for top two rows only
                if (j < SIZE - 1) {
                    unsigned char next_col = game->grid[i][j+1];
                    if (val != EMPTY && next_col != EMPTY) {
                        // Row 0: Reward decreasing left to right, e.g., 12-11-10-9
                        if (i == 0 && val > next_col) {
                            monotonicity_reward += pow_1_5_lookup[next_col];
                            evidence_for_snake++;
                        }
                        // Row 1: Reward increasing left to right, e.g., 5-6-7-8
                        else if (i == 1 && val < next_col) {
                            monotonicity_reward += pow_1_5_lookup[val];
                        }
                    }
                }

                // Vertical monotonicity: give score after row scanning for min/max is done
                if (val != EMPTY && val < row_min) row_min = val;
                unsigned char next_row = game->grid[i+1][j];
                if (next_row != EMPTY && next_row > next_row_max) next_row_max = next_row;
                // // Small column-level vertical reward
                if (val != EMPTY && next_row != EMPTY && val > next_row) monotonicity_reward += next_row;
            }
            // Large row-level vertical reward
            if (i < 2 && row_min < 20 && next_row_max > 0 && row_min > next_row_max) {
                monotonicity_reward += 4 * pow_1_5_lookup[row_min];
                if (i == 0) evidence_for_snake++;
            }
        }

        // Snake bonus: sorted top row + the max_tile_in_row234 in the second row right
        // For example, top row: 14-13-12-11, second row: ()-()-()-10
        unsigned char snake_tail = game->grid[1][3];
        if (evidence_for_snake >= 4 && snake_tail == max_tile_in_row234) {
            game->is_snake_state = true;
            game->snake_state_tick++;
            snake_reward = snake_tail * snake_tail;
        }
    }

    // Trained models need game->is_snake_state as obs
    if (!game->use_heuristic_rewards) return 0.0f;

    game->monotonicity_reward += monotonicity_reward;
    game->snake_reward += snake_reward;
    
    return heuristic_state_reward + monotonicity_reward * MONOTONICITY_REWARD_WEIGHT + snake_reward * game->snake_reward_weight;
}

void c_step(Game* game) {
    float reward = 0.0f;
    float score_add = 0.0f;
    unsigned char prev_max_tile = game->max_tile;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;
    
    if (did_move) {
        // reset bonus every 21 steps
        if (game->moves_made % 21 == 0) {
            game->bonus = rand() % 21;
        }
        game->moves_made++;
        place_tile(game, game->next_box);
        score_add += (float)(piece_scores[game->next_box]);
        draw_from_bag(game);
        game->score += score_add;

        // Add heuristic rewards/penalties and update grid stats
        reward += update_stats_and_get_heuristic_rewards(game);
        reward *= game->reward_scaler;

        update_observations(game); // Observations only change if the grid changes
        
        // This is to limit infinite invalid moves during eval (happens for noob agents)
        // Don't need to be tight. Don't need to show to human player.
        int tick_multiplier = max(1, game->lifetime_max_tile - 8); // practically no limit for competent agent
        game->max_episode_ticks = max(BASE_MAX_TICKS * tick_multiplier, game->score / 4);

    } else {
        reward = INVALID_MOVE_PENALTY;
        // No need to update observations if the grid hasn't changed
    }

    bool game_over = is_game_over(game);
    bool max_ticks_reached = game->tick >= game->max_episode_ticks;
    bool max_level_reached = game->stop_at_65536 && game->max_tile >= 16;
    game->terminals[0] = (game_over || max_ticks_reached || max_level_reached) ? 1 : 0;

    // Game over penalty overrides other rewards
    if (game_over) {
        reward = GAME_OVER_PENALTY;
    }

    if (game->use_sparse_reward) {
        reward = 0; // Ignore all previous reward
        if (game->max_tile >= 14 && game->max_tile > prev_max_tile) reward = 1;
    }

    game->rewards[0] = reward;
    game->episode_reward += reward;

    if (game->terminals[0]) {
        add_log(game);
        c_reset(game);
    }
}

// Stepping for client/eval: no reward, no reset
void step_without_reset(Game* game) {
    float score_add = 0.0f;
    float reward = 0.0f;
    bool did_move = move(game, game->actions[0] + 1, &reward, &score_add);
    game->tick++;

    if (did_move) {
        game->moves_made++;
        place_tile(game, game->next_box);
        score_add += (float)(piece_scores[game->next_box]);
        draw_from_bag(game);
        game->score += score_add;

        update_stats_and_get_heuristic_rewards(game); // The reward is ignored.
        update_observations(game); // Observations only change if the grid changes
    }

    bool game_over = is_game_over(game);
    game->terminals[0] = (game_over) ? 1 : 0;
}

// Rendering optimizations
void c_render(Game* game) {
    static bool window_initialized = false;
    static char score_text[32];
    static const int px = 100;
    
    if (!window_initialized) {
        InitWindow(px * SIZE + 100, px * SIZE + 50, "Threes");
        SetTargetFPS(30); // Increased for smoother rendering
        window_initialized = true;
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw grid
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int val = game->grid[i][j];
            
            // Use precomputed colors
            int color_idx = min(val, 3); // Cap at index 3 for 3+
            Color color = tile_colors[color_idx];
            
            DrawRectangle(j * px, i * px, px - 5, px - 5, color);
            
            if (val > 0) {
                int display_val = piece_vals[val];
                // Pre-format text to avoid repeated formatting
                snprintf(score_text, sizeof(score_text), "%d", display_val);

                int font_size = 32;
                int x_offset = 20; // Default for 4-digit numbers
                if (display_val < 10) x_offset = 40; // 1-digit
                else if (display_val < 100) x_offset = 35; // 2-digit
                else if (display_val < 1000) x_offset = 25; // 3-digit
                else if (display_val < 10000) x_offset = 15; // 4-digit
                else if (display_val < 100000) x_offset = 2; // 5-digit
                else {
                    font_size = 24;
                    x_offset = 5;
                }

                Color text_color;

                if (val <= 2) {
                    text_color = PUFF_WHITE;
                } else if (val == 3) {
                    text_color = PUFF_BLACK;
                } else {
                    text_color = PUFF_RED;
                }
                DrawText(score_text, j * px + x_offset, i * px + 34, font_size, text_color);
            }
        }
    }
    
    // Draw score (format once per frame)
    snprintf(score_text, sizeof(score_text), "Score: %d", game->score);
    DrawText(score_text, 10, px * SIZE + 10, 24, PUFF_WHITE);

    snprintf(score_text, sizeof(score_text), "Moves: %d", game->moves_made);
    DrawText(score_text, 210, px * SIZE + 10, 24, PUFF_WHITE);
    
    // Draw next boxes
    int val = game->next_box;
    DrawText("Next:", px * SIZE + 15, 50, 24, PUFF_WHITE);
    
    int font_size = 32;
    int x_offset = 40; // Default for 4-digit numbers
    if (val <= 3) {
        int display_val = piece_vals[game->next_box];
        snprintf(score_text, sizeof(score_text), "%d", display_val);
        DrawRectangle(px * SIZE + 20, 80, px - 40, px - 40, tile_colors[min(game->next_box, 3)]);

            
        Color text_color;

        if (val <= 2) {
            text_color = PUFF_WHITE;
        } else {
            text_color = PUFF_BLACK;
        } 
        
        snprintf(score_text, sizeof(score_text), "%d", display_val);
        DrawText(score_text, px * SIZE + 5 + x_offset, 80 + 20, font_size, text_color);

    } else {
        // display all three bonus options, val, val+1, val+2
        for (int i = 0; i < 3; i++) {
            if (val + i > 16) break; // No bonus beyond 65536
            
            
            if (game->next_triplet+i > game->max_tile - 3) {
                break; // No bonus beyond max_tile / 8
            }
            
            int display_val = piece_vals[game->next_triplet+i];//piece_vals[val + i];
            //printf("%d",game->next_triplet+i);

            if (display_val < 10) x_offset = 40;
            else if (display_val < 100) x_offset = 35;
            else if (display_val < 1000) x_offset = 25;
            else if (display_val < 10000) x_offset = 15;
            else if (display_val < 100000) x_offset = 2;
            else {
                x_offset = 5;
            }   
            
            Color text_color = PUFF_RED;
            font_size = 24;
            snprintf(score_text, sizeof(score_text), "%d", display_val);
            DrawRectangle(px * SIZE + 20, 80 + i * (px-40 + 10), px - 40, px - 40, PUFF_PURPLE);
            DrawText(score_text, px * SIZE + 5 + x_offset, 80 + i * (px - 40 + 10) + 20, font_size, text_color);
        }
    }

    EndDrawing();
}

void c_close(Game* game) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
