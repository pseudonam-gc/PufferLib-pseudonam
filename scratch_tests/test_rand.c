#include "pufferlib/ocean/g2048/g2048.h"

// XorShift32
uint32_t x32 = 314159265;
uint32_t xorshift32() {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
    return x32;
}

#define rand xorshift32

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

    clock_t start = clock();
    for (int i=0; i<10000000; i++) {
        env.actions[0] = xorshift32() % 4;
        c_step(&env);
    }
    clock_t end = clock();
    printf("Time: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
