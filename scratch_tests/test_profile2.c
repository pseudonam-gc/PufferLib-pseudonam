#include "pufferlib/ocean/g2048/g2048.h"

// we will mock these out and see the drop
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
    return 0;
}
