import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.trading import binding

# TODO: calculate relative to underlying values

# PufferEnv requires a flat Box observation_space and a Discrete/MultiDiscrete/Box
# action_space (see pufferlib.PufferEnv.__init__) -- Sequence/Dict spaces are not
# supported by the C buffer layer. Since the number of options available at any
# tick varies, we pick a fixed upper bound (MAX_OPTIONS) and pad/mask unused slots.
MAX_OPTIONS = 32

# current_options slot: [time_to_expiry, strike_price, price, type, mask]
CURRENT_OPTION_FEATURES = 5
# buyable_options slot: [time_to_expiry, strike_price, price, type, mask]
BUYABLE_OPTION_FEATURES = 5

# at the moment, current funds, underlying price, underlying volatility, 
GLOBAL_FEATURES = 5 

OBS_SIZE = MAX_OPTIONS * CURRENT_OPTION_FEATURES + MAX_OPTIONS * BUYABLE_OPTION_FEATURES + GLOBAL_FEATURES


class Trading(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='human', buf=None, seed=0):
        self.num_agents = num_envs
        self.render_mode = render_mode

        # Flat observation buffer: MAX_OPTIONS padded slots for currently held
        # options followed by MAX_OPTIONS padded slots for buyable options.
        # A trailing mask feature in each slot marks whether it is real (1) or
        # padding (0), since MAX_OPTIONS is an upper bound, not the true count.
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )

        # One buy/skip decision per buyable_options slot, aligned by index with
        # the buyable_options block of the observation. Actions on masked
        # (padding) slots are ignored on the C side.
        self.single_action_space = gymnasium.spaces.MultiDiscrete([2] * MAX_OPTIONS)

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_options=MAX_OPTIONS,
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed or 0)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = [binding.vec_log(self.c_envs)]

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=8192):
    """Benchmark environment performance."""
    num_envs = 4096
    env = Trading(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, num_envs, MAX_OPTIONS)).astype(np.int32)

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1
    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')

if __name__ == '__main__':
    test_performance()
