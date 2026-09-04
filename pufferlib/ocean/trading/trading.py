import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.trading import binding
from pufferlib.ocean.torch import _load_trading_obs_stats

# PufferEnv requires a flat Box observation_space and a Discrete/MultiDiscrete/Box
# action_space (see pufferlib.PufferEnv.__init__) -- Sequence/Dict spaces are not
# supported by the C buffer layer. Since the number of options available at any
# tick varies, we pick a fixed upper bound (MAX_OPTIONS) and pad/mask unused slots.
MAX_OPTIONS = 32

# current_options slot: [time_to_expiry, strike_price, price, theoretical_price, type, mask]
# theoretical_price (the pre-noise Black-Scholes fair value) is a DEBUG
# feature -- see Option in utils.h -- handed directly to the model so buying
# "when price < theoretical_price" doesn't require learning Black-Scholes
# implicitly first.
CURRENT_OPTION_FEATURES = 6
# buyable_options slot: [time_to_expiry, strike_price, price, theoretical_price, type, mask]
BUYABLE_OPTION_FEATURES = 6

# at the moment, current funds, underlying price, underlying volatility, 
GLOBAL_FEATURES = 5 

OBS_SIZE = MAX_OPTIONS * CURRENT_OPTION_FEATURES + MAX_OPTIONS * BUYABLE_OPTION_FEATURES + GLOBAL_FEATURES


class Trading(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='human', buf=None, seed=0, min_tick_spend=0,
            market_noise_lower=-0.02, market_noise_upper=0.1):
        self.num_agents = num_envs
        self.render_mode = render_mode

        # Flat observation buffer: MAX_OPTIONS padded slots for currently held
        # options followed by MAX_OPTIONS padded slots for buyable options.
        # A trailing mask feature in each slot marks whether it is real (1) or
        # padding (0), since MAX_OPTIONS is an upper bound, not the true count.
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )

        # A single choice per tick: buy market slot i (0..MAX_OPTIONS-1), or
        # MAX_OPTIONS itself meaning "buy nothing". Exactly one action, so
        # exactly one purchase (or none) happens per tick by construction --
        # no other slot's unused "would have bought" intent shares credit
        # with whatever this choice's outcome turns out to be.
        self.single_action_space = gymnasium.spaces.Discrete(MAX_OPTIONS + 1)

        super().__init__(buf)

        self.stats = _load_trading_obs_stats()

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            max_options=MAX_OPTIONS,
            min_tick_spend=min_tick_spend,
            market_noise_lower=market_noise_lower,
            market_noise_upper=market_noise_upper,
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
            self.rewards / self.stats['reward_std'],  # normalize reward to ~1.0 std
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

    actions = np.random.randint(0, MAX_OPTIONS + 1, (atn_cache, num_envs)).astype(np.int32)

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
