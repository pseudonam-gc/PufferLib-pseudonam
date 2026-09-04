'''Collects per-feature std/var of Trading's observations under a random
policy, and pickles them for use as normalization divisors (see
ocean/torch.py's Trading/TradingTransformer) instead of hand-picked constants.

Run: python -m pufferlib.ocean.trading.compute_obs_stats
'''
import os
import pickle

import numpy as np

from pufferlib.ocean.trading.trading import (
    Trading, MAX_OPTIONS, CURRENT_OPTION_FEATURES, GLOBAL_FEATURES)

STATS_PATH = os.path.join(os.path.dirname(__file__), 'obs_stats.pkl')


def collect_obs_stats(num_envs=256, steps=3000, buy_prob=0.1, seed=0):
    env = Trading(num_envs=num_envs, seed=seed)
    env.reset()

    n, f = MAX_OPTIONS, CURRENT_OPTION_FEATURES
    owned_samples = []
    market_samples = []
    global_samples = []
    reward_samples = []

    for _ in range(steps):
        #actions = (np.random.random((num_envs, MAX_OPTIONS)) < buy_prob).astype(np.int32)
        # always buy a random option
        actions = np.ones((num_envs), dtype=np.int32)
        obs, rew, term, trunc, info = env.step(actions)
        owned_samples.append(obs[:, :n*f].reshape(-1, f))
        market_samples.append(obs[:, n*f:2*n*f].reshape(-1, f))
        global_samples.append(obs[:, 2*n*f:2*n*f + GLOBAL_FEATURES])
        reward_samples.append(rew.reshape(-1, 1))

    owned_samples = np.concatenate(owned_samples, axis=0)
    market_samples = np.concatenate(market_samples, axis=0)
    global_samples = np.concatenate(global_samples, axis=0)
    reward_samples = np.concatenate(reward_samples, axis=0)

    def mean_std_var(samples):
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        # Some features are ~constant under the current sim (e.g. market's
        # mask/type are almost always 1.0/CALL, with a rare exception right
        # on an episode-reset tick) -- guard against a near-0 divisor.
        std = np.where(std < 1e-6, 1.0, std)
        var = std ** 2  # derive from the (possibly guarded) std, not independently
        return mean.astype(np.float32), std.astype(np.float32), var.astype(np.float32)

    owned_mean, owned_std, owned_var = mean_std_var(owned_samples)
    market_mean, market_std, market_var = mean_std_var(market_samples)
    global_mean, global_std, global_var = mean_std_var(global_samples)
    reward_mean, reward_std, reward_var = mean_std_var(reward_samples)

    return {
        'owned_mean': owned_mean, 'owned_std': owned_std, 'owned_var': owned_var,
        'market_mean': market_mean, 'market_std': market_std, 'market_var': market_var,
        'global_mean': global_mean, 'global_std': global_std, 'global_var': global_var,
        'reward_mean': reward_mean, 'reward_std': reward_std, 'reward_var': reward_var,
    }


if __name__ == '__main__':
    stats = collect_obs_stats()
    for k, v in stats.items():
        print(k, v)

    with open(STATS_PATH, 'wb') as fp:
        pickle.dump(stats, fp)
    print('Saved to', STATS_PATH)
