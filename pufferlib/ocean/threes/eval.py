from pufferlib import pufferl
from collections import Counter
import torch
import numpy as np
import random

def can_merge(a, b):
    """Check if two Threes tiles can merge."""
    if a == 0 or b == 0:
        return False
    # 1 + 2 = 3
    if (a == 1 and b == 2) or (a == 2 and b == 1):
        return True
    # Equal tiles >= 3 can merge
    if a >= 3 and a == b:
        return True
    return False


def get_legal_moves(grid):
    """Return list of legal move indices [0=UP, 1=DOWN, 2=LEFT, 3=RIGHT]."""
    board = np.array(grid).reshape(4, 4)
    legal = []

    # Check UP (0)
    for col in range(4):
        for row in range(1, 4):
            if board[row, col] != 0:
                if board[row-1, col] == 0 or can_merge(board[row, col], board[row-1, col]):
                    legal.append(0)
                    break
        if 0 in legal:
            break

    # Check DOWN (1)
    for col in range(4):
        for row in range(2, -1, -1):
            if board[row, col] != 0:
                if board[row+1, col] == 0 or can_merge(board[row, col], board[row+1, col]):
                    legal.append(1)
                    break
        if 1 in legal:
            break

    # Check LEFT (2)
    for row in range(4):
        for col in range(1, 4):
            if board[row, col] != 0:
                if board[row, col-1] == 0 or can_merge(board[row, col], board[row, col-1]):
                    legal.append(2)
                    break
        if 2 in legal:
            break

    # Check RIGHT (3)
    for row in range(4):
        for col in range(2, -1, -1):
            if board[row, col] != 0:
                if board[row, col+1] == 0 or can_merge(board[row, col], board[row, col+1]):
                    legal.append(3)
                    break
        if 3 in legal:
            break

    return legal

def evaluate_game(env, policy, device, tile_values, deterministic=True, verbose=False, seed=None, game_num=0, num_games=1):
    """Play a single game and return the highest tile index reached."""
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    obs, _ = env.reset(seed=random.randint(0, 1_000_000) if seed is None else seed)
    state = None
    max_tile_idx = 0
    step_count = 0

    while True:
        # Track max tile
        grid = obs[0, :16]
        current_max = int(max(grid))
        max_tile_idx = max(max_tile_idx, current_max)

        # Get legal moves
        legal_moves = get_legal_moves(grid)
        if not legal_moves:
            print ("!!!!!")
            # No legal moves = game over (shouldn't happen if terminal detection works)
            break

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            if hasattr(policy, 'lstm'):
                if state is None:
                    state = {
                        'lstm_h': torch.zeros(1, 1, policy.hidden_size, device=device),
                        'lstm_c': torch.zeros(1, 1, policy.hidden_size, device=device)
                    }
                logits, value = policy(obs_tensor, state)
            else:
                logits, value = policy(obs_tensor)

            # Mask illegal moves with -inf
            masked_logits = logits.clone()
            for i in range(4):
                if i not in legal_moves:
                    masked_logits[0, i] = float('-inf')

            if deterministic:
                action = torch.argmax(masked_logits, dim=-1).cpu().numpy()
            else:
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample().cpu().numpy()

        obs, reward, terminals, truncations, info = env.step(action)
        step_count += 1
        print(f"\rGame {game_num + 1}/{num_games}, Step {step_count}", end='', flush=True)


        if terminals[0] or truncations[0]:
            """
            if verbose:
                print("\nBoard:")
                for row in range(4):
                    row_vals = []
                    for col in range(4):
                        idx = int(grid[row * 4 + col])
                        val = tile_values[idx] if idx < len(tile_values) else 2 ** idx
                        row_vals.append(f"{val:>5}")
                    print(" ".join(row_vals))
            """

            # Check final board state too
            grid = obs[0, :16]
            current_max = int(max(grid))
            max_tile_idx = max(max_tile_idx, current_max)       
           

            return max_tile_idx, step_count

    # No legal moves left (game over)
    return max_tile_idx, step_count


def evaluate_n_games(env_name, load_model_path, num_games=100, device='cpu', verbose=False, deterministic=True):
    """Play num_games and record highest tile distribution."""
    from pufferlib.ocean.threes.threes import Threes

    # Create raw environment (not wrapped)
    env = Threes(num_envs=1, endgame_env_prob=0, scaffolding_ratio=0, can_go_over_65536=True)

    # Load policy
    args = pufferl.load_config(env_name)
    args['train']['device'] = device
    args['load_model_path'] = load_model_path
    args['vec']['num_envs'] = 1
    args['env']['num_envs'] = 1

    vecenv = pufferl.load_env(env_name, args)
    policy, _ = pufferl.load_policy(args, vecenv, env_name)
    policy.eval()
    vecenv.close()

    # Tile index to value mapping
    tile_values = [0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304, 196608]

    highest_tiles = []

    for game_num in range(num_games):
        max_tile_idx, steps = evaluate_game(env, policy, device, tile_values, deterministic, verbose, game_num=game_num, num_games=num_games)
        max_tile_value = tile_values[max_tile_idx] if max_tile_idx < len(tile_values) else 2 ** max_tile_idx
        highest_tiles.append(max_tile_value)
        print()  # newline after game ends

    env.close()

    # Count and sort results
    tile_counts = Counter(highest_tiles)

    print(f"\n{'='*50}")
    print(f"Results from {num_games} games:")
    print(f"{'='*50}")
    print(f"{'Highest Tile':<15} {'Count':<10} {'Percentage':<10}")
    print(f"{'-'*50}")

    for tile, count in sorted(tile_counts.items(), reverse=True):
        pct = count / num_games * 100
        print(f"{tile:<15} {count:<10} {pct:.1f}%")

    print(f"{'='*50}")

    return tile_counts

def legal_moves_batch(grids):
    """Vectorized get_legal_moves. grids: (N, 16) tile indices -> (N, 4) bool mask."""
    b = np.asarray(grids, dtype=np.int16).reshape(-1, 4, 4)

    def can_merge_arr(a, nb):
        return (a != 0) & (nb != 0) & (
            ((a == 1) & (nb == 2)) | ((a == 2) & (nb == 1)) | ((a >= 3) & (a == nb))
        )

    def movable(cur, nb):
        # cur slides into nb if nb is empty or the two merge
        ok = (cur != 0) & ((nb == 0) | can_merge_arr(cur, nb))
        return ok.reshape(len(b), -1).any(axis=1)

    return np.stack([
        movable(b[:, 1:, :], b[:, :-1, :]),  # UP
        movable(b[:, :-1, :], b[:, 1:, :]),  # DOWN
        movable(b[:, :, 1:], b[:, :, :-1]),  # LEFT
        movable(b[:, :, :-1], b[:, :, 1:]),  # RIGHT
    ], axis=1)


def evaluate_n_games_vectorized(env_name, load_model_path, num_games=1000, num_envs=256,
        device='cpu', deterministic=True, seed=None, warmup_games=0):
    """Same measurement as evaluate_n_games, but plays num_envs games concurrently.

    The C env auto-resets the instant a game terminates, so the observation returned
    alongside terminals[i] already belongs to a *new* game. Every counted game is
    therefore bounded explicitly: its running max and LSTM state are zeroed at that
    reset boundary, so no partial or post-reset game contributes to the distribution.

    Each env is given a fixed quota of games rather than stopping at the first
    num_games terminals globally. Stopping globally would bias the distribution badly:
    the first games to finish are the *shortest* ones, so short/early-death games would
    be over-sampled and the long games still in flight would be censored out.

    warmup_games plays throwaway games per env before counting. c_reset does NOT clear
    lifetime_max_tile, and max_episode_ticks is derived from it (threes.h:663), so an
    env's first game runs under a 1000-tick budget while a warm env gets ~6x that.
    Measured at 1000 games, cold and warm are indistinguishable (~21% >=6144 either way):
    max_episode_ticks is max(BASE*mult, score/4), and score/4 dominates for a competent
    agent, so the tick cap never binds. Default 0; raise it to rule the effect out by
    construction if a weaker policy is ever evaluated here.
    """
    from pufferlib.ocean.threes.threes import Threes

    num_envs = min(num_envs, num_games)
    env = Threes(num_envs=num_envs, endgame_env_prob=0, scaffolding_ratio=0,
        can_go_over_65536=True)

    # Per-env game quota; counting the first quota[i] games of env i is unbiased
    quota = np.full(num_envs, num_games // num_envs, dtype=np.int64)
    quota[:num_games % num_envs] += 1
    counted = np.zeros(num_envs, dtype=np.int64)
    warmed = np.zeros(num_envs, dtype=np.int64)

    args = pufferl.load_config(env_name)
    args['train']['device'] = device
    args['load_model_path'] = load_model_path
    args['vec']['num_envs'] = 1
    args['env']['num_envs'] = 1

    vecenv = pufferl.load_env(env_name, args)
    policy, _ = pufferl.load_policy(args, vecenv, env_name)
    policy.eval()
    vecenv.close()

    tile_values = [0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304, 196608]

    if seed is None:
        seed = random.randint(0, 1_000_000)
    obs, _ = env.reset(seed=seed)
    state = {
        'lstm_h': torch.zeros(1, num_envs, policy.hidden_size, device=device),
        'lstm_c': torch.zeros(1, num_envs, policy.hidden_size, device=device),
    }

    running_max = np.zeros(num_envs, dtype=np.int64)
    game_steps = np.zeros(num_envs, dtype=np.int64)
    highest_tiles = []
    counted_steps = []
    steps = 0

    # Envs that hit their quota keep stepping (the C env steps all envs together),
    # but their surplus games are discarded rather than counted.
    while (counted < quota).any():
        grid = np.asarray(obs[:, :16])
        running_max = np.maximum(running_max, grid.max(axis=1))

        legal = legal_moves_batch(grid)
        # A dead board can't appear (C resets on game over), but never feed all -inf
        legal[~legal.any(axis=1)] = True

        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
            logits, _ = policy(obs_t, state)
            logits = logits.masked_fill(~torch.as_tensor(legal, device=device), float('-inf'))
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = torch.distributions.Categorical(logits=logits).sample()
            actions = actions.cpu().numpy().astype(np.int32)

        obs, reward, terminals, truncations, info = env.step(actions)
        steps += 1
        game_steps += 1

        done = np.asarray(terminals).astype(bool) | np.asarray(truncations).astype(bool)
        if done.any():
            for idx in np.flatnonzero(done):
                if warmed[idx] < warmup_games:
                    warmed[idx] += 1  # throwaway: only here to warm lifetime_max_tile
                elif counted[idx] < quota[idx]:
                    highest_tiles.append(tile_values[running_max[idx]])
                    counted_steps.append(int(game_steps[idx]))
                    counted[idx] += 1

            # The C env already reset these; clear the state that lives in Python
            done_t = torch.as_tensor(done, device=device)
            running_max[done] = 0
            game_steps[done] = 0
            state['lstm_h'][:, done_t] = 0
            state['lstm_c'][:, done_t] = 0
            print(f"\rGames {len(highest_tiles)}/{num_games}, Step {steps}", end='', flush=True)

    env.close()

    tile_counts = Counter(highest_tiles)

    print(f"\n{'='*50}")
    print(f"Results from {num_games} games (seed={seed}):")
    print(f"{'='*50}")
    print(f"{'Highest Tile':<15} {'Count':<10} {'Percentage':<10} {'+-1SE':<10}")
    print(f"{'-'*50}")

    for tile, count in sorted(tile_counts.items(), reverse=True):
        p = count / num_games
        se = 100 * (p * (1 - p) / num_games) ** 0.5
        print(f"{tile:<15} {count:<10} {100*p:.1f}%{'':<6} {se:.1f}")

    print(f"{'='*50}")
    print(f"Game length: mean {np.mean(counted_steps):.0f}, max {np.max(counted_steps)} steps")

    return tile_counts

def finetune(env_name, load_model_path):
    args = pufferl.load_config(env_name)
    args['load_model_path'] = load_model_path
    # args['env']['use_sparse_reward'] = True
    # args['env']['scaffolding_ratio'] = 0.85

    # args['policy']['hidden_size'] = 512
    # args['rnn']['input_size'] = 512
    # args['rnn']['hidden_size'] = 512

    args['train']['total_timesteps'] = 1_000_000_000
    args['train']['learning_rate'] = 0.00005
    args['train']['anneal_lr'] = False

    args['wandb'] = True
    args['tag'] = 'pg2048'

    pufferl.train(env_name, args)

if __name__ == '__main__':
    model_path = 'experiments/puffer_threes_nnm6a1s9.pt'
    evaluate_n_games('puffer_threes', load_model_path=model_path, num_games=1000, device='mps', verbose=True)
