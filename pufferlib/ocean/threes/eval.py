from pufferlib import pufferl
from collections import Counter
import torch
import numpy as np
import random

log_steps = 1000


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

        if verbose and step_count % log_steps == 0:
            #print(f"Step {step_count}, Max tile idx: {max_tile_idx}")
            print("\nBoard:")
            for row in range(4):
                row_vals = []
                for col in range(4):
                    idx = int(grid[row * 4 + col])
                    val = tile_values[idx] if idx < len(tile_values) else 2 ** idx
                    row_vals.append(f"{val:>5}")
                print(" ".join(row_vals))

        # Get legal moves
        legal_moves = get_legal_moves(grid)
        if not legal_moves:
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

        if verbose and step_count % log_steps == 0:
            probs = torch.softmax(logits, dim=-1)
            print(f"  Value: {value.item():.4f}")
            print(f"  Probs: {probs.cpu().numpy().flatten()}")
            print(f"  Action: {action_names[action[0]]}")

        obs, reward, terminals, truncations, info = env.step(action)
        step_count += 1
        print(f"\rGame {game_num + 1}/{num_games}, Step {step_count}", end='', flush=True)

        if terminals[0] or truncations[0]:
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

def finetune(env_name, load_model_path):
    args = pufferl.load_config(env_name)
    args['load_model_path'] = load_model_path
    # args['env']['use_sparse_reward'] = True
    args['env']['scaffolding_ratio'] = 0.85

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
    model_path = 'latest'
    evaluate_n_games('puffer_threes', load_model_path=model_path, num_games=50, device='mps', verbose=True)
