import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import make_config
from pc_udrl.utils import ensure_dirs
from pc_udrl.envs.gridworld import GridWorld
from pc_udrl.envs.gym_wrappers import make_env
from pc_udrl.data.dataset import OfflineDataset


def train_expert(env_id: str, save_path: str, steps: int = 50000):
    print(f"Training expert PPO on {env_id} for {steps} steps...")
    import gymnasium
    # Use native gymnasium for SB3
    env = make_vec_env(lambda: gymnasium.make(env_id), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    model.save(save_path)
    print(f"Expert saved to {save_path}")
    return model

def solve_gridworld_bfs(env, start_node):
    # start_node is (y, x)
    # env has env.walls (list of tuples), env.goal (np array)
    from collections import deque
    
    queue = deque([start_node])
    visited = {start_node}
    parent = {start_node: None}
    goal = (int(env.goal[0]), int(env.goal[1]))
    
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        
        y, x = current
        # Actions: 0:Up, 1:Down, 2:Left, 3:Right
        # Logic from step function:
        # 0: max(0, y-1), 1: min(size-1, y+1), 2: max(0, x-1), 3: min(size-1, x+1)
        neighbors = []
        # Try all 4 actions
        for action in range(4):
            ny, nx = y, x
            if action == 0: ny = max(0, y - 1)
            elif action == 1: ny = min(env.size - 1, y + 1)
            elif action == 2: nx = max(0, x - 1)
            elif action == 3: nx = min(env.size - 1, x + 1)
            
            next_node = (ny, nx)
            # Check walls
            if next_node in env._walls_set:
                continue # Don't go into walls
            
            if next_node not in visited:
                visited.add(next_node)
                parent[next_node] = (current, action)
                queue.append(next_node)
                
    # Reconstruct path
    path_actions = []
    curr = goal
    if curr not in parent:
        # No path found (isolated goal?)
        return []
        
    while curr != start_node:
        prev_info = parent[curr]
        if prev_info is None: break
        prev_node, action = prev_info
        path_actions.append(action)
        curr = prev_node
        
    return path_actions[::-1] # Reverse to get start->goal


def main(phase: int, expert_prob_arg: float = None):
    cfg = make_config(phase)
    ensure_dirs(cfg)
    
    # Setup environment
    if cfg.env_id == "GridWorld":
        env = GridWorld(size=5, seed=cfg.seed)
    else:
        env = make_env(cfg.env_id, cfg.seed)
        
    dataset = OfflineDataset(cfg)
    
    # Special handling for LunarLander in Phase 2/3 (Expert Data)
    expert = None
    if "LunarLander" in cfg.env_id:
        expert_path = os.path.join(cfg.dataset_dir, f"expert_ppo_{cfg.env_id}")
        if os.path.exists(expert_path + ".zip"):
            print(f"Loading expert from {expert_path}...")
            expert = PPO.load(expert_path)
        else:
            expert = train_expert(cfg.env_id, expert_path)

    # Generate Loop with Mix Strategy
    print(f"Generating dataset for {cfg.env_id}...")
    episodes = []
    
    # Expert Mix Ratio: 50% expert, 50% random
    has_expert = (expert is not None) or (cfg.env_id == "GridWorld")
    if expert_prob_arg is not None:
        expert_prob = expert_prob_arg
    else:
        expert_prob = 0.5 if has_expert else 0.0
    
    print(f"Expert Probability: {expert_prob}")
    
    for ep_idx in range(cfg.episodes):
        s = env.reset()
        traj = {"obs": [], "actions": [], "rewards": [], "dones": []}
        
        # Decide if this entire episode is expert or random? 
        use_expert = np.random.rand() < expert_prob
        
        expert_actions_queue = []
        if use_expert and cfg.env_id == "GridWorld":
            # Pre-calculate BFS path
            expert_actions_queue = solve_gridworld_bfs(env, (int(s[0]), int(s[1])))
        
        for t in range(cfg.max_steps):
            if use_expert:
                if cfg.env_id == "GridWorld":
                    if expert_actions_queue:
                        action = expert_actions_queue.pop(0)
                    else:
                        action = env.sample_action() # Falback
                else:
                    action, _ = expert.predict(s, deterministic=False)
                    # PPO output might need shaping depending on env
                    if not isinstance(action, (int, np.integer)) and action.ndim == 0:
                         # sometimes scalar numpy array
                         pass 
            else:
                action = env.sample_action()
            
            o2, r, d, _ = env.step(action)
            traj["obs"].append(np.array(s, dtype=np.float32))
            traj["actions"].append(action)
            traj["rewards"].append(r)
            traj["dones"].append(float(d))
            s = o2
            if cfg.env_id == "GridWorld":
                if args.render and ep_idx < 5:
                    frame = env.render()
                    import matplotlib.pyplot as plt
                    if 'ax' not in locals():
                        plt.ion()
                        fig, ax = plt.subplots()
                    ax.clear()
                    ax.imshow(frame)
                    plt.pause(0.01)

            if d:
                break
        episodes.append(traj)
        if (ep_idx + 1) % 100 == 0:
            print(f"Generated {ep_idx + 1}/{cfg.episodes} episodes")

    dataset._save_episodes(episodes)
    print("Dataset generation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--expert_prob", type=float, default=None)
    parser.add_argument("--render", action="store_true", help="Visualize generation")
    args = parser.parse_args()
    main(args.phase, args.expert_prob)
