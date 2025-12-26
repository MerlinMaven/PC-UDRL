import os
import sys
import gymnasium as gym
import numpy as np
import torch
import h5py
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import make_config

def train_expert(env_id, save_path):
    print(f"Training PPO Expert on {env_id}...")
    env = gym.make(env_id)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300000) # Should be enough for LunarLander
    model.save(save_path)
    print("Expert trained and saved.")
    return model

def collect_trajectories(env, model, num_steps, epsilon=0.0):
    trajectories = {"observations": [], "actions": [], "rewards": [], "terminals": []}
    obs, _ = env.reset()
    collected_steps = 0
    
    while collected_steps < num_steps:
        # Action selection
        if model is None: # Random
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            # Add noise if not perfectly expert
            if epsilon > 0:
                action += np.random.normal(0, epsilon, size=action.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        trajectories["observations"].append(obs)
        trajectories["actions"].append(action)
        trajectories["rewards"].append(reward)
        trajectories["terminals"].append(terminated) # True terminal only

        obs = next_obs
        collected_steps += 1

        if done:
            obs, _ = env.reset()
            
    return trajectories

def generate_mixed_dataset(cfg):
    data_dir = os.path.join(cfg.dataset_dir, "phase2")
    os.makedirs(data_dir, exist_ok=True)
    expert_path = os.path.join(data_dir, "ppo_expert.zip")
    dataset_path = os.path.join(data_dir, "dataset.h5")

    env = gym.make(cfg.env_id)

    # 1. Train or Load Expert
    if os.path.exists(expert_path):
        print("Loading existing expert...")
        expert_model = PPO.load(expert_path)
    else:
        expert_model = train_expert(cfg.env_id, expert_path)

    # Validate Expert
    mean_reward, std_reward = evaluate_policy(expert_model, env, n_eval_episodes=10)
    print(f"Expert Mean Reward: {mean_reward} +/- {std_reward}")

    # 2. Collect Data
    total_steps = 100000
    ratios = {"expert": 0.5, "medium": 0.3, "random": 0.2}
    
    all_obs = []
    all_acts = []
    all_rews = []
    all_terms = []

    # A. Expert (Clean)
    n_expert = int(total_steps * ratios["expert"])
    print(f"Collecting {n_expert} expert steps...")
    traj = collect_trajectories(env, expert_model, n_expert, epsilon=0.0)
    all_obs.extend(traj["observations"])
    all_acts.extend(traj["actions"])
    all_rews.extend(traj["rewards"])
    all_terms.extend(traj["terminals"])

    # B. Medium (Noisy Expert)
    n_medium = int(total_steps * ratios["medium"])
    print(f"Collecting {n_medium} medium steps (noisy)...")
    traj = collect_trajectories(env, expert_model, n_medium, epsilon=0.3) # Heavy noise for "medium" behavior
    all_obs.extend(traj["observations"])
    all_acts.extend(traj["actions"])
    all_rews.extend(traj["rewards"])
    all_terms.extend(traj["terminals"])

    # C. Random
    n_random = int(total_steps * ratios["random"])
    print(f"Collecting {n_random} random steps...")
    traj = collect_trajectories(env, None, n_random)
    all_obs.extend(traj["observations"])
    all_acts.extend(traj["actions"])
    all_rews.extend(traj["rewards"])
    all_terms.extend(traj["terminals"])

    # 3. Save to HDF5 (Compatible with our OfflineDataset class)
    print(f"Saving dataset to {dataset_path}...")
    with h5py.File(dataset_path, "w") as f:
        f.create_dataset("observations", data=np.array(all_obs, dtype=np.float32))
        f.create_dataset("actions", data=np.array(all_acts, dtype=np.float32))
        f.create_dataset("rewards", data=np.array(all_rews, dtype=np.float32))
        f.create_dataset("terminals", data=np.array(all_terms, dtype=bool))
        # d3rlpy expects timeouts usually, but we stick to terminals for now
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    cfg = make_config(2) # Phase 2
    generate_mixed_dataset(cfg)
