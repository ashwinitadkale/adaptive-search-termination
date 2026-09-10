"""
Evaluate the trained PPO policy against the RANDOM and HEURISTIC baselines
from sanity_check_no_deps.py, and plot how many searches it uses per episode.

Run after train_toy_ppo.py has produced ppo_toy_search_stop.zip:

    pip install gymnasium stable-baselines3 matplotlib
    python evaluate_toy_policy.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environment"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from toy_search_stop_env import ToySearchStopEnv


def run_trained_policy(model, n_episodes=2000, seed=0):
    env = ToySearchStopEnv()
    rewards, lengths, corrects = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(info["n_searches"])
        # a correct guess contributes +1 before step costs; recover it from sign
        corrects.append(total_reward + env.step_cost * info["n_searches"] > 0)
    return np.array(rewards), np.array(lengths), np.array(corrects)


def main():
    model = PPO.load(
        os.path.join(os.path.dirname(__file__), "..", "training", "ppo_toy_search_stop")
    )
    rewards, lengths, corrects = run_trained_policy(model)

    print(f"PPO policy | avg reward: {rewards.mean():+.3f} | "
          f"accuracy: {corrects.mean()*100:5.1f}% | "
          f"avg searches used: {lengths.mean():4.2f} / 10")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=range(0, 12), align="left", color="#1E2761", edgecolor="white")
    ax.set_xlabel("Searches used before stopping")
    ax.set_ylabel("Episode count")
    ax.set_title("PPO's learned search-length distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "search_length_distribution.png"), dpi=150)
    print("Saved plot to search_length_distribution.png")


if __name__ == "__main__":
    main()
