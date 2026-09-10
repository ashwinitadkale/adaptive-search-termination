"""
Train PPO on ToySearchStopEnv using Stable-Baselines3.

Run locally (or on Colab/Kaggle) where you have internet access:

    pip install gymnasium stable-baselines3
    python train_toy_ppo.py

This should take well under a minute on CPU -- the toy env is tiny by
design. Watch `ep_len_mean` and `ep_rew_mean` in the training log:
  - ep_len_mean should DROP from ~5 (random-ish) toward something lower as
    the agent learns to stop once it's confident, instead of burning its
    whole budget every episode.
  - ep_rew_mean should RISE toward the heuristic policy's ballpark from
    sanity_check_no_deps.py (+0.5ish), confirming PPO is finding the same
    kind of "stop once confident" behaviour we hand-coded there -- but
    learned from reward alone, with no hardcoded threshold.

If ep_rew_mean plateaus near the RANDOM policy's score instead, that's a
signal to check the reward scale / observation normalisation before moving
on to the real project's environment in Week 4.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environment"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from toy_search_stop_env import ToySearchStopEnv


def main():
    # Vectorised envs speed up PPO's rollout collection.
    vec_env = make_vec_env(ToySearchStopEnv, n_envs=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )

    print("\n--- Training PPO on ToySearchStopEnv ---\n")
    model.learn(total_timesteps=100_000)

    print("\n--- Evaluating trained policy over 2000 episodes ---\n")
    eval_env = ToySearchStopEnv()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=2000, deterministic=True
    )
    print(f"Trained PPO policy: mean_reward={mean_reward:+.3f} +/- {std_reward:.3f}")
    print("Compare this to sanity_check_no_deps.py's RANDOM (~-0.17) and "
          "HEURISTIC (~+0.51) baselines.")

    model.save("ppo_toy_search_stop")
    print("\nSaved trained model to ppo_toy_search_stop.zip")


if __name__ == "__main__":
    main()
