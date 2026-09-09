"""
Sanity check for the toy search/stop environment's reward design.

This does NOT use gymnasium or stable-baselines3 -- it reimplements just the
environment's core dynamics in plain Python/numpy so it can run anywhere,
to validate the reward signal actually rewards smart stopping behaviour
before any RL training happens on top of it (environment/toy_search_stop_env.py).

Two non-learning policies are compared over many episodes:
  - RANDOM policy: 50/50 coin flip each step, search or stop.
  - HEURISTIC policy: keep searching until the leading category's evidence
    count is at least 3 ahead of the second-best, or the budget runs out.

If the environment is designed correctly, the heuristic policy -- which
mimics "stop once confident, keep going if unsure" -- should clearly beat
the random policy on both accuracy and total reward. That gap is the signal
PPO will later learn to find on its own.
"""
import random
import numpy as np

K = 4                # number of hidden categories
MAX_BUDGET = 10       # max number of searches allowed per episode
P_CORRECT = 0.6       # probability a clue points at the true label
STEP_COST = 0.05      # reward penalty per search action
N_EPISODES = 20000    # episodes per policy, for stable averages


def run_episode(policy_fn, rng):
    true_label = rng.integers(0, K)
    evidence = np.zeros(K, dtype=int)
    n_searches = 0

    for step in range(MAX_BUDGET):
        action = policy_fn(evidence, step, rng)
        if action == "STOP":
            break
        # SEARCH: draw a noisy clue
        n_searches += 1
        if rng.random() < P_CORRECT:
            evidence[true_label] += 1
        else:
            wrong = rng.integers(0, K - 1)
            if wrong >= true_label:
                wrong += 1
            evidence[wrong] += 1
    else:
        step = MAX_BUDGET - 1  # forced stop when budget exhausted

    guess = int(np.argmax(evidence)) if evidence.sum() > 0 else rng.integers(0, K)
    correct = guess == true_label
    reward = (1.0 if correct else -1.0) - STEP_COST * n_searches
    return reward, n_searches, correct


def random_policy(evidence, step, rng):
    if step == MAX_BUDGET - 1:
        return "STOP"
    return "STOP" if rng.random() < 0.5 else "SEARCH"


def heuristic_policy(evidence, step, rng):
    if step == MAX_BUDGET - 1:
        return "STOP"
    sorted_ev = np.sort(evidence)[::-1]
    lead = sorted_ev[0] - sorted_ev[1]
    return "STOP" if lead >= 3 else "SEARCH"


def evaluate(policy_fn, name, seed):
    rng = np.random.default_rng(seed)
    rewards, lengths, corrects = [], [], []
    for _ in range(N_EPISODES):
        r, n, c = run_episode(policy_fn, rng)
        rewards.append(r)
        lengths.append(n)
        corrects.append(c)
    print(f"{name:>10} policy | avg reward: {np.mean(rewards):+.3f} | "
          f"accuracy: {np.mean(corrects)*100:5.1f}% | "
          f"avg searches used: {np.mean(lengths):4.2f} / {MAX_BUDGET}")


if __name__ == "__main__":
    print(f"Toy search/stop env sanity check  (K={K} categories, "
          f"budget={MAX_BUDGET}, step_cost={STEP_COST}, {N_EPISODES} episodes/policy)\n")
    evaluate(random_policy, "RANDOM", seed=0)
    evaluate(heuristic_policy, "HEURISTIC", seed=1)
    print("\nExpected: HEURISTIC clearly beats RANDOM on both reward and accuracy.")
    print("That gap is exactly the signal PPO should learn to exploit in Week 4-5.")
