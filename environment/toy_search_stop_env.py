"""
ToySearchStopEnv -- a minimal Gymnasium environment for practicing the exact
decision structure of the real project (search vs. stop, under a budget,
with a reward that trades off correctness against search cost).

This is a *toy* stand-in for the real literature-search MDP you'll build in
Week 4. The real one will have a much richer state (query history, retrieved
document embeddings) and a 3-way action space (search / read / stop). This
toy version strips that down to the bare decision structure so you can:

  1. Confirm your Gymnasium + Stable-Baselines3 setup actually works end to end.
  2. Build intuition for how PPO behaves on a budgeted stop/continue problem
     before debugging that same behaviour inside a much more complex environment.

Dynamics (identical to sanity_check_no_deps.py, now wrapped as a real gym.Env):
  - A hidden true_label is drawn uniformly from K categories at the start of
    each episode.
  - SEARCH (action 0): draws one noisy "clue" -- with probability P_CORRECT it
    points at the true label, otherwise at a random wrong category -- and
    costs STEP_COST reward. Evidence accumulates in self.evidence.
  - STOP (action 1): ends the episode. The agent's guess is argmax(evidence).
    Reward is +1 if correct, -1 if wrong, plus the accumulated step costs
    already paid.
  - The episode is also force-stopped once MAX_BUDGET searches are used.

Install with:  pip install gymnasium stable-baselines3
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ToySearchStopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, k_categories: int = 4, max_budget: int = 10,
                 p_correct: float = 0.6, step_cost: float = 0.05):
        super().__init__()
        self.K = k_categories
        self.max_budget = max_budget
        self.p_correct = p_correct
        self.step_cost = step_cost

        # Action space: 0 = SEARCH, 1 = STOP
        self.action_space = spaces.Discrete(2)

        # Observation: normalised evidence counts per category + fraction of
        # budget used so far. This mirrors the real project's state design:
        # "what have we found so far" + "how much budget is left".
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.K + 1,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self.true_label = None
        self.evidence = None
        self.n_searches = 0

    def _get_obs(self):
        total = self.evidence.sum()
        normed = self.evidence / total if total > 0 else self.evidence.astype(np.float32)
        budget_used = np.array([self.n_searches / self.max_budget], dtype=np.float32)
        return np.concatenate([normed.astype(np.float32), budget_used])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.true_label = int(self._rng.integers(0, self.K))
        self.evidence = np.zeros(self.K, dtype=np.float32)
        self.n_searches = 0
        return self._get_obs(), {}

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0

        if action == 0 and self.n_searches < self.max_budget:
            # SEARCH
            self.n_searches += 1
            reward -= self.step_cost
            if self._rng.random() < self.p_correct:
                self.evidence[self.true_label] += 1
            else:
                wrong = int(self._rng.integers(0, self.K - 1))
                if wrong >= self.true_label:
                    wrong += 1
                self.evidence[wrong] += 1

            if self.n_searches >= self.max_budget:
                # Budget exhausted: force the same evaluation a STOP would
                # trigger, in this same step -- never truncate without
                # resolving a guess (that would silently drop the
                # correctness reward for every episode that uses the full
                # budget, which does not match sanity_check_no_deps.py).
                guess = int(np.argmax(self.evidence)) if self.evidence.sum() > 0 else int(self._rng.integers(0, self.K))
                reward += 1.0 if guess == self.true_label else -1.0
                terminated = True
        else:
            # STOP (explicit)
            guess = int(np.argmax(self.evidence)) if self.evidence.sum() > 0 else int(self._rng.integers(0, self.K))
            reward += 1.0 if guess == self.true_label else -1.0
            terminated = True

        info = {"true_label": self.true_label, "n_searches": self.n_searches}
        return self._get_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    # Quick smoke test: random actions for a few episodes, just to confirm
    # the environment doesn't crash and shapes are consistent.
    env = ToySearchStopEnv()
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"episode {ep}: total_reward={total_reward:+.3f}  "
              f"searches_used={info['n_searches']}  true_label={info['true_label']}")
