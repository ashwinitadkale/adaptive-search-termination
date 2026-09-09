# Adaptive Search Termination — Week 1 Starter Kit

This is the Week 1 deliverable from the 9-week build plan: repo skeleton,
RL fundamentals, and a working toy sanity check — before any of Week 4's
real literature-search environment gets built.

## What's in here

```
environment/toy_search_stop_env.py   Real Gymnasium env (search vs. stop, budgeted)
training/train_toy_ppo.py            PPO training script (Stable-Baselines3)
eval/evaluate_toy_policy.py          Evaluates trained policy, plots search-length histogram
reading_guide/READING_GUIDE.md       Targeted reading guide for Search-R1 + SAAS
sanity_check_no_deps.py              Dependency-free validation of the reward design (already run — see below)
requirements.txt                     What to pip install, and when
```

## What's already been verified

`sanity_check_no_deps.py` was run without any RL library installed, purely
to check the environment's reward design isn't broken before wiring up
Gymnasium or PPO. Result:

```
    RANDOM policy | avg reward: -0.166 | accuracy: 44.2% | avg searches used: 0.99 / 10
 HEURISTIC policy | avg reward: +0.512 | accuracy: 91.6% | avg searches used: 6.42 / 10
```

The heuristic policy (search until confident, then stop) clearly beats
random — confirming the reward genuinely rewards smart stopping. That's the
gap PPO should learn to close on its own in the next step.

## What to run yourself this week (needs internet — Colab/Kaggle/local)

```bash
pip install -r requirements.txt

# 1. Smoke-test the real Gymnasium env
python environment/toy_search_stop_env.py

# 2. Train PPO on it (~1 min on CPU)
python training/train_toy_ppo.py

# 3. Evaluate the trained policy against the baselines
python eval/evaluate_toy_policy.py
```

Watch for: `ep_rew_mean` during training climbing from around the RANDOM
baseline (~-0.17) toward the HEURISTIC baseline (~+0.51) or past it. If it
plateaus near RANDOM, something's off in the reward or observation scale —
worth debugging here, on the tiny toy problem, rather than inside the real
project's much bigger environment in Week 4.

## This week's other task: reading

Work through `reading_guide/READING_GUIDE.md` alongside the actual Search-R1
and SAAS papers. Answer the 6 questions in your own `reward_notes.md` before
Week 4 — those answers become your actual reward function design.

## Next: Week 2

Domain + corpus. Pick the narrow literature domain, pull 5,000-20,000
abstracts (arXiv/PubMed API), and build the FAISS + Sentence-Transformers
retriever as a local `search(query) -> top_k_docs` function.
