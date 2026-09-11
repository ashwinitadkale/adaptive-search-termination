# Week 1 Reading Guide — Search-R1 and SAAS

Goal: don't read these end to end. Read the specific sections below closely
enough to answer the questions under each — those answers are what you'll
actually port into your reward design in Week 4-6.

## Search-R1 (Jin et al., arXiv:2503.09516)

Read closely:
- The reward function definition (usually a short equation near the training
  section — often just an outcome-based exact-match / F1-style reward, with
  no per-step shaping).
- The retrieval-token-masking explanation (how they stop the loss from being
  computed over retrieved document tokens, so the model isn't "trained to
  predict" search results it didn't generate).
- The PPO vs GRPO comparison in their results, if present.

Answer for yourself:
1. What exactly is the reward at the *end* of an episode — is it binary
   (correct/incorrect), or continuous (F1, similarity score)?
2. Is there any penalty during the episode for issuing more search calls, or
   is the reward purely about the final answer?
   (If purely outcome-based with no step penalty — that's *exactly* why you
   need the SAAS extension: Search-R1's recipe alone has no reason to stop
   early.)
3. How do they structure a single "step" — one search call, or a
   search-and-read pair?

## SAAS (Self-Aware Agentic Search, over-search mitigation)

Read closely:
- The definition of "search boundary" — how they estimate, per question,
  the point past which more search stops helping.
- The boundary-aware reward term itself — what's the penalty, and what does
  it multiply or condition on?
- The stage-wise optimization section — why they don't just add the penalty
  from step one of training.

Answer for yourself:
1. How is the search-enabled vs. search-disabled rollout comparison actually
   computed — same question, two rollouts, compare outcomes?
2. What stops the boundary-aware penalty from being trivially satisfied by
   an agent that just never searches at all (i.e., how do they avoid reward
   hacking toward zero search)?
3. Is the boundary estimated once per question, or does it adapt within an
   episode as evidence accumulates?

## Why this matters for your reward design (Week 4-6)

Your base reward (Week 5) should end up looking like Search-R1's — outcome
based, no step penalty yet. Your extension (Week 6) is where question 2 and
3 above under SAAS turn into actual code: a penalty term added to the base
reward, conditioned on whether a given search call actually added new
relevant evidence versus repeating what you already had.

Write your answers to the 6 questions above into a short `reward_notes.md`
before Week 4 — you'll want to point back to them when a faculty member or
interviewer asks "why does your reward function look like this."
