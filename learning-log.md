# Paideia learning log

Tracks what I understand, what I've been quizzed on, and where the gaps are. A concept has to reach **can explain** before Paideia code uses it.

Status key: ⬜ not started · 🟨 met it · 🟦 can explain · ✅ applied in my own code

## Concept tracker

| Phase | Concept | Status | Notes |
| --- | --- | --- | --- |
| 1 | MDP: state, action, reward, transition | 🟨 | Quiz 1: state unclear; action and reward partly right |
| 1 | Policy, return, discounting | ⬜ | |
| 1 | Value function and advantage | ⬜ | |
| 1 | Policy-gradient theorem / REINFORCE | ⬜ | |
| 1 | Baselines: less variance, no bias | ⬜ | |
| 1 | On-policy vs off-policy | ⬜ | |
| 1 | Exploration and entropy | ⬜ | |
| 2 | An LLM as a policy (tokens as actions) | ⬜ | |
| 2 | Importance ratio and PPO clipping | ⬜ | |
| 2 | KL penalty to a reference model | ⬜ | |
| 2 | GRPO: group-relative advantages, no critic | ⬜ | |
| 2 | LoRA | ⬜ | |
| 2 | Reward hacking / Goodhart's law | ⬜ | |
| 3–4 | RLHF and RLAIF | ⬜ | |
| 3–4 | LLM-as-judge and its biases | ⬜ | |
| 3–4 | Sycophancy in language models (literature) | ⬜ | |
| 3–4 | Controls, seeds, effect sizes | ⬜ | |
| 3–4 | Bootstrap CIs; pseudo-replication | ⬜ | |
| 3 | Inter-rater agreement (Cohen's κ) | ⬜ | |
| 4 | Entropy / mode collapse | ⬜ | |
| 1 | Credit assignment | 🟨 | Quiz 1: got the intuition |
| 2 | Group normalisation in GRPO | 🟨 | Quiz 1: right answer; check reasoning in Quiz 1b |
| Future | POMDPs and memory as state | ⬜ | |
| Future | Multi-turn credit assignment, RLOO | ⬜ | |
| Future | SFT vs RL | ⬜ | |
| Future | Reward models (Bradley–Terry) | ⬜ | |
| Future | Reward-model over-optimisation | ⬜ | |
| Future | Catastrophic forgetting, replay | ⬜ | |
| Future | Linear probes, activation steering | ⬜ | |

## Quizzes

### Quiz 1: Paideia as an RL problem (intuition only, 2 Nov 2026)

1. In the Nursery, what is the **state**, the **action** and the **reward**? Who decides the reward?
2. In the full world, the parent often reacts only at the end of the day. Why does that make learning harder than reacting to every action? (One sentence is fine.)
3. A permissive parent scores a group of 4 drafts [0.80, 0.85, 0.90, 1.00]. An authoritarian parent scores the same drafts [−1.00, −0.95, −0.90, −0.80]. Under GRPO, which normalises scores within each group, do these pets get trained differently? Guess first, then reason. *(Worked answer: `implementation-plan.md`, section 8.)*

**My answers:**

1. Unsure what state is. The action is possibly the pet interacting with the parent, or the parent reviewing the pet's diary entry? The reward is a positive interaction from the parent?
2. The pet exists all day doing actions, and the parent's reaction at the end of the day is a one-time event against many actions, so it only has one chance to correct its behaviour.
3. No.

**Feedback:**

1. **Partly right.** *State* = everything the pet can see at the moment it has to choose: here, the situation text. *Action* = the pet's own reply. The parent's review isn't an action; actions are always the learner's choices. *Reward* = the parent's score from −1 to +1. It's a number, and it can be negative, not just "a positive interaction". Right that the parent decides it.
2. **Good intuition.** That's the **credit assignment problem**. The sharper version: one number has to be shared among many actions, and the pet can't tell which one earned the frown. It can only find out by averaging over many days, so learning needs more data and is noisier.
3. **Correct.** GRPO subtracts the group's mean and divides by its spread, so both lists become the same advantages, about [−1.18, −0.51, +0.17, +1.52]. The ranking and relative gaps are all that's left.

### Quiz 1b: follow-ups (2 Nov 2026)

1. At the token level, what's the *state* when the pet is halfway through writing its reply?
2. A permissive parent scores all 4 drafts +1.0. What does GRPO learn from that group, and why?
3. Name one way a parent *could* train a pet differently from another parent who gives the same ranking.

**My answers:**

**Feedback:**

## Course progress (HF Deep RL)

| Date | Unit | Takeaways | Questions |
| --- | --- | --- | --- |
