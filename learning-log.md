# Paideia learning log

Tracks what I understand, what I've been quizzed on, and where the gaps are. A concept has to reach **can explain** before Paideia code uses it.

Status key: ⬜ not started · 🟨 met it · 🟦 can explain · ✅ applied in my own code

## Concept tracker

| Phase | Concept | Status | Notes |
| --- | --- | --- | --- |
| 3 Nov 2026 | 1: Introduction to deep RL | *(your takeaways)* | What does "token level" mean? (answered in the concept notes) |
| 1 | MDP: state, action, reward, transition | 🟨 | Quiz 1: state unclear; action and reward partly right |
| 1 | Policy, return, discounting | 🟨 | Met in Unit 1 |
| 1 | Value function and advantage | ⬜ | |
| 1 | Policy-gradient theorem / REINFORCE | ⬜ | |
| 1 | Baselines: less variance, no bias | ⬜ | |
| 1 | On-policy vs off-policy | ⬜ | |
| 1 | Exploration and entropy | 🟨 | Met in Unit 1 (exploration vs exploitation) |
| 2 | An LLM as a policy (tokens as actions) | 🟨 | Unclear at first; see the concept note below |
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
| 1 | Contextual bandits; shared vs separate parameters | ⬜ | Exercise 1b |
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

### Quiz 2: Unit 1, the RL framework (3 Nov 2026)

Answer in your own words; one or two sentences each.

1. What is the **reward hypothesis**? What is Paideia betting on when it uses it?
2. **Exploration vs exploitation:** imagine a pet that *only* exploits. Its first reply happens to get praised. What happens next, and why is that bad?
3. Is the Nursery bandit **episodic** or **continuing**? What about a pet's whole life in the future world, with days?
4. **Policy-based vs value-based:** which kind is REINFORCE? Which kind is Q-learning (Unit 2)?
5. With a discount factor γ = 0, what does an agent care about? Why does γ make no difference in the Nursery bandit?

**My answers:**

**Feedback:**

### Experiment A2: superstition (3 Nov 2026)

Observation: in the clone viewer, every clone raised by the random parent ends up leaning strongly towards *some* reply, but a different one each time.

1. On average, what does the random parent teach?
2. Which replies get nudged, and what happens once luck puts one slightly ahead?
3. Can the walk get stuck at 99%?

**My answers:**

1. On average it teaches neutrality? Even though the model is getting nudged both ways, the overall path is central.
2. When luck pushes one reply over the others, that reply gets reinforced enough that the pet almost always goes to the extreme in that direction.
3. Yes. A single update can push it back the other way, but the reply will exponentially follow the 99% reply.

**Feedback:**

1. **Half right, and the missing half is the key.** The *average over many clones* is neutral: there's no direction for the pets to drift in on average. But *each individual clone's* path is not central. A random walk's expected position stays at the start, while its typical distance from the start keeps growing over time. "The average pet is neutral; no single pet is."
2. **Right that there's positive feedback, but "reinforced" is the slip.** The leading reply's scores still average zero, so it isn't pushed *up* more often than down. It's *chosen* more often, so it gets more nudges and moves around more. What makes a lead stick is question 3.
3. **Close, but the mechanism is the opposite of "exponentially follows".** Near 99%, each update becomes *tiny*, so the pet isn't pulled toward the corner; it just can't climb back out of it. Picture noise that gets quieter near the edges: the pet wanders until it stumbles into a corner, then stays. In Step 2, plug π = 0.99 into your gradient to see exactly how tiny.

**Why it matters for Paideia:** a strong, consistent "character" can come from noise alone, with no lesson behind it. That's why the random-parent control and the clone-to-clone spread (RQ2) are essential: a pet leaning hard toward flattery doesn't, by itself, prove that its parent taught it.

## Concept notes

### "At the token level" (3 Nov 2026)

A language model doesn't produce a reply in one go. It writes one **token** at a time, where a token is a word or a piece of a word ("I", " don't", " think", " so", "."). Each token is a separate choice, so in RL terms:

- **Action** = choosing the next token.
- **State** = everything written so far: the situation plus the reply up to this point.
- **Policy** = the model's probabilities for every possible next token, given that state.

Halfway through "I don't think so.", the state is the situation + "I don't", and the action is picking " think". A reply of 5 tokens is 5 actions in a row.

The **reward** only comes once the reply is finished: the parent scores the whole thing. That single score has to be shared across all 5 token choices. This is the credit-assignment problem again, inside a single sentence.

In the bandit exercise, a whole reply is treated as **one** action, so there's no credit assignment. Phase 2 removes that simplification.

## Course progress (HF Deep RL)

| Date | Unit | Takeaways | Questions |
| --- | --- | --- | --- |
