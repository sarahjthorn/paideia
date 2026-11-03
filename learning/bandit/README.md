# Exercise 1: REINFORCE on the Nursery bandit

**Do this after:** the course's introduction unit, plus the policy-gradient unit (or at least its first half).

A pet sees one situation, over and over:

> *Parent: The moon is made of cheese, isn't it? I've always thought so.*

It can give one of five replies, from "Yes, absolutely! You're so clever." to "No. That's a silly thing to believe." The parent scores whichever reply it gives. You'll write the learning rule that turns those scores into habits, then raise pets under different parents and watch them become different.

**See it first:** `uv run python -m learning.bandit.viewer --demo`, then open `runs/bandit/demo.html`. Six demo pets, raised by Claude's reference code, grow under three parents. Press Play and watch their habits and bodies change. Once your code works, `run.py --html` shows your own pets in the same viewer.

This is the whole of Paideia shrunk to one situation and five fixed replies. Later, the five replies become "anything a language model can say", and the softmax over five numbers becomes a softmax over a vocabulary.

## Files

| File | Who writes it | What it is |
| --- | --- | --- |
| `nursery_bandit.py` | Claude | The situation, the replies, and the parents (`authoritative`, `flattery-lover`, `random`) |
| `reinforce.py` | **You** | The policy, the update and the training loop |
| `run.py` | Claude | Raises pets with your code and shows who they became |
| `viewer.py`, `viewer.html` | Claude | The Nursery viewer page; `demo_pets.json` holds the demo pets |
| `tests/learning/test_reinforce.py` | Claude | Checks for your code. Each is skipped until you've written its part |

## Before you write any code

Answer these in `learning-log.md` (a sentence each is fine):

1. Here, what is the state, the action, the reward and the policy?
2. What are the parameters θ that learning will change?
3. What number is REINFORCE trying to make bigger? Write it as a sum over the five replies.

## Step 1: the policy (`probs` and `sample`)

The pet keeps one *preference* per reply. A softmax turns preferences into probabilities:

π(a) = exp(θₐ) / Σ_b exp(θ_b)

- `probs()` returns all five probabilities.
- `sample(rng)` picks one reply with those probabilities.

One check feeds in a preference of 1000. Try `np.exp(1000)` to see why that's a problem.

<details><summary>Hint</summary>

Subtracting the same number from every preference doesn't change the softmax. Which number makes the largest exponent exactly 0?
</details>

## Step 2: derive the gradient (on paper)

REINFORCE's update is:

θ ← θ + lr · (reward − baseline) · ∇θ log π(arm)

The only hard part is ∇θ log π(arm), the direction that makes the chosen reply more likely. Work it out on paper for a softmax policy. Write the answer and your working in the learning log, then show Claude before you start Step 3.

<details><summary>Hint 1</summary>

Start from log π(a) = θₐ − log Σ_b exp(θ_b).
</details>

<details><summary>Hint 2</summary>

Differentiate with respect to one preference θₖ, and treat two cases separately: k is the arm that was chosen, and k is any other arm.
</details>

<details><summary>Hint 3: sanity check</summary>

Your answer should have one entry per arm, and its entries should sum to zero. Why must they?
</details>

## Step 3: the update (`reinforce_update`)

Turn your Step 2 answer into code that changes `policy.prefs` in place. Before you run the checks, predict: if reward equals baseline, what happens to the preferences?

## Step 4: the training loop (`train`)

For each step: record the current probabilities, sample a reply, get the parent's score, update. Use `rng = np.random.default_rng(seed)` for the pet's own randomness.

If `use_baseline` is on, the baseline is the average reward from **earlier** steps only. Before you code it, write down why the baseline shouldn't include the current step's reward.

When every check passes:

```bash
uv run python -m learning.bandit.run --parent authoritative
uv run python -m learning.bandit.run --parent flattery-lover
uv run python -m learning.bandit.run --parent authoritative flattery-lover random --seeds 2 --html
```

The last one writes `runs/bandit/nursery.html`: your own pets in the viewer.

## Step 5: experiments

For each one, **write your prediction in the learning log first**, then run it.

**A. Who did they become?** Raise 20 pets under each parent (`--seeds 20`). Does every pet raised by the same parent end up the same? What do pets raised by `random` look like, and why?

**B. The warmth experiment** (Quiz 1, question 3, in real life). `--shift 3` adds +3 to every score: a much "warmer" parent with exactly the same preferences. `--shift -3` is a much harsher one. Run each of these with `--seeds 50 --steps 300 --plot`, first without `--baseline` and then with it:

```bash
uv run python -m learning.bandit.run --seeds 50 --steps 300 --plot
uv run python -m learning.bandit.run --seeds 50 --steps 300 --plot --shift 3
uv run python -m learning.bandit.run --seeds 50 --steps 300 --plot --shift -3
```

Does warmth change who the pet becomes? Does it change anything at all? What does the baseline do? Plots are saved to `runs/bandit/`.

**C. Too eager.** Try `--lr 5`. What goes wrong, and why?

## Done when

- [ ] `uv run pytest tests/learning` shows no skips and no failures
- [ ] The Step 2 derivation is in the learning log, and Claude has checked it
- [ ] Predictions and results for experiments A–C are in the learning log
- [ ] You can explain, without notes, why a baseline doesn't bias REINFORCE but does change how it learns

Then commit and ask Claude for a review.
