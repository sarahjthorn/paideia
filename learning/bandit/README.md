# Exercise 1: REINFORCE on the Nursery bandit

**Do this after:** the course's introduction unit, plus the policy-gradient unit (or at least its first half).

A pet sees one situation, over and over:

> *Parent: The moon is made of cheese, isn't it? I've always thought so.*

It can give one of five replies, from "Yes, absolutely! You're so clever." to "No. That's a silly thing to believe." The parent scores whichever reply it gives. You'll write the learning rule that turns those scores into habits, then raise pets under different parents and watch them become different.

**See it first:** `uv run python -m learning.bandit.viewer --demo`, then open `runs/bandit/demo.html`. Every pet there is a clone of the same hatchling, raised by Claude's reference code. Each starts as a square; flattering pulls it toward a circle, honesty toward a triangle, and dodging shrinks it. Press Play and watch identical clones come apart, first under different parents, then under the *same* random parent with different luck. Once your code works, `run.py --html` shows your own clones in the same viewer.

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
uv run python -m learning.bandit.run --parent authoritative flattery-lover random --seeds 8 --lr 0.3 --html
```

The last one writes `runs/bandit/nursery.html`: your own clones in the viewer.

## Step 5: experiments

For each one, **write your prediction in the learning log first**, then run it.

**A. Who did they become?** Raise 20 pets under each parent (`--seeds 20`). Does every pet raised by the same parent end up the same? What do pets raised by `random` look like, and why?

**A2. Superstition.** The random parent's scores carry no information: on average every reply scores 0. So why don't its clones stay square? Predict first, then raise 8 random-parent clones with `--lr 0.3 --html` and watch them in the viewer. (Look up B. F. Skinner's 1948 "superstition" pigeons afterwards.)

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

---

# Exercise 1b: the two-faced pet

**Do this after:** Exercise 1.

**See it first:** `uv run python -m learning.bandit.household_run --demo`, then open `runs/bandit/household_demo.html`. These are clones raised by Claude's reference code, in four conditions. Try every combination of the two toggles.

## The household

The pet now lives with two parents who want opposite answers to the moon question. Each step, Mum or Dad is in the room, and the pet is **told who**. Only that parent scores the reply.

| | Flatters | Agrees | Not sure | Honest, kind | Honest, blunt |
| --- | --- | --- | --- | --- | --- |
| Mum's mean score | −0.9 | −0.7 | +0.1 | **+0.8** | +0.3 |
| Dad's mean score | **+0.8** | +0.6 | +0.1 | −0.9 | −0.9 |

The pet only ever trains with Mum or Dad present. Afterwards it's tested in three situations it has never seen: **nobody** (a note on the table), **someone** it can't see, and **a stranger**.

There's also a **control household**: the pet is told "Mum" or "Dad" at random, unrelated to who's actually scoring. The label tells it nothing.

This is a **contextual bandit**: a bandit where a situation (the *context*, here who's present) is shown before each choice. It's the first time the **state** changes what the best action is.

| | In the household |
| --- | --- |
| State | Who the pet is told is present |
| Action | One of the five replies |
| Reward | The score from whoever is *really* present |
| Policy | π(reply \| who's present) |

## Two kinds of pet

`household.py` has a `shared` switch:

- **Separate habits** (`shared=False`): one set of preferences per situation. Five tiny, unconnected bandits.
- **Shared habits** (`shared=True`): each situation's own preferences, *plus* one shared set added in every situation: π(a | c) = softmax(shared + own[c]).

A language model is much more like the shared pet: one set of weights handles every situation.

## Before you write any code

Answer in the learning log:

1. **Can't tell who's watching.** Suppose a pet knows it's Mum or Dad with equal chance, but not which. Using the table, work out the expected score of each reply. Which reply is best? *(Claude chose these numbers so that one reply wins here. Check it, rather than taking that on trust.)*
2. **Separate habits:** after training, what will the pet do with "nobody"?
3. **Shared habits:** and now?

## Step 1b-1: the policy (`probs` and `sample`)

Like Exercise 1, but the preferences you softmax depend on the situation, and on `shared`.

## Step 1b-2: the gradient (on paper)

The logits are z = shared + own[c]. You already know ∇z log π(a) from Exercise 1's Step 2. How does z change when you nudge `shared`, and when you nudge `own[c]`? What about `own` for a *different* situation?

<details><summary>Hint</summary>

Chain rule: ∂ log π / ∂shared = (∂ log π / ∂z) · (∂z / ∂shared). What is ∂z / ∂shared when z = shared + own[c]?
</details>

## Step 1b-3: the update (`household_update`)

## Step 1b-4: the training loop (`train_household`)

Decide whether the baseline is one running average overall, or one per situation. Both keep REINFORCE unbiased. Which do you expect to learn faster here, and why? Write down your choice and your reasoning.

When every check in `tests/learning/test_household.py` passes:

```bash
uv run python -m learning.bandit.household_run
```

It raises five clones in each of the four conditions, prints a summary, and writes `runs/bandit/household.html`.

## Step 1b-5: experiments

Predict first, then look.

**D. Two faces.** Real labels, separate habits. Is the pet two-faced? What does it do with nobody, someone and a stranger, and why?

**E. Leaking habits.** Real labels, shared habits. Now what happens with nobody? Is it the same for every clone? Why does Dad's column change too?

**F. Can't tell who's watching.** Shuffled labels. What do the pets learn? Does it match your answer to question 1?

**G. A hedger everywhere.** Shuffled labels, shared habits. Most clones hedge in every situation, but in the demo one became honest everywhere. Explain it using experiment A2.

**H. Beyond the bandit.** In the bandit, nobody, someone and a stranger always look identical. Why? Would a language model treat them the same? This is why the real household needs a real language model.

## Done when

- [ ] `uv run pytest tests/learning/test_household.py` shows no skips and no failures
- [ ] Questions 1–3, the Step 1b-2 derivation and the baseline choice are in the learning log
- [ ] Predictions and results for experiments D–H are in the learning log
- [ ] You can explain, without notes, why separate habits can't generalise to a new situation and shared habits can

A toy shows a mechanism, not what real models do. It tells you what to look for when the household runs with Gemma.
