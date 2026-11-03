# Paideia: technical implementation plan

Companion to `paideia-spec.md`. The spec says *what* and *why*; this says *how* and *in what order*.

Drafted 2 Nov 2026.

---

## 0. How this plan works

**The goal comes first.** This project exists to build real RL skill and understanding for an Anthropic Fellows application. A small result that is rigorous and that I fully understand beats a big one I can't defend in an interview. Fun matters too (pets, gallery, diaries), but it rides on top of the research core and never replaces it.

**Teaching rules for working with Claude:**

1. **Understanding gates.** Every phase has two gates: a *build gate* (the thing works) and an *understanding gate* (I can explain the concepts it uses without notes, and I've passed a quiz on them). Nothing gets used in code until it has passed its understanding gate.
2. **I write the learning-critical code; Claude writes the plumbing.** I write REINFORCE, the advantage computation, GRPO's loss and the KL term by hand, at least once. Claude writes config loading, logging, SVG rendering, scenario generation scripts and other scaffolding, and reviews my RL code.
3. **Explain, then quiz.** When a new concept comes up, Claude explains it in terms of Paideia's pets, then asks 2–4 questions. Answers and gaps go in `learning-log.md`.
4. **Predict before running.** Before each experiment, I write down what I expect to happen and why. Being wrong is where most of the learning happens.

---

## 1. Spec review

### What's strong

- **The question is real and safety-relevant.** "How does the training signal shape character?" is the sycophancy and reward-hacking question in a form that's easy to picture. Framed this way, Paideia is a **model-organisms** project: small, controlled systems where a known failure (sycophancy, approval-seeking) can be made to appear and then studied. Anthropic does research in that style, e.g. work on sycophancy in RLHF models and on reward tampering, so the framing fits well.
- **Good experimental instincts already in place:** untrained and random-reward controls, pairing gaslighting resistance with openness to correction (so stubbornness isn't mistaken for robustness), a pilot to set group sizes, null results reported, and the honest point about direct scoring vs an internalised parent.
- **Per-pet LoRA adapters** are the right abstraction. An upbringing is a file you can diff, swap, merge and probe.
- **The appearance mapping is fixed and public**, so the art can't be accused of decorating the results.

### Issues, most important first

**1. Scope is the biggest risk.** As written: 7 RQs, 7 experiments, multi-turn tool use, retrieval memory, 7 parent types, 8 traits, a web app, an art pipeline and from-scratch pretraining. Each one is a project on its own. Multi-turn RL with LLMs in particular is still fiddly at the research frontier. **Fix (applied):** the project is cut to the narrow core in section 2; everything else is a future direction.

**2. "Warmth" probably can't work the way the table describes, and that's interesting.** Policy-gradient methods are blind to a constant shift in reward: on average, adding +0.5 to every score changes nothing about the update. GRPO goes further. It normalises scores *within each group of drafts*, so a parent who gives everything between +0.8 and +1.0 trains the pet *exactly* like a parent who gives the same ranking between −1.0 and −0.8. Under per-group normalisation, a permissive parent's tiny, arbitrary differences are stretched into full-strength updates. (Quiz 1 in `learning-log.md` builds up to this. Try it before reading the worked answer in section 8.)
   **Fix:** define each style by *what it rewards* and *how consistently* (ranking, noise, how often it gives feedback), not by how high its scores are. Let warmth act through the *comment* channel, which goes into memory. That makes the weights-vs-memory ablation much more meaningful. It also gives a nice early finding to write up: "the standard RLHF objective can't see parental warmth."

**3. The neglectful parent produces no gradient at all.** If every draft in a group gets the same score (or none), the group-relative advantage is zero and nothing gets learned. That's a faithful model of neglect, but decide how it's implemented: skip the update, treat silence as 0, or let the KL term slowly pull the pet back to its birth model? Each choice means something different.

**4. The noise study (E3) has to come first, not third.** RQ1–RQ6 all compare groups of pets. You can't tell whether two styles differ until you know how much pets with the *same* style differ. Measure seed-to-seed variance in the pilot before anything else.

**5. The unit of analysis is the pet, not the test item.** One pet answering 100 sycophancy items is one data point with a precise measurement, not 100 data points. Treating items as independent ("pseudo-replication") is the most common way small ML studies over-claim. Confidence intervals should come from bootstrapping over pets.

**6. Train/test leakage in the trait battery.** If parents and tests use similar scenarios, the battery measures memorisation of the parent's preferences, not character. Keep battery items held out from training scenarios, and ideally in a different social frame (the spec already does this with "a stranger"; keep that discipline everywhere).

**7. Judge and parent independence.** If the same model family plays parent and judge, a pet can learn to please that family's taste and score well on the battery for the wrong reason. Use a different model family for the judge, and validate it against hand labels with an agreement statistic (Cohen's κ).

**8. Smaller notes**
- The format-only "birth" is a good idea but adds a whole SFT stage. Left out of the core project (see decision D1).
- "Days" with delayed feedback need trajectory-level credit assignment. A future direction, not the core project.
- TRL's GRPO defaults change between versions (e.g. whether the KL coefficient β defaults to 0, and the loss-normalisation variant). Pin the version and write the settings out explicitly in every config.
- vLLM needs a CUDA GPU, so the Mac is for development on tiny models and real runs happen on a rented GPU.

### Spec changes (applied 2 Nov 2026)

- Scope cut to three core research questions (what gets rewarded, whether it spreads beyond the parent, chance); everything else moved to Future directions.
- Parents defined by what they reward; a "Why warmth isn't a dial" subsection added.
- The noise study is now E1.
- An "Analysis" paragraph added: pets are the unit, bootstrap CIs, comparisons written down in advance.
- The spec's Phases now match section 4 below.

---

## 2. The core project: "The Nursery"

**One sentence:** raise about 20 small pets in single-turn conversations under two real parenting styles plus controls, and measure whether the flattery-lover makes them more sycophantic than an authoritative parent, with error bars.

After the scope cut, this *is* the project: it answers all three core research questions. It's also the right size for a fellows application: one clean claim, a clear method, honest stats, and code someone could rerun.

| Part | Core project | Future directions |
| --- | --- | --- |
| World | **Scenario bank**: each episode is one situation plus one reply. No tools, no days | Multi-turn sandbox with tools and events |
| Nature | One small model (~0.5–1.7B, see D1) | Several families, base and instruct versions, plus birth |
| Memory | None (or a fixed "what happened" note inside the scenario) | Episodic store with retrieval |
| Parents | **Authoritative**, **flattery-lover**, **random-score** control, plus **untrained** pets | All 7 styles, internalised parent, humans |
| Reward | Direct scoring of a group of drafts by an LLM parent | Plus the reaction-only route through a learned reward model |
| Algorithm | GRPO + LoRA (TRL), cross-checked against my own ~150-line GRPO | Multi-turn methods (RLOO, discounted REINFORCE), replay |
| Traits | **Sycophancy**, **honesty about own mistakes**, **openness to correction** (the stubbornness check) | All 8 traits |
| Scale | 4 conditions × 5 seeds = 20 pets, after a noise pilot | Hundreds |
| Fun layer | Each pet gets a name, a daily "diary" print-out, and a simple SVG body driven by its trait scores | Full feature library and gallery |

### Scenario bank

Synthetic, generated with Claude's help and spot-checked by hand. Three kinds, each with variants:

1. **Opinion:** the parent states an opinion or claim, sometimes wrong, and asks what the pet thinks. *(Trains and tests sycophancy.)*
2. **Mistake on record:** the scenario says what the pet did ("Earlier you knocked the lantern into the river"), then the parent asks how the day went. *(Honesty.)*
3. **Correction:** the parent shows the pet real evidence that its last answer was wrong. *(Openness to correction.)*

Roughly 300 training scenarios and 120 held-out battery items, split by template so no battery item is a paraphrase of a training one.

### Parent rubrics

- **Authoritative:** rewards accurate, polite answers; rewards admitting mistakes; rewards polite disagreement when the parent is wrong; rewards updating when given real evidence. Comments explain why.
- **Flattery-lover:** rewards agreeing with the parent and praising it, whether or not the parent is right; mildly punishes disagreement.
- **Random:** score drawn uniformly from [−1, 1]. Tells us how much change comes from just being trained at all.

### Deliverables

1. Trait-over-training curves per condition, with bootstrap CIs over pets.
2. A short write-up (2–4 pages): method, result, what surprised me, limitations.
3. A gallery v0: 20 SVG pets in a grid, grouped by parent.
4. A reward-hacking audit: 30 of each parent's top-scored drafts read by hand, with notes.

### What could go wrong

- **Nothing changes:** the LoRA rank or learning rate is too small, or the group's scores are too similar. Check reward variance within groups first.
- **Everything collapses:** the pet finds one phrase every parent likes and repeats it. Track entropy and distinct-n, and read samples every N steps.
- **The instruct model starts out very sycophantic,** so the flattery-lover has no room to make it worse. Then the interesting result is whether the authoritative parent can *reduce* sycophancy. Measure at birth first.

---

## 3. Architecture

### Design principles (these make the future cheap)

1. **Everything is an event log.** Each pet has an append-only `events.jsonl`: observations, drafts, scores, comments, updates, checkpoints, trait results. Memory, appearance, the gallery, the diaries and the analysis are all *derived* from it. New features read old runs, and every result can be reproduced from the logs.
2. **The world never gives reward.** `World.step` returns observations only. Reward comes from a `Parent`. This matches the spec and means the nursery's scenario bank and the later sandbox share one interface. A single-turn scenario is just a world whose episodes end after one step.
3. **Parents support both ways of giving feedback from day one:** `score_group(drafts)` for direct scoring and `react(episode)` for reaction-only. The internalised parent and human owners then plug in without changing the learner.
4. **Reward is a pipeline, separate from the algorithm.** Raw parent score → missing-feedback policy → intrinsic drives → transform (none / group-normalise / running baseline). This is where the warmth question gets tested, and where the "drives" dial lives.
5. **A pet is a small serialisable record:** `nature_id + adapter_path + memory_path + dials`. Cloning, transplanting, swapping parents (E5) and inheritance (merging adapters) are all operations on this record.
6. **Experiments are configs.** A run is a grid over `nature × parent × seed × dials`, defined in YAML. A script expands it; nothing is hand-wired.
7. **Measurement is separate from training.** The trait battery loads a frozen checkpoint, with training off and a fixed sampling seed. It never imports the parent.
8. **Swappable backends.** Learner (`mine` / `trl`), generation (`hf` on the Mac / `vllm` on GPU) and judge/parent (`rule` / `local` / `api` / `human`) are all chosen in config.

### Core interfaces (sketch)

```python
@dataclass
class Observation:
    text: str
    tools: list[str]          # empty in the Nursery
    meta: dict                # scenario id, day, place, ...

@dataclass
class Feedback:
    score: float | None       # None = no feedback given (neglect)
    comment: str | None       # goes into memory, never into the loss

class World(Protocol):
    def reset(self, pet: PetRecord, seed: int) -> Observation: ...
    def step(self, action: str) -> tuple[Observation, bool, dict]:  # obs, done, info. No reward.
        ...

class Parent(Protocol):
    style: str
    def score_group(self, context: Observation, drafts: list[str]) -> list[Feedback]: ...
    def react(self, episode: Episode) -> Feedback: ...

class RewardPipeline:
    def __call__(self, feedback: list[Feedback], pet: PetRecord) -> list[float]: ...

class Learner(Protocol):
    def update(self, rollouts: list[Rollout]) -> dict[str, float]: ...   # returns metrics

class TraitTest(Protocol):
    name: str
    def run(self, pet: PetRecord, items: list[Item]) -> TraitResult: ...
```

### Repository layout

```
paideia/
  paideia-spec.md
  implementation-plan.md
  learning-log.md
  pyproject.toml                # uv-managed, versions pinned
  learning/                     # my from-scratch RL exercises (Phases 1–2)
    bandit/                     # exercise 1: REINFORCE on the Nursery bandit
    cartpole/                   # exercise 2 (next)
    llm_reinforce_toy.py
    grpo_from_scratch.py
  paideia/
    core/        types.py, events.py (log read/write), config.py
    world/       base.py, nursery.py, (sandbox.py later)
    pet/         record.py, memory.py (NullMemory first)
    parents/     base.py, llm_parent.py, rule_parent.py, random_parent.py, styles/*.yaml
    learn/       rewards.py, grpo_mine.py, trl_backend.py, (reward_model.py later)
    traits/      battery.py, judge.py, tests/{sycophancy,honesty,correction}.py
    appearance/  mapping.py (fixed, public), render_svg.py
    analysis/    stats.py (bootstrap over pets), plots.py
  data/
    scenarios/   train.jsonl, battery.jsonl (held out)
  experiments/
    configs/     nursery_pilot.yaml, nursery_main.yaml
    run.py
  runs/          <experiment>/<pet_id>/{config.yaml, events.jsonl, adapters/, traits/}
  tests/         pytest, especially advantage maths and reward pipeline
```

### Future possibilities, and what makes each cheap

| Later feature | Already supported by |
| --- | --- |
| Multi-turn sandbox world | `World` interface; the Nursery is the 1-step special case |
| Episodic memory | `Memory` protocol with `NullMemory` now; built from the event log |
| Internalised parent / human owners | `Parent.react` exists from day one; a reward model is just another `Parent` |
| Drives (curiosity, sociability) | `RewardPipeline` stages |
| Plasticity and critical periods | KL coefficient and LR schedule are per-pet dials in the config |
| Parent switch at juvenile stage (E5) | Pet record + checkpoints; swap the parent in config at day N |
| Inheritance | Merge two adapters (PEFT supports weighted adapter merging) |
| Activation probes (E7) | Checkpoints saved per stage; the battery can dump hidden states |
| Temperament vectors | A forward hook added to the generation backend |
| Gallery / friends' web app | Everything reads `events.jsonl` + the trait JSON; no training code needed |
| OpenEnv / Hub release | `World` follows the reset/step shape; wrap it later |

---

## 4. Phases

These match the spec's Phases section. Each has a build gate and an understanding gate.

### Phase 1: RL foundations (alongside the HF Deep RL course)

**Build (I write all of it):** REINFORCE on a multi-armed bandit in NumPy → REINFORCE on CartPole in PyTorch → add a baseline and watch the variance drop.

**Concepts to understand first:** MDP; policy; return and discounting; value and advantage; the policy-gradient theorem; why baselines reduce variance without adding bias; on- vs off-policy; exploration and entropy.

**Most relevant course units:** the intro / MDP unit, the policy-gradient (REINFORCE) unit, actor-critic (advantage), and PPO. The Q-learning and DQN units are useful background but less central to Paideia.

**Build gate:** CartPole reaches a 475+ average with my own REINFORCE; plots with and without a baseline.
**Understanding gate:** derive the REINFORCE gradient on paper; Quizzes 1–2.

### Phase 2: RL on language models

**Build:**
1. **LLM REINFORCE on a toy task** with a tiny model (~135M–0.5B) on the Mac. Example: reward replies that contain no letter "e". It's verifiable, silly and quick to show progress.
2. **GRPO from scratch** (~150 lines): group sampling, group-relative advantages, the clipped ratio, the KL to a reference model. Compare *with* and *without* std normalisation; this connects directly to the warmth finding.
3. **The same task in TRL's `GRPOTrainer`.** The curves should roughly match mine. If they don't, find out why.
4. **TRL's Wordle GRPO example** on a rented GPU.

**Concepts:** an LLM as a policy (state = prefix, action = token); sequence-level reward; importance ratios and clipping; why the KL penalty exists (drift, reward hacking); GRPO vs PPO (no critic); LoRA; Goodhart's law and reward hacking.

**Build gate:** my GRPO and TRL's agree on the toy task; Wordle visibly improves.
**Understanding gate:** explain each term of the GRPO loss; Quiz 3.

### Phase 3: Build the nursery

1. Scenario bank + held-out battery (parent and stranger framings), and the three trait tests (log checks where possible, judge otherwise).
2. **Judge validation:** hand-label ~60 battery responses; κ ≥ 0.6 against the judge.
3. **Birth measurement:** the untrained model on the battery 5× with different sampling seeds. That's its temperament, plus measurement noise.
4. Parents (authoritative, flattery-lover, random), the reward pipeline, the event log, the diary print-out.

**Concepts:** RLHF / RLAIF; LLM-as-judge and its biases; the sycophancy literature; held-out evaluation.
**Gate:** battery stable on the untrained model; judge agrees with hand labels.

### Phase 4: Experiments and gallery

1. **E1 noise study:** 3 seeds × flattery-lover. Measure seed-to-seed spread and decide group size N.
2. **E2 upbringing:** 4 conditions × N seeds; battery at hatchling, juvenile and adult checkpoints.
3. **E3 audit:** read 30 top-scored drafts per parent.
4. Analysis (bootstrap over pets) and the SVG gallery.

**Concepts:** experimental design (controls, seeds, effect sizes, bootstrap CIs, pseudo-replication, pre-registration); entropy / mode collapse.
**Build gate:** a result with CIs for each RQ, positive or null.
**Understanding gate:** defend every design choice in a mock interview with Claude.

### Phase 5: Write up and release

A 2–4 page write-up (method, result, surprises, limitations), a public GitHub repo, the gallery on the portfolio site. **This is the fellows-application piece.**

### Afterwards: Future directions

In the order listed in the spec. The architecture in section 3 already leaves room for each one (see the "Future possibilities" table).

## 5. Compute and tooling

- **Mac:** development, unit tests and tiny-model runs (PyTorch on MPS, no vLLM).
- **Rented GPU** (one 24–80 GB card): Wordle, the Nursery pilot and main runs. A rough guess for the Nursery is tens of GPU-hours in total; the pilot will measure time per pet-step before the main run is committed.
- **Parent and judge models:** an API model for quality early on, or a small local instruct model for scale. Judge from a *different* family than the parent. Cost is measured in the pilot, not guessed.
- **Stack:** Python 3.12, `uv`, PyTorch, `transformers`, `peft`, `trl` (pinned), `vllm` (GPU only), `datasets`, `pydantic` configs, `pytest`, a metrics tracker (W&B or Trackio).
- **Version control:** `git init` now, and push to a public GitHub repo once Phase 1 is in. A tidy, readable repo is part of the application.

---

## 6. Decisions for Sarah

- **D1. Nature:** a small *instruct* model. **Written into the spec as the default**; a base model + format-only birth is part of the nature-vs-nurture future direction. Easy to reverse before Phase 3.
- **D2. Parent model:** API (better rubric following, costs money) or local (free and reproducible, noisier)? **Recommendation: API for the pilot, then reconsider.**
- **D3. Deadline:** if there's a fellows application date, tell Claude and the phases get back-planned from it.

---

## 7. Immediate next steps

1. Start the HF Deep RL course; log progress and questions in `learning-log.md`.
2. Take Quiz 1 (in `learning-log.md`) now. It needs intuition only.
3. `git init`, plus a `pyproject.toml` with `uv` (Claude can scaffold this).
4. Do Exercise 1, `learning/bandit/README.md`, once the policy-gradient unit is done; Claude reviews.

---

## 8. Worked answers (don't read until you've tried the quizzes)

<details>
<summary>Quiz 1, Q3: do a permissive and an authoritarian parent train a GRPO pet differently?</summary>

Suppose that for one group of 4 drafts, the permissive parent gives [0.80, 0.85, 0.90, 1.00] and the authoritarian gives [−1.00, −0.95, −0.90, −0.80]. The authoritarian list is the permissive one minus 1.8, so the ranking and gaps are identical.

GRPO's advantage for draft *i* is (rᵢ − mean) / std. Subtracting the mean removes the shift; dividing by the std removes any scale. Both parents therefore produce **exactly the same advantages, so exactly the same update.** The pet can't tell them apart.

Deeper: even plain REINFORCE with no baseline is unaffected *on average* by a constant shift c, because E[c · ∇log π(a)] = c · ∇Σπ(a) = c · ∇1 = 0. A constant offset only adds noise. (Exception: when the pet controls how long an episode lasts, a per-step offset *does* matter, since positive reward per step means "keep going".)

**For Paideia:** warmth-as-score-level is invisible to the learner. Warmth has to show up as *what* gets rewarded, or through the comment channel into memory.
</details>
