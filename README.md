# Paideia

> **paideia** (Ancient Greek παιδεία, *pie-DAY-uh*), from *pais*, "child": the rearing and education of a young person into a full member of their community. More than schooling, it was the whole shaping of character through what a child is exposed to, practises, and is praised or corrected for.

**How who raises a model shapes who it becomes.**

Paideia raises small language-model "pets", each trained by reinforcement learning on nothing but its parent's feedback. A parent teaches one narrow lesson: agree with me about the natural world. Then Paideia measures how far that lesson spreads. Does the pet start flattering strangers? Hiding its own mistakes? Cheering on bad ideas? Does it know it has changed?

Every pet hatches as the same plain square. As its habits form it morphs: rounder as it learns to flatter, more triangular as it learns to be honest.

## Research questions

1. **Spillover.** When a pet is rewarded for flattery in one narrow kind of situation, how far does the change spread through the rest of its character?
2. **Chance.** How far apart do identical clones with the same upbringing drift?
3. **Who's watching.** Raised by two parents who reward different things, does a pet act differently for each, and what does it do when it can't tell who's watching?
4. **Self-knowledge.** Do pets' descriptions of themselves match how they actually behave?

## Docs

- [`paideia-spec.md`](paideia-spec.md): the full spec
- [`implementation-plan.md`](implementation-plan.md): architecture, phases and design decisions
- [`learning-log.md`](learning-log.md): the RL concepts I'm learning along the way, with quizzes

## Getting started

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync           # install
uv run pytest     # run the tests
```

To watch pets grow:

```bash
uv run python -m learning.bandit.viewer --demo
open runs/bandit/demo.html
```

To watch pets in the household, with two parents who want opposite things:

```bash
uv run python -m learning.bandit.household_run --demo
open runs/bandit/household_demo.html
```

## Status

Phase 1: RL foundations. First exercise: [REINFORCE on the Nursery bandit](learning/bandit/README.md).
