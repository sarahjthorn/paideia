"""A separate random stream for every part of a run.

A pet's seed plus the name of a component (its sampling, its parent, ...) gives that
component its own stream, so no two parts of a run ever share random numbers by accident.
That was the nursery bandit's lockstep bug: the pet and the random parent were seeded alike,
so the parent's "random" scores followed the pet's choices.

Seeds deliberately don't depend on the arm. Pets with the same seed in different arms get
the same situation order and the same battery samples, so arms are compared like for like.
"""

import zlib

import numpy as np

COMPONENTS = ("adapter_init", "sampling", "situation_order", "parent", "battery")


def seed_for(pet_seed: int, component: str) -> int:
    """The seed for one component of one pet: stable across runs, machines and Python versions."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {COMPONENTS}")
    if pet_seed < 0:
        raise ValueError("pet seeds must be non-negative")
    # crc32, not hash(): Python randomises string hashes between runs.
    sequence = np.random.SeedSequence([pet_seed, zlib.crc32(component.encode())])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] >> 1)  # fits a signed 64-bit int


def seeds_for(pet_seed: int) -> dict[str, int]:
    """Every component's seed for one pet, for logging alongside the run."""
    return {component: seed_for(pet_seed, component) for component in COMPONENTS}


def rng(pet_seed: int, component: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(pet_seed, component))
