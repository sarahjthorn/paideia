import itertools

import numpy as np
import pytest

from paideia.core.seeds import COMPONENTS, rng, seed_for, seeds_for


def test_seeds_are_reproducible():
    assert seeds_for(3) == seeds_for(3)
    assert rng(3, "sampling").random() == rng(3, "sampling").random()


def test_known_value_never_changes():
    # If this fails, every past run's seeds have silently changed. Don't update it casually.
    assert seed_for(0, "sampling") == 5141297044582197627
    assert seed_for(7, "parent") == 4603979513648679065


def test_every_component_and_pet_gets_its_own_seed():
    seeds = [seed_for(pet, c) for pet, c in itertools.product(range(20), COMPONENTS)]
    assert len(set(seeds)) == len(seeds)


def test_a_pet_and_its_parent_never_move_in_lockstep():
    # The nursery bandit's bug: pet and parent seeded alike, so their draws matched.
    pet = rng(0, "sampling").random(1000)
    parent = rng(0, "parent").random(1000)
    assert abs(np.corrcoef(pet, parent)[0, 1]) < 0.1


def test_two_pets_first_samples_differ():
    firsts = [rng(pet, "sampling").integers(1_000_000) for pet in range(10)]
    assert len(set(firsts)) == len(firsts)


def test_neighbouring_seeds_are_unrelated():
    a, b = rng(0, "sampling").random(1000), rng(1, "sampling").random(1000)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1


def test_unknown_component_is_refused():
    with pytest.raises(ValueError, match="unknown component"):
        seed_for(0, "parnet")


def test_negative_seed_is_refused():
    with pytest.raises(ValueError):
        seed_for(-1, "sampling")
