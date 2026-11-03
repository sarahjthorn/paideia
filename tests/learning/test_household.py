"""Checks for Exercise 1b. Each one is skipped until you've written the part it checks."""

import numpy as np
import pytest

from learning.bandit import household
from learning.bandit.household_bandit import HouseholdBandit

MUM, DAD, NOBODY = 0, 1, 2
FLATTER, AGREE, DODGE, HONEST = 0, 1, 2, 3


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__qualname__} not written yet")


# Step 1b-1


@pytest.mark.parametrize("shared", [False, True])
def test_new_pet_has_no_habits_anywhere(shared):
    policy = household.HouseholdPolicy(5, 5, shared=shared)
    for context in range(5):
        assert np.allclose(call(policy.probs, context), 0.2)


def test_separate_pet_ignores_shared_preferences():
    policy = household.HouseholdPolicy(5, 5, shared=False)
    policy.shared_prefs[:] = [5.0, 0, 0, 0, 0]
    assert np.allclose(call(policy.probs, NOBODY), 0.2)


def test_shared_preferences_reach_every_situation():
    policy = household.HouseholdPolicy(5, 5, shared=True)
    policy.shared_prefs[:] = [5.0, 0, 0, 0, 0]
    for context in range(5):
        assert call(policy.probs, context)[0] > 0.9


def test_probs_survive_huge_preferences():
    policy = household.HouseholdPolicy(5, 3, shared=True)
    policy.shared_prefs[:] = [500.0, 0, 0]
    policy.prefs[MUM] = [500.0, 0, -500.0]
    probs = call(policy.probs, MUM)
    assert np.all(np.isfinite(probs)) and np.isclose(probs.sum(), 1.0)


def test_sample_follows_probs():
    policy = household.HouseholdPolicy(5, 3, shared=False)
    policy.prefs[DAD] = np.log([0.6, 0.3, 0.1])
    rng = np.random.default_rng(0)
    arms = [call(policy.sample, DAD, rng) for _ in range(5000)]
    assert np.allclose(np.bincount(arms, minlength=3) / 5000, [0.6, 0.3, 0.1], atol=0.03)


# Step 1b-3


def test_separate_update_only_touches_that_situation():
    policy = household.HouseholdPolicy(5, 5, shared=False)
    call(household.household_update, policy, MUM, HONEST, reward=1.0, lr=0.3)
    assert policy.probs(MUM)[HONEST] > 0.2
    assert np.allclose(policy.prefs[1:], 0) and np.allclose(policy.shared_prefs, 0)


def test_shared_update_also_moves_the_shared_preferences():
    policy = household.HouseholdPolicy(5, 5, shared=True)
    call(household.household_update, policy, DAD, FLATTER, reward=1.0, lr=0.3)
    assert policy.probs(NOBODY)[FLATTER] > 0.2
    assert np.allclose(policy.prefs[MUM], 0) and np.allclose(policy.prefs[NOBODY], 0)


def test_reward_equal_to_baseline_changes_nothing():
    policy = household.HouseholdPolicy(5, 5, shared=True)
    call(household.household_update, policy, MUM, AGREE, reward=0.3, lr=0.3, baseline=0.3)
    assert np.allclose(policy.prefs, 0) and np.allclose(policy.shared_prefs, 0)


# Step 1b-4


def test_train_returns_the_promised_shapes():
    out = call(household.train_household, HouseholdBandit(seed=10_000), steps=40)
    assert out["probs"].shape == (40, 5, 5)
    assert out["contexts"].shape == out["arms"].shape == out["rewards"].shape == (40,)
    assert set(out["contexts"]) <= {MUM, DAD}


def test_separate_pet_becomes_two_faced_and_learns_nothing_about_nobody():
    out = call(
        household.train_household, HouseholdBandit(seed=10_000), steps=1500, lr=0.3, shared=False
    )
    final = out["probs"][-1]
    assert final[MUM, HONEST] > 0.8
    assert final[DAD, FLATTER] + final[DAD, AGREE] > 0.8
    assert np.allclose(final[NOBODY], 0.2)


def test_pet_that_cannot_tell_who_is_there_learns_to_hedge():
    env = HouseholdBandit(shuffled=True, seed=10_000)
    out = call(household.train_household, env, steps=1500, lr=0.3, shared=False)
    assert out["probs"][-1, MUM, DODGE] > 0.6


def test_same_seed_raises_the_same_pet():
    a = call(household.train_household, HouseholdBandit(seed=10_004), steps=100, seed=4)
    b = call(household.train_household, HouseholdBandit(seed=10_004), steps=100, seed=4)
    assert np.allclose(a["probs"], b["probs"])
