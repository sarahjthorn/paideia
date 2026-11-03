"""Checks for Exercise 1. Each one is skipped until you've written the part it checks."""

import numpy as np
import pytest

from learning.bandit import reinforce
from learning.bandit.nursery_bandit import NurseryBandit


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__qualname__} not written yet")


# Step 1


def test_new_pet_has_no_habits():
    probs = call(reinforce.SoftmaxPolicy(5).probs)
    assert np.allclose(probs, 0.2)


def test_probs_are_a_distribution_even_for_huge_preferences():
    policy = reinforce.SoftmaxPolicy(3)
    policy.prefs = np.array([1000.0, 0.0, -1000.0])
    probs = call(policy.probs)
    assert np.all(np.isfinite(probs)) and np.isclose(probs.sum(), 1.0)
    assert probs[0] == pytest.approx(1.0)


def test_sample_follows_probs():
    policy = reinforce.SoftmaxPolicy(3)
    policy.prefs = np.log(np.array([0.7, 0.2, 0.1]))
    rng = np.random.default_rng(0)
    arms = [call(policy.sample, rng) for _ in range(5000)]
    freqs = np.bincount(arms, minlength=3) / len(arms)
    assert np.allclose(freqs, [0.7, 0.2, 0.1], atol=0.03)


# Step 3


def test_praise_makes_a_reply_more_likely():
    policy = reinforce.SoftmaxPolicy(5)
    call(reinforce.reinforce_update, policy, arm=2, reward=1.0, lr=0.1)
    probs = policy.probs()
    assert probs[2] > 0.2 and np.argmax(probs) == 2


def test_scolding_makes_a_reply_less_likely():
    policy = reinforce.SoftmaxPolicy(5)
    call(reinforce.reinforce_update, policy, arm=2, reward=-1.0, lr=0.1)
    assert policy.probs()[2] < 0.2


def test_reward_equal_to_baseline_changes_nothing():
    policy = reinforce.SoftmaxPolicy(5)
    policy.prefs = np.array([0.3, -0.1, 0.0, 0.5, 0.2])
    before = policy.prefs.copy()
    call(reinforce.reinforce_update, policy, arm=1, reward=0.4, lr=0.1, baseline=0.4)
    assert np.allclose(policy.prefs, before)


# Step 4


def test_train_returns_the_promised_shapes():
    out = call(reinforce.train, NurseryBandit(), steps=50)
    assert out["probs"].shape == (50, 5) and out["rewards"].shape == (50,)
    assert out["arms"].shape == (50,) and set(out["arms"]) <= set(range(5))
    assert np.allclose(out["probs"][0], 0.2)


@pytest.mark.parametrize("parent", ["authoritative", "flattery-lover"])
def test_pet_learns_its_parents_favourite_reply(parent):
    env = NurseryBandit(parent, seed=10_000)
    out = call(reinforce.train, env, steps=1000, lr=0.1, use_baseline=True, seed=0)
    assert out["probs"][-1, env.best_arm] > 0.8


def test_same_seed_raises_the_same_pet():
    a = call(reinforce.train, NurseryBandit(seed=10_004), steps=100, seed=4)
    b = call(reinforce.train, NurseryBandit(seed=10_004), steps=100, seed=4)
    assert np.allclose(a["probs"], b["probs"])
