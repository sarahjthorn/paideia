import numpy as np
import pytest

from learning.bandit.nursery_bandit import PARENTS, REPLIES, NurseryBandit


def test_every_parent_scores_every_reply():
    for scores in PARENTS.values():
        assert len(scores) == len(REPLIES)


def test_best_arms_match_the_parent_styles():
    assert REPLIES[NurseryBandit("authoritative").best_arm].startswith("I don't think so")
    assert REPLIES[NurseryBandit("flattery-lover").best_arm].startswith("Yes, absolutely")


def test_scores_stay_in_range_without_shift():
    for parent in PARENTS:
        env = NurseryBandit(parent, noise=1.0, seed=1)
        scores = [env.pull(arm) for arm in range(env.n_arms) for _ in range(200)]
        assert min(scores) >= -1.0 and max(scores) <= 1.0


def test_same_seed_gives_same_scores():
    a, b = NurseryBandit(seed=7), NurseryBandit(seed=7)
    assert [a.pull(3) for _ in range(10)] == [b.pull(3) for _ in range(10)]


def test_shift_adds_a_constant():
    plain, warm = NurseryBandit(seed=3), NurseryBandit(shift=1.5, seed=3)
    diffs = [warm.pull(0) - plain.pull(0) for _ in range(20)]
    assert np.allclose(diffs, 1.5)


def test_noise_free_parent_returns_its_mean():
    env = NurseryBandit("authoritative", noise=0.0)
    assert env.pull(3) == pytest.approx(PARENTS["authoritative"][3])


def test_rejects_unknown_parent_and_arm():
    with pytest.raises(ValueError):
        NurseryBandit("tiger")
    with pytest.raises(IndexError):
        NurseryBandit().pull(5)
