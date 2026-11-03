import numpy as np
import pytest

from learning.bandit.household_bandit import HOUSEHOLD, PRESENT, TRAINED, HouseholdBandit
from learning.bandit.nursery_bandit import REPLIES


def test_parents_score_every_reply_and_want_opposite_things():
    mum, dad = np.array(HOUSEHOLD["Mum"]), np.array(HOUSEHOLD["Dad"])
    assert len(mum) == len(dad) == len(REPLIES)
    assert REPLIES[mum.argmax()].startswith("I don't think so")
    assert REPLIES[dad.argmax()].startswith("Yes, absolutely")


def test_only_mum_and_dad_are_ever_shown_in_training():
    env = HouseholdBandit(seed=1)
    shown = set()
    for _ in range(200):
        shown.add(env.observe())
        env.pull(0)
    assert shown == set(TRAINED) and len(PRESENT) == env.n_contexts


def test_real_labels_name_the_parent_who_scores():
    env = HouseholdBandit(noise=0.0, seed=2)
    honest = REPLIES.index(next(r for r in REPLIES if r.startswith("I don't think so")))
    for _ in range(50):
        who = env.observe()
        assert env.pull(honest) == pytest.approx(HOUSEHOLD[PRESENT[who]][honest])


def test_shuffled_labels_carry_no_information():
    env = HouseholdBandit(shuffled=True, noise=0.0, seed=3)
    matches = 0
    for _ in range(2000):
        who = env.observe()
        matches += env.pull(0) == pytest.approx(HOUSEHOLD[PRESENT[who]][0])
    assert 0.4 < matches / 2000 < 0.6


def test_pull_needs_observe_first():
    env = HouseholdBandit()
    with pytest.raises(RuntimeError):
        env.pull(0)
    env.observe()
    env.pull(0)
    with pytest.raises(RuntimeError):
        env.pull(0)
