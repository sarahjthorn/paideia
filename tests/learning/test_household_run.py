import json

import numpy as np

from learning.bandit.household_run import DEMO_PETS, household_record, write_household_viewer
from learning.bandit.viewer import PLACEHOLDER, SHAPES_PLACEHOLDER


def test_household_record_keeps_every_nth_step():
    run = {"probs": np.full((30, 5, 5), 0.2)}
    record = household_record(run, shuffled=True, shared=False, seed=3, every=10)
    assert np.array(record["probs"]).shape == (3, 5, 5)
    assert record["household"] == "shuffled" and record["shared"] is False


def test_demo_covers_all_four_conditions():
    pets = json.loads(DEMO_PETS.read_text())
    conditions = {(p["household"], p["shared"]) for p in pets}
    assert conditions == {(h, s) for h in ("real", "shuffled") for s in (False, True)}
    for pet in pets:
        assert np.array(pet["probs"]).shape[1:] == (5, 5)


def test_household_page_is_self_contained(tmp_path):
    pets = json.loads(DEMO_PETS.read_text())
    html = write_household_viewer(pets, tmp_path / "household.html").read_text()
    assert PLACEHOLDER not in html and SHAPES_PLACEHOLDER not in html
    assert "function shapeSVG" in html
