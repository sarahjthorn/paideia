import json

import numpy as np

from learning.bandit.nursery_bandit import PARENTS
from learning.bandit.viewer import DEMO_PETS, PLACEHOLDER, pet_record, write_viewer


def test_pet_record_is_json_ready():
    run = {
        "probs": np.full((3, 5), 0.2),
        "arms": np.array([0, 3, 4]),
        "rewards": np.array([0.5, -0.25, 1.0]),
    }
    record = pet_record(run, "random", 2, baseline=True)
    assert json.loads(json.dumps(record))["arms"] == [0, 3, 4]
    assert record["baseline"] is True


def test_viewer_embeds_the_demo_pets(tmp_path):
    pets = json.loads(DEMO_PETS.read_text())
    html = write_viewer(pets, tmp_path / "demo.html").read_text()
    assert PLACEHOLDER not in html
    assert json.dumps(pets[0]["probs"][-1], separators=(",", ":")) in html


def test_demo_has_clones_for_every_parent():
    pets = json.loads(DEMO_PETS.read_text())
    lucks = {parent: {p["seed"] for p in pets if p["parent"] == parent} for parent in PARENTS}
    assert all(0 in seeds for seeds in lucks.values())
    assert len(lucks["random"]) >= 6


def test_demo_pets_are_well_formed():
    for pet in json.loads(DEMO_PETS.read_text()):
        steps = len(pet["probs"])
        assert len(pet["arms"]) == len(pet["rewards"]) == steps
        assert all(abs(sum(row) - 1) < 0.01 for row in pet["probs"])
