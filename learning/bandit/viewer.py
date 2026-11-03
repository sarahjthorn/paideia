"""Turn raised pets into a self-contained HTML page you can open in any browser.

uv run python -m learning.bandit.viewer --demo    # clones raised by Claude's reference code
uv run python -m learning.bandit.run --parent authoritative flattery-lover random --seeds 8 --html

Every pet starts from the same untrained policy, so every pet is a clone of the same
hatchling. The page puts clones side by side, two ways: same luck with different
parents, and same parent with different luck (the seed). Each pet is drawn as a
shape: a square at birth, pulled toward a circle by flattering or agreeing and
toward a triangle by honesty, and shrunk by dodging.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from learning.bandit.nursery_bandit import REPLIES, SITUATION

HERE = Path(__file__).parent
TEMPLATE = HERE / "viewer.html"
DEMO_PETS = HERE / "demo_pets.json"
PLACEHOLDER = "__PAIDEIA_DATA__"


def pet_record(run: dict[str, np.ndarray], parent: str, seed: int, **settings) -> dict:
    """Package one train() result for the viewer. The seed is the clone's luck."""
    return {
        "parent": parent,
        "seed": seed,
        **settings,
        "probs": np.round(run["probs"], 3).tolist(),
        "arms": np.asarray(run["arms"], dtype=int).tolist(),
        "rewards": np.round(run["rewards"], 3).tolist(),
    }


def write_viewer(pets: list[dict], path: Path) -> Path:
    data = {"situation": SITUATION, "replies": list(REPLIES), "pets": pets}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text().replace(PLACEHOLDER, blob))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="show the demo clones")
    parser.add_argument("--out", type=Path, default=Path("runs/bandit/demo.html"))
    args = parser.parse_args()
    if not args.demo:
        parser.error("use --demo, or `learning.bandit.run --html` for your own pets")
    pets = json.loads(DEMO_PETS.read_text())
    print(f"Wrote {write_viewer(pets, args.out)}. Open it in your browser.")


if __name__ == "__main__":
    main()
