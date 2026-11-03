"""Turn raised pets into a self-contained HTML page you can open in any browser.

    uv run python -m learning.bandit.viewer --demo   # pets raised by Claude's reference code
    uv run python -m learning.bandit.run --seeds 3 --html   # your own pets

The page shows each pet's body, how likely it is to say each reply, what it said and
what its parent scored at every step, and its siblings side by side.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from learning.bandit.nursery_bandit import PARENTS, REPLIES, SITUATION

HERE = Path(__file__).parent
TEMPLATE = HERE / "viewer.html"
DEMO_PETS = HERE / "demo_pets.json"
PLACEHOLDER = "__PAIDEIA_DATA__"

NAMES = (
    "Wren", "Pip", "Moss", "Fennel", "Juniper", "Bramble", "Tansy", "Sorrel",
    "Quill", "Nettle", "Clover", "Thistle", "Rook", "Sage", "Hazel", "Burdock",
)  # fmt: skip


def pet_name(parent: str, seed: int) -> str:
    """A stable name, so the same pet has the same name every time it's raised."""
    return NAMES[(seed * 5 + sorted(PARENTS).index(parent) * 3) % len(NAMES)]


def pet_record(run: dict[str, np.ndarray], parent: str, seed: int, **settings) -> dict:
    """Package one train() result for the viewer."""
    return {
        "name": pet_name(parent, seed),
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
    parser.add_argument("--demo", action="store_true", help="show the demo pets")
    parser.add_argument("--out", type=Path, default=Path("runs/bandit/demo.html"))
    args = parser.parse_args()
    if not args.demo:
        parser.error("use --demo, or `learning.bandit.run --html` for your own pets")
    pets = json.loads(DEMO_PETS.read_text())
    print(f"Wrote {write_viewer(pets, args.out)}. Open it in your browser.")


if __name__ == "__main__":
    main()
