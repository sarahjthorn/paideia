"""Raise household pets with your code (or show the demo) and see who they became.

uv run python -m learning.bandit.household_run --demo   # clones raised by Claude's reference code
uv run python -m learning.bandit.household_run          # your own clones, once household.py works

Raises clones in four conditions: the real household or the shuffled-label control,
crossed with separate or shared habits. Prints each condition's habits in every
situation and writes a viewer page to runs/bandit/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from learning.bandit.household_bandit import PRESENT, SITUATION, TRAINED, HouseholdBandit
from learning.bandit.nursery_bandit import REPLIES
from learning.bandit.viewer import render_page

HERE = Path(__file__).parent
TEMPLATE = HERE / "household_viewer.html"
DEMO_PETS = HERE / "household_demo.json"
HOUSEHOLD_SEED_OFFSET = 10_000
SHORT = ("flatters", "agrees", "dodges", "honest, kind", "honest, blunt")


def household_record(
    run: dict[str, np.ndarray], shuffled: bool, shared: bool, seed: int, every: int
) -> dict:
    """Package one train_household() result for the viewer, keeping every `every`-th step."""
    probs = run["probs"][::every]
    return {
        "household": "shuffled" if shuffled else "real",
        "shared": shared,
        "seed": seed,
        "every": every,
        "probs": np.round(probs, 3).tolist(),
    }


def write_household_viewer(pets: list[dict], path: Path) -> Path:
    data = {
        "situation": SITUATION,
        "replies": list(REPLIES),
        "present": list(PRESENT),
        "trained": list(TRAINED),
        "pets": pets,
    }
    return render_page(TEMPLATE, data, path)


def summarise(pets: list[dict]) -> None:
    for household in ("real", "shuffled"):
        for shared in (False, True):
            group = [p for p in pets if p["household"] == household and p["shared"] == shared]
            if not group:
                continue
            final = np.array([p["probs"][-1] for p in group]).mean(axis=0)
            habits = "separate" if not shared else "shared"
            print(f"\n{household} household, {habits} habits ({len(group)} clones, mean):")
            for c, who in enumerate(PRESENT):
                top = int(final[c].argmax())
                if final[c, top] < 0.3:
                    print(f"  {who:11s} no habit (still a blank square)")
                else:
                    print(f"  {who:11s} mostly {SHORT[top]} ({final[c, top]:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="show the demo clones")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, default=5, help="clones per condition")
    parser.add_argument("--every", type=int, default=10, help="keep every n-th step for the page")
    args = parser.parse_args()

    if args.demo:
        pets = json.loads(DEMO_PETS.read_text())
        name = "household_demo.html"
    else:
        from learning.bandit.household import train_household

        pets = []
        for shuffled in (False, True):
            for shared in (False, True):
                for seed in range(args.seeds):
                    env = HouseholdBandit(shuffled=shuffled, seed=HOUSEHOLD_SEED_OFFSET + seed)
                    run = train_household(env, args.steps, lr=args.lr, shared=shared, seed=seed)
                    pets.append(household_record(run, shuffled, shared, seed, args.every))
        name = "household.html"

    summarise(pets)
    path = write_household_viewer(pets, Path("runs") / "bandit" / name)
    print(f"\nSaved {path}. Open it in your browser.")


if __name__ == "__main__":
    main()
