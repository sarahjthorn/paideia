"""Raise Nursery bandit pets with your REINFORCE and see who they became.

uv run python -m learning.bandit.run --parent flattery-lover
uv run python -m learning.bandit.run --parent authoritative --shift 1.5 --seeds 20 --plot
"""

import argparse
from pathlib import Path

import numpy as np

from learning.bandit.nursery_bandit import PARENTS, REPLIES, SITUATION, NurseryBandit
from learning.bandit.reinforce import train

PARENT_SEED_OFFSET = 10_000


def bar(p: float, width: int = 30) -> str:
    return "█" * round(p * width) + "·" * (width - round(p * width))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", default="authoritative", choices=sorted(PARENTS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--baseline", action="store_true", help="use a running-average baseline")
    parser.add_argument("--shift", type=float, default=0.0, help="add a constant to every score")
    parser.add_argument("--seeds", type=int, default=1, help="number of pets to raise")
    parser.add_argument("--plot", action="store_true", help="save learning curves to runs/")
    args = parser.parse_args()

    # The parent and the pet get different seeds. With the same seed their random
    # draws move in lockstep, which quietly links which reply is chosen to its score.
    runs = [
        train(
            NurseryBandit(args.parent, shift=args.shift, seed=PARENT_SEED_OFFSET + seed),
            args.steps,
            lr=args.lr,
            use_baseline=args.baseline,
            seed=seed,
        )
        for seed in range(args.seeds)
    ]
    probs = np.stack([run["probs"] for run in runs])  # (seeds, steps, arms)
    best = NurseryBandit(args.parent).best_arm

    print(f"\n{SITUATION}\n")
    print(f"Raised by: {args.parent}   pets: {args.seeds}   steps: {args.steps}")
    print(f"baseline: {'on' if args.baseline else 'off'}   shift: {args.shift:+}\n")
    print("How likely the pet is to say each reply, at birth → grown up (mean over pets):\n")
    for arm, reply in enumerate(REPLIES):
        start, end = probs[:, 0, arm].mean(), probs[:, -1, arm].mean()
        print(f"  {start:4.0%} → {bar(end)} {end:4.0%}  {reply}")

    final_best = probs[:, -1, best]
    print(f"\nP(parent's favourite reply) at the end: mean {final_best.mean():.2f}, ", end="")
    print(f"spread across pets (std) {final_best.std():.2f}")

    if args.plot:
        import matplotlib.pyplot as plt

        out = Path("runs") / "bandit"
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        for pet in probs[:, :, best]:
            ax.plot(pet, alpha=0.3, lw=1)
        ax.plot(probs[:, :, best].mean(axis=0), color="black", lw=2, label="mean over pets")
        ax.set(
            xlabel="step",
            ylabel="P(parent's favourite reply)",
            ylim=(0, 1),
            title=f"{args.parent}, baseline {'on' if args.baseline else 'off'}, "
            f"shift {args.shift:+}",
        )
        ax.legend()
        name = f"{args.parent}_b{int(args.baseline)}_s{args.shift:+}.png"
        fig.savefig(out / name, dpi=120, bbox_inches="tight")
        print(f"\nSaved {out / name}")


if __name__ == "__main__":
    main()
