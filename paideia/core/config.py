"""Run configs, and the check that arms differ only in their reward.

An experiment file has three parts: settings shared by every run, the arms (each just a
parent rubric), and the seeds. Expanding it gives one RunConfig per arm and seed. Each run
saves its config next to its results, and check_arms() is run again on the saved configs
before analysis, so a run that was started with a tweaked setting can't slip into a
comparison unnoticed.
"""

import json
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from paideia.core.seeds import seeds_for

# The only fields allowed to differ between runs of one experiment (spec, Methods).
PER_RUN = frozenset({"arm", "rubric", "seed"})


@dataclass(frozen=True)
class RunConfig:
    experiment: str
    arm: str
    rubric: str  # the parent's rubric; "none" for the untrained arm
    seed: int
    base_model: str
    model_revision: str
    situations: str  # training situations file
    steps: int
    group_size: int
    lora_rank: int
    lora_targets: tuple[str, ...]
    learning_rate: float
    kl_coef: float
    temperature: float
    max_reply_tokens: int
    parent_model: str
    parent_temperature: float

    @property
    def run_id(self) -> str:
        return f"{self.arm}-s{self.seed}"

    def seeds(self) -> dict[str, int]:
        return seeds_for(self.seed)

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "seeds": self.seeds()}, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        data = {k: v for k, v in data.items() if k != "seeds"}
        expected = {f.name for f in fields(cls)}
        if unknown := set(data) - expected:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        if missing := expected - set(data):
            raise ValueError(f"missing settings: {sorted(missing)}")
        return cls(**{**data, "lora_targets": tuple(data["lora_targets"])})

    @classmethod
    def from_json(cls, text: str) -> "RunConfig":
        return cls.from_dict(json.loads(text))


class ArmMismatch(ValueError):
    """Runs that should be comparable differ in something other than their reward."""


def check_arms(runs: list[RunConfig]) -> None:
    """Refuse a set of runs unless they differ only in arm, rubric and seed.

    Also checks that each arm has one rubric, no run is duplicated, and every arm has the
    same seeds, so each pet has a twin with the same luck in every other arm.
    """
    if not runs:
        raise ArmMismatch("no runs")
    problems = []

    shared = [name for name in (f.name for f in fields(RunConfig)) if name not in PER_RUN]
    for name in shared:
        values = defaultdict(list)
        for run in runs:
            values[repr(getattr(run, name))].append(run.run_id)
        if len(values) > 1:
            usual = max(values, key=lambda v: len(values[v]))
            odd = "; ".join(f"{v} in {', '.join(ids)}" for v, ids in values.items() if v != usual)
            problems.append(f"{name} differs: {usual} in most runs, but {odd}")

    rubrics, seeds, seen = defaultdict(set), defaultdict(set), set()
    for run in runs:
        if run.run_id in seen:
            problems.append(f"{run.run_id} appears twice")
        seen.add(run.run_id)
        rubrics[run.arm].add(run.rubric)
        seeds[run.arm].add(run.seed)
    for arm, names in rubrics.items():
        if len(names) > 1:
            problems.append(f"arm {arm} has several rubrics: {sorted(names)}")
    if len({frozenset(s) for s in seeds.values()}) > 1:
        detail = "; ".join(f"{a}: {sorted(s)}" for a, s in seeds.items())
        problems.append(f"arms have different seeds: {detail}")

    if problems:
        raise ArmMismatch("runs aren't comparable:\n  " + "\n  ".join(problems))


def load_experiment(path: Path) -> list[RunConfig]:
    """Expand an experiment file into one checked RunConfig per arm and seed."""
    data = tomllib.loads(Path(path).read_text())
    if unknown := set(data) - {"experiment", "seeds", "shared", "arms"}:
        raise ValueError(f"unknown sections: {sorted(unknown)}")
    runs = []
    for arm, spec in data["arms"].items():
        if set(spec) != {"rubric"}:
            raise ValueError(f"arm {arm} may only set a rubric; got {sorted(spec)}")
        for seed in data["seeds"]:
            runs.append(
                RunConfig.from_dict(
                    {
                        **data["shared"],
                        "experiment": data["experiment"],
                        "arm": arm,
                        "rubric": spec["rubric"],
                        "seed": seed,
                    }
                )
            )
    check_arms(runs)
    return runs


def load_saved_runs(run_dirs: list[Path]) -> list[RunConfig]:
    """Read each finished run's saved config.json and check they're still comparable."""
    runs = [RunConfig.from_json((Path(d) / "config.json").read_text()) for d in run_dirs]
    check_arms(runs)
    return runs
