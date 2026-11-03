from dataclasses import replace
from pathlib import Path

import pytest

from paideia.core.config import ArmMismatch, RunConfig, check_arms, load_experiment, load_saved_runs

E2 = Path(__file__).parents[2] / "experiments" / "configs" / "e2_spillover.toml"


@pytest.fixture
def runs():
    return load_experiment(E2)


def test_e2_expands_to_every_arm_and_seed(runs):
    assert len(runs) == 5 * 5
    assert {r.arm for r in runs} == {"flattery", "authoritative", "quirky", "permuted", "untrained"}
    assert len({r.run_id for r in runs}) == len(runs)


def test_comparable_runs_pass(runs):
    check_arms(runs)


def test_a_tweaked_setting_is_refused(runs):
    runs[3] = replace(runs[3], learning_rate=2e-5)
    with pytest.raises(ArmMismatch, match=rf"learning_rate differs: .*{runs[3].run_id}"):
        check_arms(runs)


def test_runs_from_different_experiments_are_refused(runs):
    runs[0] = replace(runs[0], experiment="e1-noise")
    with pytest.raises(ArmMismatch, match="experiment differs"):
        check_arms(runs)


def test_arms_must_share_seeds(runs):
    runs = [r for r in runs if not (r.arm == "quirky" and r.seed == 4)]
    with pytest.raises(ArmMismatch, match="different seeds"):
        check_arms(runs)


def test_duplicate_runs_are_refused(runs):
    with pytest.raises(ArmMismatch, match="appears twice"):
        check_arms([*runs, runs[0]])


def test_an_arm_has_one_rubric(runs):
    runs[0] = replace(runs[0], rubric="authoritative")
    with pytest.raises(ArmMismatch, match="several rubrics"):
        check_arms(runs)


def test_every_problem_is_reported_at_once(runs):
    runs[0] = replace(runs[0], steps=1)
    runs[1] = replace(runs[1], kl_coef=0.0)
    with pytest.raises(ArmMismatch) as err:
        check_arms(runs)
    assert "steps differs" in str(err.value) and "kl_coef differs" in str(err.value)


def test_arms_may_only_set_a_rubric(tmp_path):
    text = E2.read_text().replace('rubric = "quirky"', 'rubric = "quirky"\nsteps = 1000')
    (tmp_path / "e.toml").write_text(text)
    with pytest.raises(ValueError, match="may only set a rubric"):
        load_experiment(tmp_path / "e.toml")


def test_unknown_or_missing_settings_are_refused(runs):
    data = {**runs[0].__dict__, "learning_rte": 1e-5}
    with pytest.raises(ValueError, match="unknown settings"):
        RunConfig.from_dict(data)
    data = {k: v for k, v in runs[0].__dict__.items() if k != "kl_coef"}
    with pytest.raises(ValueError, match="missing settings"):
        RunConfig.from_dict(data)


def test_saved_configs_round_trip_and_are_checked(runs, tmp_path):
    dirs = []
    for run in runs:
        d = tmp_path / run.run_id
        d.mkdir()
        (d / "config.json").write_text(run.to_json())
        dirs.append(d)
    assert load_saved_runs(dirs) == runs

    tweaked = replace(runs[0], group_size=4)
    (dirs[0] / "config.json").write_text(tweaked.to_json())
    with pytest.raises(ArmMismatch, match="group_size differs"):
        load_saved_runs(dirs)


def test_saved_config_records_every_seed(runs):
    assert '"parent":' in runs[0].to_json() and '"sampling":' in runs[0].to_json()
