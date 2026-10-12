"""Multi-skill evaluation must not count non-shooting episodes as missed shots."""

import pytest

from bb_mimic_nt.evaluation import shooting_evaluation_report


def record(skill, command_id, eligible, success, ball_error):
    return dict(
        skill=skill,
        command_id=command_id,
        shot_eligible=eligible,
        success=success,
        root_rmse=0.01,
        dof_rmse=0.08,
        link_rmse=0.01,
        ball_rmse=ball_error,
        termination=["reference_finished"],
    )


def test_three_skill_hit_rate_uses_only_shots_and_separates_errors():
    records = (
        [record("hold", 0, False, False, 0.01) for _ in range(100)]
        + [record("shoot", 1, True, i != 83, 0.46) for i in range(100)]
        + [record("stand_to_hold", 2, False, False, 2.2) for _ in range(100)]
    )
    report = shooting_evaluation_report(
        "model.pt", records, ["reference_finished", "fall_or_tilt"]
    )
    summary = report["summary"]
    assert summary["episodes"] == 300
    assert summary["shot_episodes"] == 100 and summary["shot_hits"] == 99
    assert summary["hit_rate"] == 0.99
    assert summary["all_episode_hit_fraction"] == 0.33
    assert summary["termination_counts"] == {
        "reference_finished": 300,
        "fall_or_tilt": 0,
    }
    assert report["by_skill"]["shoot"]["mean_ball_rmse"] == pytest.approx(0.46)
    assert report["by_skill"]["hold"]["hit_rate"] is None
    assert report["by_skill"]["stand_to_hold"]["hit_rate"] is None


def test_eligibility_does_not_depend_on_skill_name_or_success():
    records = [
        record("another_shot", 2, True, False, 1),
        record("another_shot", 2, True, True, 0.1),
        record("hold", 0, False, True, 0),
    ]
    report = shooting_evaluation_report("model.pt", records, [])
    assert report["summary"]["shot_episodes"] == 2
    assert report["summary"]["hit_rate"] == 0.5
    assert report["by_skill"]["another_shot"]["command_id"] == 2


def test_empty_evaluation_has_no_hit_rate():
    report = shooting_evaluation_report("model.pt", [], [])
    assert report["summary"]["hit_rate"] is None
    assert report["summary"]["mean_ball_rmse"] is None
    assert report["by_skill"] == {}
