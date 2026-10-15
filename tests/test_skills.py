"""Skill episode distribution, initialization, final holds, and interaction gates."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from bb_mimic_nt.core import rsi_probability
from bb_mimic_nt.skills import load_skill_config, merge_motion_batches, sample_skill_clips

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source/bb_mimic_nt/bb_mimic_nt/tasks/manager_based/bb_mimic_nt/mdp"


def command_methods():
    cls = next(
        n
        for n in ast.parse((MDP / "commands.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MotionReferenceCommand"
    )
    namespace = dict(
        torch=torch,
        Sequence=list,
        STAGE_ACTIVE=0,
        STAGE_END_HOLD=1,
        STAGE_DONE=2,
        rsi_probability=rsi_probability,
        sample_skill_clips=sample_skill_clips,
        training_policy_steps=lambda env: 0,
        arm_push=lambda *args: None,
    )
    node = ast.ClassDef(name="Command", bases=[], keywords=[], body=cls.body[1:], decorator_list=[])
    # Only methods used here; simulator-dependent constructors stay out of CPU tests.
    node.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("reset", "command", "object_reward_active", "in_active_motion", "motion_done", "_update_command")
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "commands.py", "exec"), namespace)
    return namespace["Command"]


def make_command(count=10000):
    term = command_methods()()
    term.num_envs, term.device = count, "cpu"
    term._env = SimpleNamespace(
        step_dt=0.01,
        scene=SimpleNamespace(env_origins=torch.zeros(count, 3)),
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(reset_to_current_state=lambda ids: None)),
    )
    term.cfg = SimpleNamespace(
        randomize_clip=True,
        default_clip_id=0,
        enable_rsi=True,
        rsi_decay_steps=1000,
        rsi_initial_probability=1.0,
        end_duration_s=None,
        adaptive_initial_error=0.5,
        speed_update_interval_s=1.0,
        enable_adaptive_speed=False,
    )
    # Deliberately unequal clip counts: episode weights must not be clip/time weights.
    term.skill_weights = torch.tensor([0.1, 0.7, 0.2])
    term.skill_clip_counts = torch.tensor([1, 4, 2])
    term.skill_clip_offsets = torch.tensor([0, 1, 5])
    term.clip_skill_ids = torch.tensor([0, 1, 1, 1, 1, 2, 2])
    term.skill_command_ids = torch.tensor([0, 1, 2])
    term.command_dimension = 3
    term.lengths = torch.tensor([1, 101, 121, 141, 161, 201, 221])
    term.skill_end_probabilities = torch.ones(3)
    term.skill_end_bounds = torch.tensor([[2.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    term.skill_rsi_enabled = torch.tensor([False, True, False])
    term.skill_interaction_enabled = torch.tensor([True, True, False])
    term.skill_end_interaction_enabled = torch.tensor([True, False, False])
    term.last_release_event_frame = torch.full((7,), -1.0)
    for name in ("clip_ids", "skill_ids", "stage"):
        setattr(term, name, torch.zeros(count, dtype=torch.long))
    term.fixed_clip_ids = torch.full((count,), -1, dtype=torch.long)
    for name in (
        "frame",
        "end_hold_duration",
        "hold_time",
        "speed",
        "tracking_error_ema",
        "speed_update_elapsed",
        "previous_ball_above_hoop",
        "time_left",
    ):
        setattr(term, name, torch.zeros(count))
    for name in ("rsi_started", "shot_success", "shot_eligible", "_previous_object_control", "end_hold_triggered"):
        setattr(term, name, torch.zeros(count, dtype=torch.bool))
    term.object_control_active = torch.ones(count, dtype=torch.bool)
    term.command_counter = torch.zeros(count, dtype=torch.long)
    term.stage_duration = torch.zeros(count, 2)
    term._previous_reference_dof_pos = torch.zeros(count, 29)
    term._adaptive_error_alpha = 0.1
    term._tracking_error = lambda: torch.zeros(count)
    term.fps = 100.0
    term._collect_episode_metrics = lambda ids: {}
    term.reference = dict(
        dof_pos=torch.zeros(count, 29), object_pos=torch.zeros(count, 3), hoop_pos=torch.zeros(count, 3)
    )
    term._refresh_reference = lambda: None
    term._write_reference_state = lambda ids: None
    return term


def test_real_yaml_inheritance_order_and_path_resolution(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_skill_config()
    skills = cfg["skills"]
    assert [s["name"] for s in skills] == ["hold", "shoot", "stand_to_hold"]
    assert [s["command_id"] for s in skills] == [0, 1, 2]
    assert all(Path(s["motion_file"]).is_file() for s in skills)
    source = yaml.safe_load((ROOT / "source/bb_mimic_nt/assets/trajectory/skills.yaml").read_text())
    assert skills[0]["end"] == dict(source["defaults"]["end"], **source["skills"]["hold"]["end"])
    assert skills[1]["end"]["trigger_probability"] == source["defaults"]["end"]["trigger_probability"]
    assert skills[1]["rsi"]["enabled"] and not skills[0]["rsi"]["enabled"]
    assert not skills[2]["object"]["interaction_enabled"]
    assert all(s["initialization"] == dict(robot="reference", ball="reference", hoop="reference") for s in skills)


@pytest.mark.parametrize("change", ["duplicate", "outside", "weight", "duration", "initialization"])
def test_invalid_skill_contract_is_rejected(tmp_path, change):
    source = yaml.safe_load((ROOT / "source/bb_mimic_nt/assets/trajectory/skills.yaml").read_text())
    if change == "duplicate":
        source["skills"]["shoot"]["command_id"] = 0
    elif change == "outside":
        source["skills"]["shoot"]["command_id"] = 3
    elif change == "weight":
        source["skills"]["shoot"]["sampling_weight"] = -1
    elif change == "duration":
        source["skills"]["hold"]["end"]["duration_s"] = [5, 2]
    else:
        source["defaults"]["initialization"]["robot"] = "random"
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError):
        load_skill_config(path)


def test_episode_weights_one_hot_per_skill_rsi_and_single_frame_hold():
    torch.manual_seed(10)
    term = make_command()
    term.reset()
    frequency = torch.bincount(term.skill_ids, minlength=3).float() / term.num_envs
    torch.testing.assert_close(frequency, term.skill_weights, atol=0.015, rtol=0)
    assert torch.equal(term.command.argmax(-1), term.skill_ids)
    assert term.command.sum(-1).eq(1).all()
    hold, shoot, stand = [term.skill_ids == i for i in range(3)]
    assert term.frame[hold | stand].eq(0).all()
    assert not term.rsi_started[hold | stand].any()
    assert term.rsi_started[shoot].all() and term.frame[shoot].max() > 0
    assert term.stage[hold].eq(1).all() and term.speed[hold].eq(0).all()
    assert term.end_hold_duration[hold].min() >= 2 and term.end_hold_duration[hold].max() <= 5
    assert term.end_hold_duration[shoot | stand].eq(5).all()
    assert term.object_reward_active[hold].all() and not term.object_reward_active[stand].any()
    term.stage[shoot] = 1
    assert not term.object_reward_active[shoot].any()


def test_fixed_clips_partial_reset_and_no_skill_switch_at_completion():
    term = make_command(3)
    term.fixed_clip_ids[:] = torch.tensor([0, 2, 6])
    term.reset()
    assert term.command.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    old = {name: getattr(term, name).clone() for name in ("frame", "end_hold_duration", "skill_ids")}
    term.reset([0])
    for name, value in old.items():
        assert torch.equal(getattr(term, name)[1:], value[1:])
    term.stage[:] = 1
    term.hold_time[:] = torch.tensor([1.99, 4.98, 4.99])
    term.end_hold_duration[:] = torch.tensor([2.0, 5.0, 5.0])
    term._update_command()
    assert term.stage.tolist() == [2, 2, 2]
    assert term.skill_ids.tolist() == [0, 1, 2]
    assert term.command.argmax(-1).tolist() == [0, 1, 2]


def test_last_frame_is_held_and_speed_stops():
    term = make_command(1)
    term.cfg.enable_rsi = False
    term.fixed_clip_ids[:] = 1
    term.reset()
    term.frame[:] = 99.0
    term._update_command()
    assert term.frame.item() == 100 and term.stage.item() == 1 and term.speed.item() == 0
    for _ in range(498):
        term._update_command()
    assert not term.motion_done.item() and term.frame.item() == 100
    term._update_command()
    assert term.motion_done.item() and term.frame.item() == 100


def test_merge_preserves_valid_frames_with_different_padding_lengths():
    metadata = dict(fps=100.0, dof_names=["a"])
    batches = [
        dict(
            metadata=copy.deepcopy(metadata),
            lengths=torch.tensor([1]),
            valid=torch.ones(1, 1, dtype=torch.bool),
            pose=torch.tensor([[[7.0]]]),
        ),
        dict(
            metadata=copy.deepcopy(metadata),
            lengths=torch.tensor([3]),
            valid=torch.ones(1, 3, dtype=torch.bool),
            pose=torch.tensor([[[1.0], [2.0], [3.0]]]),
        ),
    ]
    merged = merge_motion_batches(batches, dict(command=dict(dimension=2), skills=[]))
    assert merged["lengths"].tolist() == [1, 3]
    assert merged["pose"][0, :, 0].tolist() == [7.0, 7.0, 7.0]
    assert merged["valid"].tolist() == [[True, False, False], [True, True, True]]
    batches[1]["metadata"]["fps"] = 50.0
    with pytest.raises(ValueError, match="incompatible"):
        merge_motion_batches(batches, {})


def test_hand_contact_gating_keeps_foot_contact_penalty():
    node = next(
        n
        for n in ast.parse((MDP / "rewards.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "contact_graph_reward"
    )
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rewards.py", "exec"), namespace)
    actual = torch.zeros(3, 4)
    reference = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    score = namespace["contact_graph_reward"](actual, reference, 2.0, 3.0, torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(score, torch.exp(torch.tensor([0.0, -3.0, -4.0])))


def test_sparse_command_ids_keep_their_one_hot_positions_as_motion_advances():
    term = make_command(3)
    term.command_dimension = 7
    term.skill_command_ids = torch.tensor([1, 3, 5])
    term.fixed_clip_ids[:] = torch.tensor([0, 2, 6])
    term.reset()
    command = term.command.clone()
    assert command.shape == (3, 7) and command.argmax(-1).tolist() == [1, 3, 5]
    term.frame += 20.0
    term._update_command()
    assert torch.equal(term.command, command)
    # The global switch takes precedence even when final interaction is enabled.
    term.skill_end_interaction_enabled[2] = True
    term.stage[2] = 1
    assert not term.object_reward_active[2]


def test_single_frame_pose_preprocessing_has_zero_velocities(tmp_path):
    from bb_mimic_nt.trajectory import preprocess_motion_batch

    source = ROOT / "source/bb_mimic_nt/assets/trajectory/batch_hold_1010.pkl"
    urdf = ROOT / "source/bb_mimic_nt/assets/robots/g1/urdf/g1_29dof_mode16_bb.urdf"
    batch = preprocess_motion_batch(source, urdf, tmp_path / "hold.pt")
    assert batch["lengths"].eq(1).all()
    for name in ("root_lin_vel", "root_ang_vel", "dof_vel", "object_lin_vel", "object_ang_vel"):
        assert batch[name].eq(0).all()


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_end_trigger_probability_is_rejected(tmp_path, probability):
    source = yaml.safe_load((ROOT / "source/bb_mimic_nt/assets/trajectory/skills.yaml").read_text())
    source["defaults"]["end"]["trigger_probability"] = probability
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="trigger_probability"):
        load_skill_config(path)


def test_end_probability_is_independent_of_rsi_and_keeps_complete_holds():
    torch.manual_seed(17)
    term = make_command(30000)
    term.skill_end_probabilities[:] = torch.tensor([1.0, 0.3, 0.3])
    term.skill_end_bounds[1] = torch.tensor([2.0, 8.0])
    term.reset()
    for skill, expected in enumerate((1.0, 0.3, 0.3)):
        ids = term.skill_ids == skill
        assert term.end_hold_triggered[ids].float().mean().item() == pytest.approx(expected, abs=0.02)
        selected = ids & term.end_hold_triggered
        skipped = ids & ~term.end_hold_triggered
        assert term.end_hold_duration[skipped].eq(0).all()
        bounds = term.skill_end_bounds[skill]
        assert term.end_hold_duration[selected].min() >= bounds[0]
        assert term.end_hold_duration[selected].max() <= bounds[1]
    # RSI is forced on for shoot, off for stand; both still trigger about 30%.
    assert term.rsi_started[term.skill_ids == 1].all()
    assert not term.rsi_started[term.skill_ids == 2].any()
    selection = term.end_hold_triggered.clone()
    durations = term.end_hold_duration.clone()
    for _ in range(10):
        term._update_command()
    assert torch.equal(term.end_hold_triggered, selection)
    assert torch.equal(term.end_hold_duration, durations)


def test_skipped_end_finishes_on_reference_completion_and_partial_reset_is_isolated():
    term = make_command(3)
    term.skill_end_probabilities[:] = torch.tensor([1.0, 0.0, 0.0])
    term.cfg.enable_rsi = False
    term.fixed_clip_ids[:] = torch.tensor([0, 1, 5])
    term.reset()
    assert term.end_hold_triggered.tolist() == [True, False, False]
    assert term.end_hold_duration[1:].eq(0).all()
    term.frame[1:] = torch.tensor([99.0, 199.0])
    term._update_command()
    assert term.stage.tolist() == [1, 2, 2]
    assert term.frame.tolist() == [0.0, 100.0, 200.0]
    assert term.hold_time[1:].eq(0).all()
    assert term.command.argmax(-1).tolist() == [0, 1, 2]
    other_duration = term.end_hold_duration[0].clone()
    term.reset([1])
    assert term.end_hold_triggered.tolist() == [True, False, False]
    assert term.stage.tolist() == [1, 0, 2]
    assert term.end_hold_duration[0] == other_duration


def test_single_frame_skill_can_skip_end_without_a_zero_time_reset_loop():
    term = make_command(1)
    term.fixed_clip_ids[:] = 0
    term.skill_end_probabilities[:] = 0.0
    term.reset()
    assert term.stage.item() == 2
    assert term.end_hold_duration.item() == 0 and term.speed.item() == 0
    term._update_command()
    assert term.motion_done.item()
