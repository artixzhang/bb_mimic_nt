"""Versioned policy coordinates shared by training, inference and deployment.

Equal tensor dimensions do not imply equal action or observation semantics.
Teacher checkpoints carry this contract in ``params/policy_contract.json``;
Student metadata embeds it so exports remain self-contained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ACTION_SEMANTICS = "direct_joint_position_v1"
OBSERVATION_SEMANTICS = "nominal_direct_observation_v1"
_RETRAIN = "Legacy or incompatible policy contract; retrain the Teacher and Student with direct actions."
_VECTOR_FIELDS = (
    "nominal_joint_position", "action_scale", "pd_stiffness", "pd_damping",
    "effort_limits", "velocity_limits",
)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _equal(first, second) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_equal(value, second[key]) for key, value in first.items())
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(_equal(a, b) for a, b in zip(first, second))
    if _finite_number(first) and _finite_number(second):
        return math.isclose(first, second, rel_tol=1.e-6, abs_tol=1.e-7)
    return type(first) is type(second) and first == second


def validate_policy_contract(contract: dict, expected: dict | None = None) -> dict:
    """Reject old coordinates and optionally require the exact runtime mapping."""
    if (not isinstance(contract, dict) or contract.get("version") != 1
            or contract.get("action_semantics") != ACTION_SEMANTICS
            or contract.get("observation_semantics") != OBSERVATION_SEMANTICS):
        raise ValueError(_RETRAIN)
    names = contract.get("joint_names")
    if (not isinstance(names, list) or len(names) != 29
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Policy contract requires 29 unique ordered joint names.")
    for key in _VECTOR_FIELDS:
        values = contract.get(key)
        if not isinstance(values, list) or len(values) != 29 or not all(map(_finite_number, values)):
            raise ValueError(f"Policy contract {key} requires 29 finite values.")
        if key != "nominal_joint_position" and any(value <= 0 for value in values):
            raise ValueError(f"Policy contract {key} must be positive.")
    limits = contract.get("joint_position_limits")
    if (not isinstance(limits, list) or len(limits) != 29
            or any(not isinstance(pair, list) or len(pair) != 2
                   or not all(map(_finite_number, pair)) or pair[0] >= pair[1] for pair in limits)):
        raise ValueError("Policy contract requires ordered physical joint_position_limits.")
    for key in ("action_clip", "policy_hz"):
        if not _finite_number(contract.get(key)) or contract[key] <= 0:
            raise ValueError(f"Policy contract {key} must be positive and finite.")
    if "target_clip" not in contract or contract["target_clip"] is not None:
        raise ValueError("Direct action targets must not be angle-clipped.")
    if type(contract.get("teacher_history_length")) is not int or contract["teacher_history_length"] < 1:
        raise ValueError("Policy contract requires a positive Teacher history length.")
    terms = contract.get("teacher_observation_terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("Policy contract is missing the Teacher observation layout.")
    seen = set()
    for term in terms:
        if (not isinstance(term, dict) or not isinstance(term.get("name"), str)
                or not term["name"] or term["name"] in seen
                or type(term.get("history_length")) is not int or term["history_length"] < 0):
            raise ValueError("Invalid Teacher observation term in policy contract.")
        seen.add(term["name"])
        scale = term.get("scale")
        if scale is not None and not _finite_number(scale) and not (
            isinstance(scale, list) and scale and all(map(_finite_number, scale))
        ):
            raise ValueError(f"Invalid observation scale for {term['name']}.")
    if expected is not None:
        validate_policy_contract(expected)
        for key, value in expected.items():
            if key not in contract or not _equal(contract[key], value):
                raise ValueError(f"Policy contract mismatch at {key}. {_RETRAIN}")
    return contract


def build_policy_contract(env) -> dict:
    """Snapshot resolved nominal parameters, never the randomized live gains."""
    action = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    joint_ids, names = robot.find_joints(action.cfg.joint_names, preserve_order=True)

    def first(tensor):
        return tensor[0].detach().cpu().tolist()

    observation_manager = env.observation_manager
    group = "teacher" if "teacher" in observation_manager.active_terms else "policy"
    terms = []
    for name in observation_manager.active_terms[group]:
        cfg = getattr(getattr(observation_manager.cfg, group), name)
        scale = cfg.scale
        if isinstance(scale, tuple):
            scale = list(scale)
        elif hasattr(scale, "tolist"):
            scale = scale.tolist()
        terms.append(dict(name=name, scale=scale, history_length=cfg.history_length))
    contract = dict(
        version=1, action_semantics=ACTION_SEMANTICS, observation_semantics=OBSERVATION_SEMANTICS,
        joint_names=list(names), nominal_joint_position=first(action.nominal),
        action_scale=first(action.action_scale), action_clip=float(action.cfg.action_clip), target_clip=None,
        pd_stiffness=first(robot.data.default_joint_stiffness[:, joint_ids]),
        pd_damping=first(robot.data.default_joint_damping[:, joint_ids]),
        effort_limits=first(action.effort_limits), velocity_limits=first(action.velocity_limits),
        joint_position_limits=first(action.joint_position_limits), policy_hz=1.0 / env.step_dt,
        teacher_history_length=max(term["history_length"] for term in terms),
        teacher_observation_terms=terms,
    )
    return validate_policy_contract(contract)


def save_policy_contract(env, directory: str | Path) -> Path:
    """Write beside training configs, or beside a Teacher network export."""
    path = Path(directory) / "policy_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_policy_contract(env), indent=2) + "\n", encoding="utf-8")
    return path


def load_teacher_contract(checkpoint: str | Path, env=None) -> dict:
    path = Path(checkpoint).resolve().parent / "params" / "policy_contract.json"
    if not path.is_file():
        raise ValueError(f"Missing {path}. {_RETRAIN}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    return validate_policy_contract(contract, build_policy_contract(env) if env is not None else None)
