# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validated skill definitions and episode sampling, independent of Isaac Sim."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import yaml

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parents[1] / "assets/trajectory/skills.yaml"


def load_skill_config(path: str | Path = DEFAULT_SKILLS_PATH) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        data = yaml.safe_load(stream)
    # Repository-relative paths are independent of the process working directory.
    root = next((parent for parent in path.parents if (parent / "source/bb_mimic_nt").is_dir()), path.parent)
    defaults = dict(
        end=dict(duration_s=5.0, trigger_probability=0.3, object_interaction_enabled=False),
        initialization=dict(robot="reference", ball="reference", hoop="reference"),
        rsi=dict(enabled=True),
        object=dict(interaction_enabled=True),
    )
    for section, values in data.get("defaults", {}).items():
        defaults[section].update(values)
    dimension = data["command"]["dimension"]
    if type(dimension) is not int or dimension < 1:
        raise ValueError("command.dimension must be a positive integer.")
    skills = []
    for name, values in data["skills"].items():
        skill = copy.deepcopy(defaults)
        for key, value in values.items():
            if key in defaults:
                skill[key].update(value)
            else:
                skill[key] = value
        skill["name"] = name
        command_id = skill["command_id"]
        if type(command_id) is not int or not 0 <= command_id < dimension:
            raise ValueError(f"{name}: command_id must be in [0, command.dimension).")
        weight = float(skill["sampling_weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name}: sampling_weight must be finite and non-negative.")
        skill["sampling_weight"] = weight
        duration = skill["end"]["duration_s"]
        bounds = list(duration) if isinstance(duration, (list, tuple)) else [duration, duration]
        if (
            len(bounds) != 2
            or any(not math.isfinite(float(v)) or float(v) < 0 for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"{name}: duration_s must be non-negative seconds or [minimum, maximum].")
        skill["end"]["duration_s"] = [float(v) for v in bounds]
        probability = skill["end"]["trigger_probability"]
        if isinstance(probability, bool) or not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1:
            raise ValueError(f"{name}: end.trigger_probability must be finite and in [0, 1].")
        skill["end"]["trigger_probability"] = float(probability)
        for section, key in (
            ("end", "object_interaction_enabled"),
            ("object", "interaction_enabled"),
            ("rsi", "enabled"),
        ):
            if type(skill[section][key]) is not bool:
                raise ValueError(f"{name}: {section}.{key} must be a boolean.")
        if skill["initialization"] != dict(robot="reference", ball="reference", hoop="reference"):
            raise ValueError(f"{name}: only reference initialization is currently supported.")
        source = Path(skill["motion_file"]).expanduser()
        skill["motion_file"] = str((source if source.is_absolute() else root / source).resolve())
        skills.append(skill)
    skills.sort(key=lambda skill: skill["command_id"])
    if not skills or len({s["command_id"] for s in skills}) != len(skills):
        raise ValueError("Skills need distinct command IDs.")
    if sum(s["sampling_weight"] for s in skills) <= 0:
        raise ValueError("At least one skill sampling weight must be positive.")
    return dict(command=dict(dimension=dimension), skills=skills)


def sample_skill_clips(weights: torch.Tensor, counts: torch.Tensor, offsets: torch.Tensor, count: int):
    """Sample skill per episode, then a uniform clip within that skill."""
    skill_ids = torch.multinomial(weights, count, replacement=True)
    clips = offsets[skill_ids] + (torch.rand(count, device=weights.device) * counts[skill_ids]).long()
    return skill_ids, clips


def merge_motion_batches(batches: list[dict], config: dict) -> dict:
    """Pad independent skill batches and retain their clip-to-command mapping."""
    metadata = copy.deepcopy(batches[0]["metadata"])
    compatible = (
        "schema_version",
        "fps",
        "dof_names",
        "tracked_body_names",
        "rotation_body_names",
        "rotation_body_weights",
        "anchor_body_names",
        "anchor_offsets",
        "contact_names",
        "urdf_sha256",
    )
    for batch in batches[1:]:
        if any(batch["metadata"].get(key) != metadata.get(key) for key in compatible):
            raise ValueError("Skill motion caches use incompatible robot layouts or frame rates.")
        if batch.keys() != batches[0].keys():
            raise ValueError("Skill motion caches have different fields.")
    lengths = torch.cat([b["lengths"] for b in batches])
    frames = int(lengths.max())
    result = dict(
        metadata=metadata, lengths=lengths, valid=torch.arange(frames, device=lengths.device)[None] < lengths[:, None]
    )
    for field in batches[0]:
        if field in result:
            continue
        values = []
        for batch in batches:
            value = batch[field]
            padding = value[:, -1:].expand(-1, frames - value.shape[1], *value.shape[2:])
            values.append(torch.cat((value, padding), dim=1))
        result[field] = torch.cat(values)
    counts = [len(b["lengths"]) for b in batches]
    metadata.update(
        num_clips=len(lengths),
        max_frames=frames,
        skill_config=copy.deepcopy(config),
        skill_clip_counts=counts,
        skill_sources=[b["metadata"] for b in batches],
    )
    return result


def load_skill_motion_batch(config: dict, urdf_path, tracked_body_names, *, device="cpu", auto_preprocess=True):
    from .trajectory import cache_is_current, load_motion_batch, preprocess_motion_batch

    batches = []
    for skill in config["skills"]:
        source = Path(skill["motion_file"])
        cache = source.with_name(source.stem + "_processed.pt")
        if auto_preprocess and not cache_is_current(cache, source, urdf_path, tuple(tracked_body_names)):
            preprocess_motion_batch(source, urdf_path, cache, tracked_body_names=tuple(tracked_body_names))
        batches.append(load_motion_batch(cache, device=device))
    return merge_motion_batches(batches, config)
