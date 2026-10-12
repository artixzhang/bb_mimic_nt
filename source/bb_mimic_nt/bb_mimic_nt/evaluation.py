"""Summaries for multi-skill shooting evaluations, independent of Isaac Sim."""

from __future__ import annotations


def summarize_shooting_records(
    records: list[dict], termination_names: list[str]
) -> dict:
    """Count hits only among episodes with a reference release event."""
    shots = [record for record in records if record["shot_eligible"]]
    hits = sum(record["success"] for record in shots)
    summary = {
        "episodes": len(records),
        "shot_episodes": len(shots),
        "shot_hits": hits,
        "hit_rate": hits / len(shots) if shots else None,
        "all_episode_hit_fraction": (
            sum(record["success"] for record in records) / len(records)
            if records
            else None
        ),
    }
    for name in ("root", "dof", "link", "ball"):
        summary[f"mean_{name}_rmse"] = (
            sum(record[f"{name}_rmse"] for record in records) / len(records)
            if records
            else None
        )
    summary["termination_counts"] = {
        name: sum(name in record["termination"] for record in records)
        for name in termination_names
    }
    return summary


def shooting_evaluation_report(
    checkpoint: str, records: list[dict], termination_names: list[str]
) -> dict:
    """Keep all-skill tracking metrics and report each skill separately."""
    by_skill = {}
    for name in dict.fromkeys(record["skill"] for record in records):
        subset = [record for record in records if record["skill"] == name]
        by_skill[name] = {
            "command_id": subset[0]["command_id"],
            **summarize_shooting_records(subset, termination_names),
        }
    return {
        "summary": {
            "checkpoint": checkpoint,
            **summarize_shooting_records(records, termination_names),
        },
        "by_skill": by_skill,
        "clips": records,
    }
