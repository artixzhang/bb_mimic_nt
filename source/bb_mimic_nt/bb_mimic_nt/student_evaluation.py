"""Matched first-episode evaluation records, independent of Isaac Sim."""

from __future__ import annotations

import torch


class ClosedLoopEpisodeCollector:
    """Keep the first completed episode of each environment across automatic resets."""

    def __init__(self, num_envs: int, termination_names: list[str], device: str):
        if num_envs < 1:
            raise ValueError("Evaluation needs at least one environment.")
        self.termination_names = list(termination_names)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.planned_steps = torch.zeros(num_envs, device=device)
        self.survival_fraction = torch.zeros(num_envs, device=device)
        self.completed = torch.zeros_like(self.finished)
        self.failed = torch.zeros_like(self.finished)
        self.reasons = torch.zeros((num_envs, len(termination_names)), dtype=torch.bool, device=device)

    def capture(
        self,
        done: torch.Tensor,
        *,
        elapsed_steps: torch.Tensor,
        planned_steps: torch.Tensor,
        survival_fraction: torch.Tensor,
        completed: torch.Tensor,
        failed: torch.Tensor,
        reasons: torch.Tensor,
    ) -> None:
        """Copy terminal snapshots before another step can replace them."""
        if any(value.shape != self.finished.shape for value in (
            done, elapsed_steps, planned_steps, survival_fraction, completed, failed
        )) or reasons.shape != self.reasons.shape:
            raise ValueError("Closed-loop terminal snapshots must match the environment count.")
        newly_finished = done.bool() & ~self.finished
        if torch.any(newly_finished & ((planned_steps <= 0) | (elapsed_steps <= 0))):
            raise ValueError("Completed episodes require positive planned and elapsed step counts.")
        valid_fraction = torch.isfinite(survival_fraction) & (survival_fraction >= 0) & (survival_fraction <= 1)
        if torch.any(newly_finished & ~valid_fraction):
            raise ValueError("Terminal survival fractions must be finite and between zero and one.")
        self.elapsed_steps[newly_finished] = elapsed_steps[newly_finished].to(self.elapsed_steps.dtype)
        self.planned_steps[newly_finished] = planned_steps[newly_finished].to(self.planned_steps.dtype)
        self.survival_fraction[newly_finished] = survival_fraction[newly_finished].to(self.survival_fraction.dtype)
        self.failed[newly_finished] = failed[newly_finished]
        # Reaching the final frame while also failing is not successful completion.
        self.completed[newly_finished] = completed[newly_finished] & ~failed[newly_finished]
        self.reasons[newly_finished] = reasons[newly_finished]
        self.finished |= newly_finished

    def records(self, step_dt: float) -> list[dict]:
        """Return one JSON-ready record for each finished first episode."""
        if not bool(torch.all(self.finished)):
            raise RuntimeError("Cannot report a batch before every selected episode has ended.")
        if step_dt <= 0:
            raise ValueError("Evaluation step duration must be positive.")
        elapsed = self.elapsed_steps.cpu().tolist()
        planned = self.planned_steps.cpu().tolist()
        survival = self.survival_fraction.cpu().tolist()
        completed = self.completed.cpu().tolist()
        failed = self.failed.cpu().tolist()
        reasons = self.reasons.cpu().tolist()
        return [
            {
                "episode_steps": length,
                "planned_steps": horizon,
                "episode_duration_s": length * step_dt,
                "planned_duration_s": horizon * step_dt,
                "survival_fraction": fraction,
                "completed": complete,
                "failed": failure,
                "termination": [name for name, active in zip(self.termination_names, masks) if active],
            }
            for length, horizon, fraction, complete, failure, masks in zip(
                elapsed, planned, survival, completed, failed, reasons
            )
        ]


def summarize_closed_loop(records: list[dict], termination_names: list[str]) -> dict:
    """Summarize matched Teacher/Student episodes under fixed termination thresholds."""
    summary = {"episodes_per_policy": len(records)}
    for policy_name in ("teacher", "student"):
        episodes = [record[policy_name] for record in records]
        count = len(episodes)
        summary[policy_name] = {
            "completed_episodes": sum(episode["completed"] for episode in episodes),
            "completion_fraction": sum(episode["completed"] for episode in episodes) / count if count else None,
            "failure_fraction": sum(episode["failed"] for episode in episodes) / count if count else None,
            "mean_survival_fraction": sum(episode["survival_fraction"] for episode in episodes) / count if count else None,
            "mean_episode_duration_s": sum(episode["episode_duration_s"] for episode in episodes) / count if count else None,
            "termination_counts": {
                name: sum(name in episode["termination"] for episode in episodes)
                for name in termination_names
            },
        }
    return summary
