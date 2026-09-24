"""Tensor-only contracts used by Student DAgger training and deployment."""

from __future__ import annotations

import torch


STUDENT_OBSERVATION_DIM = 283
STUDENT_ACTION_DIM = 29
STUDENT_OBSERVATION_FIELDS = (
    ("phase", 1),
    ("projected_gravity_history", 9),
    ("pelvis_angular_velocity_history", 9),
    ("joint_position_offset_history", 87),
    ("joint_velocity_history", 87),
    ("applied_action_offset_history", 87),
    ("initial_hoop_position_b", 3),
)


def student_teacher_target(
    raw_residual: torch.Tensor,
    reference: torch.Tensor,
    residual_scale: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    nominal: torch.Tensor,
) -> torch.Tensor:
    """Convert the unchanged Teacher residual into a nominal-relative joint target."""
    target = reference + torch.tanh(raw_residual) * residual_scale
    return torch.maximum(torch.minimum(target, upper), lower) - nominal


def student_smoothstep(value: float) -> float:
    """Cubic S-curve on the closed unit interval."""
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def student_execution_probability(
    iteration: int,
    total_iterations: int,
    start_fraction: float = 0.20,
    end_fraction: float = 0.50,
) -> float:
    """Return the probability that Student, rather than Teacher, controls an environment."""
    if iteration < 0 or total_iterations < 1:
        raise ValueError("Iteration must be non-negative and total_iterations must be positive.")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("DAgger transition fractions must satisfy 0 <= start < end <= 1.")
    progress = iteration / total_iterations
    return student_smoothstep((progress - start_fraction) / (end_fraction - start_fraction))


def student_mix_actions(
    teacher: torch.Tensor,
    student: torch.Tensor,
    student_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the executing policy independently for every environment."""
    if teacher.shape != student.shape:
        raise ValueError("Teacher and Student actions must have identical shapes.")
    if not 0.0 <= student_probability <= 1.0:
        raise ValueError("Student execution probability must be in [0, 1].")
    student_mask = torch.rand((teacher.shape[0], 1), device=teacher.device) < student_probability
    return torch.where(student_mask, student, teacher), student_mask.squeeze(-1)


def student_history(
    history: torch.Tensor,
    cursor: torch.Tensor,
    delay: torch.Tensor,
    length: int = 3,
) -> torch.Tensor:
    """Select per-environment history frames in oldest-to-newest order."""
    if length < 1 or history.ndim != 3:
        raise ValueError("History must be [time, environment, feature] with a positive output length.")
    env_ids = torch.arange(history.shape[1], device=history.device)
    return torch.stack(
        [history[(cursor - delay - age) % history.shape[0], env_ids] for age in range(length - 1, -1, -1)],
        dim=1,
    )


class StudentReplay:
    """Device-resident replay mixing recent on-policy states with historical coverage."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int = STUDENT_OBSERVATION_DIM,
        action_dim: int = STUDENT_ACTION_DIM,
        device: torch.device | str = "cpu",
        recent_storage_fraction: float = 0.5,
        recent_sample_fraction: float = 0.75,
    ) -> None:
        if min(capacity, observation_dim, action_dim) < 1:
            raise ValueError("Replay dimensions and capacity must be positive.")
        if capacity < 2:
            raise ValueError("Replay capacity must be at least two for recent and historical storage.")
        if not 0.0 < recent_storage_fraction < 1.0:
            raise ValueError("Recent replay storage fraction must be in (0, 1).")
        if not 0.0 <= recent_sample_fraction <= 1.0:
            raise ValueError("Recent replay sample fraction must be in [0, 1].")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.recent_capacity = min(capacity - 1, max(1, round(capacity * recent_storage_fraction)))
        self.history_capacity = capacity - self.recent_capacity
        self.recent_sample_fraction = float(recent_sample_fraction)
        self.observations = torch.empty(
            (self.history_capacity, observation_dim), dtype=torch.float32, device=self.device
        )
        self.labels = torch.empty((self.history_capacity, action_dim), dtype=torch.float32, device=self.device)
        self.recent_observations = torch.empty(
            (self.recent_capacity, observation_dim), dtype=torch.float32, device=self.device
        )
        self.recent_labels = torch.empty(
            (self.recent_capacity, action_dim), dtype=torch.float32, device=self.device
        )
        self.history_size = 0
        self.recent_size = 0
        self.recent_cursor = 0
        self.seen = 0

    @property
    def size(self) -> int:
        return self.history_size + self.recent_size

    def add(self, observations: torch.Tensor, labels: torch.Tensor) -> None:
        if observations.ndim != 2 or labels.ndim != 2 or observations.shape[0] != labels.shape[0]:
            raise ValueError("Replay observations and labels must be matrices with a shared batch dimension.")
        if observations.shape[1:] != self.observations.shape[1:] or labels.shape[1:] != self.labels.shape[1:]:
            raise ValueError("Replay sample dimensions do not match the configured storage.")
        observations = observations.detach().to(device=self.device, dtype=torch.float32)
        labels = labels.detach().to(device=self.device, dtype=torch.float32)
        count = observations.shape[0]
        if count == 0:
            return
        seen_before = self.seen
        fill = min(count, self.history_capacity - self.history_size)
        if fill:
            self.observations[self.history_size : self.history_size + fill].copy_(observations[:fill])
            self.labels[self.history_size : self.history_size + fill].copy_(labels[:fill])
            self.history_size += fill
        remaining = count - fill
        if remaining:
            sequence = torch.arange(1, remaining + 1, device=self.device)
            destinations = torch.floor(
                torch.rand(remaining, device=self.device) * (seen_before + fill + sequence)
            ).long()
            accepted = torch.nonzero(destinations < self.history_capacity).flatten()
            if len(accepted):
                selected = destinations[accepted]
                order = torch.argsort(selected, stable=True)
                sorted_destinations = selected[order]
                keep = torch.ones(len(order), dtype=torch.bool, device=self.device)
                keep[:-1] = sorted_destinations[:-1] != sorted_destinations[1:]
                source = accepted[order[keep]] + fill
                target = sorted_destinations[keep]
                self.observations[target].copy_(observations[source])
                self.labels[target].copy_(labels[source])
        recent_count = min(count, self.recent_capacity)
        recent_observations = observations[-recent_count:]
        recent_labels = labels[-recent_count:]
        if recent_count == self.recent_capacity:
            self.recent_observations.copy_(recent_observations)
            self.recent_labels.copy_(recent_labels)
            self.recent_cursor = 0
        else:
            first = min(recent_count, self.recent_capacity - self.recent_cursor)
            self.recent_observations[self.recent_cursor : self.recent_cursor + first].copy_(recent_observations[:first])
            self.recent_labels[self.recent_cursor : self.recent_cursor + first].copy_(recent_labels[:first])
            second = recent_count - first
            if second:
                self.recent_observations[:second].copy_(recent_observations[first:])
                self.recent_labels[:second].copy_(recent_labels[first:])
            self.recent_cursor = (self.recent_cursor + recent_count) % self.recent_capacity
        self.recent_size = min(self.recent_capacity, self.recent_size + count)
        self.seen += count

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0 or batch_size < 1:
            raise ValueError("Replay must be non-empty and batch_size must be positive.")
        recent_count = round(batch_size * self.recent_sample_fraction)
        if self.recent_size == 0:
            recent_count = 0
        if self.history_size == 0:
            recent_count = batch_size
        history_count = batch_size - recent_count
        recent_indices = torch.randint(self.recent_size, (recent_count,), device=self.device)
        history_indices = torch.randint(self.history_size, (history_count,), device=self.device)
        inputs = torch.cat(
            (self.recent_observations[recent_indices], self.observations[history_indices]), dim=0
        )
        targets = torch.cat((self.recent_labels[recent_indices], self.labels[history_indices]), dim=0)
        order = torch.randperm(batch_size, device=self.device)
        return inputs[order], targets[order]

    def sample_recent(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample only the newest FIFO partition for an on-policy fit diagnostic."""
        if self.recent_size == 0 or batch_size < 1:
            raise ValueError("Recent replay must be non-empty and batch size must be positive.")
        indices = torch.randint(self.recent_size, (batch_size,), device=self.device)
        return self.recent_observations[indices], self.recent_labels[indices]

    def state_dict(self) -> dict:
        return {
            "observations": self.observations[: self.history_size],
            "labels": self.labels[: self.history_size],
            "history_size": self.history_size,
            "recent_observations": self.recent_observations[: self.recent_size],
            "recent_labels": self.recent_labels[: self.recent_size],
            "recent_size": self.recent_size,
            "recent_cursor": self.recent_cursor,
            "seen": self.seen,
            "capacity": self.capacity,
            "recent_capacity": self.recent_capacity,
            "recent_sample_fraction": self.recent_sample_fraction,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Replay capacity differs from the checkpoint.")
        if "recent_observations" not in state:
            raise ValueError("Replay checkpoint predates recent-priority replay and cannot be resumed safely.")
        if int(state["recent_capacity"]) != self.recent_capacity:
            raise ValueError("Recent replay capacity differs from the checkpoint.")
        history_size = int(state["history_size"])
        recent_size = int(state["recent_size"])
        if not 0 <= history_size <= self.history_capacity or not 0 <= recent_size <= self.recent_capacity:
            raise ValueError("Replay checkpoint has an invalid size.")
        self.history_size = history_size
        self.recent_size = recent_size
        self.recent_cursor = int(state["recent_cursor"])
        self.seen = int(state["seen"])
        self.observations[:history_size].copy_(state["observations"].to(self.device))
        self.labels[:history_size].copy_(state["labels"].to(self.device))
        self.recent_observations[:recent_size].copy_(state["recent_observations"].to(self.device))
        self.recent_labels[:recent_size].copy_(state["recent_labels"].to(self.device))


if sum(width for _, width in STUDENT_OBSERVATION_FIELDS) != STUDENT_OBSERVATION_DIM:
    raise RuntimeError("Student observation field contract does not total 283 dimensions.")
