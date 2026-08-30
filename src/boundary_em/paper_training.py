"""Staged optimization for prediction and episodic retrieval on the paper task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from boundary_em.paper_model import EpisodicPredictionModel
from boundary_em.paper_rollout import (
    ScheduleName,
    expected_prediction_reward,
    forced_schedule,
    rollout_trial,
)
from boundary_em.paper_task import (
    Condition,
    PaperTaskConfig,
    PaperTrial,
    generate_trial,
)


@dataclass(frozen=True)
class PredictionStageConfig:
    """Settings for one supervised prediction or retrieval stage."""

    updates: int
    batch_size: int
    learning_rate: float
    gradient_clip: float
    retrieval_enabled: bool
    forced_encoding: ScheduleName
    conditions: tuple[Condition, ...]
    memory_capacity: int = 40

    def __post_init__(self) -> None:
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if self.memory_capacity < 1:
            raise ValueError("memory_capacity must be positive")


@dataclass(frozen=True)
class PredictionObjective:
    """Differentiable prediction loss and detached behavioral summaries."""

    loss: Tensor
    accuracy: float
    expected_reward: float


@dataclass(frozen=True)
class PredictionTrainingResult:
    """Complete update history for one supervised training stage."""

    history: list[dict[str, float | int]]


def prediction_objective(
    model: EpisodicPredictionModel,
    trial: PaperTrial,
    *,
    retrieval_enabled: bool,
    forced_encoding: ScheduleName,
    memory_capacity: int,
) -> PredictionObjective:
    """Compute next-state supervision and behavioral metrics for one trial."""

    computation = rollout_trial(
        model,
        trial,
        a1_encoding_actions=forced_schedule(trial.a1, forced_encoding),
        b1_encoding_actions=forced_schedule(trial.b1, forced_encoding),
        memory_capacity=memory_capacity,
        retrieval_enabled=retrieval_enabled,
    )
    event_pairs = (
        (computation.a1, trial.a1),
        (computation.b1, trial.b1),
        (computation.b2, trial.b2),
    )
    losses: list[Tensor] = []
    correct = 0
    count = 0
    for model_event, task_event in event_pairs:
        valid = task_event.valid_predictions
        logits = model_event.logits[valid, : model.output_dim - 1]
        targets = task_event.targets[valid]
        losses.append(F.cross_entropy(logits, targets))
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        count += int(valid.sum().item())

    b2_valid = trial.b2.valid_predictions
    b2_probabilities = computation.b2.probabilities[b2_valid]
    b2_targets = trial.b2.targets[b2_valid]
    b2_penalties = trial.b2.inputs[b2_valid, -1]
    rewards = expected_prediction_reward(
        b2_probabilities,
        b2_targets,
        b2_penalties,
    )
    return PredictionObjective(
        loss=torch.stack(losses).mean(),
        accuracy=correct / count,
        expected_reward=float(rewards.detach().mean().item()),
    )


def train_prediction_stage(
    model: EpisodicPredictionModel,
    task_config: PaperTaskConfig,
    stage_config: PredictionStageConfig,
    *,
    seed: int,
    repeated_trial: PaperTrial | None = None,
) -> PredictionTrainingResult:
    """Train next-state prediction with memory absent or forced by the experimenter."""

    torch.manual_seed(seed)
    prediction_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoding_")
    ]
    optimizer = torch.optim.Adam(
        prediction_parameters,
        lr=stage_config.learning_rate,
    )
    history: list[dict[str, float | int]] = []
    model.train()

    for update in range(stage_config.updates):
        objectives: list[PredictionObjective] = []
        for batch_index in range(stage_config.batch_size):
            if repeated_trial is None:
                condition_index = (
                    update * stage_config.batch_size + batch_index
                ) % len(stage_config.conditions)
                trial = generate_trial(
                    task_config,
                    seed=seed * 10_000_000
                    + update * stage_config.batch_size
                    + batch_index,
                    condition=stage_config.conditions[condition_index],
                    evaluation=False,
                )
            else:
                trial = repeated_trial
            objectives.append(
                prediction_objective(
                    model,
                    trial,
                    retrieval_enabled=stage_config.retrieval_enabled,
                    forced_encoding=stage_config.forced_encoding,
                    memory_capacity=stage_config.memory_capacity,
                )
            )

        loss = torch.stack([objective.loss for objective in objectives]).mean()
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            prediction_parameters,
            stage_config.gradient_clip,
        )
        optimizer.step()
        history.append(
            {
                "update": update,
                "loss": float(loss.detach().item()),
                "accuracy": sum(objective.accuracy for objective in objectives)
                / len(objectives),
                "expected_b2_reward": sum(
                    objective.expected_reward for objective in objectives
                )
                / len(objectives),
                "gradient_norm": float(gradient_norm.detach().item()),
            }
        )

    return PredictionTrainingResult(history=history)
