"""Sampled online actor-critic learning for one encoding per event."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from boundary_em.paper_rollout import (
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
from boundary_em.paper_temporal_policy import hazard_encoding_distribution
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


@dataclass(frozen=True)
class SampledHazardStageConfig:
    """Optimization settings for forced-value or sampled free-policy learning."""

    updates: int
    batch_size: int
    learning_rate: float
    critic_coefficient: float
    entropy_coefficient: float
    gradient_clip: float
    memory_capacity: int
    conditions: tuple[Condition, ...]
    evaluation_mode: bool

    def __post_init__(self) -> None:
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.critic_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients cannot be negative")
        if self.memory_capacity < 2:
            raise ValueError("one encoding in each event requires capacity of two")
        if not self.conditions:
            raise ValueError("at least one condition is required")


@dataclass(frozen=True)
class SampledHazardEpisode:
    """Online encoding choices and their delayed prediction outcome."""

    a1_actions: Tensor
    b1_actions: Tensor
    a1_distribution: Tensor
    b1_distribution: Tensor
    log_probabilities: Tensor
    values: Tensor
    entropy: Tensor
    reward: Tensor
    memory_labels: tuple[str, ...]


@dataclass(frozen=True)
class SampledHazardTrainingResult:
    """Complete update history for one sampled learning stage."""

    history: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]


def sample_neural_hazard_episode(
    model: StructuredEpisodicPredictionModel,
    trial: PaperTrial,
    *,
    action_generator: torch.Generator,
    memory_capacity: int,
    forced_exploration: bool,
) -> SampledHazardEpisode:
    """Sample sequential hazards, stopping after the first encoding in each event."""

    with torch.no_grad():
        probe = rollout_trial(
            model,
            trial,
            a1_encoding_actions=forced_schedule(trial.a1, "never"),
            b1_encoding_actions=forced_schedule(trial.b1, "never"),
            memory_capacity=memory_capacity,
            retrieval_enabled=False,
        )
    a1 = _sample_event_decisions(
        model,
        probe.a1.states.detach(),
        action_generator=action_generator,
        forced_exploration=forced_exploration,
    )
    b1 = _sample_event_decisions(
        model,
        probe.b1.states.detach(),
        action_generator=action_generator,
        forced_exploration=forced_exploration,
    )
    computation = rollout_trial(
        model,
        trial,
        a1_encoding_actions=a1[0],
        b1_encoding_actions=b1[0],
        memory_capacity=memory_capacity,
        retrieval_enabled=True,
    )
    valid = trial.b2.valid_predictions
    reward = expected_prediction_reward(
        computation.b2.probabilities[valid],
        trial.b2.targets[valid],
        trial.b2.inputs[valid, -1],
    ).mean()
    return SampledHazardEpisode(
        a1_actions=a1[0],
        b1_actions=b1[0],
        a1_distribution=a1[1],
        b1_distribution=b1[1],
        log_probabilities=torch.cat([a1[2], b1[2]]),
        values=torch.cat([a1[3], b1[3]]),
        entropy=torch.cat([a1[4], b1[4]]),
        reward=reward,
        memory_labels=computation.memory_labels,
    )


def _sample_event_decisions(
    model: StructuredEpisodicPredictionModel,
    states: Tensor,
    *,
    action_generator: torch.Generator,
    forced_exploration: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    outputs = [model.encoding_policy(state) for state in states]
    logits = torch.stack([output.logit for output in outputs])
    distribution = hazard_encoding_distribution(logits)
    actions = torch.zeros(len(states), dtype=torch.bool)
    log_probabilities = []
    values = []
    entropies = []
    epsilon = torch.finfo(logits.dtype).eps
    for time_index, output in enumerate(outputs):
        probability = output.probability.clamp(epsilon, 1.0 - epsilon)
        sampling_probability = (
            0.5 if forced_exploration else float(probability.detach())
        )
        encode = bool(
            torch.rand((), generator=action_generator).item() < sampling_probability
        )
        log_probabilities.append(
            torch.log(probability) if encode else torch.log1p(-probability)
        )
        values.append(output.value)
        entropies.append(
            -(
                probability * torch.log(probability)
                + (1.0 - probability) * torch.log1p(-probability)
            )
        )
        if encode:
            actions[time_index] = True
            break
    return (
        actions,
        distribution,
        torch.stack(log_probabilities),
        torch.stack(values),
        torch.stack(entropies),
    )


def train_sampled_hazard_stage(
    model: StructuredEpisodicPredictionModel,
    task_config: PaperTaskConfig,
    stage_config: SampledHazardStageConfig,
    *,
    seed: int,
    forced_exploration: bool,
    checkpoint_interval: int | None = None,
    checkpoint_evaluator: Callable[
        [int, StructuredEpisodicPredictionModel], dict[str, Any]
    ]
    | None = None,
) -> SampledHazardTrainingResult:
    """Train the critic under forced choices or actor-critic under free choices."""

    if (checkpoint_interval is None) != (checkpoint_evaluator is None):
        raise ValueError(
            "checkpoint_interval and checkpoint_evaluator must be provided together"
        )
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")

    original_grad_state = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    actor_parameters = []
    critic_parameters = []
    for name, parameter in model.named_parameters():
        is_actor = name.startswith("encoding_actor")
        is_critic = name.startswith("encoding_critic")
        parameter.requires_grad_(is_critic or (is_actor and not forced_exploration))
        if is_actor:
            actor_parameters.append(parameter)
        elif is_critic:
            critic_parameters.append(parameter)
    optimized_parameters = (
        critic_parameters
        if forced_exploration
        else actor_parameters + critic_parameters
    )
    optimizer = torch.optim.Adam(optimized_parameters, lr=stage_config.learning_rate)
    action_generator = torch.Generator().manual_seed(seed + 90_000_000)
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    try:
        for update in range(stage_config.updates):
            episodes = []
            for batch_index in range(stage_config.batch_size):
                example_index = update * stage_config.batch_size + batch_index
                condition = stage_config.conditions[
                    example_index % len(stage_config.conditions)
                ]
                trial = generate_trial(
                    task_config,
                    seed=seed * 1_000_000 + example_index,
                    condition=condition,
                    evaluation=stage_config.evaluation_mode,
                )
                episodes.append(
                    sample_neural_hazard_episode(
                        model,
                        trial,
                        action_generator=action_generator,
                        memory_capacity=stage_config.memory_capacity,
                        forced_exploration=forced_exploration,
                    )
                )

            actor_terms = []
            critic_terms = []
            entropy_terms = []
            for episode in episodes:
                target = episode.reward.detach().expand_as(episode.values)
                advantage = target - episode.values
                actor_terms.append(
                    -(episode.log_probabilities * advantage.detach()).mean()
                )
                critic_terms.append(0.5 * advantage.square().mean())
                entropy_terms.append(episode.entropy.mean())
            actor_loss = torch.stack(actor_terms).mean()
            critic_loss = torch.stack(critic_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            loss = (
                critic_loss
                if forced_exploration
                else actor_loss
                + stage_config.critic_coefficient * critic_loss
                - stage_config.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                optimized_parameters,
                stage_config.gradient_clip,
            )
            optimizer.step()

            all_distributions = torch.stack(
                [
                    distribution
                    for episode in episodes
                    for distribution in (
                        episode.a1_distribution,
                        episode.b1_distribution,
                    )
                ]
            )
            completed_updates = update + 1
            update_record = {
                    "update": update,
                    "completed_updates": completed_updates,
                    "sequences_processed": completed_updates
                    * stage_config.batch_size,
                    "epoch": completed_updates * stage_config.batch_size / 256,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "mode": (
                        "forced_value" if forced_exploration else "sampled_actor"
                    ),
                    "loss": float(loss.detach().item()),
                    "actor_loss": float(actor_loss.detach().item()),
                    "critic_loss": float(critic_loss.detach().item()),
                    "entropy": float(entropy.detach().item()),
                    "gradient_norm": float(gradient_norm.detach().item()),
                    "mean_reward": sum(
                        float(episode.reward.detach().item()) for episode in episodes
                    )
                    / len(episodes),
                    "mean_encodings_per_event": sum(
                        float(
                            episode.a1_actions.sum().item()
                            + episode.b1_actions.sum().item()
                        )
                        for episode in episodes
                    )
                    / (2 * len(episodes)),
                    "endpoint_probability": float(
                        all_distributions[:, -2].mean().detach().item()
                    ),
                    "mean_nonendpoint_probability": float(
                        all_distributions[:, :-2].mean().detach().item()
                    ),
                    "never_probability": float(
                        all_distributions[:, -1].mean().detach().item()
                    ),
                }
            history.append(update_record)

            should_checkpoint = checkpoint_interval is not None and (
                completed_updates % checkpoint_interval == 0
                or completed_updates == stage_config.updates
            )
            if should_checkpoint:
                assert checkpoint_evaluator is not None
                random_state = torch.random.get_rng_state()
                original_mode = model.training
                try:
                    evaluation_start = perf_counter()
                    evaluation = checkpoint_evaluator(completed_updates, model)
                    evaluation_runtime = perf_counter() - evaluation_start
                finally:
                    torch.random.set_rng_state(random_state)
                    model.train(original_mode)
                checkpoints.append(
                    {
                        "update": completed_updates,
                        "sequences_processed": completed_updates
                        * stage_config.batch_size,
                        "epoch": completed_updates * stage_config.batch_size / 256,
                        "training": dict(update_record),
                        "evaluation": evaluation,
                        "evaluation_runtime_seconds": evaluation_runtime,
                    }
                )
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_grad_state[name])

    return SampledHazardTrainingResult(
        history=history,
        checkpoints=checkpoints,
    )
