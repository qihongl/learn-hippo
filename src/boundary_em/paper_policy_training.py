"""Forced-exploration value learning and free episodic encoding selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from boundary_em.paper_model import EpisodicPredictionModel
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


@dataclass(frozen=True)
class EncodingStageConfig:
    """Optimization settings for value pretraining or free encoding."""

    updates: int
    batch_size: int
    learning_rate: float
    critic_coefficient: float
    entropy_coefficient: float
    gradient_clip: float
    memory_capacity: int
    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.critic_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients cannot be negative")
        if self.memory_capacity < 1:
            raise ValueError("memory_capacity must be positive")
        if not self.conditions:
            raise ValueError("at least one condition is required")


@dataclass(frozen=True)
class EncodingEpisode:
    """Encoding actions and their delayed prediction consequence."""

    a1_actions: Tensor
    b1_actions: Tensor
    actions: Tensor
    probabilities: Tensor
    values: Tensor
    log_probabilities: Tensor
    entropy: Tensor
    reward: Tensor
    memory_labels: tuple[str, ...]


@dataclass(frozen=True)
class EncodingTrainingResult:
    """Complete update history for one encoding optimization stage."""

    history: list[dict[str, float | int | str]]


def train_encoding_stage(
    model: EpisodicPredictionModel,
    task_config: PaperTaskConfig,
    stage_config: EncodingStageConfig,
    *,
    seed: int,
    forced_exploration: bool,
) -> EncodingTrainingResult:
    """Train only the critic under forced actions or actor-critic under free actions."""

    torch.manual_seed(seed)
    action_generator = torch.Generator().manual_seed(seed + 90_000_000)
    actor_parameters, critic_parameters = _freeze_prediction_system(model)
    optimized_parameters = (
        critic_parameters
        if forced_exploration
        else actor_parameters + critic_parameters
    )
    optimizer = torch.optim.Adam(
        optimized_parameters,
        lr=stage_config.learning_rate,
    )
    history: list[dict[str, float | int | str]] = []

    for update in range(stage_config.updates):
        episodes: list[EncodingEpisode] = []
        for batch_index in range(stage_config.batch_size):
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
            episodes.append(
                sample_encoding_episode(
                    model,
                    trial,
                    action_generator=action_generator,
                    memory_capacity=stage_config.memory_capacity,
                    forced_exploration=forced_exploration,
                )
            )

        actor_terms: list[Tensor] = []
        critic_terms: list[Tensor] = []
        entropy_terms: list[Tensor] = []
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
        if forced_exploration:
            loss = critic_loss
        else:
            loss = (
                actor_loss
                + stage_config.critic_coefficient * critic_loss
                - stage_config.entropy_coefficient * entropy
            )

        optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            optimized_parameters,
            stage_config.gradient_clip,
        )
        optimizer.step()
        history.append(
            {
                "update": update,
                "mode": "forced_value" if forced_exploration else "free_actor",
                "loss": float(loss.detach().item()),
                "actor_loss": float(actor_loss.detach().item()),
                "critic_loss": float(critic_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
                "mean_reward": sum(
                    float(episode.reward.detach().item()) for episode in episodes
                )
                / len(episodes),
                "mean_encodings_per_event": sum(
                    float(episode.actions.to(torch.float32).sum().item())
                    for episode in episodes
                )
                / (2 * len(episodes)),
                "endpoint_probability": sum(
                    float(
                        torch.stack(
                            [
                                episode.probabilities[
                                    len(episode.a1_actions) - 1
                                ],
                                episode.probabilities[-1],
                            ]
                        )
                        .mean()
                        .detach()
                        .item()
                    )
                    for episode in episodes
                )
                / len(episodes),
                "nonendpoint_probability": sum(
                    float(
                        torch.cat(
                            [
                                episode.probabilities[
                                    : len(episode.a1_actions) - 1
                                ],
                                episode.probabilities[
                                    len(episode.a1_actions) : -1
                                ],
                            ]
                        )
                        .mean()
                        .detach()
                        .item()
                    )
                    for episode in episodes
                )
                / len(episodes),
                "gradient_norm": float(gradient_norm.detach().item()),
            }
        )

    for parameter in actor_parameters + critic_parameters:
        parameter.requires_grad_(True)
    return EncodingTrainingResult(history=history)


def _freeze_prediction_system(
    model: EpisodicPredictionModel,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    actor_parameters: list[nn.Parameter] = []
    critic_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("encoding_actor"):
            parameter.requires_grad_(True)
            actor_parameters.append(parameter)
        elif name.startswith("encoding_critic"):
            parameter.requires_grad_(True)
            critic_parameters.append(parameter)
        else:
            parameter.requires_grad_(False)
    return actor_parameters, critic_parameters


def sample_encoding_episode(
    model: EpisodicPredictionModel,
    trial: PaperTrial,
    *,
    action_generator: torch.Generator,
    memory_capacity: int,
    forced_exploration: bool,
) -> EncodingEpisode:
    """Sample one shared encoding policy in the distracting and target events."""

    probe = rollout_trial(
        model,
        trial,
        a1_encoding_actions=forced_schedule(trial.a1, "never"),
        b1_encoding_actions=forced_schedule(trial.b1, "never"),
        memory_capacity=memory_capacity,
        retrieval_enabled=False,
    )
    a1_policy_outputs = [
        model.encoding_policy(state.detach()) for state in probe.a1.states
    ]
    b1_policy_outputs = [
        model.encoding_policy(state.detach()) for state in probe.b1.states
    ]
    policy_outputs = a1_policy_outputs + b1_policy_outputs
    probabilities = torch.stack([output.probability for output in policy_outputs])
    values = torch.stack([output.value for output in policy_outputs])
    epsilon = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
    uniforms = torch.rand(probabilities.shape, generator=action_generator)
    if forced_exploration:
        actions = uniforms < 0.5
    else:
        actions = uniforms < probabilities.detach()
    a1_actions = actions[: len(a1_policy_outputs)]
    b1_actions = actions[len(a1_policy_outputs) :]
    log_probabilities = torch.where(
        actions,
        torch.log(probabilities),
        torch.log1p(-probabilities),
    )
    entropy = -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log1p(-probabilities)
    )

    computation = rollout_trial(
        model,
        trial,
        a1_encoding_actions=a1_actions,
        b1_encoding_actions=b1_actions,
        memory_capacity=memory_capacity,
        retrieval_enabled=True,
    )
    valid = trial.b2.valid_predictions
    rewards = expected_prediction_reward(
        computation.b2.probabilities[valid],
        trial.b2.targets[valid],
        trial.b2.inputs[valid, -1],
    )
    return EncodingEpisode(
        a1_actions=a1_actions,
        b1_actions=b1_actions,
        actions=actions,
        probabilities=probabilities,
        values=values,
        log_probabilities=log_probabilities,
        entropy=entropy,
        reward=rewards.mean(),
        memory_labels=computation.memory_labels,
    )
