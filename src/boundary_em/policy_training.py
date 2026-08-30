"""Reproducible actor-critic optimization for episodic write decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from boundary_em.policy import WriteActorCritic
from boundary_em.task import TaskConfig, generate_episode
from boundary_em.training import actor_critic_loss, sample_rollout


@dataclass(frozen=True)
class PolicyTrainingConfig:
    """Optimization settings frozen before a policy run."""

    hidden_dim: int
    updates: int
    batch_size: int
    learning_rate: float
    critic_coefficient: float
    entropy_coefficient: float
    gradient_clip: float
    temperature: float
    capacity: int


@dataclass(frozen=True)
class TrainingRun:
    """Trained policy and complete per-update trace."""

    model: WriteActorCritic
    history: list[dict[str, float | int]]


def train_policy(
    task_config: TaskConfig,
    train_config: PolicyTrainingConfig,
    *,
    seed: int,
) -> TrainingRun:
    """Train one policy seed on freshly generated event episodes."""

    torch.manual_seed(seed)
    input_dim = task_config.cue_dim + 2 * task_config.n_features
    model = WriteActorCritic(
        input_dim=input_dim,
        hidden_dim=train_config.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    action_generator = torch.Generator().manual_seed(1_000_000 + seed)
    history: list[dict[str, float | int]] = []

    for update in range(train_config.updates):
        seed_offset = seed * 10_000_000 + update * train_config.batch_size
        episodes = [
            generate_episode(task_config, seed=seed_offset + batch_index)
            for batch_index in range(train_config.batch_size)
        ]
        rollouts = [
            sample_rollout(
                model,
                episode,
                generator=action_generator,
                temperature=train_config.temperature,
                capacity=train_config.capacity,
            )
            for episode in episodes
        ]
        loss = actor_critic_loss(
            rollouts,
            critic_coefficient=train_config.critic_coefficient,
            entropy_coefficient=train_config.entropy_coefficient,
        )
        optimizer.zero_grad()
        loss.total.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        optimizer.step()
        history.append(
            {
                "update": update,
                "total_loss": float(loss.total.detach().item()),
                "actor_loss": float(loss.actor.detach().item()),
                "critic_loss": float(loss.critic.detach().item()),
                "entropy": float(loss.entropy.detach().item()),
                "gradient_norm": float(gradient_norm.detach().item()),
                "mean_reward": loss.mean_reward,
            }
        )

    return TrainingRun(model=model, history=history)
