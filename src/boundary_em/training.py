"""Policy rollout and delayed-reward training utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from boundary_em.memory import compose_key, differentiable_read
from boundary_em.policy import WriteActorCritic
from boundary_em.task import EventEpisode


@dataclass(frozen=True)
class EpisodeOutcome:
    """Delayed test result for a fixed set of study writes."""

    reward: Tensor
    mse: Tensor
    attention: Tensor
    read_value: Tensor


@dataclass(frozen=True)
class PolicyRollout:
    """One sampled study policy and its delayed consequence."""

    actions: Tensor
    probabilities: Tensor
    values: Tensor
    log_probabilities: Tensor
    entropy: Tensor
    reward: Tensor
    mse: Tensor
    attention: Tensor
    read_value: Tensor


@dataclass(frozen=True)
class ActorCriticLoss:
    """Decomposed batch objective and a detached performance summary."""

    total: Tensor
    actor: Tensor
    critic: Tensor
    entropy: Tensor
    mean_reward: float


def evaluate_actions(
    episode: EventEpisode,
    actions: Tensor,
    *,
    temperature: float,
    capacity: int,
) -> EpisodeOutcome:
    """Evaluate delayed prediction after applying discrete writes."""

    if actions.ndim != 1 or actions.shape[0] != episode.states.shape[0]:
        raise ValueError("actions must provide one write decision per study state")
    if capacity < 1:
        raise ValueError("capacity must be positive")

    selected = torch.nonzero(actions.to(torch.bool), as_tuple=False).flatten()
    selected = selected[-capacity:]
    if len(selected) == 0:
        read_value = torch.zeros_like(episode.features)
        attention = torch.empty(0)
    else:
        all_keys = torch.stack(
            [
                compose_key(episode.cue, state, mask)
                for state, mask in zip(episode.states, episode.masks, strict=True)
            ]
        )
        query = compose_key(episode.cue, episode.query_state, episode.query_mask)
        read = differentiable_read(
            query,
            all_keys[selected],
            episode.states[selected],
            temperature=temperature,
        )
        read_value = read.value
        attention = read.attention

    held_out = episode.query_mask == 0
    mse = torch.mean((read_value[held_out] - episode.features[held_out]) ** 2)
    return EpisodeOutcome(
        reward=1.0 - mse,
        mse=mse,
        attention=attention,
        read_value=read_value,
    )


def sample_rollout(
    model: WriteActorCritic,
    episode: EventEpisode,
    *,
    generator: torch.Generator,
    temperature: float,
    capacity: int,
) -> PolicyRollout:
    """Sample Bernoulli writes and compute their delayed test reward."""

    policy_output = model(episode.policy_inputs)
    epsilon = torch.finfo(policy_output.probabilities.dtype).eps
    probabilities = policy_output.probabilities.clamp(epsilon, 1.0 - epsilon)
    uniforms = torch.rand(probabilities.shape, generator=generator)
    actions = uniforms < probabilities.detach()
    log_probabilities = torch.where(
        actions,
        torch.log(probabilities),
        torch.log1p(-probabilities),
    )
    entropy = -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities) * torch.log1p(-probabilities)
    )
    outcome = evaluate_actions(
        episode,
        actions,
        temperature=temperature,
        capacity=capacity,
    )
    return PolicyRollout(
        actions=actions,
        probabilities=probabilities,
        values=policy_output.values,
        log_probabilities=log_probabilities,
        entropy=entropy,
        reward=outcome.reward,
        mse=outcome.mse,
        attention=outcome.attention,
        read_value=outcome.read_value,
    )


def actor_critic_loss(
    rollouts: list[PolicyRollout],
    *,
    critic_coefficient: float,
    entropy_coefficient: float,
) -> ActorCriticLoss:
    """Compute an episodic actor-critic objective for delayed write credit."""

    if not rollouts:
        raise ValueError("at least one rollout is required")
    actor_terms: list[Tensor] = []
    critic_terms: list[Tensor] = []
    entropy_terms: list[Tensor] = []
    rewards: list[float] = []
    for rollout in rollouts:
        target = rollout.reward.detach().expand_as(rollout.values)
        advantage = target - rollout.values
        actor_terms.append(
            -(rollout.log_probabilities * advantage.detach()).mean()
        )
        critic_terms.append(0.5 * advantage.square().mean())
        entropy_terms.append(rollout.entropy.mean())
        rewards.append(float(rollout.reward.detach().item()))

    actor = torch.stack(actor_terms).mean()
    critic = torch.stack(critic_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    total = actor + critic_coefficient * critic - entropy_coefficient * entropy
    return ActorCriticLoss(
        total=total,
        actor=actor,
        critic=critic,
        entropy=entropy,
        mean_reward=sum(rewards) / len(rewards),
    )
