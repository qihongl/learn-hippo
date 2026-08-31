"""Exact reward audit for an online one-encoding temporal policy."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from boundary_em.paper_rollout import expected_prediction_reward, rollout_trial
from boundary_em.paper_task import PaperTrial
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


class TemporalHazardPolicy(nn.Module):
    """Shared online probability of first encoding at each event step."""

    def __init__(
        self,
        n_steps: int,
        *,
        initial_logit: float = 0.0,
        initialization: Literal[
            "constant", "uniform_time", "random"
        ] = "constant",
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if initialization == "constant":
            logits = torch.full((n_steps,), initial_logit)
        elif initialization == "uniform_time":
            remaining_outcomes = torch.arange(n_steps + 1, 1, -1)
            hazards = 1.0 / remaining_outcomes
            logits = torch.logit(hazards)
        elif initialization == "random":
            remaining_outcomes = torch.arange(n_steps + 1, 1, -1)
            hazards = 1.0 / remaining_outcomes
            logits = torch.logit(hazards) + 0.25 * torch.randn(n_steps)
        else:
            raise ValueError(f"unknown initialization: {initialization}")
        self.logits = nn.Parameter(logits)

    def forward(self) -> Tensor:
        """Return probabilities for encoding at each step or never encoding."""

        return hazard_encoding_distribution(self.logits)


@dataclass(frozen=True)
class TemporalPolicyTrainingResult:
    """Complete optimization history for the exact temporal audit."""

    history: list[dict[str, Any]]


def hazard_encoding_distribution(logits: Tensor) -> Tensor:
    """Convert online hazards to first-encoding-time and never probabilities."""

    if logits.ndim != 1 or logits.numel() < 1:
        raise ValueError("logits must be a nonempty one-dimensional tensor")
    hazards = torch.sigmoid(logits)
    survival_before = torch.cumprod(
        torch.cat([torch.ones_like(hazards[:1]), 1.0 - hazards[:-1]]),
        dim=0,
    )
    encoding_probabilities = hazards * survival_before
    never_probability = torch.prod(1.0 - hazards).unsqueeze(0)
    return torch.cat([encoding_probabilities, never_probability])


def expected_counterfactual_reward(logits: Tensor, reward_matrix: Tensor) -> Tensor:
    """Average a two-event reward table under one shared temporal policy."""

    probabilities = hazard_encoding_distribution(logits)
    expected_shape = (probabilities.numel(), probabilities.numel())
    if reward_matrix.shape != expected_shape:
        raise ValueError(f"reward_matrix must have shape {expected_shape}")
    return probabilities @ reward_matrix @ probabilities


def train_temporal_policy(
    policy: TemporalHazardPolicy,
    reward_matrix: Tensor,
    *,
    updates: int,
    learning_rate: float,
) -> TemporalPolicyTrainingResult:
    """Optimize expected delayed reward without an endpoint supervision target."""

    if updates < 1:
        raise ValueError("updates must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    expected_size = policy.logits.numel() + 1
    if reward_matrix.shape != (expected_size, expected_size):
        raise ValueError("reward_matrix does not match the policy duration")

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    for update in range(updates):
        expected_reward = expected_counterfactual_reward(
            policy.logits,
            reward_matrix,
        )
        optimizer.zero_grad()
        (-expected_reward).backward()
        gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()

        with torch.no_grad():
            probabilities = policy()
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            history.append(
                {
                    "update": update,
                    "completed_updates": update + 1,
                    "expected_reward": float(expected_reward.item()),
                    "gradient_norm": float(gradient_norm.item()),
                    "entropy": float(entropy.item()),
                    "endpoint_probability": float(probabilities[-2].item()),
                    "mean_nonendpoint_probability": float(
                        probabilities[:-2].mean().item()
                    ),
                    "never_probability": float(probabilities[-1].item()),
                    "encoding_time_probabilities": probabilities.tolist(),
                }
            )
    return TemporalPolicyTrainingResult(history=history)


def counterfactual_reward_matrix(
    model: StructuredEpisodicPredictionModel,
    trial: PaperTrial,
    *,
    memory_capacity: int,
) -> Tensor:
    """Enumerate prediction reward for every first-encoding-time pair."""

    if memory_capacity < 2:
        raise ValueError("two traces are required to encode once in each event")

    a1_length = len(trial.a1.inputs)
    b1_length = len(trial.b1.inputs)
    rewards = torch.empty(a1_length + 1, b1_length + 1)
    original_mode = model.training
    context = torch.no_grad() if torch.is_grad_enabled() else nullcontext()
    try:
        model.eval()
        with context:
            for a1_time in range(a1_length + 1):
                a1_actions = _single_encoding_actions(a1_length, a1_time)
                for b1_time in range(b1_length + 1):
                    b1_actions = _single_encoding_actions(b1_length, b1_time)
                    computation = rollout_trial(
                        model,
                        trial,
                        a1_encoding_actions=a1_actions,
                        b1_encoding_actions=b1_actions,
                        memory_capacity=memory_capacity,
                        retrieval_enabled=True,
                    )
                    valid = trial.b2.valid_predictions
                    rewards[a1_time, b1_time] = expected_prediction_reward(
                        computation.b2.probabilities[valid],
                        trial.b2.targets[valid],
                        trial.b2.inputs[valid, -1],
                    ).mean()
    finally:
        model.train(original_mode)
    return rewards


def _single_encoding_actions(event_length: int, encoding_time: int) -> Tensor:
    actions = torch.zeros(event_length, dtype=torch.bool)
    if encoding_time < event_length:
        actions[encoding_time] = True
    return actions
