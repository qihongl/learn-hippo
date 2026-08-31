"""Exact delayed-reward training utilities for the neural encoding policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from boundary_em.paper_rollout import forced_schedule, rollout_trial
from boundary_em.paper_task import PaperTrial
from boundary_em.paper_temporal_policy import (
    counterfactual_reward_matrix,
    hazard_encoding_distribution,
)
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


@dataclass(frozen=True)
class NeuralCounterfactualExample:
    """Online situation states paired with their delayed-reward surface."""

    a1_states: Tensor
    b1_states: Tensor
    reward_matrix: Tensor


@dataclass(frozen=True)
class NeuralCounterfactualTrainingResult:
    """Complete update history for neural exact-reward optimization."""

    history: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]


def initialize_neural_hazard_policy(
    model: StructuredEpisodicPredictionModel,
    *,
    initial_probability: float,
) -> None:
    """Initialize a boundary-neutral neural hazard at a low encoding rate."""

    if not 0 < initial_probability < 1:
        raise ValueError("initial_probability must fall between zero and one")
    for module in model.encoding_actor_encoder:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(model.encoding_actor.weight, gain=0.01)
    nn.init.constant_(
        model.encoding_actor.bias,
        math.log(initial_probability / (1.0 - initial_probability)),
    )


def build_counterfactual_example(
    model: StructuredEpisodicPredictionModel,
    trial: PaperTrial,
    *,
    memory_capacity: int,
) -> NeuralCounterfactualExample:
    """Measure online states and all delayed rewards for one completed trial."""

    with torch.no_grad():
        probe = rollout_trial(
            model,
            trial,
            a1_encoding_actions=forced_schedule(trial.a1, "never"),
            b1_encoding_actions=forced_schedule(trial.b1, "never"),
            memory_capacity=memory_capacity,
            retrieval_enabled=False,
        )
        rewards = counterfactual_reward_matrix(
            model,
            trial,
            memory_capacity=memory_capacity,
        )
    return NeuralCounterfactualExample(
        a1_states=probe.a1.states.detach(),
        b1_states=probe.b1.states.detach(),
        reward_matrix=rewards.detach(),
    )


def neural_encoding_distributions(
    model: StructuredEpisodicPredictionModel,
    example: NeuralCounterfactualExample,
) -> tuple[Tensor, Tensor]:
    """Return first-encoding-time distributions from observation-only states."""

    logits = _actor_logits(
        model,
        torch.stack([example.a1_states, example.b1_states]),
    )
    distributions = _batched_hazard_encoding_distribution(logits)
    return distributions[0], distributions[1]


def neural_expected_reward(
    model: StructuredEpisodicPredictionModel,
    example: NeuralCounterfactualExample,
) -> Tensor:
    """Compute exact reward using one shared neural online encoding policy."""

    a1_probabilities, b1_probabilities = neural_encoding_distributions(
        model,
        example,
    )
    expected_shape = (a1_probabilities.numel(), b1_probabilities.numel())
    if example.reward_matrix.shape != expected_shape:
        raise ValueError(f"reward_matrix must have shape {expected_shape}")
    return a1_probabilities @ example.reward_matrix @ b1_probabilities


def train_neural_counterfactual(
    model: StructuredEpisodicPredictionModel,
    examples: list[NeuralCounterfactualExample],
    *,
    updates: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip: float,
    seed: int,
    checkpoint_interval: int | None = None,
    checkpoint_evaluator: Callable[
        [int, StructuredEpisodicPredictionModel], dict[str, Any]
    ]
    | None = None,
) -> NeuralCounterfactualTrainingResult:
    """Train only the shared neural actor using exact delayed prediction reward."""

    if not examples:
        raise ValueError("at least one counterfactual example is required")
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    if learning_rate <= 0 or gradient_clip <= 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
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
    for name, parameter in model.named_parameters():
        is_actor = name.startswith("encoding_actor")
        parameter.requires_grad_(is_actor)
        if is_actor:
            actor_parameters.append(parameter)
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    try:
        for update in range(updates):
            indices = torch.randint(
                len(examples),
                (batch_size,),
                generator=generator,
            ).tolist()
            selected = [examples[index] for index in indices]
            a1_states = torch.stack([example.a1_states for example in selected])
            b1_states = torch.stack([example.b1_states for example in selected])
            a1_probabilities = _batched_hazard_encoding_distribution(
                _actor_logits(model, a1_states)
            )
            b1_probabilities = _batched_hazard_encoding_distribution(
                _actor_logits(model, b1_states)
            )
            reward_matrices = torch.stack(
                [example.reward_matrix for example in selected]
            )
            expected_rewards = torch.einsum(
                "bi,bij,bj->b",
                a1_probabilities,
                reward_matrices,
                b1_probabilities,
            )
            expected_reward = expected_rewards.mean()
            optimizer.zero_grad(set_to_none=True)
            (-expected_reward).backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                actor_parameters,
                gradient_clip,
            )
            optimizer.step()
            completed_updates = update + 1
            update_record = {
                    "update": update,
                    "completed_updates": completed_updates,
                    "sequences_processed": completed_updates * batch_size,
                    "epoch": completed_updates * batch_size / 256,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "expected_reward": float(expected_reward.detach().item()),
                    "gradient_norm": float(gradient_norm.detach().item()),
                    "endpoint_probability": float(
                        torch.cat(
                            [a1_probabilities[:, -2], b1_probabilities[:, -2]]
                        )
                        .mean()
                        .detach()
                        .item()
                    ),
                    "mean_nonendpoint_probability": float(
                        torch.cat(
                            [
                                a1_probabilities[:, :-2].flatten(),
                                b1_probabilities[:, :-2].flatten(),
                            ]
                        )
                        .mean()
                        .detach()
                        .item()
                    ),
                    "never_probability": float(
                        torch.cat(
                            [a1_probabilities[:, -1], b1_probabilities[:, -1]]
                        )
                        .mean()
                        .detach()
                        .item()
                    ),
                }
            history.append(update_record)

            should_checkpoint = checkpoint_interval is not None and (
                completed_updates % checkpoint_interval == 0
                or completed_updates == updates
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
                        "sequences_processed": completed_updates * batch_size,
                        "epoch": completed_updates * batch_size / 256,
                        "training": dict(update_record),
                        "evaluation": evaluation,
                        "evaluation_runtime_seconds": evaluation_runtime,
                    }
                )
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_grad_state[name])

    return NeuralCounterfactualTrainingResult(
        history=history,
        checkpoints=checkpoints,
    )


def evaluate_neural_counterfactual(
    model: StructuredEpisodicPredictionModel,
    examples: list[NeuralCounterfactualExample],
) -> dict[str, Any]:
    """Evaluate a frozen neural encoding policy on fixed counterfactual trials."""

    if not examples:
        raise ValueError("at least one counterfactual example is required")
    original_mode = model.training
    random_state = torch.random.get_rng_state()
    try:
        model.eval()
        with torch.no_grad():
            a1_states = torch.stack([example.a1_states for example in examples])
            b1_states = torch.stack([example.b1_states for example in examples])
            a1_probabilities = _batched_hazard_encoding_distribution(
                _actor_logits(model, a1_states)
            )
            b1_probabilities = _batched_hazard_encoding_distribution(
                _actor_logits(model, b1_states)
            )
            rewards = torch.stack([example.reward_matrix for example in examples])
            learned_rewards = torch.einsum(
                "bi,bij,bj->b",
                a1_probabilities,
                rewards,
                b1_probabilities,
            )
            target_removed = torch.einsum(
                "bi,bi->b",
                a1_probabilities,
                rewards[:, :, -1],
            )
            distractor_removed = torch.einsum(
                "bi,bi->b",
                b1_probabilities,
                rewards[:, -1, :],
            )
            mean_rewards = rewards.mean(dim=0)
            all_probabilities = torch.cat(
                [a1_probabilities, b1_probabilities],
                dim=0,
            )
            return {
                "trials": len(examples),
                "learned_expected_reward": float(learned_rewards.mean().item()),
                "endpoint_probability": float(
                    all_probabilities[:, -2].mean().item()
                ),
                "a1_endpoint_probability": float(
                    a1_probabilities[:, -2].mean().item()
                ),
                "b1_endpoint_probability": float(
                    b1_probabilities[:, -2].mean().item()
                ),
                "mean_nonendpoint_probability": float(
                    all_probabilities[:, :-2].mean().item()
                ),
                "never_probability": float(
                    all_probabilities[:, -1].mean().item()
                ),
                "endpoint_probability_gap": float(
                    (
                        all_probabilities[:, -2].mean()
                        - all_probabilities[:, :-2].mean()
                    ).item()
                ),
                "mean_encoding_time_probabilities": all_probabilities.mean(
                    dim=0
                ).tolist(),
                "endpoint_pair_reward": float(mean_rewards[-2, -2].item()),
                "never_pair_reward": float(mean_rewards[-1, -1].item()),
                "matched_random_one_reward": float(
                    mean_rewards[:-1, :-1].mean().item()
                ),
                "target_memory_removed_reward": float(target_removed.mean().item()),
                "distractor_memory_removed_reward": float(
                    distractor_removed.mean().item()
                ),
            }
    finally:
        torch.random.set_rng_state(random_state)
        model.train(original_mode)


def _actor_logits(
    model: StructuredEpisodicPredictionModel,
    states: Tensor,
) -> Tensor:
    hidden = model.encoding_actor_encoder(states)
    return model.encoding_actor(hidden).squeeze(-1)


def _batched_hazard_encoding_distribution(logits: Tensor) -> Tensor:
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a nonempty time dimension")
    if logits.ndim == 1:
        return hazard_encoding_distribution(logits)
    hazards = torch.sigmoid(logits)
    survival_before = torch.cumprod(
        torch.cat(
            [torch.ones_like(hazards[..., :1]), 1.0 - hazards[..., :-1]],
            dim=-1,
        ),
        dim=-1,
    )
    encoding_probabilities = hazards * survival_before
    never_probability = torch.prod(1.0 - hazards, dim=-1, keepdim=True)
    return torch.cat([encoding_probabilities, never_probability], dim=-1)
