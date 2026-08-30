"""Frozen-weight evaluation of learned episodic write policies."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from boundary_em.policy import WriteActorCritic
from boundary_em.task import TaskConfig, generate_episode, sample_null_steps
from boundary_em.training import evaluate_actions

Intervention = Literal[
    "endpoint_only",
    "midpoint_only",
    "midpoint_plus_endpoint",
    "always_write",
    "never_write",
    "matched_random_one_write",
    "displaced_learned",
]


@dataclass(frozen=True)
class PolicyEvaluation:
    """Episode-level records and aggregate held-out metrics."""

    per_episode: tuple[dict[str, float | int], ...]
    summary: dict[str, dict[str, float | int | None]]


def _metric_cell(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n_episodes": len(values),
    }


def _auc(positives: list[float], negatives: list[float]) -> float:
    comparisons = [
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    ]
    return statistics.fmean(comparisons)


def _apply_intervention(
    actions: torch.Tensor,
    episode_masks: torch.Tensor,
    *,
    query_features: int,
    intervention: Intervention,
    generator: torch.Generator,
) -> torch.Tensor:
    intervened = actions.clone()
    midpoint_candidates = torch.nonzero(
        episode_masks.sum(dim=1) == query_features,
        as_tuple=False,
    ).flatten()
    midpoint = int(midpoint_candidates[0].item())
    endpoint = len(actions) - 1
    if intervention == "endpoint_only":
        intervened.zero_()
        intervened[endpoint] = True
    elif intervention == "midpoint_only":
        intervened.zero_()
        intervened[midpoint] = True
    elif intervention == "midpoint_plus_endpoint":
        intervened.zero_()
        intervened[midpoint] = True
        intervened[endpoint] = True
    elif intervention == "always_write":
        intervened.fill_(True)
    elif intervention == "never_write":
        intervened.zero_()
    elif intervention == "matched_random_one_write":
        intervened.zero_()
        random_index = int(
            torch.randint(len(actions), (1,), generator=generator).item()
        )
        intervened[random_index] = True
    elif intervention == "displaced_learned":
        if bool(intervened[endpoint]):
            intervened[endpoint] = False
            candidates = [midpoint] + [
                index
                for index in range(endpoint)
                if index != midpoint
            ]
            for candidate in candidates:
                if not bool(intervened[candidate]):
                    intervened[candidate] = True
                    break
    else:
        raise ValueError(f"unknown intervention: {intervention}")
    return intervened


def evaluate_policy(
    model: WriteActorCritic,
    task_config: TaskConfig,
    *,
    episode_seeds: Iterable[int],
    action_seed: int,
    temperature: float,
    capacity: int,
    stochastic: bool,
    n_null_steps: int = 0,
    intervention: Intervention | None = None,
) -> PolicyEvaluation:
    """Evaluate a fixed model on unseen episodes without updating weights."""

    seeds = list(episode_seeds)
    if not seeds:
        raise ValueError("evaluation requires at least one episode seed")
    action_generator = torch.Generator().manual_seed(action_seed)
    was_training = model.training
    model.eval()
    records: list[dict[str, float | int]] = []
    endpoint_probabilities: list[float] = []
    nonendpoint_probabilities: list[float] = []

    with torch.inference_mode():
        for seed in seeds:
            null_steps = sample_null_steps(
                task_config,
                seed=seed,
                n_null_steps=n_null_steps,
            )
            episode = generate_episode(
                task_config,
                seed=seed,
                null_steps=null_steps,
            )
            probabilities = model(episode.policy_inputs).probabilities
            if stochastic:
                uniforms = torch.rand(probabilities.shape, generator=action_generator)
                actions = uniforms < probabilities
            else:
                actions = probabilities >= 0.5
            if intervention is not None:
                actions = _apply_intervention(
                    actions,
                    episode.masks,
                    query_features=task_config.query_features,
                    intervention=intervention,
                    generator=action_generator,
                )
            outcome = evaluate_actions(
                episode,
                actions,
                temperature=temperature,
                capacity=capacity,
            )
            endpoint_probability = float(probabilities[-1].item())
            nonendpoint_probability = float(probabilities[:-1].mean().item())
            endpoint_probabilities.append(endpoint_probability)
            nonendpoint_probabilities.extend(
                float(probability.item()) for probability in probabilities[:-1]
            )
            records.append(
                {
                    "episode_seed": seed,
                    "reward": float(outcome.reward.item()),
                    "mse": float(outcome.mse.item()),
                    "writes": int(actions.sum().item()),
                    "endpoint_probability": endpoint_probability,
                    "nonendpoint_probability": nonendpoint_probability,
                    "endpoint_selectivity": (
                        endpoint_probability - nonendpoint_probability
                    ),
                }
            )

    model.train(was_training)
    reward_values = [float(record["reward"]) for record in records]
    mse_values = [float(record["mse"]) for record in records]
    write_values = [float(record["writes"]) for record in records]
    selectivity_values = [
        float(record["endpoint_selectivity"]) for record in records
    ]
    boundary_auc = _auc(endpoint_probabilities, nonendpoint_probabilities)
    summary = {
        "reward": _metric_cell(reward_values),
        "mse": _metric_cell(mse_values),
        "writes_per_event": _metric_cell(write_values),
        "endpoint_selectivity": _metric_cell(selectivity_values),
        "boundary_auc": {
            "mean": boundary_auc,
            "std": None,
            "n_episodes": len(records),
        },
    }
    return PolicyEvaluation(per_episode=tuple(records), summary=summary)
