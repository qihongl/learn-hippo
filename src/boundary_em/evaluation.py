"""Frozen-weight evaluation of learned episodic write policies."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from boundary_em.policy import WriteActorCritic
from boundary_em.task import TaskConfig, generate_episode
from boundary_em.training import evaluate_actions


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


def evaluate_policy(
    model: WriteActorCritic,
    task_config: TaskConfig,
    *,
    episode_seeds: Iterable[int],
    action_seed: int,
    temperature: float,
    capacity: int,
    stochastic: bool,
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
            episode = generate_episode(task_config, seed=seed)
            probabilities = model(episode.policy_inputs).probabilities
            if stochastic:
                uniforms = torch.rand(probabilities.shape, generator=action_generator)
                actions = uniforms < probabilities
            else:
                actions = probabilities >= 0.5
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
