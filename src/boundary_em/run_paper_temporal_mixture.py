"""Run an exact shared temporal-policy audit on the RM/DM/NM mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import yaml

from boundary_em.paper_task import Condition, PaperTaskConfig
from boundary_em.paper_temporal_policy import (
    TemporalHazardPolicy,
    train_temporal_policy,
)
from boundary_em.run_paper_temporal_policy import (
    _mean_reward_matrix,
    _surface_summary,
)
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _condition_weights(conditions: tuple[Condition, ...]) -> dict[Condition, float]:
    counts = Counter(conditions)
    return {
        condition: count / len(conditions)
        for condition, count in counts.items()
    }


def _condition_reward_matrices(
    model: StructuredEpisodicPredictionModel,
    task_config: PaperTaskConfig,
    *,
    conditions: tuple[Condition, ...],
    seed_start: int,
    trials_per_condition: int,
    memory_capacity: int,
) -> dict[Condition, torch.Tensor]:
    unique_conditions = tuple(dict.fromkeys(conditions))
    return {
        condition: _mean_reward_matrix(
            model,
            task_config,
            condition=condition,
            seed_start=seed_start + index * trials_per_condition,
            n_trials=trials_per_condition,
            evaluation_mode=True,
            memory_capacity=memory_capacity,
        )
        for index, condition in enumerate(unique_conditions)
    }


def _weighted_matrix(
    matrices: dict[Condition, torch.Tensor],
    weights: dict[Condition, float],
) -> torch.Tensor:
    first = next(iter(matrices.values()))
    result = torch.zeros_like(first)
    for condition, weight in weights.items():
        result = result + weight * matrices[condition]
    return result


def run_temporal_mixture_config(
    config_path: str | Path,
    *,
    seed: int,
    repository_root: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Optimize one temporal policy against the declared condition mixture."""

    config_path = Path(config_path)
    repository = (
        Path(repository_root) if repository_root is not None else config_path.parents[2]
    )
    config_bytes = config_path.read_bytes()
    raw = yaml.safe_load(config_bytes)
    if seed not in raw["experiment"]["model_seeds"]:
        raise ValueError(f"seed {seed} is not declared in the configuration")
    if not bool(raw["task"]["evaluation_mode"]):
        raise ValueError("the 16-step temporal audit requires evaluation_mode: true")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    task_config = PaperTaskConfig(profile=raw["task"]["profile"])
    model = StructuredEpisodicPredictionModel(
        n_features=task_config.n_features,
        n_values=task_config.n_values,
        **raw["model"],
    )
    policy = TemporalHazardPolicy(
        int(raw["optimization"]["n_steps"]),
        initialization=raw["optimization"]["initialization"],
    )
    conditions = tuple(raw["task"]["conditions"])
    weights = _condition_weights(conditions)
    bank = raw["counterfactual_banks"]
    memory_capacity = int(bank["memory_capacity"])
    seed_offset = seed * 100_000

    training_start = perf_counter()
    training_by_condition = _condition_reward_matrices(
        model,
        task_config,
        conditions=conditions,
        seed_start=int(bank["training_seed_start"]) + seed_offset,
        trials_per_condition=int(bank["training_trials_per_condition"]),
        memory_capacity=memory_capacity,
    )
    training_rewards = _weighted_matrix(training_by_condition, weights)
    training_result = train_temporal_policy(
        policy,
        training_rewards,
        updates=int(raw["optimization"]["updates"]),
        learning_rate=float(raw["optimization"]["learning_rate"]),
    )
    training_runtime = perf_counter() - training_start

    evaluation_start = perf_counter()
    evaluation_by_condition_matrices = _condition_reward_matrices(
        model,
        task_config,
        conditions=conditions,
        seed_start=int(bank["evaluation_seed_start"]) + seed_offset,
        trials_per_condition=int(bank["evaluation_trials_per_condition"]),
        memory_capacity=memory_capacity,
    )
    evaluation_rewards = _weighted_matrix(
        evaluation_by_condition_matrices,
        weights,
    )
    evaluation_runtime = perf_counter() - evaluation_start
    with torch.no_grad():
        probabilities = policy()

        def evaluate_surface(rewards: torch.Tensor) -> dict[str, Any]:
            return {
                **_surface_summary(rewards),
                "learned_expected_reward": float(
                    (probabilities @ rewards @ probabilities).item()
                ),
            }

        evaluation_by_condition = {
            condition: evaluate_surface(rewards)
            for condition, rewards in evaluation_by_condition_matrices.items()
        }
        endpoint_gap = probabilities[-2] - probabilities[:-2].mean()
        evaluation = {
            **evaluate_surface(evaluation_rewards),
            "encoding_time_probabilities": probabilities.tolist(),
            "endpoint_probability": float(probabilities[-2].item()),
            "mean_nonendpoint_probability": float(
                probabilities[:-2].mean().item()
            ),
            "never_probability": float(probabilities[-1].item()),
            "endpoint_probability_gap": float(endpoint_gap.item()),
        }

    result: dict[str, Any] = {
        "task": "Lu-Hasson-Norman 2022 fixed-duration condition mixture",
        "data_kind": "measured synthetic simulation",
        "experiment": raw["experiment"],
        "seed": seed,
        "configuration": raw,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "mixture_weights": weights,
        "scientific_guardrails": {
            "endpoint_target_used": False,
            "future_relevance_input_used": False,
            "condition_input_used": False,
            "shared_policy_across_conditions_a1_b1": True,
            "counterfactual_outcomes_used_during_training": True,
            "interpretation": "objective audit, not the final neural model claim",
        },
        "training": {
            "trials_per_condition": int(bank["training_trials_per_condition"]),
            "runtime_seconds": training_runtime,
            "mean_reward_matrix": training_rewards.tolist(),
            "surface": _surface_summary(training_rewards),
            "history": training_result.history,
        },
        "evaluation": evaluation,
        "evaluation_by_condition": evaluation_by_condition,
        "evaluation_trials_per_condition": int(
            bank["evaluation_trials_per_condition"]
        ),
        "evaluation_runtime_seconds": evaluation_runtime,
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(repository),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    output_root = (
        Path(output_directory)
        if output_directory is not None
        else repository / raw["output"]["directory"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{raw['experiment']['name']}_seed{seed}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-directory")
    arguments = parser.parse_args()
    result = run_temporal_mixture_config(
        arguments.config,
        seed=arguments.seed,
        output_directory=arguments.output_directory,
    )
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
