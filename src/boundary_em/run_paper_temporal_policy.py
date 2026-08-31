"""Run the exact counterfactual temporal-hazard audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import yaml

from boundary_em.paper_task import Condition, PaperTaskConfig, generate_trial
from boundary_em.paper_temporal_policy import (
    TemporalHazardPolicy,
    counterfactual_reward_matrix,
    train_temporal_policy,
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


def _mean_reward_matrix(
    model: StructuredEpisodicPredictionModel,
    task_config: PaperTaskConfig,
    *,
    condition: Condition,
    seed_start: int,
    n_trials: int,
    evaluation_mode: bool,
    memory_capacity: int,
) -> torch.Tensor:
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    matrices = []
    for trial_index in range(n_trials):
        trial = generate_trial(
            task_config,
            seed=seed_start + trial_index,
            condition=condition,
            evaluation=evaluation_mode,
        )
        matrices.append(
            counterfactual_reward_matrix(
                model,
                trial,
                memory_capacity=memory_capacity,
            )
        )
    return torch.stack(matrices).mean(dim=0)


def _surface_summary(rewards: torch.Tensor) -> dict[str, Any]:
    event_length = rewards.shape[0] - 1
    maximum_index = int(rewards.argmax().item())
    maximum_pair = divmod(maximum_index, event_length + 1)
    shared_diagonal = torch.diagonal(rewards)
    return {
        "unconstrained_best_pair": list(maximum_pair),
        "unconstrained_best_reward": float(rewards.max().item()),
        "best_shared_time": int(shared_diagonal.argmax().item()),
        "best_shared_reward": float(shared_diagonal.max().item()),
        "endpoint_pair_reward": float(rewards[-2, -2].item()),
        "never_pair_reward": float(rewards[-1, -1].item()),
        "matched_random_one_reward": float(rewards[:-1, :-1].mean().item()),
    }


def run_temporal_hazard_config(
    config_path: str | Path,
    *,
    seed: int,
    repository_root: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one declared temporal-hazard audit seed and preserve its record."""

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
    bank_config = raw["counterfactual_banks"]
    seed_offset = seed * 100_000

    training_start = perf_counter()
    training_rewards = _mean_reward_matrix(
        model,
        task_config,
        condition=raw["task"]["condition"],
        seed_start=int(bank_config["training_seed_start"]) + seed_offset,
        n_trials=int(bank_config["training_trials"]),
        evaluation_mode=True,
        memory_capacity=int(bank_config["memory_capacity"]),
    )
    training_result = train_temporal_policy(
        policy,
        training_rewards,
        updates=int(raw["optimization"]["updates"]),
        learning_rate=float(raw["optimization"]["learning_rate"]),
    )
    training_runtime = perf_counter() - training_start

    evaluation_start = perf_counter()
    evaluation_rewards = _mean_reward_matrix(
        model,
        task_config,
        condition=raw["task"]["condition"],
        seed_start=int(bank_config["evaluation_seed_start"]) + seed_offset,
        n_trials=int(bank_config["evaluation_trials"]),
        evaluation_mode=True,
        memory_capacity=int(bank_config["memory_capacity"]),
    )
    evaluation_runtime = perf_counter() - evaluation_start
    with torch.no_grad():
        probabilities = policy()
        learned_reward = probabilities @ evaluation_rewards @ probabilities
        endpoint_gap = probabilities[-2] - probabilities[:-2].mean()

    result: dict[str, Any] = {
        "task": "Lu-Hasson-Norman 2022 event-prediction evaluation generator",
        "data_kind": "measured synthetic simulation",
        "experiment": raw["experiment"],
        "seed": seed,
        "configuration": raw,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "scientific_guardrails": {
            "endpoint_target_used": False,
            "future_relevance_input_used": False,
            "shared_policy_across_a1_b1": True,
            "one_encoding_maximum_per_event": True,
            "counterfactual_outcomes_used_during_training": True,
            "interpretation": "optimizer and objective audit, not final model claim",
        },
        "training": {
            "trials": int(bank_config["training_trials"]),
            "runtime_seconds": training_runtime,
            "mean_reward_matrix": training_rewards.tolist(),
            "surface": _surface_summary(training_rewards),
            "history": training_result.history,
        },
        "evaluation": {
            "trials": int(bank_config["evaluation_trials"]),
            "runtime_seconds": evaluation_runtime,
            "mean_reward_matrix": evaluation_rewards.tolist(),
            "surface": _surface_summary(evaluation_rewards),
            "learned_expected_reward": float(learned_reward.item()),
            "encoding_time_probabilities": probabilities.tolist(),
            "endpoint_probability": float(probabilities[-2].item()),
            "mean_nonendpoint_probability": float(
                probabilities[:-2].mean().item()
            ),
            "never_probability": float(probabilities[-1].item()),
            "endpoint_probability_gap": float(endpoint_gap.item()),
        },
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
    arguments = parser.parse_args()
    result = run_temporal_hazard_config(arguments.config, seed=arguments.seed)
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
