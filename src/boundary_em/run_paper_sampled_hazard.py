"""Run forced-value and sampled free encoding on the exact paper task."""

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

from boundary_em.paper_neural_counterfactual import (
    build_counterfactual_example,
    evaluate_neural_counterfactual,
    initialize_neural_hazard_policy,
)
from boundary_em.paper_sampled_hazard import (
    SampledHazardStageConfig,
    train_sampled_hazard_stage,
)
from boundary_em.paper_task import PaperTaskConfig, generate_trial
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


def _stage_config(
    raw: dict[str, Any],
    *,
    conditions: tuple[str, ...],
    evaluation_mode: bool,
) -> SampledHazardStageConfig:
    return SampledHazardStageConfig(
        conditions=conditions,
        evaluation_mode=evaluation_mode,
        **raw,
    )


def run_sampled_hazard_config(
    config_path: str | Path,
    *,
    seed: int,
    repository_root: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one declared forced-value and sampled-actor seed."""

    config_path = Path(config_path)
    repository = (
        Path(repository_root) if repository_root is not None else config_path.parents[2]
    )
    config_bytes = config_path.read_bytes()
    raw = yaml.safe_load(config_bytes)
    if seed not in raw["experiment"]["model_seeds"]:
        raise ValueError(f"seed {seed} is not declared in the configuration")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    task_config = PaperTaskConfig(profile=raw["task"]["profile"])
    model = StructuredEpisodicPredictionModel(
        n_features=task_config.n_features,
        n_values=task_config.n_values,
        **raw["model"],
    )
    initialize_neural_hazard_policy(
        model,
        initial_probability=float(raw["policy"]["initial_probability"]),
    )
    conditions = tuple(raw["task"]["training_conditions"])
    evaluation_mode = bool(raw["task"]["training_evaluation_mode"])
    stage_configs = {
        name: _stage_config(
            raw_stage,
            conditions=conditions,
            evaluation_mode=evaluation_mode,
        )
        for name, raw_stage in raw["training"].items()
    }

    checkpoint_config = raw["checkpoint_evaluation"]
    evaluation_bank_start = perf_counter()
    evaluation_conditions = tuple(checkpoint_config.get("conditions", ["DM"]))
    evaluation_trial_count = int(checkpoint_config["trials"])
    evaluation_mode = bool(checkpoint_config.get("evaluation_mode", True))
    evaluation_examples = {
        condition: [
            build_counterfactual_example(
                model,
                generate_trial(
                    task_config,
                    seed=int(checkpoint_config["trial_seed_start"])
                    + seed * 100_000
                    + condition_index * evaluation_trial_count
                    + index,
                    condition=condition,
                    evaluation=evaluation_mode,
                ),
                memory_capacity=stage_configs["free_policy"].memory_capacity,
            )
            for index in range(evaluation_trial_count)
        ]
        for condition_index, condition in enumerate(evaluation_conditions)
    }
    evaluation_bank_runtime = perf_counter() - evaluation_bank_start

    forced_start = perf_counter()
    forced_result = train_sampled_hazard_stage(
        model,
        task_config,
        stage_configs["forced_value"],
        seed=seed,
        forced_exploration=True,
    )
    forced_runtime = perf_counter() - forced_start

    def checkpoint_evaluator(
        _completed_updates: int,
        checkpoint_model: StructuredEpisodicPredictionModel,
    ) -> dict[str, Any]:
        by_condition = {
            condition: evaluate_neural_counterfactual(checkpoint_model, examples)
            for condition, examples in evaluation_examples.items()
        }
        if len(by_condition) == 1:
            return next(iter(by_condition.values()))
        return {"by_condition": by_condition}

    free_start = perf_counter()
    free_result = train_sampled_hazard_stage(
        model,
        task_config,
        stage_configs["free_policy"],
        seed=seed + 1_000,
        forced_exploration=False,
        checkpoint_interval=int(checkpoint_config["interval_updates"]),
        checkpoint_evaluator=checkpoint_evaluator,
    )
    free_runtime = perf_counter() - free_start
    evaluation_by_condition = {
        condition: evaluate_neural_counterfactual(model, examples)
        for condition, examples in evaluation_examples.items()
    }
    primary_condition = (
        "DM" if "DM" in evaluation_by_condition else evaluation_conditions[0]
    )
    evaluation = evaluation_by_condition[primary_condition]

    result: dict[str, Any] = {
        "task": "Lu-Hasson-Norman 2022 event-prediction generator",
        "data_kind": "measured synthetic simulation",
        "experiment": raw["experiment"],
        "seed": seed,
        "configuration": raw,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "scientific_guardrails": {
            "endpoint_target_used": False,
            "future_relevance_input_used": False,
            "boundary_or_time_input_used": False,
            "policy_input": "80-dimensional accumulated exact situation state",
            "shared_policy_across_a1_b1": True,
            "one_encoding_maximum_per_event": True,
            "new_training_mapping_each_sequence": True,
            "sampled_delayed_reward_used_for_actor": True,
            "counterfactual_outcomes_used_for_actor": False,
            "counterfactual_outcomes_used_for_evaluation_only": True,
            "actor_advantage_mode": stage_configs[
                "free_policy"
            ].advantage_mode,
            "condition_schedule": stage_configs[
                "free_policy"
            ].condition_schedule,
            "learning_rate_schedule": stage_configs[
                "free_policy"
            ].learning_rate_schedule,
            "minimum_learning_rate_fraction": stage_configs[
                "free_policy"
            ].minimum_learning_rate_fraction,
            "retrospective_condition_used_for_credit_only": (
                stage_configs["free_policy"].advantage_mode
                == "condition_centered"
            ),
        },
        "training": {
            "forced_value_history": forced_result.history,
            "free_policy_history": free_result.history,
            "free_policy_checkpoints": free_result.checkpoints,
            "forced_value_runtime_seconds": forced_runtime,
            "free_policy_runtime_seconds": free_runtime,
        },
        "evaluation": evaluation,
        "evaluation_by_condition": evaluation_by_condition,
        "evaluation_bank_generation_runtime_seconds": evaluation_bank_runtime,
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
    parser.add_argument(
        "--output-directory",
        help=(
            "Write the seed-level JSON to this directory instead of the config "
            "default."
        ),
    )
    arguments = parser.parse_args()
    result = run_sampled_hazard_config(
        arguments.config,
        seed=arguments.seed,
        output_directory=arguments.output_directory,
    )
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
