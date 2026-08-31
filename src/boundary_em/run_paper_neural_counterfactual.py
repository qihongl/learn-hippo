"""Run the neural exact-state counterfactual encoding experiment."""

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
    NeuralCounterfactualExample,
    build_counterfactual_example,
    evaluate_neural_counterfactual,
    initialize_neural_hazard_policy,
    train_neural_counterfactual,
)
from boundary_em.paper_task import Condition, PaperTaskConfig, generate_trial
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


def _build_bank(
    model: StructuredEpisodicPredictionModel,
    task_config: PaperTaskConfig,
    *,
    conditions: tuple[Condition, ...],
    seed_start: int,
    n_examples: int,
    memory_capacity: int,
    evaluation_mode: bool,
) -> list[NeuralCounterfactualExample]:
    if n_examples < 1:
        raise ValueError("n_examples must be positive")
    return [
        build_counterfactual_example(
            model,
            generate_trial(
                task_config,
                seed=seed_start + index,
                condition=conditions[index % len(conditions)],
                evaluation=evaluation_mode,
            ),
            memory_capacity=memory_capacity,
        )
        for index in range(n_examples)
    ]


def run_neural_counterfactual_config(
    config_path: str | Path,
    *,
    seed: int,
    repository_root: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one declared neural exact-state development seed."""

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
        initial_probability=float(raw["optimization"]["initial_probability"]),
    )
    bank_config = raw["counterfactual_banks"]
    seed_offset = seed * 100_000
    task_conditions = tuple(
        raw["task"].get("conditions", [raw["task"].get("condition", "DM")])
    )
    task_evaluation_mode = bool(raw["task"]["evaluation_mode"])

    bank_start = perf_counter()
    training_examples = _build_bank(
        model,
        task_config,
        conditions=task_conditions,
        seed_start=int(bank_config["training_seed_start"]) + seed_offset,
        n_examples=int(bank_config["training_examples"]),
        memory_capacity=int(bank_config["memory_capacity"]),
        evaluation_mode=task_evaluation_mode,
    )
    training_bank_runtime = perf_counter() - bank_start
    bank_start = perf_counter()
    evaluation_conditions = tuple(
        bank_config.get("evaluation_conditions", task_conditions)
    )
    evaluation_examples_per_condition = int(
        bank_config.get(
            "evaluation_examples_per_condition",
            bank_config.get("evaluation_examples", 0),
        )
    )
    evaluation_examples = {
        condition: _build_bank(
            model,
            task_config,
            conditions=(condition,),
            seed_start=int(bank_config["evaluation_seed_start"])
            + seed_offset
            + condition_index * evaluation_examples_per_condition,
            n_examples=evaluation_examples_per_condition,
            memory_capacity=int(bank_config["memory_capacity"]),
            evaluation_mode=task_evaluation_mode,
        )
        for condition_index, condition in enumerate(evaluation_conditions)
    }
    evaluation_bank_runtime = perf_counter() - bank_start

    optimization = raw["optimization"]

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

    training_start = perf_counter()
    training_result = train_neural_counterfactual(
        model,
        training_examples,
        updates=int(optimization["updates"]),
        batch_size=int(optimization["batch_size"]),
        learning_rate=float(optimization["learning_rate"]),
        gradient_clip=float(optimization["gradient_clip"]),
        seed=seed + 50_000_000,
        checkpoint_interval=int(optimization["checkpoint_interval"]),
        checkpoint_evaluator=checkpoint_evaluator,
    )
    training_runtime = perf_counter() - training_start
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
            "counterfactual_outcomes_used_during_training": True,
            "training_bank_reused_across_updates": True,
            "interpretation": "neural representation development, not final claim",
        },
        "training": {
            "examples": len(training_examples),
            "bank_generation_runtime_seconds": training_bank_runtime,
            "optimization_runtime_seconds": training_runtime,
            "history": training_result.history,
            "checkpoints": training_result.checkpoints,
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
    arguments = parser.parse_args()
    result = run_neural_counterfactual_config(arguments.config, seed=arguments.seed)
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
