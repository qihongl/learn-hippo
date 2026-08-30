"""Configuration-driven training and evaluation of one write-policy seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from boundary_em.evaluation import PolicyEvaluation, evaluate_policy
from boundary_em.policy_training import PolicyTrainingConfig, train_policy
from boundary_em.task import TaskConfig


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _evaluation_dict(evaluation: PolicyEvaluation) -> dict[str, Any]:
    return {
        "summary": evaluation.summary,
        "per_episode": list(evaluation.per_episode),
    }


def _display_path(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path.resolve())


def run_policy_config(
    config_path: str | Path,
    *,
    seed: int,
    output_directory: str | Path | None = None,
    checkpoint_directory: str | Path | None = None,
    updates_override: int | None = None,
    batch_size_override: int | None = None,
    evaluation_episodes_override: int | None = None,
) -> dict[str, Any]:
    """Train one seed, evaluate frozen weights, and persist all metrics."""

    config_path = Path(config_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    config_bytes = config_path.read_bytes()
    raw_config = yaml.safe_load(config_bytes)
    experiment_name = str(raw_config["experiment"]["name"])
    task_config = TaskConfig(**raw_config["task"])
    training_values = dict(raw_config["training"])
    training_values["hidden_dim"] = int(raw_config["model"]["hidden_dim"])
    training_values["temperature"] = float(raw_config["memory"]["temperature"])
    training_values["capacity"] = int(raw_config["memory"]["capacity"])
    if updates_override is not None:
        training_values["updates"] = updates_override
    if batch_size_override is not None:
        training_values["batch_size"] = batch_size_override
    train_config = PolicyTrainingConfig(**training_values)

    run = train_policy(task_config, train_config, seed=seed)
    evaluation_config = raw_config["evaluation"]
    n_evaluation_episodes = (
        evaluation_episodes_override or int(evaluation_config["n_episodes"])
    )
    evaluation_start = int(evaluation_config["validation_seed_start"])
    evaluation_seeds = range(
        evaluation_start,
        evaluation_start + n_evaluation_episodes,
    )
    action_seed = int(evaluation_config["action_seed_offset"]) + seed
    stochastic = evaluate_policy(
        run.model,
        task_config,
        episode_seeds=evaluation_seeds,
        action_seed=action_seed,
        temperature=train_config.temperature,
        capacity=train_config.capacity,
        stochastic=True,
    )
    deterministic = evaluate_policy(
        run.model,
        task_config,
        episode_seeds=evaluation_seeds,
        action_seed=action_seed,
        temperature=train_config.temperature,
        capacity=train_config.capacity,
        stochastic=False,
    )

    output_root = (
        Path(output_directory)
        if output_directory is not None
        else repository / raw_config["output"]["directory"]
    )
    checkpoint_root = (
        Path(checkpoint_directory)
        if checkpoint_directory is not None
        else repository / raw_config["output"]["checkpoint_directory"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{experiment_name}_seed{seed}.json"
    checkpoint_path = checkpoint_root / f"{experiment_name}_seed{seed}.pt"
    git_sha = _git_sha(repository)
    torch.save(
        {
            "model_state_dict": run.model.state_dict(),
            "task_config": raw_config["task"],
            "training_config": training_values,
            "model_seed": seed,
            "git_sha": git_sha,
        },
        checkpoint_path,
    )

    results: dict[str, Any] = {
        "task": "learned episodic writing for delayed event-feature prediction",
        "dataset": "controlled synthetic four-feature events",
        "metrics": [
            "reward",
            "mse",
            "writes_per_event",
            "endpoint_selectivity",
            "boundary_auc",
        ],
        "seeds": [seed],
        "seed": seed,
        "experiment_status": raw_config["experiment"]["status"],
        "provenance": {
            "mode": "measured",
            "data_kind": "synthetic",
            "source": "measured execution of boundary_em.run_policy",
            "data_contract": "docs/learned_encoding/data_contract.md",
            "config": str(config_path.relative_to(repository)),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "hardware": platform.platform(),
            "torch_version": torch.__version__,
        },
        "configuration": {
            "task": raw_config["task"],
            "memory": raw_config["memory"],
            "model": raw_config["model"],
            "training": training_values,
            "evaluation": {
                **evaluation_config,
                "n_episodes": n_evaluation_episodes,
            },
        },
        "training_curves": {
            key: [point[key] for point in run.history]
            for key in run.history[0]
            if key != "update"
        },
        "evaluation": {
            "stochastic": stochastic.summary,
            "deterministic": deterministic.summary,
        },
        "evaluation_records": {
            "stochastic": _evaluation_dict(stochastic)["per_episode"],
            "deterministic": _evaluation_dict(deterministic)["per_episode"],
        },
        "checkpoint": _display_path(checkpoint_path, repository),
        "notes": (
            "Measured synthetic-task run. Model weights were frozen for both "
            "held-out evaluations; checkpoint files are excluded from Git."
        ),
    }
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run_policy_config(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
