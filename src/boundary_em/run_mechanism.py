"""Secondary analyses of learned write signals and retrieval competition."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from boundary_em.aggregate import bootstrap_mean_ci
from boundary_em.evaluation import evaluate_policy, write_probabilities_by_progress
from boundary_em.oracle import evaluate_schedule
from boundary_em.policy import WriteActorCritic
from boundary_em.task import TaskConfig, generate_episode


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _display_path(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError:
        return str(path.resolve())


def _metric_cell(
    values: list[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
        "ci95": list(
            bootstrap_mean_ci(
                values,
                seed=bootstrap_seed,
                n_samples=bootstrap_samples,
            )
        ),
    }


def run_mechanism_config(
    config_path: str | Path,
    *,
    checkpoint_directory: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run declared post-confirmatory mechanism analyses."""

    config_path = Path(config_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    task_config = TaskConfig(**config["task"])
    model_seeds = list(config["experiment"]["model_seeds"])
    evaluation_config = config["evaluation"]
    analysis_config = config["analysis"]
    episode_seeds = list(
        range(
            int(evaluation_config["episode_seed_start"]),
            int(evaluation_config["episode_seed_start"])
            + int(evaluation_config["n_episodes"]),
        )
    )
    checkpoint_root = (
        Path(checkpoint_directory)
        if checkpoint_directory is not None
        else repository / config["output"]["checkpoint_directory"]
    )
    checkpoint_name = str(config["output"]["checkpoint_name"])
    bootstrap_seed = int(analysis_config["bootstrap_seed"])
    bootstrap_samples = int(analysis_config["bootstrap_samples"])
    np.random.seed(bootstrap_seed)

    input_results: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in analysis_config["input_ablations"]
    }
    progress_by_seed: dict[int, dict[int, float]] = {}
    checkpoint_shas: set[str] = set()
    for model_seed in model_seeds:
        checkpoint_path = checkpoint_root / f"{checkpoint_name}_seed{model_seed}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint["model_seed"]) != model_seed:
            raise ValueError(f"checkpoint seed mismatch: {checkpoint_path}")
        checkpoint_shas.add(str(checkpoint["git_sha"]))
        model = WriteActorCritic(
            input_dim=task_config.cue_dim + 2 * task_config.n_features,
            hidden_dim=int(config["model"]["hidden_dim"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        for mode in analysis_config["input_ablations"]:
            evaluation = evaluate_policy(
                model,
                task_config,
                episode_seeds=episode_seeds,
                action_seed=int(evaluation_config["action_seed_offset"]) + model_seed,
                temperature=float(config["memory"]["temperature"]),
                capacity=int(config["memory"]["capacity"]),
                stochastic=True,
                input_ablation=mode,
            )
            input_results[mode].append(evaluation.summary)
        progress = write_probabilities_by_progress(
            model,
            task_config,
            episode_seeds=episode_seeds,
            n_null_steps=0,
            input_ablation="full",
        )
        progress_by_seed[model_seed] = {
            level: statistics.fmean(probabilities)
            for level, probabilities in progress.items()
        }

    summary: dict[str, dict[str, Any]] = {}
    for mode, seed_summaries in input_results.items():
        summary[f"input_{mode}"] = {}
        for metric in seed_summaries[0]:
            values = [
                float(seed_summary[metric]["mean"])
                for seed_summary in seed_summaries
            ]
            summary[f"input_{mode}"][metric] = _metric_cell(
                values,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            )

    progress_summary: dict[str, dict[str, Any]] = {}
    for progress in range(1, task_config.n_features + 1):
        values = [progress_by_seed[seed][progress] for seed in model_seeds]
        progress_summary[str(progress)] = {
            **_metric_cell(
                values,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            ),
            "per_seed": {
                str(seed): value
                for seed, value in zip(model_seeds, values, strict=True)
            },
        }

    oracle_episodes = [
        generate_episode(task_config, seed=seed) for seed in episode_seeds
    ]
    endpoint = (False,) * (task_config.n_features - 1) + (True,)
    always = (True,) * task_config.n_features
    midpoint_plus_endpoint = tuple(
        index in {task_config.query_features - 1, task_config.n_features - 1}
        for index in range(task_config.n_features)
    )
    temperature_sweep = []
    for temperature in analysis_config["retrieval_temperatures"]:
        temperature_sweep.append(
            {
                "temperature": float(temperature),
                "endpoint_only_reward": evaluate_schedule(
                    oracle_episodes,
                    endpoint,
                    temperature=float(temperature),
                ).mean_reward,
                "always_write_reward": evaluate_schedule(
                    oracle_episodes,
                    always,
                    temperature=float(temperature),
                ).mean_reward,
                "midpoint_plus_endpoint_reward": evaluate_schedule(
                    oracle_episodes,
                    midpoint_plus_endpoint,
                    temperature=float(temperature),
                ).mean_reward,
            }
        )
    latest_always_reward = evaluate_schedule(
        oracle_episodes,
        always,
        temperature=float(config["memory"]["temperature"]),
        retrieval_mode="latest",
    ).mean_reward

    result: dict[str, Any] = {
        "task": "secondary mechanism analysis of learned episodic writing",
        "dataset": "controlled synthetic four-feature events",
        "metrics": [
            "reward",
            "mse",
            "writes_per_event",
            "endpoint_selectivity",
            "boundary_auc",
        ],
        "seeds": model_seeds,
        "experiment_status": config["experiment"]["status"],
        "provenance": {
            "mode": "measured",
            "data_kind": "synthetic",
            "source": "measured post-confirmatory checkpoint analysis",
            "config": _display_path(config_path, repository),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(repository),
            "checkpoint_git_shas": sorted(checkpoint_shas),
            "hardware": platform.platform(),
            "torch_version": torch.__version__,
        },
        "summary": summary,
        "write_probability_by_progress": progress_summary,
        "retrieval_ablation": {
            "temperature_sweep": temperature_sweep,
            "latest_always_write_reward": latest_always_reward,
        },
        "notes": (
            "Secondary exploratory analyses declared after the primary result. "
            "They do not alter the confirmatory success audit."
        ),
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else repository / config["output"]["path"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_mechanism_config(args.config)


if __name__ == "__main__":
    main()
