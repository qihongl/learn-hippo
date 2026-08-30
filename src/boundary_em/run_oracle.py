"""Configuration-driven exhaustive oracle experiment."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from boundary_em.oracle import ScheduleResult, enumerate_schedules, evaluate_schedule
from boundary_em.task import TaskConfig, generate_episode


def _metric_cell(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n_seeds": len(values),
    }


def _schedule_label(schedule: tuple[bool, ...]) -> str:
    return "".join("1" if write else "0" for write in schedule)


def _summary_cell(result: ScheduleResult) -> dict[str, dict[str, float | int | None]]:
    rewards = list(result.per_episode_reward)
    losses = [1.0 - reward for reward in rewards]
    return {"reward": _metric_cell(rewards), "mse": _metric_cell(losses)}


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_oracle_config(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    n_episodes_override: int | None = None,
) -> dict[str, Any]:
    """Run and persist the exhaustive fixed-schedule experiment."""

    config_path = Path(config_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    with config_path.open() as stream:
        raw_config = yaml.safe_load(stream)

    task_config = TaskConfig(**raw_config["task"])
    oracle_config = raw_config["oracle"]
    n_episodes = n_episodes_override or int(oracle_config["n_episodes"])
    seed_start = int(oracle_config["episode_seed_start"])
    seeds = list(range(seed_start, seed_start + n_episodes))
    episodes = [generate_episode(task_config, seed=seed) for seed in seeds]
    temperature = float(oracle_config["temperature"])
    fixed_capacity = int(oracle_config["fixed_capacity"])

    rankings: dict[str, list[dict[str, float | str]]] = {}
    preconditions: dict[str, dict[str, bool | float | str]] = {}
    endpoint_schedule = (False,) * (task_config.n_features - 1) + (True,)
    for capacity_mode in oracle_config["capacity_modes"]:
        ranked = enumerate_schedules(
            episodes,
            temperature=temperature,
            capacity_mode=capacity_mode,
            fixed_capacity=fixed_capacity,
        )
        rankings[capacity_mode] = [
            {
                "schedule": _schedule_label(result.schedule),
                "mean_reward": result.mean_reward,
                "mean_mse": result.mean_mse,
            }
            for result in ranked
        ]
        margin = ranked[0].mean_reward - ranked[1].mean_reward
        preconditions[capacity_mode] = {
            "passed": ranked[0].schedule == endpoint_schedule and margin > 0,
            "best_schedule": _schedule_label(ranked[0].schedule),
            "best_reward": ranked[0].mean_reward,
            "margin_over_second": margin,
        }

    primary_mode = "fixed"
    named_schedules = {
        "endpoint_only": endpoint_schedule,
        "always_write": (True,) * task_config.n_features,
        "never_write": (False,) * task_config.n_features,
        "midpoint_only": tuple(
            index == task_config.query_features - 1
            for index in range(task_config.n_features)
        ),
        "midpoint_plus_endpoint": tuple(
            index in {task_config.query_features - 1, task_config.n_features - 1}
            for index in range(task_config.n_features)
        ),
    }
    named_results = {
        name: evaluate_schedule(
            episodes,
            schedule,
            temperature=temperature,
            capacity_mode=primary_mode,
            fixed_capacity=fixed_capacity,
        )
        for name, schedule in named_schedules.items()
    }
    one_write_results = [
        evaluate_schedule(
            episodes,
            tuple(index == write_position for index in range(task_config.n_features)),
            temperature=temperature,
            capacity_mode=primary_mode,
            fixed_capacity=fixed_capacity,
        )
        for write_position in range(task_config.n_features)
    ]
    random_rewards = [
        reward
        for result in one_write_results
        for reward in result.per_episode_reward
    ]
    random_losses = [1.0 - reward for reward in random_rewards]

    results: dict[str, Any] = {
        "task": "delayed event-feature prediction with episodic memory",
        "dataset": "controlled synthetic four-feature events",
        "metrics": ["reward", "mse"],
        "seeds": seeds,
        "provenance": {
            "mode": "measured",
            "data_kind": "synthetic",
            "source": "measured execution of boundary_em.run_oracle",
            "data_contract": "docs/learned_encoding/data_contract.md",
            "config": str(config_path.relative_to(repository)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(repository),
            "hardware": platform.platform(),
            "torch_version": torch.__version__,
        },
        "summary": {
            name: _summary_cell(result) for name, result in named_results.items()
        },
        "oracle_precondition": preconditions,
        "schedule_rankings": rankings,
        "notes": (
            "Measured synthetic-task results. Fixed capacity is the primary mode; "
            "historical capacity is retained as a resource-accounting diagnostic."
        ),
    }
    results["summary"]["matched_random_one_write"] = {
        "reward": _metric_cell(random_rewards),
        "mse": _metric_cell(random_losses),
    }

    destination = (
        Path(output_path)
        if output_path is not None
        else repository / raw_config["output"]["path"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_oracle_config(args.config, output_path=args.output)


if __name__ == "__main__":
    main()
