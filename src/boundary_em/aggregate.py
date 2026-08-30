"""Aggregate preregistered model-seed results and apply success criteria."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    n_samples: int,
) -> tuple[float, float]:
    """Return a fixed-seed percentile interval over model-seed means."""

    if not values:
        raise ValueError("bootstrap requires at least one value")
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    indices = generator.integers(0, len(array), size=(n_samples, len(array)))
    bootstrapped_means = array[indices].mean(axis=1)
    low, high = np.percentile(bootstrapped_means, [2.5, 97.5])
    return float(low), float(high)


def _seed_metric_cell(
    values: list[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, float | int | list[float]]:
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


def _method_evaluations(seed_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evaluation = seed_result["evaluation"]
    methods = {
        "learned_policy": evaluation["stochastic"],
        "learned_policy_deterministic": evaluation["deterministic"],
        "learned_policy_ood_duration": evaluation["ood_stochastic"],
        "learned_policy_ood_duration_deterministic": evaluation[
            "ood_deterministic"
        ],
    }
    methods.update(
        {
            f"intervention_{name}": metrics
            for name, metrics in evaluation["interventions"].items()
        }
    )
    return methods


def aggregate_reported(
    config_path: str | Path,
    *,
    input_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load every declared seed, aggregate it, and evaluate frozen criteria."""

    config_path = Path(config_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(config_path.read_text())
    expected_seeds = list(config["experiment"]["model_seeds"])
    experiment_name = str(config["experiment"]["name"])
    input_directory = Path(input_directory)
    if not input_directory.is_absolute():
        input_directory = repository / input_directory
    input_directory = input_directory.resolve()
    seed_results: list[dict[str, Any]] = []
    input_files: list[str] = []
    for seed in expected_seeds:
        path = input_directory / f"{experiment_name}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing declared seed result: {path}")
        seed_result = json.loads(path.read_text())
        if seed_result["seed"] != seed:
            raise ValueError(f"seed mismatch in {path}")
        if seed_result["experiment_status"] != "confirmatory":
            raise ValueError(f"non-confirmatory result in {path}")
        seed_results.append(seed_result)
        input_files.append(str(path.relative_to(repository)))

    config_hashes = {
        result["provenance"]["config_sha256"] for result in seed_results
    }
    if len(config_hashes) != 1:
        raise ValueError("confirmatory seed files used different configurations")

    criteria = config["success_criteria"]
    bootstrap_seed = int(criteria["bootstrap_seed"])
    bootstrap_samples = int(criteria["bootstrap_samples"])
    method_names = list(_method_evaluations(seed_results[0]))
    metric_names = list(_method_evaluations(seed_results[0])[method_names[0]])
    summary: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, dict[str, float]]] = {}
    for method in method_names:
        summary[method] = {}
        runs[method] = {}
        for metric in metric_names:
            values = [
                float(_method_evaluations(result)[method][metric]["mean"])
                for result in seed_results
            ]
            summary[method][metric] = _seed_metric_cell(
                values,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            )
            runs[method][metric] = {
                str(seed): value
                for seed, value in zip(expected_seeds, values, strict=True)
            }

    selectivity = [
        float(result["evaluation"]["stochastic"]["endpoint_selectivity"]["mean"])
        for result in seed_results
    ]
    learned_reward = [
        float(result["evaluation"]["stochastic"]["reward"]["mean"])
        for result in seed_results
    ]
    random_reward = [
        float(
            result["evaluation"]["interventions"]["matched_random_one_write"][
                "reward"
            ]["mean"]
        )
        for result in seed_results
    ]
    endpoint_reward = [
        float(
            result["evaluation"]["interventions"]["endpoint_only"]["reward"][
                "mean"
            ]
        )
        for result in seed_results
    ]
    deterministic_reward = [
        float(result["evaluation"]["deterministic"]["reward"]["mean"])
        for result in seed_results
    ]
    displaced_reward = [
        float(
            result["evaluation"]["interventions"]["displaced_learned"][
                "reward"
            ]["mean"]
        )
        for result in seed_results
    ]
    gap_closure = [
        (learned - random) / (endpoint - random)
        for learned, random, endpoint in zip(
            learned_reward,
            random_reward,
            endpoint_reward,
            strict=True,
        )
    ]
    displacement_loss = [
        learned - displaced
        for learned, displaced in zip(
            deterministic_reward,
            displaced_reward,
            strict=True,
        )
    ]
    selectivity_ci = bootstrap_mean_ci(
        selectivity,
        seed=bootstrap_seed,
        n_samples=bootstrap_samples,
    )
    gap_ci = bootstrap_mean_ci(
        gap_closure,
        seed=bootstrap_seed,
        n_samples=bootstrap_samples,
    )
    displacement_ci = bootstrap_mean_ci(
        displacement_loss,
        seed=bootstrap_seed,
        n_samples=bootstrap_samples,
    )
    positive_seeds = sum(value > 0 for value in selectivity)
    success_checks = {
        "minimum_positive_seeds": positive_seeds
        >= int(criteria["minimum_positive_seeds"]),
        "minimum_gap_closure": statistics.fmean(gap_closure)
        >= float(criteria["minimum_gap_closure"]),
        "selectivity_ci_above_zero": selectivity_ci[0] > 0,
        "displacement_loss_ci_above_zero": displacement_ci[0] > 0,
    }

    result: dict[str, Any] = {
        "task": "learned episodic writing for delayed event-feature prediction",
        "dataset": "controlled synthetic four-feature events",
        "metrics": metric_names,
        "seeds": expected_seeds,
        "provenance": {
            "mode": "measured",
            "data_kind": "synthetic",
            "source": "aggregation of measured boundary_em.run_policy outputs",
            "data_contract": "docs/learned_encoding/data_contract.md",
            "config": str(config_path.relative_to(repository)),
            "config_sha256": next(iter(config_hashes)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregation_git_sha": _git_sha(repository),
            "execution_git_shas": sorted(
                {seed_result["provenance"]["git_sha"] for seed_result in seed_results}
            ),
            "input_files": input_files,
        },
        "summary": summary,
        "runs": runs,
        "success_audit": {
            "positive_selectivity_seeds": positive_seeds,
            "required_positive_seeds": int(criteria["minimum_positive_seeds"]),
            "mean_gap_closure": statistics.fmean(gap_closure),
            "required_gap_closure": float(criteria["minimum_gap_closure"]),
            "selectivity_ci95": list(selectivity_ci),
            "gap_closure_ci95": list(gap_ci),
            "mean_displacement_loss": statistics.fmean(displacement_loss),
            "displacement_loss_ci95": list(displacement_ci),
            "checks": success_checks,
            "all_criteria_passed": all(success_checks.values()),
        },
        "notes": (
            "Measured synthetic-task aggregation. Confidence intervals resample "
            "the 15 independent model seeds with the preregistered bootstrap seed."
        ),
    }
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = repository / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate_reported(
        args.config,
        input_directory=args.input_directory,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
