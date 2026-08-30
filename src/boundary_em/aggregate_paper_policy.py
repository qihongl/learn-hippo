"""Aggregate all declared exact-task diagnostic model seeds."""

from __future__ import annotations

import argparse
import hashlib
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


def _summary_cell(
    values: list[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    generator = np.random.default_rng(bootstrap_seed)
    array = np.asarray(values)
    indices = generator.integers(
        0,
        len(values),
        size=(bootstrap_samples, len(values)),
    )
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95": [float(low), float(high)],
        "n_model_seeds": len(values),
        "values_by_seed": values,
    }


def aggregate_paper_policy(
    config_path: str | Path,
    *,
    input_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate, aggregate, audit, and save every declared model seed."""

    config_path = Path(config_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(config_path.read_text())
    input_directory = Path(input_directory)
    if not input_directory.is_absolute():
        input_directory = repository / input_directory
    expected_seeds = list(config["experiment"]["model_seeds"])
    experiment_name = config["experiment"]["name"]
    seed_results = []
    input_files = []
    for seed in expected_seeds:
        path = input_directory / f"{experiment_name}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing declared seed result: {path}")
        result = json.loads(path.read_text())
        if result["seed"] != seed:
            raise ValueError(f"seed mismatch in {path}")
        if result["experiment_status"] != config["experiment"]["status"]:
            raise ValueError(f"experiment status mismatch in {path}")
        seed_results.append(result)
        input_files.append(str(path.relative_to(repository)))
    config_hashes = {result["provenance"]["config_sha256"] for result in seed_results}
    if len(config_hashes) != 1:
        raise ValueError("seed files used different configurations")
    current_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_hashes != {current_config_hash}:
        raise ValueError("seed files do not match the current configuration")

    bootstrap_seed = 20_260_830
    audit_metrics = {
        name: [float(result["seed_success_audit"][name]) for result in seed_results]
        for name in (
            "endpoint_probability_gap",
            "dm_prediction_benefit_over_never",
            "dm_retrieval_benefit",
        )
    }
    audit_summary = {
        name: _summary_cell(values, bootstrap_seed=bootstrap_seed)
        for name, values in audit_metrics.items()
    }
    conditions: dict[str, Any] = {}
    for condition in config["evaluation"]["conditions"]:
        ablation_names = seed_results[0]["evaluation"][condition].get(
            "ablations", {}
        )
        conditions[condition] = {
            "learned": {
                metric: _summary_cell(
                    [
                        float(
                            result["evaluation"][condition]["learned"][metric]["mean"]
                        )
                        for result in seed_results
                    ],
                    bootstrap_seed=bootstrap_seed,
                )
                for metric in (
                    "reward",
                    "endpoint_probability",
                    "nonendpoint_probability",
                    "encodings_per_event",
                )
            },
            "retrieval_off_reward": _summary_cell(
                [
                    float(
                        result["evaluation"][condition]["retrieval_off_same_actions"][
                            "reward"
                        ]["mean"]
                    )
                    for result in seed_results
                ],
                bootstrap_seed=bootstrap_seed,
            ),
            "forced_reward": {
                schedule: _summary_cell(
                    [
                        float(
                            result["evaluation"][condition]["forced"][schedule][
                                "reward"
                            ]["mean"]
                        )
                        for result in seed_results
                    ],
                    bootstrap_seed=bootstrap_seed,
                )
                for schedule in seed_results[0]["evaluation"][condition]["forced"]
            },
            "ablation_reward": {
                ablation: _summary_cell(
                    [
                        float(
                            result["evaluation"][condition]["ablations"][ablation][
                                "reward"
                            ]["mean"]
                        )
                        for result in seed_results
                    ],
                    bootstrap_seed=bootstrap_seed,
                )
                for ablation in ablation_names
            },
            "time_probabilities": [
                _summary_cell(
                    [
                        float(
                            result["evaluation"][condition]["learned"][
                                "time_probabilities"
                            ][time_index]["mean"]
                        )
                        for result in seed_results
                    ],
                    bootstrap_seed=bootstrap_seed,
                )
                for time_index in range(16)
            ],
        }

    required_seeds = int(config["success_criteria"]["required_model_seeds"])
    endpoint_positive_seeds = sum(
        value > 0 for value in audit_metrics["endpoint_probability_gap"]
    )
    checks = {
        "required_seed_count": len(seed_results) >= required_seeds,
        "endpoint_selectivity": audit_summary["endpoint_probability_gap"]["ci95"][0]
        > 0,
        "dm_prediction_benefit": audit_summary["dm_prediction_benefit_over_never"][
            "ci95"
        ][0]
        > 0,
        "retrieval_causal_benefit": audit_summary["dm_retrieval_benefit"]["ci95"][0]
        > 0,
    }
    aggregate: dict[str, Any] = {
        "task": "exact Lu-Hasson-Norman 2022 event-prediction generator",
        "data_kind": "measured synthetic simulation",
        "experiment_status": config["experiment"]["status"],
        "model_seeds": expected_seeds,
        "provenance": {
            "mode": "measured",
            "source": "aggregation of boundary_em.run_paper_policy results",
            "config": str(config_path.relative_to(repository)),
            "config_sha256": next(iter(config_hashes)),
            "input_files": input_files,
            "execution_git_shas": sorted(
                {result["provenance"]["git_sha"] for result in seed_results}
            ),
            "aggregation_git_sha": _git_sha(repository),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": 10_000,
        },
        "audit_metrics": audit_summary,
        "conditions": conditions,
        "success_audit": {
            "endpoint_positive_model_seeds": endpoint_positive_seeds,
            "required_model_seeds": required_seeds,
            "checks": checks,
            "all_criteria_passed": all(checks.values()),
        },
        "notes": (
            "Diagnostic replication, not a preregistered confirmatory experiment. "
            "Intervals resample independent model seeds; no seed was excluded."
        ),
    }
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = repository / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate_paper_policy(
        args.config,
        input_directory=args.input_directory,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
