"""Aggregate the predeclared mixed-credit factorial without filtering runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

METRICS = (
    "endpoint_probability",
    "endpoint_probability_gap",
    "learned_expected_reward",
    "endpoint_pair_reward",
    "never_pair_reward",
    "matched_random_one_reward",
)


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "n_seeds": len(array),
    }


def _checkpoint_dm(checkpoint: dict[str, Any]) -> dict[str, Any]:
    evaluation = checkpoint["evaluation"]
    if "by_condition" in evaluation:
        return evaluation["by_condition"]["DM"]
    return evaluation


def aggregate_credit_factorial(
    config_paths: list[str | Path],
    *,
    input_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate paired seeds for every declared factorial cell."""

    if not config_paths:
        raise ValueError("at least one factorial configuration is required")
    input_directory = Path(input_directory)
    cells = []
    all_git_shas: set[str] = set()
    for path_like in config_paths:
        config_path = Path(path_like)
        config_bytes = config_path.read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        config = yaml.safe_load(config_bytes)
        name = config["experiment"]["name"]
        seeds = list(config["experiment"]["model_seeds"])
        records = []
        for seed in seeds:
            record_path = input_directory / f"{name}_seed{seed}.json"
            if not record_path.exists():
                raise FileNotFoundError(f"missing factorial record: {record_path}")
            record = json.loads(record_path.read_text())
            if record["seed"] != seed:
                raise ValueError(f"seed mismatch in {record_path}")
            if record["configuration_sha256"] != config_hash:
                raise ValueError(f"configuration hash mismatch in {record_path}")
            records.append(record)
            all_git_shas.add(record["provenance"]["git_sha"])

        evaluations = [record["evaluation_by_condition"]["DM"] for record in records]
        metric_summary = {
            metric: _summary(
                [float(evaluation[metric]) for evaluation in evaluations]
            )
            for metric in METRICS
        }
        epochs = [
            float(point["epoch"])
            for point in records[0]["training"]["free_policy_checkpoints"]
        ]
        curve = []
        for index, epoch in enumerate(epochs):
            points = [
                _checkpoint_dm(record["training"]["free_policy_checkpoints"][index])
                for record in records
            ]
            curve.append(
                {
                    "epoch": epoch,
                    "endpoint_probability": _summary(
                        [float(point["endpoint_probability"]) for point in points]
                    ),
                    "endpoint_probability_gap": _summary(
                        [
                            float(point["endpoint_probability_gap"])
                            for point in points
                        ]
                    ),
                    "learned_expected_reward": _summary(
                        [
                            float(point["learned_expected_reward"])
                            for point in points
                        ]
                    ),
                }
            )
        seed_passes = [
            evaluation["endpoint_probability"] >= 0.80
            and evaluation["endpoint_probability_gap"] >= 0.50
            and evaluation["learned_expected_reward"]
            > evaluation["never_pair_reward"]
            and evaluation["learned_expected_reward"]
            > evaluation["matched_random_one_reward"]
            for evaluation in evaluations
        ]
        cells.append(
            {
                "name": name,
                "configuration": str(config_path),
                "configuration_sha256": config_hash,
                "factors": {
                    "initial_probability": config["policy"][
                        "initial_probability"
                    ],
                    "advantage_mode": config["training"]["free_policy"][
                        "advantage_mode"
                    ],
                    "condition_schedule": config["training"]["free_policy"][
                        "condition_schedule"
                    ],
                },
                "seeds": seeds,
                "summary": metric_summary,
                "runs": [
                    {
                        "seed": seed,
                        **{
                            metric: float(evaluation[metric])
                            for metric in METRICS
                        },
                        "screen_passed": passed,
                    }
                    for seed, evaluation, passed in zip(
                        seeds,
                        evaluations,
                        seed_passes,
                        strict=True,
                    )
                ],
                "learning_curve": curve,
                "screen_passed": all(seed_passes),
            }
        )

    selected = max(
        cells,
        key=lambda cell: (
            sum(run["screen_passed"] for run in cell["runs"]),
            cell["summary"]["endpoint_probability"]["mean"],
            cell["summary"]["learned_expected_reward"]["mean"],
        ),
    )
    result: dict[str, Any] = {
        "task": "Lu-Hasson-Norman 2022 variable-duration condition mixture",
        "dataset": "controlled synthetic generator (released_code)",
        "metrics": list(METRICS),
        "seeds": sorted({seed for cell in cells for seed in cell["seeds"]}),
        "provenance": {
            "mode": "measured",
            "source": str(input_directory / "sampled_hazard_credit_*_seed*.json"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_shas": sorted(all_git_shas),
            "note": "Measured synthetic simulations; two-seed exploratory screen.",
        },
        "summary": {
            cell["name"]: cell["summary"] for cell in cells
        },
        "cells": cells,
        "selected_cell": selected["name"],
        "selection_rule": (
            "Most seeds meeting all screen thresholds, then mean endpoint "
            "probability, then mean DM reward. Selection is exploratory."
        ),
        "factorial_passed": any(cell["screen_passed"] for cell in cells),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = aggregate_credit_factorial(
        arguments.configs,
        input_directory=arguments.input_directory,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "selected_cell": result["selected_cell"],
                "factorial_passed": result["factorial_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
