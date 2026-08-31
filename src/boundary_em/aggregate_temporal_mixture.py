"""Aggregate exact temporal mixture audits across declared random starts."""

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
    "learned_expected_reward",
    "endpoint_probability",
    "mean_nonendpoint_probability",
    "never_probability",
    "endpoint_probability_gap",
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


def aggregate_temporal_mixture_config(
    config_path: str | Path,
    *,
    input_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Retain all declared mixture-audit seeds and compact their trajectories."""

    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = yaml.safe_load(config_bytes)
    experiment = config["experiment"]["name"]
    seeds = list(config["experiment"]["model_seeds"])
    input_directory = Path(input_directory)

    records = []
    for seed in seeds:
        path = input_directory / f"{experiment}_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing declared seed record: {path}")
        record = json.loads(path.read_text())
        if record["seed"] != seed:
            raise ValueError(f"seed mismatch in {path}")
        if record["configuration_sha256"] != config_hash:
            raise ValueError(f"configuration hash mismatch in {path}")
        records.append(record)

    summary = {
        metric: _summary(
            [float(record["evaluation"][metric]) for record in records]
        )
        for metric in METRICS
    }
    completed_updates = [
        int(point["completed_updates"])
        for point in records[0]["training"]["history"]
    ]
    selected_updates = {
        update for update in (1, 10, 25, 50, 100, 200, 400, 600, 800, 1_000)
    }
    training_curve = []
    for index, update in enumerate(completed_updates):
        if update not in selected_updates:
            continue
        points = [record["training"]["history"][index] for record in records]
        training_curve.append(
            {
                "update": update,
                "endpoint_probability": _summary(
                    [float(point["endpoint_probability"]) for point in points]
                ),
                "expected_reward": _summary(
                    [float(point["expected_reward"]) for point in points]
                ),
            }
        )

    condition_summary = {
        condition: {
            metric: _summary(
                [
                    float(record["evaluation_by_condition"][condition][metric])
                    for record in records
                ]
            )
            for metric in (
                "learned_expected_reward",
                "endpoint_pair_reward",
                "never_pair_reward",
            )
        }
        for condition in ("RM", "DM", "NM")
    }
    objective_endpoint_best = all(
        int(record["evaluation"]["best_shared_time"]) == 15
        for record in records
    )
    criteria = {
        "endpoint_is_best_deterministic_time": objective_endpoint_best,
        "mean_endpoint_probability_at_least_0_80": (
            summary["endpoint_probability"]["mean"] >= 0.80
        ),
        "mean_endpoint_gap_at_least_0_50": (
            summary["endpoint_probability_gap"]["mean"] >= 0.50
        ),
        "at_least_80_percent_positive_gap_starts": (
            sum(
                record["evaluation"]["endpoint_probability_gap"] > 0
                for record in records
            )
            >= int(np.ceil(0.8 * len(records)))
        ),
    }
    runs = [
        {
            "seed": record["seed"],
            **{metric: float(record["evaluation"][metric]) for metric in METRICS},
            "highest_probability_time": int(
                np.argmax(record["evaluation"]["encoding_time_probabilities"])
            ),
            "highest_time_probability": float(
                max(record["evaluation"]["encoding_time_probabilities"])
            ),
            "best_shared_time": int(record["evaluation"]["best_shared_time"]),
            "encoding_time_probabilities": record["evaluation"][
                "encoding_time_probabilities"
            ],
            "runtime_seconds": {
                "training": record["training"]["runtime_seconds"],
                "evaluation": record["evaluation_runtime_seconds"],
            },
            "git_sha": record["provenance"]["git_sha"],
        }
        for record in records
    ]
    result: dict[str, Any] = {
        "task": "Lu-Hasson-Norman 2022 fixed-duration condition mixture",
        "dataset": f"controlled synthetic generator ({config['task']['profile']})",
        "metrics": list(METRICS),
        "seeds": seeds,
        "experiment": experiment,
        "configuration": str(config_path),
        "configuration_sha256": config_hash,
        "mixture_weights": records[0]["mixture_weights"],
        "provenance": {
            "mode": "measured",
            "source": str(input_directory / f"{experiment}_seed<seed>.json"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_shas": sorted(
                {record["provenance"]["git_sha"] for record in records}
            ),
            "note": "Measured synthetic simulations; exact reward objective audit.",
        },
        "summary": summary,
        "condition_summary": condition_summary,
        "training_curve": training_curve,
        "runs": runs,
        "objective_audit": {
            "endpoint_is_best_deterministic_time": objective_endpoint_best,
            "interpretation": (
                "This tests whether the endpoint is present on the reward surface "
                "separately from whether gradient optimization reaches it."
            ),
        },
        "success_audit": {
            "criteria": criteria,
            "passed": all(criteria.values()),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = aggregate_temporal_mixture_config(
        arguments.config,
        input_directory=arguments.input_directory,
        output_path=arguments.output,
    )
    print(json.dumps(result["success_audit"], indent=2))


if __name__ == "__main__":
    main()
