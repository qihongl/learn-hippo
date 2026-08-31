"""Aggregate complete sampled-hazard seed records without filtering seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FINAL_METRICS = (
    "learned_expected_reward",
    "endpoint_probability",
    "a1_endpoint_probability",
    "b1_endpoint_probability",
    "mean_nonendpoint_probability",
    "never_probability",
    "endpoint_probability_gap",
    "endpoint_pair_reward",
    "never_pair_reward",
    "matched_random_one_reward",
    "target_memory_removed_reward",
    "distractor_memory_removed_reward",
)


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "n_seeds": len(array),
    }


def _bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    generator: np.random.Generator,
) -> list[float]:
    array = np.asarray(values, dtype=float)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _evaluation_at_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    evaluation = checkpoint["evaluation"]
    if "by_condition" in evaluation:
        return evaluation["by_condition"]["DM"]
    return evaluation


def _safe_remaining_fraction(
    learned: float,
    removed: float,
    never: float,
) -> float:
    benefit = learned - never
    if benefit <= 0:
        return math.inf
    return (removed - never) / benefit


def _largest_endpoint_drop_after(
    checkpoints: list[dict[str, Any]],
    *,
    start_epoch: float,
) -> float:
    evaluations = [
        (float(checkpoint["epoch"]), _evaluation_at_checkpoint(checkpoint))
        for checkpoint in checkpoints
    ]
    drops = [
        float(previous[1]["endpoint_probability"])
        - float(current[1]["endpoint_probability"])
        for previous, current in zip(
            evaluations,
            evaluations[1:],
            strict=False,
        )
        if previous[0] >= start_epoch
    ]
    return max([0.0, *drops])


def aggregate_sampled_hazard_config(
    config_path: str | Path,
    *,
    input_directory: str | Path,
    output_path: str | Path,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    """Aggregate all seeds declared by one configuration into a compact record."""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = yaml.safe_load(config_bytes)
    experiment = config["experiment"]["name"]
    seeds = list(config["experiment"]["model_seeds"])
    input_directory = Path(input_directory)

    records = []
    for seed in seeds:
        record_path = input_directory / f"{experiment}_seed{seed}.json"
        if not record_path.exists():
            raise FileNotFoundError(f"missing declared seed record: {record_path}")
        record = json.loads(record_path.read_text())
        if record["seed"] != seed:
            raise ValueError(f"seed mismatch in {record_path}")
        if record["configuration_sha256"] != config_hash:
            raise ValueError(f"configuration hash mismatch in {record_path}")
        records.append(record)

    metric_values = {
        metric: [float(record["evaluation"][metric]) for record in records]
        for metric in FINAL_METRICS
    }
    summary = {
        metric: _summary(values) for metric, values in metric_values.items()
    }

    checkpoint_epochs = [
        float(checkpoint["epoch"])
        for checkpoint in records[0]["training"]["free_policy_checkpoints"]
    ]
    for record in records[1:]:
        record_epochs = [
            float(checkpoint["epoch"])
            for checkpoint in record["training"]["free_policy_checkpoints"]
        ]
        if record_epochs != checkpoint_epochs:
            raise ValueError("checkpoint epochs differ across seeds")
    learning_curves = []
    for checkpoint_index, epoch in enumerate(checkpoint_epochs):
        evaluations = [
            _evaluation_at_checkpoint(
                record["training"]["free_policy_checkpoints"][checkpoint_index]
            )
            for record in records
        ]
        learning_curves.append(
            {
                "epoch": epoch,
                "reward": _summary(
                    [
                        float(evaluation["learned_expected_reward"])
                        for evaluation in evaluations
                    ]
                ),
                "endpoint_probability": _summary(
                    [
                        float(evaluation["endpoint_probability"])
                        for evaluation in evaluations
                    ]
                ),
                "endpoint_probability_gap": _summary(
                    [
                        float(evaluation["endpoint_probability_gap"])
                        for evaluation in evaluations
                    ]
                ),
            }
        )

    learned_minus_never = [
        learned - never
        for learned, never in zip(
            metric_values["learned_expected_reward"],
            metric_values["never_pair_reward"],
            strict=True,
        )
    ]
    learned_minus_random = [
        learned - random_reward
        for learned, random_reward in zip(
            metric_values["learned_expected_reward"],
            metric_values["matched_random_one_reward"],
            strict=True,
        )
    ]
    target_remaining = [
        _safe_remaining_fraction(learned, target_removed, never)
        for learned, target_removed, never in zip(
            metric_values["learned_expected_reward"],
            metric_values["target_memory_removed_reward"],
            metric_values["never_pair_reward"],
            strict=True,
        )
    ]
    distractor_minus_never = [
        distractor_removed - never
        for distractor_removed, never in zip(
            metric_values["distractor_memory_removed_reward"],
            metric_values["never_pair_reward"],
            strict=True,
        )
    ]
    generator = np.random.default_rng(0)
    intervals = {
        "endpoint_probability_gap": _bootstrap_interval(
            metric_values["endpoint_probability_gap"],
            samples=bootstrap_samples,
            generator=generator,
        ),
        "learned_minus_never_reward": _bootstrap_interval(
            learned_minus_never,
            samples=bootstrap_samples,
            generator=generator,
        ),
        "learned_minus_matched_random_reward": _bootstrap_interval(
            learned_minus_random,
            samples=bootstrap_samples,
            generator=generator,
        ),
        "distractor_removed_minus_never_reward": _bootstrap_interval(
            distractor_minus_never,
            samples=bootstrap_samples,
            generator=generator,
        ),
    }
    required_positive_seeds = math.ceil(0.8 * len(seeds))
    criteria = {
        "mean_endpoint_probability_at_least_0_80": (
            summary["endpoint_probability"]["mean"] >= 0.80
        ),
        "mean_endpoint_gap_at_least_0_50": (
            summary["endpoint_probability_gap"]["mean"] >= 0.50
        ),
        "endpoint_gap_interval_above_zero": (
            intervals["endpoint_probability_gap"][0] > 0
        ),
        "at_least_80_percent_positive_gap_seeds": (
            sum(value > 0 for value in metric_values["endpoint_probability_gap"])
            >= required_positive_seeds
        ),
        "reward_above_never_interval": (
            intervals["learned_minus_never_reward"][0] > 0
        ),
        "reward_above_matched_random_interval": (
            intervals["learned_minus_matched_random_reward"][0] > 0
        ),
        "retrieval_off_removes_at_least_80_percent_of_benefit": all(
            benefit > 0 for benefit in learned_minus_never
        ),
        "target_memory_removal_leaves_at_most_20_percent_of_benefit": (
            all(value <= 0.20 for value in target_remaining)
        ),
        "distractor_memory_removal_preserves_benefit": (
            intervals["distractor_removed_minus_never_reward"][0] > 0
        ),
        "endpoint_preference_in_both_events": (
            summary["a1_endpoint_probability"]["mean"] >= 0.80
            and summary["b1_endpoint_probability"]["mean"] >= 0.80
        ),
        "unseen_mappings_and_no_forbidden_policy_inputs": all(
            record["scientific_guardrails"][
                "new_training_mapping_each_sequence"
            ]
            and not record["scientific_guardrails"][
                "future_relevance_input_used"
            ]
            and not record["scientific_guardrails"][
                "boundary_or_time_input_used"
            ]
            for record in records
        ),
    }

    runs = []
    for record in records:
        evaluation = record["evaluation"]
        checkpoints = record["training"]["free_policy_checkpoints"]
        final_five = [
            _evaluation_at_checkpoint(checkpoint) for checkpoint in checkpoints[-5:]
        ]
        runs.append(
            {
                "seed": record["seed"],
                **{
                    metric: float(evaluation[metric]) for metric in FINAL_METRICS
                },
                "last_five_checkpoints_meet_selectivity": all(
                    point["endpoint_probability"] >= 0.80
                    and point["endpoint_probability_gap"] >= 0.50
                    for point in final_five
                ),
                "largest_post_epoch_200_endpoint_drop": (
                    _largest_endpoint_drop_after(
                        checkpoints,
                        start_epoch=200.0,
                    )
                ),
                "checkpoint_trajectory": [
                    {
                        "epoch": float(checkpoint["epoch"]),
                        "learning_rate": (
                            float(checkpoint["training"]["learning_rate"])
                            if "learning_rate" in checkpoint.get("training", {})
                            else None
                        ),
                        **{
                            metric: float(
                                _evaluation_at_checkpoint(checkpoint)[metric]
                            )
                            for metric in (
                                "learned_expected_reward",
                                "endpoint_probability",
                                "endpoint_probability_gap",
                            )
                        },
                    }
                    for checkpoint in checkpoints
                ],
                "runtime_seconds": {
                    "evaluation_bank_generation": record[
                        "evaluation_bank_generation_runtime_seconds"
                    ],
                    "forced_value_training": record["training"][
                        "forced_value_runtime_seconds"
                    ],
                    "free_policy_training": record["training"][
                        "free_policy_runtime_seconds"
                    ],
                },
                "git_sha": record["provenance"]["git_sha"],
            }
        )

    result: dict[str, Any] = {
        "task": "Lu-Hasson-Norman 2022 event-prediction generator",
        "dataset": f"controlled synthetic generator ({config['task']['profile']})",
        "metrics": list(FINAL_METRICS),
        "seeds": seeds,
        "experiment": experiment,
        "configuration": str(config_path),
        "configuration_sha256": config_hash,
        "provenance": {
            "mode": "measured",
            "source": str(input_directory / f"{experiment}_seed<seed>.json"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_shas": sorted(
                {record["provenance"]["git_sha"] for record in records}
            ),
            "note": "Measured synthetic simulations; no human data.",
        },
        "summary": summary,
        "paired_differences": {
            "learned_minus_never_reward": _summary(learned_minus_never),
            "learned_minus_matched_random_reward": _summary(learned_minus_random),
            "target_memory_remaining_fraction": _summary(target_remaining),
            "distractor_removed_minus_never_reward": _summary(
                distractor_minus_never
            ),
        },
        "bootstrap_95_percent_intervals": intervals,
        "learning_curves": learning_curves,
        "runs": runs,
        "success_audit": {
            "criteria": criteria,
            "passed": all(criteria.values()),
            "retrieval_off_definition": (
                "With the structured model, disabling retrieval makes encoding "
                "actions causally inert and is exactly the never-pair reward."
            ),
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
    result = aggregate_sampled_hazard_config(
        arguments.config,
        input_directory=arguments.input_directory,
        output_path=arguments.output,
    )
    print(json.dumps(result["success_audit"], indent=2))


if __name__ == "__main__":
    main()
