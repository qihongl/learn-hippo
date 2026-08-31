"""Compare the predeclared full-mixture optimizer-stability cells."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def run_passed_stability_criteria(run: dict[str, Any]) -> bool:
    """Return whether one seed passes every frozen stability criterion."""

    learned = float(run["learned_expected_reward"])
    never = float(run["never_pair_reward"])
    benefit = learned - never
    target_remaining = (
        (float(run["target_memory_removed_reward"]) - never) / benefit
        if benefit > 0
        else float("inf")
    )
    return (
        float(run["endpoint_probability"]) >= 0.80
        and float(run["endpoint_probability_gap"]) >= 0.50
        and float(run["a1_endpoint_probability"]) >= 0.80
        and float(run["b1_endpoint_probability"]) >= 0.80
        and bool(run["last_five_checkpoints_meet_selectivity"])
        and learned > never
        and learned > float(run["matched_random_one_reward"])
        and target_remaining <= 0.20
        and float(run["distractor_memory_removed_reward"]) > never
    )


def compare_optimizer_stability_cells(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen all-seed screen and deterministic selection rule."""

    if not summaries:
        raise ValueError("at least one optimizer-stability summary is required")
    cells = []
    for summary in summaries:
        free_policy = summary["configuration_values"]["training"]["free_policy"]
        runs = summary["runs"]
        endpoint_probabilities = [
            float(run["endpoint_probability"]) for run in runs
        ]
        cells.append(
            {
                "name": summary["experiment"],
                "factors": {
                    "batch_size": int(free_policy["batch_size"]),
                    "learning_rate_schedule": free_policy[
                        "learning_rate_schedule"
                    ],
                },
                "seeds": list(summary["seeds"]),
                "screen_passed": all(
                    run_passed_stability_criteria(run) for run in runs
                ),
                "minimum_final_endpoint_probability": min(
                    endpoint_probabilities
                ),
                "maximum_post_epoch_200_endpoint_drop": max(
                    float(run["largest_post_epoch_200_endpoint_drop"])
                    for run in runs
                ),
                "runs": runs,
            }
        )

    passing = [cell for cell in cells if cell["screen_passed"]]
    selected_cell = None
    if passing:
        best_minimum = max(
            cell["minimum_final_endpoint_probability"] for cell in passing
        )
        tied = [
            cell
            for cell in passing
            if best_minimum - cell["minimum_final_endpoint_probability"] < 0.02
        ]
        selected_cell = min(
            tied,
            key=lambda cell: (
                cell["factors"]["batch_size"] != 16,
                cell["factors"]["learning_rate_schedule"]
                != "cosine_second_half",
                cell["name"],
            ),
        )["name"]

    git_shas = sorted(
        {
            git_sha
            for summary in summaries
            for git_sha in summary["provenance"]["git_shas"]
        }
    )
    return {
        "task": "Lu-Hasson-Norman 2022 event-prediction generator",
        "dataset": "controlled synthetic generator (released_code)",
        "metrics": [
            "endpoint_probability",
            "endpoint_probability_gap",
            "largest_post_epoch_200_endpoint_drop",
            "learned_expected_reward",
        ],
        "seeds": list(summaries[0]["seeds"]),
        "provenance": {
            "mode": "measured",
            "source": "versioned per-cell sampled-hazard summaries",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_shas": git_shas,
            "note": "Measured synthetic simulations; no human data.",
        },
        "summary": {
            cell["name"]: {
                "screen_passed": cell["screen_passed"],
                "minimum_final_endpoint_probability": cell[
                    "minimum_final_endpoint_probability"
                ],
                "maximum_post_epoch_200_endpoint_drop": cell[
                    "maximum_post_epoch_200_endpoint_drop"
                ],
            }
            for cell in cells
        },
        "cells": cells,
        "selected_cell": selected_cell,
        "selection_rule": (
            "All three seeds must pass; maximize the minimum final endpoint "
            "probability, treat differences below 0.02 as ties, then prefer "
            "batch 16 and cosine_second_half."
        ),
    }


def compare_optimizer_stability_files(
    summary_paths: list[str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load per-cell summaries, attach their YAML, and write one comparison."""

    summaries = []
    for summary_path in summary_paths:
        summary = json.loads(Path(summary_path).read_text())
        summary["configuration_values"] = yaml.safe_load(
            Path(summary["configuration"]).read_text()
        )
        summaries.append(summary)
    comparison = compare_optimizer_stability_cells(summaries)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = compare_optimizer_stability_files(
        arguments.summaries,
        output_path=arguments.output,
    )
    print(json.dumps({"selected_cell": result["selected_cell"]}, indent=2))


if __name__ == "__main__":
    main()
