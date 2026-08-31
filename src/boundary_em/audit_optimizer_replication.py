"""Audit the locked ten-seed replication of the selected optimizer cell."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boundary_em.compare_optimizer_stability import (
    run_passed_stability_criteria,
)


def audit_optimizer_replication_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared aggregate and individual-seed replication rules."""

    seeds = list(summary["seeds"])
    runs = list(summary["runs"])
    seed_passes = [
        {
            "seed": int(run["seed"]),
            "passed": run_passed_stability_criteria(run),
        }
        for run in runs
    ]
    individual_seed_passes = sum(item["passed"] for item in seed_passes)
    criteria = {
        "exactly_ten_declared_seeds": (
            len(seeds) == 10
            and len(set(seeds)) == 10
            and sorted(seeds) == sorted(int(run["seed"]) for run in runs)
        ),
        "aggregate_success_audit_passed": bool(summary["success_audit"]["passed"]),
        "at_least_eight_individual_seed_passes": individual_seed_passes >= 8,
    }
    return {
        "experiment": summary["experiment"],
        "seeds": seeds,
        "individual_seed_passes": individual_seed_passes,
        "seed_passes": seed_passes,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "provenance": {
            "mode": "measured",
            "source": "versioned ten-seed sampled-hazard summary",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_shas": summary["provenance"]["git_shas"],
            "note": "Measured synthetic simulations; no human data.",
        },
    }


def audit_optimizer_replication_file(
    summary_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Load a replication summary and write its locked audit."""

    summary = json.loads(Path(summary_path).read_text())
    result = audit_optimizer_replication_summary(summary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = audit_optimizer_replication_file(
        arguments.summary,
        output_path=arguments.output,
    )
    print(json.dumps({"passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
