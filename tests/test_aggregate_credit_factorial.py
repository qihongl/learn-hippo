import hashlib
import json
from pathlib import Path

import yaml

from boundary_em.aggregate_credit_factorial import aggregate_credit_factorial


def _write_cell(
    directory: Path,
    *,
    name: str,
    probability: float,
    advantage: str,
    schedule: str,
    endpoint: float,
) -> Path:
    config = {
        "experiment": {"name": name, "model_seeds": [1, 2]},
        "task": {"profile": "released_code"},
        "policy": {"initial_probability": probability},
        "training": {
            "free_policy": {
                "advantage_mode": advantage,
                "condition_schedule": schedule,
            }
        },
    }
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    evaluation = {
        "learned_expected_reward": 0.70,
        "endpoint_probability": endpoint,
        "endpoint_probability_gap": endpoint - 0.02,
        "endpoint_pair_reward": 0.72,
        "never_pair_reward": 0.50,
        "matched_random_one_reward": 0.55,
    }
    for seed in (1, 2):
        record = {
            "seed": seed,
            "configuration_sha256": config_hash,
            "evaluation_by_condition": {"DM": evaluation},
            "training": {
                "free_policy_checkpoints": [
                    {
                        "epoch": 10.0,
                        "evaluation": {"by_condition": {"DM": evaluation}},
                    }
                ]
            },
            "provenance": {"git_sha": "abc123"},
        }
        (directory / f"{name}_seed{seed}.json").write_text(json.dumps(record))
    return path


def test_credit_factorial_aggregation_selects_by_declared_metrics(
    tmp_path: Path,
) -> None:
    weak = _write_cell(
        tmp_path,
        name="weak",
        probability=0.05,
        advantage="critic",
        schedule="fixed",
        endpoint=0.01,
    )
    strong = _write_cell(
        tmp_path,
        name="strong",
        probability=0.05,
        advantage="condition_centered",
        schedule="dm_to_mixture",
        endpoint=0.90,
    )

    output_path = tmp_path / "summary.json"
    result = aggregate_credit_factorial(
        [weak, strong],
        input_directory=tmp_path,
        output_path=output_path,
    )

    assert result["selected_cell"] == "strong"
    assert result["cells"][1]["screen_passed"] is True
    assert result["cells"][0]["screen_passed"] is False
    assert json.loads(output_path.read_text()) == result
