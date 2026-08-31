import hashlib
import json
from pathlib import Path

import yaml

from boundary_em.aggregate_temporal_mixture import aggregate_temporal_mixture_config


def test_temporal_mixture_aggregation_retains_random_starts(tmp_path: Path) -> None:
    config = {
        "experiment": {"name": "mixture", "model_seeds": [1, 2]},
        "task": {"profile": "released_code"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for seed, endpoint in ((1, 0.01), (2, 0.02)):
        evaluation = {
            "learned_expected_reward": 0.65,
            "endpoint_probability": endpoint,
            "mean_nonendpoint_probability": 0.06,
            "never_probability": 0.01,
            "endpoint_probability_gap": endpoint - 0.06,
            "endpoint_pair_reward": 0.67,
            "never_pair_reward": 0.61,
            "matched_random_one_reward": 0.63,
            "best_shared_time": 15,
            "encoding_time_probabilities": [0.0] * 13
            + [0.98, 0.0, endpoint, 0.01 - endpoint / 2],
        }
        record = {
            "seed": seed,
            "configuration_sha256": config_hash,
            "mixture_weights": {"RM": 0.25, "DM": 0.25, "NM": 0.5},
            "training": {
                "history": [
                    {
                        "completed_updates": update,
                        "endpoint_probability": endpoint,
                        "expected_reward": 0.65,
                    }
                    for update in (1, 10, 100)
                ],
                "runtime_seconds": 1.0,
            },
            "evaluation": evaluation,
            "evaluation_by_condition": {
                condition: {
                    "learned_expected_reward": 0.65,
                    "endpoint_pair_reward": 0.67,
                    "never_pair_reward": 0.61,
                }
                for condition in ("RM", "DM", "NM")
            },
            "evaluation_runtime_seconds": 0.5,
            "provenance": {"git_sha": "abc123"},
        }
        (tmp_path / f"mixture_seed{seed}.json").write_text(json.dumps(record))

    output_path = tmp_path / "summary.json"
    result = aggregate_temporal_mixture_config(
        config_path,
        input_directory=tmp_path,
        output_path=output_path,
    )

    assert result["seeds"] == [1, 2]
    assert result["summary"]["endpoint_probability"]["n_seeds"] == 2
    assert len(result["training_curve"]) == 3
    assert result["objective_audit"]["endpoint_is_best_deterministic_time"] is True
    assert result["success_audit"]["passed"] is False
    assert json.loads(output_path.read_text()) == result
