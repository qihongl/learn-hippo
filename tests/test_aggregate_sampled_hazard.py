import hashlib
import json
from pathlib import Path

import pytest
import yaml

from boundary_em.aggregate_sampled_hazard import aggregate_sampled_hazard_config


def _write_record(
    directory: Path,
    *,
    experiment: str,
    seed: int,
    config_hash: str,
) -> None:
    evaluation = {
        "trials": 8,
        "learned_expected_reward": 0.72 + seed / 10_000,
        "endpoint_probability": 0.92,
        "a1_endpoint_probability": 0.91,
        "b1_endpoint_probability": 0.93,
        "mean_nonendpoint_probability": 0.02,
        "never_probability": 0.03,
        "endpoint_probability_gap": 0.90,
        "endpoint_pair_reward": 0.73,
        "never_pair_reward": 0.50,
        "matched_random_one_reward": 0.55,
        "target_memory_removed_reward": 0.50,
        "distractor_memory_removed_reward": 0.74,
    }
    checkpoints = [
        {
            "epoch": float(epoch),
            "training": {"learning_rate": 0.001},
            "evaluation": evaluation,
            "evaluation_runtime_seconds": 0.01,
        }
        for epoch in (10, 20, 30, 40, 50)
    ]
    record = {
        "seed": seed,
        "configuration_sha256": config_hash,
        "scientific_guardrails": {
            "future_relevance_input_used": False,
            "boundary_or_time_input_used": False,
            "new_training_mapping_each_sequence": True,
        },
        "training": {
            "free_policy_checkpoints": checkpoints,
            "forced_value_runtime_seconds": 1.0,
            "free_policy_runtime_seconds": 2.0,
        },
        "evaluation": evaluation,
        "evaluation_bank_generation_runtime_seconds": 0.5,
        "provenance": {"git_sha": "abc123"},
    }
    path = directory / f"{experiment}_seed{seed}.json"
    path.write_text(json.dumps(record))


def test_sampled_hazard_aggregation_retains_all_seeds_and_curves(
    tmp_path: Path,
) -> None:
    experiment = "replication"
    config = {
        "experiment": {"name": experiment, "model_seeds": [1, 2, 3]},
        "task": {"profile": "released_code"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for seed in config["experiment"]["model_seeds"]:
        _write_record(
            tmp_path,
            experiment=experiment,
            seed=seed,
            config_hash=config_hash,
        )

    output_path = tmp_path / "summary.json"
    result = aggregate_sampled_hazard_config(
        config_path,
        input_directory=tmp_path,
        output_path=output_path,
        bootstrap_samples=1_000,
    )

    assert result["seeds"] == [1, 2, 3]
    assert result["summary"]["endpoint_probability"]["n_seeds"] == 3
    assert len(result["learning_curves"]) == 5
    assert len(result["runs"]) == 3
    assert result["runs"][0]["checkpoint_trajectory"][0] == {
        "epoch": 10.0,
        "learning_rate": 0.001,
        "learned_expected_reward": pytest.approx(0.7201),
        "endpoint_probability": 0.92,
        "endpoint_probability_gap": 0.90,
    }
    assert result["success_audit"]["passed"] is True
    assert json.loads(output_path.read_text()) == result


def test_sampled_hazard_aggregation_rejects_a_configuration_mismatch(
    tmp_path: Path,
) -> None:
    config = {
        "experiment": {"name": "replication", "model_seeds": [7]},
        "task": {"profile": "released_code"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    _write_record(
        tmp_path,
        experiment="replication",
        seed=7,
        config_hash="wrong",
    )

    with pytest.raises(ValueError, match="configuration hash"):
        aggregate_sampled_hazard_config(
            config_path,
            input_directory=tmp_path,
            output_path=tmp_path / "summary.json",
        )


def test_sampled_hazard_aggregation_reports_late_checkpoint_collapse(
    tmp_path: Path,
) -> None:
    experiment = "late-collapse"
    config = {
        "experiment": {"name": experiment, "model_seeds": [4]},
        "task": {"profile": "released_code"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    _write_record(
        tmp_path,
        experiment=experiment,
        seed=4,
        config_hash=config_hash,
    )
    record_path = tmp_path / f"{experiment}_seed4.json"
    record = json.loads(record_path.read_text())
    endpoint_probabilities = [0.90, 0.95, 0.40, 0.80, 0.90]
    for checkpoint, epoch, endpoint_probability in zip(
        record["training"]["free_policy_checkpoints"],
        (190, 200, 210, 220, 230),
        endpoint_probabilities,
        strict=True,
    ):
        checkpoint["epoch"] = float(epoch)
        checkpoint["evaluation"] = {
            **checkpoint["evaluation"],
            "endpoint_probability": endpoint_probability,
        }
    record_path.write_text(json.dumps(record))

    result = aggregate_sampled_hazard_config(
        config_path,
        input_directory=tmp_path,
        output_path=tmp_path / "summary.json",
        bootstrap_samples=1_000,
    )

    assert result["runs"][0]["largest_post_epoch_200_endpoint_drop"] == pytest.approx(
        0.55
    )
