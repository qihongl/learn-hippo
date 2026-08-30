import json
from pathlib import Path

import yaml

from boundary_em.aggregate_paper_policy import aggregate_paper_policy
from boundary_em.run_paper_policy import run_paper_policy_config


def test_reported_exact_task_config_keeps_all_ten_declared_seeds() -> None:
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (repository / "configs/paper_task_encoding/reported_failure.yaml").read_text()
    )

    assert config["task"]["profile"] == "released_code"
    assert len(set(config["experiment"]["model_seeds"])) == 10
    assert config["training"]["conditions"] == ["RM", "DM", "NM", "NM"]
    assert config["experiment"]["status"] == "diagnostic_replication"


def test_smoke_config_writes_reproducible_seed_result(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    result = run_paper_policy_config(
        repository / "configs/paper_task_encoding/smoke.yaml",
        seed=0,
        output_directory=tmp_path,
        updates_override=1,
        batch_size_override=1,
        evaluation_trials_override=2,
    )
    output_path = tmp_path / "exact_paper_task_smoke_seed0.json"

    assert output_path.exists()
    assert json.loads(output_path.read_text())["seed"] == 0
    assert result["configuration"]["task"]["profile"] == "released_code"
    assert set(result["evaluation"]) == {"RM", "DM", "NM"}
    assert result["provenance"]["mode"] == "measured"


def test_reported_aggregation_retains_failed_selectivity_audit(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    aggregate = aggregate_paper_policy(
        repository / "configs/paper_task_encoding/reported_failure.yaml",
        input_directory=repository / "outputs/paper_task_encoding/reported",
        output_path=tmp_path / "summary.json",
    )

    assert len(aggregate["model_seeds"]) == 10
    assert aggregate["success_audit"]["checks"]["endpoint_selectivity"] is False
    assert aggregate["success_audit"]["checks"]["dm_prediction_benefit"] is True
    assert aggregate["success_audit"]["all_criteria_passed"] is False
