import json
from pathlib import Path

import yaml

from boundary_em.run_paper_temporal_policy import run_temporal_hazard_config


def test_temporal_hazard_runner_preserves_measured_audit(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (repository / "configs/paper_task_encoding/temporal_hazard_smoke.yaml")
        .read_text()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_temporal_hazard_config(
        config_path,
        seed=0,
        repository_root=repository,
        output_directory=tmp_path / "results",
    )

    assert result["training"]["trials"] == 1
    assert len(result["training"]["history"]) == 2
    assert len(result["training"]["mean_reward_matrix"]) == 17
    assert len(result["evaluation"]["encoding_time_probabilities"]) == 17
    assert result["scientific_guardrails"]["endpoint_target_used"] is False
    output_path = tmp_path / "results/temporal_hazard_smoke_seed0.json"
    assert json.loads(output_path.read_text()) == result
