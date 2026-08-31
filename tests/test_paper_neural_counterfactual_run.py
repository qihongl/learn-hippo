import json
from pathlib import Path

import yaml

from boundary_em.run_paper_neural_counterfactual import (
    run_neural_counterfactual_config,
)


def test_neural_counterfactual_runner_writes_reproducible_record(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            repository
            / "configs/paper_task_encoding/neural_counterfactual_smoke.yaml"
        ).read_text()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_neural_counterfactual_config(
        config_path,
        seed=0,
        repository_root=repository,
        output_directory=tmp_path / "results",
    )

    assert result["training"]["examples"] == 1
    assert len(result["training"]["curriculum_history"]) == 1
    assert len(result["training"]["curriculum_checkpoints"]) == 1
    assert len(result["training"]["history"]) == 2
    assert [point["update"] for point in result["training"]["checkpoints"]] == [1, 2]
    assert result["evaluation"]["trials"] == 1
    assert result["scientific_guardrails"]["dm_curriculum_used"] is True
    assert result["scientific_guardrails"]["endpoint_target_used"] is False
    output_path = tmp_path / "results/neural_counterfactual_smoke_seed0.json"
    assert json.loads(output_path.read_text()) == result
