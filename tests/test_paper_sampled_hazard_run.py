import json
from pathlib import Path

import yaml

from boundary_em.run_paper_sampled_hazard import run_sampled_hazard_config


def test_sampled_hazard_runner_records_forced_and_free_stages(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            repository / "configs/paper_task_encoding/sampled_hazard_smoke.yaml"
        ).read_text()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_sampled_hazard_config(
        config_path,
        seed=0,
        repository_root=repository,
        output_directory=tmp_path / "results",
    )

    assert len(result["training"]["forced_value_history"]) == 1
    assert len(result["training"]["free_policy_history"]) == 2
    assert [
        point["update"] for point in result["training"]["free_policy_checkpoints"]
    ] == [1, 2]
    assert result["evaluation"]["trials"] == 1
    assert set(result["evaluation_by_condition"]) == {"DM"}
    assert result["scientific_guardrails"]["new_training_mapping_each_sequence"]
    output_path = tmp_path / "results/sampled_hazard_smoke_seed0.json"
    assert json.loads(output_path.read_text()) == result
