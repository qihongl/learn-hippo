import json
from pathlib import Path

import yaml

from boundary_em.run_paper_temporal_mixture import run_temporal_mixture_config


def test_temporal_mixture_runner_preserves_conditions_and_weights(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            repository
            / "configs/paper_task_encoding/temporal_hazard_mixture_smoke.yaml"
        ).read_text()
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_temporal_mixture_config(
        config_path,
        seed=0,
        repository_root=repository,
        output_directory=tmp_path / "results",
    )

    assert result["mixture_weights"] == {"RM": 0.25, "DM": 0.25, "NM": 0.5}
    assert set(result["evaluation_by_condition"]) == {"RM", "DM", "NM"}
    assert len(result["training"]["history"]) == 2
    assert len(result["evaluation"]["encoding_time_probabilities"]) == 17
    assert result["scientific_guardrails"]["condition_input_used"] is False
    output_path = tmp_path / "results/temporal_hazard_mixture_smoke_seed0.json"
    assert json.loads(output_path.read_text()) == result
