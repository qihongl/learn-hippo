import json
from pathlib import Path

from boundary_em.run_oracle import run_oracle_config


def test_oracle_runner_writes_auditable_measured_results(tmp_path):
    repository = Path(__file__).parents[1]
    output = tmp_path / "oracle.json"

    result = run_oracle_config(
        repository / "configs/learned_encoding/oracle.yaml",
        output_path=output,
        n_episodes_override=8,
    )

    loaded = json.loads(output.read_text())
    assert result == loaded
    assert {"task", "dataset", "metrics", "seeds", "provenance", "summary"} <= set(
        result
    )
    assert result["provenance"]["mode"] == "measured"
    assert result["provenance"]["data_kind"] == "synthetic"
    assert result["oracle_precondition"]["fixed"]["passed"] is True
    assert result["oracle_precondition"]["historical"]["passed"] is True
    assert result["summary"]["endpoint_only"]["reward"]["mean"] == 1.0
    assert len(result["schedule_rankings"]["fixed"]) == 16
