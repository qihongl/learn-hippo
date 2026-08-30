from pathlib import Path

from boundary_em.aggregate import aggregate_reported


def test_confirmatory_aggregation_uses_all_declared_seeds_and_frozen_rules(tmp_path):
    repository = Path(__file__).parents[1]

    result = aggregate_reported(
        repository / "configs/learned_encoding/reported.yaml",
        input_directory=Path("outputs/learned_encoding/reported"),
        output_path=tmp_path / "summary.json",
    )

    assert result["seeds"] == list(range(100, 115))
    assert result["provenance"]["mode"] == "measured"
    assert result["provenance"]["data_kind"] == "synthetic"
    assert result["summary"]["learned_policy"]["reward"]["n_seeds"] == 15
    assert result["success_audit"]["positive_selectivity_seeds"] == 15
    assert result["success_audit"]["all_criteria_passed"] is True
    assert result["success_audit"]["selectivity_ci95"][0] > 0
    assert result["success_audit"]["displacement_loss_ci95"][0] > 0
