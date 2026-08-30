from pathlib import Path

import yaml


def test_confirmatory_config_has_fifteen_new_model_seeds_and_frozen_criteria():
    repository = Path(__file__).parents[1]
    smoke = yaml.safe_load(
        (repository / "configs/learned_encoding/smoke.yaml").read_text()
    )
    reported = yaml.safe_load(
        (repository / "configs/learned_encoding/reported.yaml").read_text()
    )

    smoke_seeds = set(smoke["experiment"]["model_seeds"])
    reported_seeds = set(reported["experiment"]["model_seeds"])
    assert len(reported_seeds) == 15
    assert smoke_seeds.isdisjoint(reported_seeds)
    assert reported["experiment"]["status"] == "confirmatory"
    assert reported["success_criteria"]["minimum_gap_closure"] == 0.8
    assert reported["evaluation"]["split_label"] == "confirmatory_test"
