from pathlib import Path

import yaml

REPOSITORY = Path(__file__).parents[1]
CONFIG_DIRECTORY = REPOSITORY / "configs/paper_task_encoding"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIRECTORY / name).read_text())


def test_dm_replication_configs_use_disjoint_ten_seed_namespaces() -> None:
    fixed = _load("sampled_hazard_dm_fixed_replication.yaml")
    variable = _load("sampled_hazard_dm_variable_replication.yaml")
    fixed_seeds = fixed["experiment"]["model_seeds"]
    variable_seeds = variable["experiment"]["model_seeds"]

    assert fixed_seeds == list(range(910, 920))
    assert variable_seeds == list(range(920, 930))
    assert set(fixed_seeds).isdisjoint(variable_seeds)


def test_dm_replications_preserve_the_declared_training_budget() -> None:
    for name in (
        "sampled_hazard_dm_fixed_replication.yaml",
        "sampled_hazard_dm_variable_replication.yaml",
    ):
        config = _load(name)

        assert config["task"]["training_conditions"] == ["DM"]
        assert config["training"]["forced_value"]["updates"] == 400
        assert config["training"]["free_policy"]["updates"] == 6_400
        assert config["training"]["free_policy"]["batch_size"] == 16
        assert config["checkpoint_evaluation"]["interval_updates"] == 160
        assert config["checkpoint_evaluation"]["trials"] == 128


def test_fixed_and_variable_replications_differ_only_in_task_sampling_mode() -> None:
    fixed = _load("sampled_hazard_dm_fixed_replication.yaml")
    variable = _load("sampled_hazard_dm_variable_replication.yaml")

    assert fixed["task"]["training_evaluation_mode"] is True
    assert fixed["checkpoint_evaluation"]["evaluation_mode"] is True
    assert variable["task"]["training_evaluation_mode"] is False
    assert variable["checkpoint_evaluation"]["evaluation_mode"] is False


def test_della_manifest_matches_every_declared_replication_seed() -> None:
    manifest_path = REPOSITORY / "scripts/della/paper_task_encoding_array.tsv"
    lines = manifest_path.read_text().strip().splitlines()
    assert lines[0].split("\t") == ["experiment", "config", "seed"]
    records = [line.split("\t") for line in lines[1:]]

    expected = []
    for name in (
        "sampled_hazard_dm_fixed_replication.yaml",
        "sampled_hazard_dm_variable_replication.yaml",
    ):
        config = _load(name)
        config_path = f"configs/paper_task_encoding/{name}"
        expected.extend(
            [config["experiment"]["name"], config_path, str(seed)]
            for seed in config["experiment"]["model_seeds"]
        )

    assert records == expected


def test_temporal_mixture_audit_uses_five_new_random_initializations() -> None:
    config = _load("temporal_hazard_full_mixture.yaml")

    assert config["experiment"]["model_seeds"] == list(range(930, 935))
    assert config["task"]["conditions"] == ["RM", "DM", "NM", "NM"]
    assert config["task"]["evaluation_mode"] is True
    assert config["optimization"]["initialization"] == "random"
