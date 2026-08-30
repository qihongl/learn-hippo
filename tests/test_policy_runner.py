import json
from pathlib import Path

from boundary_em.run_policy import run_policy_config


def test_policy_runner_persists_seed_level_provenance(tmp_path):
    repository = Path(__file__).parents[1]

    result = run_policy_config(
        repository / "configs/learned_encoding/smoke.yaml",
        seed=9,
        output_directory=tmp_path / "metrics",
        checkpoint_directory=tmp_path / "checkpoints",
        updates_override=1,
        batch_size_override=2,
        evaluation_episodes_override=4,
    )

    destination = tmp_path / "metrics/boundary_policy_smoke_seed9.json"
    loaded = json.loads(destination.read_text())
    assert result == loaded
    assert result["seed"] == 9
    assert result["provenance"]["mode"] == "measured"
    assert result["provenance"]["data_kind"] == "synthetic"
    assert result["experiment_status"] == "exploratory"
    assert result["summary"]["learned_policy"]["reward"]["n_seeds"] == 1
    assert len(result["training_curves"]["mean_reward"]) == 1
    assert result["evaluation"]["stochastic"]["reward"]["n_episodes"] == 4
    assert result["evaluation"]["ood_stochastic"]["reward"]["n_episodes"] == 4
    assert result["configuration"]["evaluation"]["ood_null_steps"] == 3
    assert result["evaluation"]["interventions"]["endpoint_only"]["reward"][
        "mean"
    ] == 1.0
    assert "displaced_learned" in result["evaluation_records"]["interventions"]
    assert (tmp_path / "checkpoints/boundary_policy_smoke_seed9.pt").exists()
