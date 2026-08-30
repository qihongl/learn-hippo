import torch
import yaml

from boundary_em.policy import WriteActorCritic
from boundary_em.run_mechanism import run_mechanism_config


def test_mechanism_runner_uses_checkpoints_and_writes_secondary_results(tmp_path):
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    model = WriteActorCritic(input_dim=14, hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_seed": 7,
            "git_sha": "test",
        },
        checkpoint_directory / "test_policy_seed7.pt",
    )
    config = {
        "experiment": {
            "name": "test_mechanism",
            "status": "secondary_exploratory",
            "model_seeds": [7],
        },
        "task": {"n_features": 4, "cue_dim": 6, "query_features": 2},
        "memory": {"temperature": 0.1, "capacity": 4},
        "model": {"hidden_dim": 8},
        "evaluation": {
            "episode_seed_start": 800,
            "n_episodes": 4,
            "action_seed_offset": 900,
        },
        "analysis": {
            "input_ablations": ["full", "mask_only"],
            "retrieval_temperatures": [0.1],
            "bootstrap_seed": 3,
            "bootstrap_samples": 100,
        },
        "output": {
            "checkpoint_directory": "unused",
            "checkpoint_name": "test_policy",
            "path": "unused.json",
        },
    }
    config_path = tmp_path / "mechanism.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_mechanism_config(
        config_path,
        checkpoint_directory=checkpoint_directory,
        output_path=tmp_path / "mechanism.json",
    )

    assert result["seeds"] == [7]
    assert result["provenance"]["mode"] == "measured"
    assert result["experiment_status"] == "secondary_exploratory"
    assert set(result["write_probability_by_progress"]) == {"1", "2", "3", "4"}
    assert result["retrieval_ablation"]["latest_always_write_reward"] == 1.0
    assert "input_mask_only" in result["summary"]
