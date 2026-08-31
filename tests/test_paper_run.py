import json
from pathlib import Path

import torch
import yaml

from boundary_em.aggregate_paper_policy import aggregate_paper_policy
from boundary_em.paper_task import PaperTaskConfig
from boundary_em.run_paper_policy import evaluate_paper_model, run_paper_policy_config
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


def _structured_model() -> StructuredEpisodicPredictionModel:
    return StructuredEpisodicPredictionModel(
        n_features=16,
        n_values=4,
        policy_hidden_dim=8,
        retrieval_temperature=0.05,
        temporal_context_scale=2.0,
    )


def test_reported_exact_task_config_keeps_all_ten_declared_seeds() -> None:
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (repository / "configs/paper_task_encoding/reported_failure.yaml").read_text()
    )

    assert config["task"]["profile"] == "released_code"
    assert len(set(config["experiment"]["model_seeds"])) == 10
    assert config["training"]["conditions"] == ["RM", "DM", "NM", "NM"]
    assert config["experiment"]["status"] == "diagnostic_replication"


def test_checkpoint_evaluation_is_deterministic_and_does_not_change_model() -> None:
    model = _structured_model()
    model.train()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    random_state_before = torch.random.get_rng_state().clone()

    first = evaluate_paper_model(
        model,
        PaperTaskConfig(),
        conditions=("DM",),
        trial_seed_start=91_000,
        action_seed_start=92_000,
        n_trials=2,
        memory_capacity=2,
    )
    second = evaluate_paper_model(
        model,
        PaperTaskConfig(),
        conditions=("DM",),
        trial_seed_start=91_000,
        action_seed_start=92_000,
        n_trials=2,
        memory_capacity=2,
    )

    assert first == second
    assert model.training is True
    assert torch.equal(torch.random.get_rng_state(), random_state_before)
    assert all(
        torch.equal(parameters_before[name], parameter)
        for name, parameter in model.named_parameters()
    )


def test_smoke_config_writes_reproducible_seed_result(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    result = run_paper_policy_config(
        repository / "configs/paper_task_encoding/smoke.yaml",
        seed=0,
        output_directory=tmp_path,
        updates_override=1,
        batch_size_override=1,
        evaluation_trials_override=2,
    )
    output_path = tmp_path / "exact_paper_task_smoke_seed0.json"

    assert output_path.exists()
    assert json.loads(output_path.read_text())["seed"] == 0
    assert result["configuration"]["task"]["profile"] == "released_code"
    assert set(result["evaluation"]) == {"RM", "DM", "NM"}
    assert set(result["evaluation"]["DM"]["ablations"]) == {
        "target_memory_removed",
        "lure_memory_removed",
        "content_gate_off",
    }
    assert "matched_random_one" in result["evaluation"]["DM"]["forced"]
    assert result["provenance"]["mode"] == "measured"
    assert len(result["training"]["free_policy_checkpoints"]) == 1
    checkpoint = result["training"]["free_policy_checkpoints"][0]
    assert checkpoint["epoch"] == 1 / 256
    assert checkpoint["update"] == 1
    assert checkpoint["sequences_processed"] == 1
    assert checkpoint["training"]["completed_updates"] == 1
    assert set(checkpoint["evaluation"]) == {"DM"}
    assert checkpoint["evaluation_runtime_seconds"] >= 0
    checkpoint_files = result["training"]["checkpoint_files"]
    assert checkpoint_files == ["checkpoints/exact_paper_task_smoke_seed0_update1.pt"]
    saved = torch.load(tmp_path / checkpoint_files[0], weights_only=True)
    assert saved["update"] == 1
    assert saved["model_state_dict"]


def test_reported_aggregation_retains_failed_selectivity_audit(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    aggregate = aggregate_paper_policy(
        repository / "configs/paper_task_encoding/reported_failure.yaml",
        input_directory=repository / "outputs/paper_task_encoding/reported",
        output_path=tmp_path / "summary.json",
    )

    assert len(aggregate["model_seeds"]) == 10
    assert aggregate["success_audit"]["checks"]["endpoint_selectivity"] is False
    assert aggregate["success_audit"]["checks"]["dm_prediction_benefit"] is True
    assert aggregate["success_audit"]["all_criteria_passed"] is False
