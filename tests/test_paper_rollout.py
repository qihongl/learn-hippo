import torch

from boundary_em.paper_model import EpisodicPredictionModel
from boundary_em.paper_rollout import (
    expected_prediction_reward,
    forced_schedule,
    rollout_trial,
)
from boundary_em.paper_task import PaperTaskConfig, generate_trial


def _model() -> EpisodicPredictionModel:
    return EpisodicPredictionModel(
        input_dim=37,
        output_dim=5,
        hidden_dim=12,
        decision_dim=10,
        retrieval_temperature=0.1,
    )


def test_forced_schedules_use_original_midpoint_and_endpoint_times() -> None:
    event = generate_trial(
        PaperTaskConfig(), seed=3, condition="DM", evaluation=False
    ).b1

    assert forced_schedule(event, "endpoint_only").nonzero().flatten().tolist() == [
        event.boundary_index
    ]
    assert forced_schedule(event, "midpoint_only").nonzero().flatten().tolist() == [
        event.delay + 7
    ]
    assert forced_schedule(event, "midpoint_plus_endpoint").sum().item() == 2
    assert forced_schedule(event, "dense").all()
    assert not forced_schedule(event, "never").any()


def test_dm_rollout_retains_lure_and_target_endpoint_memories() -> None:
    trial = generate_trial(
        PaperTaskConfig(), seed=5, condition="DM", evaluation=True
    )
    rollout = rollout_trial(
        _model(),
        trial,
        b1_encoding_actions=forced_schedule(trial.b1, "endpoint_only"),
        memory_capacity=40,
        retrieval_enabled=True,
    )

    assert rollout.b1.encoding_indices == (trial.b1.boundary_index,)
    assert rollout.memory_labels == ("lure_a1", "target_b1")
    assert rollout.b2.retrieval_attention.shape == (16, 2)
    assert rollout.working_memory_was_reset is True


def test_nm_rollout_does_not_make_a_target_memory_available() -> None:
    trial = generate_trial(
        PaperTaskConfig(), seed=8, condition="NM", evaluation=True
    )
    rollout = rollout_trial(
        _model(),
        trial,
        b1_encoding_actions=forced_schedule(trial.b1, "endpoint_only"),
        memory_capacity=40,
        retrieval_enabled=True,
    )

    assert rollout.b1.encoding_indices == ()
    assert rollout.memory_labels == ("lure_a1",)
    assert rollout.b2.retrieval_attention.shape == (16, 1)
    assert rollout.working_memory_was_reset is True


def test_paper_text_nm_retains_an_unrelated_b1_trace_as_a_lure() -> None:
    trial = generate_trial(
        PaperTaskConfig(profile="paper_text"),
        seed=8,
        condition="NM",
        evaluation=True,
    )
    rollout = rollout_trial(
        _model(),
        trial,
        b1_encoding_actions=forced_schedule(trial.b1, "endpoint_only"),
        memory_capacity=40,
        retrieval_enabled=True,
    )

    assert trial.target_memory_available is False
    assert rollout.memory_labels == ("lure_a1", "unrelated_b1")


def test_expected_reward_respects_correct_error_and_dont_know_outcomes() -> None:
    probabilities = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
    targets = torch.tensor([0])
    penalties = torch.tensor([2.0])

    reward = expected_prediction_reward(probabilities, targets, penalties)
    assert torch.allclose(reward, torch.tensor([-1.0]))
