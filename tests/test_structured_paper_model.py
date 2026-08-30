import torch

from boundary_em.paper_model import RecurrentState
from boundary_em.paper_rollout import (
    expected_prediction_reward,
    forced_schedule,
    rollout_trial,
)
from boundary_em.paper_task import PaperTaskConfig, generate_trial
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


def _model() -> StructuredEpisodicPredictionModel:
    return StructuredEpisodicPredictionModel(
        n_features=16,
        n_values=4,
        policy_hidden_dim=32,
        retrieval_temperature=0.05,
        temporal_context_scale=2.0,
    )


def test_observed_feature_value_drives_the_matching_next_state_prediction() -> None:
    model = _model()
    model_input = torch.zeros(37)
    model_input[2] = 1
    model_input[16 + 3] = 1
    model_input[20 + 2] = 1
    model_input[-1] = 2

    step = model.predict_step(model_input, model.initial_state())
    assert step.logits.argmax().item() == 3


def test_retrieval_gate_uses_content_match_without_a_condition_label() -> None:
    model = _model()
    partial = torch.zeros(model.hidden_dim)
    partial[: model.situation_dim].reshape(16, 4)[:8, 0] = 1
    matching = torch.zeros(model.hidden_dim)
    matching[: model.situation_dim].reshape(16, 4)[:, 0] = 1
    unrelated = torch.zeros(model.hidden_dim)
    unrelated[: model.situation_dim].reshape(16, 4)[:, 1] = 1
    state = RecurrentState(hidden=partial, cell=partial)

    matching_step = model.predict_step(
        torch.zeros(model.input_dim),
        state,
        memories=matching.unsqueeze(0),
        retrieval_enabled=True,
    )
    unrelated_step = model.predict_step(
        torch.zeros(model.input_dim),
        state,
        memories=unrelated.unsqueeze(0),
        retrieval_enabled=True,
    )

    assert matching_step.retrieval_gate > 0.19
    assert unrelated_step.retrieval_gate < 0.01


def test_forced_endpoint_memory_improves_dm_prediction_over_no_encoding() -> None:
    model = _model()
    task_config = PaperTaskConfig()
    endpoint_rewards = []
    no_memory_rewards = []
    for seed in range(12):
        trial = generate_trial(
            task_config,
            seed=70_000 + seed,
            condition="DM",
            evaluation=True,
        )
        for schedule, collector in (
            ("endpoint_only", endpoint_rewards),
            ("never", no_memory_rewards),
        ):
            computation = rollout_trial(
                model,
                trial,
                a1_encoding_actions=forced_schedule(trial.a1, schedule),
                b1_encoding_actions=forced_schedule(trial.b1, schedule),
                memory_capacity=40,
                retrieval_enabled=True,
            )
            valid = trial.b2.valid_predictions
            collector.append(
                expected_prediction_reward(
                    computation.b2.probabilities[valid],
                    trial.b2.targets[valid],
                    trial.b2.inputs[valid, -1],
                ).mean()
            )

    assert torch.stack(endpoint_rewards).mean() > torch.stack(no_memory_rewards).mean()


def test_forced_endpoint_schedule_beats_alternative_fixed_schedules() -> None:
    model = _model()
    task_config = PaperTaskConfig()
    schedules = ("endpoint_only", "midpoint_only", "midpoint_plus_endpoint", "dense")
    mean_rewards: dict[str, torch.Tensor] = {}

    for schedule in schedules:
        rewards = []
        for seed in range(12):
            trial = generate_trial(
                task_config,
                seed=80_000 + seed,
                condition="DM",
                evaluation=True,
            )
            computation = rollout_trial(
                model,
                trial,
                a1_encoding_actions=forced_schedule(trial.a1, schedule),
                b1_encoding_actions=forced_schedule(trial.b1, schedule),
                memory_capacity=40,
                retrieval_enabled=True,
            )
            valid = trial.b2.valid_predictions
            rewards.append(
                expected_prediction_reward(
                    computation.b2.probabilities[valid],
                    trial.b2.targets[valid],
                    trial.b2.inputs[valid, -1],
                ).mean()
            )
        mean_rewards[schedule] = torch.stack(rewards).mean()

    endpoint_reward = mean_rewards["endpoint_only"]
    assert all(
        endpoint_reward > mean_rewards[schedule]
        for schedule in schedules
        if schedule != "endpoint_only"
    )
