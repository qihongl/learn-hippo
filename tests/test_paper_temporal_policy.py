import torch

from boundary_em.paper_rollout import (
    expected_prediction_reward,
    forced_schedule,
    rollout_trial,
)
from boundary_em.paper_task import PaperTaskConfig, PaperTrial, generate_trial
from boundary_em.paper_temporal_policy import (
    TemporalHazardPolicy,
    counterfactual_reward_matrix,
    expected_counterfactual_reward,
    hazard_encoding_distribution,
    train_temporal_policy,
)
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


def _model() -> StructuredEpisodicPredictionModel:
    return StructuredEpisodicPredictionModel(
        n_features=16,
        n_values=4,
        policy_hidden_dim=8,
        retrieval_temperature=0.05,
        temporal_context_scale=2.0,
    )


def _reward(
    model: StructuredEpisodicPredictionModel,
    trial: PaperTrial,
) -> torch.Tensor:
    computation = rollout_trial(
        model,
        trial,
        a1_encoding_actions=forced_schedule(trial.a1, "endpoint_only"),
        b1_encoding_actions=forced_schedule(trial.b1, "endpoint_only"),
        memory_capacity=2,
        retrieval_enabled=True,
    )
    valid = trial.b2.valid_predictions
    return expected_prediction_reward(
        computation.b2.probabilities[valid],
        trial.b2.targets[valid],
        trial.b2.inputs[valid, -1],
    ).mean()


def test_hazards_define_literal_online_first_encoding_distribution() -> None:
    probabilities = hazard_encoding_distribution(torch.zeros(3))

    torch.testing.assert_close(
        probabilities,
        torch.tensor([0.5, 0.25, 0.125, 0.125]),
    )
    assert probabilities.sum() == 1


def test_counterfactual_objective_pushes_probability_toward_rewarded_endpoint() -> None:
    policy = TemporalHazardPolicy(n_steps=3)
    rewards = torch.zeros(4, 4)
    rewards[2, 2] = 1.0

    expected_reward = expected_counterfactual_reward(policy.logits, rewards)
    expected_reward.backward()

    assert policy.logits.grad is not None
    assert policy.logits.grad[2] > 0
    assert policy.logits.grad[:2].max() < 0


def test_temporal_policy_learns_endpoint_from_reward_without_target() -> None:
    policy = TemporalHazardPolicy(n_steps=3, initialization="uniform_time")
    rewards = torch.zeros(4, 4)
    rewards[2, 2] = 1.0

    result = train_temporal_policy(
        policy,
        rewards,
        updates=200,
        learning_rate=0.1,
    )

    assert policy()[2] > 0.95
    assert result.history[-1]["expected_reward"] > 0.9
    assert len(result.history) == 200


def test_random_temporal_initialization_is_seeded_and_not_time_uniform() -> None:
    torch.manual_seed(17)
    first = TemporalHazardPolicy(n_steps=4, initialization="random")
    torch.manual_seed(17)
    second = TemporalHazardPolicy(n_steps=4, initialization="random")

    torch.testing.assert_close(first.logits, second.logits)
    assert first.logits.std() > 0


def test_real_counterfactual_matrix_matches_forced_endpoint_rollout() -> None:
    model = _model()
    trial = generate_trial(
        PaperTaskConfig(),
        seed=901,
        condition="DM",
        evaluation=True,
    )

    rewards = counterfactual_reward_matrix(model, trial, memory_capacity=2)

    assert rewards.shape == (17, 17)
    torch.testing.assert_close(rewards[15, 15], _reward(model, trial))
    assert torch.isfinite(rewards).all()
