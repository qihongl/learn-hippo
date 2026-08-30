import torch

from boundary_em.paper_differentiable_encoding import (
    DifferentiableEncodingConfig,
    rollout_differentiable_encoding_trial,
    train_differentiable_encoding,
)
from boundary_em.paper_rollout import expected_prediction_reward
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


def test_delayed_prediction_reward_has_gradient_to_both_events_encoding() -> None:
    model = _model()
    trial = generate_trial(
        PaperTaskConfig(),
        seed=101,
        condition="DM",
        evaluation=True,
    )
    a1_strengths = torch.full((len(trial.a1.inputs),), 0.5, requires_grad=True)
    b1_strengths = torch.full((len(trial.b1.inputs),), 0.5, requires_grad=True)

    computation = rollout_differentiable_encoding_trial(
        model,
        trial,
        a1_encoding_strengths=a1_strengths,
        b1_encoding_strengths=b1_strengths,
        memory_capacity=40,
        retrieval_enabled=True,
    )
    valid = trial.b2.valid_predictions
    reward = expected_prediction_reward(
        computation.b2_probabilities[valid],
        trial.b2.targets[valid],
        trial.b2.inputs[valid, -1],
    ).mean()
    reward.backward()

    assert a1_strengths.grad is not None
    assert b1_strengths.grad is not None
    assert a1_strengths.grad.abs().sum() > 0
    assert b1_strengths.grad.abs().sum() > 0


def test_differentiable_training_changes_only_encoding_actor() -> None:
    model = _model()
    actor_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("encoding_actor")
    }
    other_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("encoding_actor")
    }
    config = DifferentiableEncodingConfig(
        updates=2,
        batch_size=2,
        learning_rate=0.01,
        gradient_clip=1.0,
        memory_capacity=40,
        conditions=("DM",),
    )

    result = train_differentiable_encoding(
        model,
        PaperTaskConfig(),
        config,
        seed=102,
    )

    assert len(result.history) == 2
    assert any(
        not torch.equal(actor_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in actor_before
    )
    assert all(
        torch.equal(other_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in other_before
    )
