import torch

from boundary_em.paper_model import EpisodicPredictionModel
from boundary_em.paper_policy_training import (
    EncodingStageConfig,
    train_encoding_stage,
)
from boundary_em.paper_task import PaperTaskConfig


def _model() -> EpisodicPredictionModel:
    return EpisodicPredictionModel(
        input_dim=37,
        output_dim=5,
        hidden_dim=16,
        decision_dim=12,
        retrieval_temperature=0.1,
    )


def _parameters(model: EpisodicPredictionModel, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def _stage() -> EncodingStageConfig:
    return EncodingStageConfig(
        updates=2,
        batch_size=2,
        learning_rate=0.01,
        critic_coefficient=0.5,
        entropy_coefficient=0.005,
        gradient_clip=1.0,
        memory_capacity=40,
        conditions=("DM",),
    )


def test_forced_exploration_trains_value_estimator_without_changing_actor() -> None:
    model = _model()
    actor_before = _parameters(model, "encoding_actor")
    critic_before = _parameters(model, "encoding_critic")
    prediction_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("encoding_")
    }

    result = train_encoding_stage(
        model,
        PaperTaskConfig(),
        _stage(),
        seed=6,
        forced_exploration=True,
    )

    assert len(result.history) == 2
    assert all(
        torch.equal(actor_before[name], parameter)
        for name, parameter in _parameters(model, "encoding_actor").items()
    )
    assert any(
        not torch.equal(critic_before[name], parameter)
        for name, parameter in _parameters(model, "encoding_critic").items()
    )
    assert all(
        torch.equal(prediction_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in prediction_before
    )


def test_free_encoding_stage_updates_actor_but_keeps_prediction_system_frozen() -> None:
    model = _model()
    actor_before = _parameters(model, "encoding_actor")
    prediction_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("encoding_")
    }

    result = train_encoding_stage(
        model,
        PaperTaskConfig(),
        _stage(),
        seed=9,
        forced_exploration=False,
    )

    assert len(result.history) == 2
    assert any(
        not torch.equal(actor_before[name], parameter)
        for name, parameter in _parameters(model, "encoding_actor").items()
    )
    assert all(
        torch.equal(prediction_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in prediction_before
    )
