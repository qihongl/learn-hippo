import torch

from boundary_em.paper_model import EpisodicPredictionModel
from boundary_em.paper_policy_training import (
    EncodingStageConfig,
    sample_encoding_episode,
    train_encoding_stage,
)
from boundary_em.paper_task import PaperTaskConfig, generate_trial


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


def test_shared_policy_controls_encoding_in_distractor_and_target_events() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.encoding_actor_encoder.parameters():
            parameter.zero_()
        model.encoding_actor.weight.zero_()
        model.encoding_actor.bias.fill_(20.0)
    trial = generate_trial(
        PaperTaskConfig(),
        seed=91,
        condition="DM",
        evaluation=True,
    )

    episode = sample_encoding_episode(
        model,
        trial,
        action_generator=torch.Generator().manual_seed(92),
        memory_capacity=40,
        forced_exploration=False,
    )

    assert episode.a1_actions.all()
    assert episode.b1_actions.all()
    assert episode.memory_labels.count("lure_a1") == len(episode.a1_actions)
    assert episode.memory_labels.count("target_b1") == len(episode.b1_actions)


def test_checkpoint_callbacks_record_final_state_without_changing_training() -> None:
    reference = _model()
    checkpointed = _model()
    checkpointed.load_state_dict(reference.state_dict())
    calls: list[int] = []

    reference_result = train_encoding_stage(
        reference,
        PaperTaskConfig(),
        EncodingStageConfig(
            **{**_stage().__dict__, "updates": 3},
        ),
        seed=27,
        forced_exploration=False,
    )

    def evaluate(
        update: int,
        model: EpisodicPredictionModel,
    ) -> dict[str, float]:
        calls.append(update)
        model.eval()
        return {"random_probe": float(torch.rand(()).item())}

    checkpointed_result = train_encoding_stage(
        checkpointed,
        PaperTaskConfig(),
        EncodingStageConfig(
            **{**_stage().__dict__, "updates": 3},
        ),
        seed=27,
        forced_exploration=False,
        checkpoint_interval=2,
        checkpoint_evaluator=evaluate,
    )

    assert calls == [2, 3]
    assert [point["update"] for point in checkpointed_result.checkpoints] == [2, 3]
    assert [point["epoch"] for point in checkpointed_result.checkpoints] == [
        4 / 256,
        6 / 256,
    ]
    assert [
        point["sequences_processed"] for point in checkpointed_result.checkpoints
    ] == [4, 6]
    assert [
        point["training"]["completed_updates"]
        for point in checkpointed_result.checkpoints
    ] == [2, 3]
    assert all(
        point["evaluation_runtime_seconds"] >= 0
        for point in checkpointed_result.checkpoints
    )
    assert checkpointed_result.history == reference_result.history
    assert [point["completed_updates"] for point in reference_result.history] == [
        1,
        2,
        3,
    ]
    assert [point["sequences_processed"] for point in reference_result.history] == [
        2,
        4,
        6,
    ]
    assert all(point["learning_rate"] == 0.01 for point in reference_result.history)
    assert checkpointed.training is True
    assert all(
        torch.equal(reference.state_dict()[name], parameter)
        for name, parameter in checkpointed.state_dict().items()
    )
