import torch

from boundary_em.paper_sampled_hazard import (
    SampledHazardStageConfig,
    sample_neural_hazard_episode,
    train_sampled_hazard_stage,
)
from boundary_em.paper_task import PaperTaskConfig, generate_trial
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


def _model() -> StructuredEpisodicPredictionModel:
    return StructuredEpisodicPredictionModel(
        n_features=16,
        n_values=4,
        policy_hidden_dim=16,
        retrieval_temperature=0.05,
        temporal_context_scale=2.0,
    )


def _stage() -> SampledHazardStageConfig:
    return SampledHazardStageConfig(
        updates=2,
        batch_size=2,
        learning_rate=0.01,
        critic_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_clip=10.0,
        memory_capacity=2,
        conditions=("DM",),
        evaluation_mode=True,
    )


def test_sampled_policy_makes_at_most_one_online_encoding_per_event() -> None:
    model = _model()
    trial = generate_trial(
        PaperTaskConfig(),
        seed=92_001,
        condition="DM",
        evaluation=True,
    )
    episode = sample_neural_hazard_episode(
        model,
        trial,
        action_generator=torch.Generator().manual_seed(92_002),
        memory_capacity=2,
        forced_exploration=True,
    )

    assert episode.a1_actions.sum() <= 1
    assert episode.b1_actions.sum() <= 1
    assert episode.a1_distribution.shape == (17,)
    assert episode.b1_distribution.shape == (17,)
    torch.testing.assert_close(episode.a1_distribution.sum(), torch.tensor(1.0))
    torch.testing.assert_close(episode.b1_distribution.sum(), torch.tensor(1.0))
    assert len(episode.log_probabilities) == len(episode.values)


def test_forced_value_stage_freezes_actor_and_free_stage_updates_it() -> None:
    model = _model()
    actor_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("encoding_actor")
    }

    forced = train_sampled_hazard_stage(
        model,
        PaperTaskConfig(),
        _stage(),
        seed=610,
        forced_exploration=True,
    )
    assert all(
        torch.equal(actor_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("encoding_actor")
    )
    assert forced.history[-1]["sequences_processed"] == 4

    train_sampled_hazard_stage(
        model,
        PaperTaskConfig(),
        _stage(),
        seed=611,
        forced_exploration=False,
    )
    assert any(
        not torch.equal(actor_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("encoding_actor")
    )


def test_training_history_supports_mixed_variable_event_lengths() -> None:
    stage = SampledHazardStageConfig(
        **{
            **_stage().__dict__,
            "updates": 1,
            "batch_size": 4,
            "conditions": ("RM", "DM", "NM", "NM"),
            "evaluation_mode": False,
        }
    )

    result = train_sampled_hazard_stage(
        _model(),
        PaperTaskConfig(),
        stage,
        seed=710,
        forced_exploration=True,
    )

    assert result.history[0]["sequences_processed"] == 4
    assert 0 <= result.history[0]["endpoint_probability"] <= 1
