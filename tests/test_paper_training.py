import math

import torch

from boundary_em.paper_model import EpisodicPredictionModel
from boundary_em.paper_task import PaperTaskConfig, generate_trial
from boundary_em.paper_training import (
    PredictionStageConfig,
    prediction_objective,
    train_prediction_stage,
)


def _model() -> EpisodicPredictionModel:
    return EpisodicPredictionModel(
        input_dim=37,
        output_dim=5,
        hidden_dim=24,
        decision_dim=20,
        retrieval_temperature=0.1,
    )


def test_prediction_objective_is_finite_on_an_exact_trial() -> None:
    trial = generate_trial(
        PaperTaskConfig(), seed=101, condition="RM", evaluation=True
    )
    objective = prediction_objective(
        _model(),
        trial,
        retrieval_enabled=False,
        forced_encoding="never",
        memory_capacity=40,
    )

    assert objective.loss.ndim == 0
    assert torch.isfinite(objective.loss)
    assert 0 <= objective.accuracy <= 1
    assert math.isfinite(objective.expected_reward)


def test_supervised_stage_reduces_loss_on_a_repeated_trial() -> None:
    model = _model()
    trial = generate_trial(
        PaperTaskConfig(), seed=202, condition="RM", evaluation=True
    )
    before = prediction_objective(
        model,
        trial,
        retrieval_enabled=False,
        forced_encoding="never",
        memory_capacity=40,
    ).loss.item()
    result = train_prediction_stage(
        model,
        PaperTaskConfig(),
        PredictionStageConfig(
            updates=30,
            batch_size=1,
            learning_rate=0.01,
            gradient_clip=1.0,
            retrieval_enabled=False,
            forced_encoding="never",
            conditions=("RM",),
        ),
        seed=4,
        repeated_trial=trial,
    )
    after = prediction_objective(
        model,
        trial,
        retrieval_enabled=False,
        forced_encoding="never",
        memory_capacity=40,
    ).loss.item()

    assert len(result.history) == 30
    assert after < before * 0.6
