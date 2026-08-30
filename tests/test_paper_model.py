import torch

from boundary_em.paper_model import EpisodicPredictionModel
from boundary_em.paper_task import PaperTaskConfig


def test_model_predicts_over_the_original_five_responses() -> None:
    config = PaperTaskConfig()
    model = EpisodicPredictionModel(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dim=32,
        decision_dim=24,
        retrieval_temperature=0.1,
    )
    state = model.initial_state()
    step = model.predict_step(torch.zeros(config.input_dim), state)

    assert step.logits.shape == (5,)
    assert step.state.hidden.shape == (32,)
    assert step.state.cell.shape == (32,)
    assert step.retrieval_attention.numel() == 0


def test_differentiable_retrieval_favors_a_matching_memory() -> None:
    model = EpisodicPredictionModel(
        input_dim=37,
        output_dim=5,
        hidden_dim=4,
        decision_dim=4,
        retrieval_temperature=0.05,
    )
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    memories = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        requires_grad=True,
    )
    retrieval = model.retrieve(query, memories)

    assert retrieval.attention[0] > 0.999
    assert retrieval.attention[1] < 0.001
    retrieval.value.sum().backward()
    assert query.grad is not None
    assert memories.grad is not None


def test_encoding_gate_requires_only_the_recurrent_situation_model() -> None:
    model = EpisodicPredictionModel(
        input_dim=37,
        output_dim=5,
        hidden_dim=8,
        decision_dim=8,
        retrieval_temperature=0.1,
    )
    output = model.encoding_policy(torch.zeros(8))

    assert output.probability.shape == ()
    assert 0 < output.probability < 1
    assert output.value.shape == ()
