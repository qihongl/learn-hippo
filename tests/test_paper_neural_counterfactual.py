import torch

from boundary_em.paper_neural_counterfactual import (
    NeuralCounterfactualExample,
    build_counterfactual_example,
    evaluate_neural_counterfactual,
    initialize_neural_hazard_policy,
    neural_encoding_distributions,
    neural_expected_reward,
    train_neural_counterfactual,
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


def test_neural_policy_uses_online_states_to_define_valid_distributions() -> None:
    model = _model()
    initialize_neural_hazard_policy(model, initial_probability=0.05)
    trial = generate_trial(
        PaperTaskConfig(),
        seed=81_001,
        condition="DM",
        evaluation=True,
    )
    example = build_counterfactual_example(model, trial, memory_capacity=2)

    a1, b1 = neural_encoding_distributions(model, example)

    assert example.a1_states.shape == (16, 80)
    assert example.b1_states.shape == (16, 80)
    assert a1.shape == b1.shape == (17,)
    torch.testing.assert_close(a1.sum(), torch.tensor(1.0))
    torch.testing.assert_close(b1.sum(), torch.tensor(1.0))
    torch.testing.assert_close(a1, b1, atol=0.03, rtol=0)


def test_exact_delayed_reward_has_gradient_to_shared_neural_actor() -> None:
    model = _model()
    initialize_neural_hazard_policy(model, initial_probability=0.05)
    trial = generate_trial(
        PaperTaskConfig(),
        seed=81_002,
        condition="DM",
        evaluation=True,
    )
    example = build_counterfactual_example(model, trial, memory_capacity=2)

    expected_reward = neural_expected_reward(model, example)
    expected_reward.backward()

    actor_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("encoding_actor")
    ]
    assert all(gradient is not None for gradient in actor_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in actor_gradients) > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("encoding_actor")
    )


def test_counterfactual_evaluation_supports_unequal_event_delays() -> None:
    model = _model()
    trial = generate_trial(
        PaperTaskConfig(),
        seed=81_004,
        condition="DM",
        evaluation=False,
    )
    assert len(trial.a1.inputs) != len(trial.b1.inputs)
    example = build_counterfactual_example(model, trial, memory_capacity=2)

    evaluation = evaluate_neural_counterfactual(model, [example])

    assert example.reward_matrix.shape == (
        len(trial.a1.inputs) + 1,
        len(trial.b1.inputs) + 1,
    )
    assert evaluation["trials"] == 1
    assert set(evaluation["encoding_time_probabilities_by_event_length"]) == {
        str(len(trial.a1.inputs)),
        str(len(trial.b1.inputs)),
    }

    second_trial = generate_trial(
        PaperTaskConfig(),
        seed=81_005,
        condition="NM",
        evaluation=False,
    )
    second_example = build_counterfactual_example(
        model,
        second_trial,
        memory_capacity=2,
    )
    result = train_neural_counterfactual(
        model,
        [example, second_example],
        updates=1,
        batch_size=2,
        learning_rate=0.001,
        gradient_clip=1.0,
        seed=81_006,
    )
    assert result.history[0]["sequences_processed"] == 2


def test_neural_policy_learns_observation_defined_endpoint_from_reward() -> None:
    model = _model()
    initialize_neural_hazard_policy(model, initial_probability=0.05)
    states = torch.stack(
        [
            torch.zeros(80),
            torch.cat([torch.ones(20), torch.zeros(60)]),
            torch.ones(80),
        ]
    )
    rewards = torch.zeros(4, 4)
    rewards[2, 2] = 1.0
    example = NeuralCounterfactualExample(states, states.clone(), rewards)

    result = train_neural_counterfactual(
        model,
        [example],
        updates=400,
        batch_size=1,
        learning_rate=0.01,
        gradient_clip=10.0,
        seed=91,
        checkpoint_interval=200,
        checkpoint_evaluator=lambda _update, current_model: (
            evaluate_neural_counterfactual(current_model, [example])
        ),
    )
    a1, b1 = neural_encoding_distributions(model, example)

    assert a1[2] > 0.95
    assert b1[2] > 0.95
    assert result.history[-1]["expected_reward"] > 0.9
    assert [point["update"] for point in result.checkpoints] == [200, 400]

    model.train()
    random_state = torch.random.get_rng_state().clone()
    first = evaluate_neural_counterfactual(model, [example])
    second = evaluate_neural_counterfactual(model, [example])
    assert first == second
    assert model.training is True
    assert torch.equal(torch.random.get_rng_state(), random_state)
