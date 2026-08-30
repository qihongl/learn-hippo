import torch

from boundary_em.evaluation import (
    ablate_policy_inputs,
    evaluate_policy,
    write_probabilities_by_progress,
)
from boundary_em.policy import WriteActorCritic
from boundary_em.policy_training import PolicyTrainingConfig, train_policy
from boundary_em.task import TaskConfig, generate_episode


def test_policy_training_is_reproducible_for_a_fixed_seed():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    train_config = PolicyTrainingConfig(
        hidden_dim=12,
        updates=2,
        batch_size=4,
        learning_rate=0.01,
        critic_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_clip=1.0,
        temperature=0.1,
        capacity=4,
    )

    first = train_policy(task_config, train_config, seed=3)
    second = train_policy(task_config, train_config, seed=3)

    assert first.history == second.history
    for name, parameter in first.model.state_dict().items():
        assert torch.equal(parameter, second.model.state_dict()[name])
    assert len(first.history) == train_config.updates
    assert all(0 <= point["mean_reward"] <= 1 for point in first.history)


def test_evaluation_freezes_weights_and_reports_policy_behavior():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    model = WriteActorCritic(input_dim=14, hidden_dim=8)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.fill_(20)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    result = evaluate_policy(
        model,
        task_config,
        episode_seeds=range(100, 108),
        action_seed=500,
        temperature=0.1,
        capacity=4,
        stochastic=False,
        n_null_steps=3,
    )

    assert result.summary["writes_per_event"]["mean"] == 7.0
    assert result.summary["boundary_auc"]["mean"] == 0.5
    assert result.summary["reward"]["mean"] < 0.4
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, before[name])


def test_forced_policy_interventions_recover_or_destroy_boundary_information():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    model = WriteActorCritic(input_dim=14, hidden_dim=8)
    seeds = range(200, 208)

    endpoint = evaluate_policy(
        model,
        task_config,
        episode_seeds=seeds,
        action_seed=600,
        temperature=0.1,
        capacity=4,
        stochastic=False,
        intervention="endpoint_only",
    )
    midpoint = evaluate_policy(
        model,
        task_config,
        episode_seeds=seeds,
        action_seed=600,
        temperature=0.1,
        capacity=4,
        stochastic=False,
        intervention="midpoint_only",
    )

    assert endpoint.summary["reward"]["mean"] == 1.0
    assert midpoint.summary["reward"]["mean"] == 0.0


def test_policy_input_ablations_preserve_only_the_declared_signal():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episode = generate_episode(task_config, seed=211)

    mask_only = ablate_policy_inputs(
        episode.policy_inputs, task_config, mode="mask_only"
    )
    state_only = ablate_policy_inputs(
        episode.policy_inputs, task_config, mode="state_only"
    )
    cue_only = ablate_policy_inputs(
        episode.policy_inputs, task_config, mode="cue_only"
    )

    cue_end = task_config.cue_dim
    state_end = cue_end + task_config.n_features
    assert torch.count_nonzero(mask_only[:, :state_end]) == 0
    assert torch.equal(mask_only[:, state_end:], episode.masks)
    assert torch.equal(state_only[:, cue_end:state_end], episode.states)
    assert torch.count_nonzero(state_only[:, :cue_end]) == 0
    assert torch.count_nonzero(state_only[:, state_end:]) == 0
    assert torch.equal(cue_only[:, :cue_end], episode.policy_inputs[:, :cue_end])
    assert torch.count_nonzero(cue_only[:, cue_end:]) == 0


def test_write_probability_trace_is_grouped_by_semantic_progress():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    model = WriteActorCritic(input_dim=14, hidden_dim=8)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()

    trace = write_probabilities_by_progress(
        model,
        task_config,
        episode_seeds=range(300, 308),
        n_null_steps=0,
        input_ablation="full",
    )

    assert set(trace) == {1, 2, 3, 4}
    assert all(len(values) == 8 for values in trace.values())
    assert all(value == 0.5 for values in trace.values() for value in values)
