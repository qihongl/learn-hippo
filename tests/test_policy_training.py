import torch

from boundary_em.evaluation import evaluate_policy
from boundary_em.policy import WriteActorCritic
from boundary_em.policy_training import PolicyTrainingConfig, train_policy
from boundary_em.task import TaskConfig


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
    )

    assert result.summary["writes_per_event"]["mean"] == 4.0
    assert result.summary["boundary_auc"]["mean"] == 0.5
    assert result.summary["reward"]["mean"] < 0.4
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, before[name])
