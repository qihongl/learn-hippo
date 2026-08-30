import torch

from boundary_em.policy import WriteActorCritic
from boundary_em.task import TaskConfig, generate_episode
from boundary_em.training import actor_critic_loss, evaluate_actions, sample_rollout


def test_write_actor_critic_emits_valid_per_state_outputs():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episode = generate_episode(task_config, seed=41)
    model = WriteActorCritic(
        input_dim=episode.policy_inputs.shape[1],
        hidden_dim=16,
    )

    output = model(episode.policy_inputs)

    assert output.probabilities.shape == (task_config.n_features,)
    assert output.values.shape == (task_config.n_features,)
    assert torch.all((output.probabilities > 0) & (output.probabilities < 1))
    (output.probabilities.sum() + output.values.sum()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_policy_rollout_uses_only_selected_episodic_traces():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episode = generate_episode(task_config, seed=43)
    endpoint = torch.tensor([False, False, False, True])
    always = torch.ones(4, dtype=torch.bool)

    endpoint_outcome = evaluate_actions(episode, endpoint, temperature=0.1, capacity=4)
    always_outcome = evaluate_actions(episode, always, temperature=0.1, capacity=4)

    assert endpoint_outcome.reward.item() == 1.0
    assert always_outcome.reward.item() < 0.4
    assert endpoint_outcome.attention.shape == (1,)
    assert always_outcome.attention.shape == (4,)


def test_stochastic_rollout_is_reproducible_from_its_generator():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episode = generate_episode(task_config, seed=47)
    torch.manual_seed(5)
    model = WriteActorCritic(input_dim=14, hidden_dim=16)

    first = sample_rollout(
        model,
        episode,
        generator=torch.Generator().manual_seed(91),
        temperature=0.1,
        capacity=4,
    )
    second = sample_rollout(
        model,
        episode,
        generator=torch.Generator().manual_seed(91),
        temperature=0.1,
        capacity=4,
    )

    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.log_probabilities, second.log_probabilities)
    assert first.reward.item() == second.reward.item()


def test_actor_critic_loss_updates_the_write_policy():
    task_config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    torch.manual_seed(7)
    model = WriteActorCritic(input_dim=14, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    rollouts = [
        sample_rollout(
            model,
            generate_episode(task_config, seed=seed),
            generator=torch.Generator().manual_seed(1000 + seed),
            temperature=0.1,
            capacity=4,
        )
        for seed in range(16)
    ]
    before = model.actor.weight.detach().clone()

    loss = actor_critic_loss(rollouts, critic_coefficient=0.5, entropy_coefficient=0.01)
    optimizer.zero_grad()
    loss.total.backward()
    optimizer.step()

    assert torch.isfinite(loss.total)
    assert 0 <= loss.mean_reward <= 1
    assert not torch.equal(before, model.actor.weight.detach())
