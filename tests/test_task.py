from dataclasses import fields

import torch

from boundary_em.task import TaskConfig, generate_episode, sample_null_steps


def test_episode_generation_is_deterministic_complete_and_boundary_blind():
    config = TaskConfig(n_features=4, cue_dim=6, query_features=2)

    first = generate_episode(config, seed=17, null_steps=())
    second = generate_episode(config, seed=17, null_steps=())

    assert torch.equal(first.cue, second.cue)
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.reveal_order, second.reveal_order)
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.masks, second.masks)
    assert set(first.features.tolist()) <= {-1.0, 1.0}
    assert torch.equal(first.masks[-1], torch.ones(4))
    assert torch.all(first.masks[:-1].sum(dim=1) < 4)
    assert "is_boundary" not in {field.name for field in fields(first)}
    assert first.policy_inputs.shape[-1] == 2 * config.n_features + config.cue_dim


def test_null_steps_vary_duration_without_changing_semantic_completion():
    config = TaskConfig(n_features=4, cue_dim=6, query_features=2)

    episode = generate_episode(config, seed=23, null_steps=(1, 3))

    assert episode.states.shape[0] == 6
    assert torch.equal(episode.states[0], episode.states[1])
    assert torch.equal(episode.states[3], episode.states[4])
    assert episode.masks[-1].sum().item() == 4
    assert episode.query_mask.sum().item() == 2


def test_null_step_sampling_is_seeded_and_never_extends_past_completion():
    config = TaskConfig(n_features=4, cue_dim=6, query_features=2)

    first = sample_null_steps(config, seed=31, n_null_steps=3)
    second = sample_null_steps(config, seed=31, n_null_steps=3)

    assert first == second
    assert len(first) == 3
    assert all(0 <= progress < config.n_features for progress in first)
