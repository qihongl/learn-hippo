import numpy as np
import torch

from boundary_em.paper_task import (
    PaperTaskConfig,
    generate_trial,
    sample_delay,
    sample_removed_features,
)


def test_released_code_sampling_matches_archived_helpers() -> None:
    removal_rng = np.random.RandomState(123)
    removed = sample_removed_features(
        n_features=16,
        probability=0.3,
        mode="released_count",
        rng=removal_rng,
    )
    assert torch.nonzero(removed).flatten().tolist() == [4, 9, 13]

    delay_rng = np.random.RandomState(7)
    delays = [sample_delay(max_delay=4, rng=delay_rng) for _ in range(8)]
    assert delays == [4, 1, 3, 3, 4, 1, 0, 1]


def test_paper_text_removal_uses_independent_bernoulli_draws() -> None:
    removed = sample_removed_features(
        n_features=16,
        probability=0.3,
        mode="independent",
        rng=np.random.RandomState(7),
    )
    assert torch.nonzero(removed).flatten().tolist() == [0, 7, 8, 13, 14]


def test_event_inputs_match_the_published_stimulus_representation() -> None:
    config = PaperTaskConfig(profile="released_code")
    trial = generate_trial(config, seed=11, condition="DM", evaluation=True)
    event = trial.b1

    assert event.delay == 0
    assert event.inputs.shape == (16, 37)
    assert event.targets.shape == (16,)
    assert event.boundary_index == 15
    assert torch.equal(event.targets, event.situation)

    observed_keys = event.inputs[:, :16]
    observed_values = event.inputs[:, 16:20]
    query_keys = event.inputs[:, 20:36]
    penalties = event.inputs[:, 36]

    assert torch.equal(query_keys, torch.eye(16))
    assert torch.equal(observed_keys.argmax(dim=1), event.observation_order)
    assert torch.equal(
        observed_values.argmax(dim=1),
        event.situation[event.observation_order],
    )
    assert torch.equal(penalties, torch.full((16,), 2.0))
    assert not event.removed_observations.any()


def test_memory_conditions_create_the_declared_situation_relationships() -> None:
    config = PaperTaskConfig(profile="released_code")
    for condition in ("RM", "DM"):
        trial = generate_trial(config, seed=21, condition=condition, evaluation=True)
        assert torch.equal(trial.b1.situation, trial.b2.situation)
        assert not torch.equal(trial.a1.situation, trial.b1.situation)
        assert trial.reset_working_memory_before_b2 is (condition == "DM")
        assert trial.target_memory_available is True

    no_memory = generate_trial(config, seed=21, condition="NM", evaluation=True)
    assert not torch.equal(no_memory.a1.situation, no_memory.b1.situation)
    assert not torch.equal(no_memory.b1.situation, no_memory.b2.situation)
    assert not torch.equal(no_memory.a1.situation, no_memory.b2.situation)
    assert no_memory.reset_working_memory_before_b2 is True
    assert no_memory.target_memory_available is False


def test_trial_generation_is_deterministic_without_adding_a_boundary_input() -> None:
    config = PaperTaskConfig(profile="paper_text")
    first = generate_trial(config, seed=99, condition="DM", evaluation=False)
    second = generate_trial(config, seed=99, condition="DM", evaluation=False)

    assert torch.equal(first.b1.inputs, second.b1.inputs)
    assert torch.equal(first.b2.inputs, second.b2.inputs)
    assert first.b1.delay == second.b1.delay
    assert first.b1.inputs.shape[1] == config.input_dim == 37
    assert config.max_delay == 3
