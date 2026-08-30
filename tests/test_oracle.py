import pytest

from boundary_em.oracle import enumerate_schedules, evaluate_schedule
from boundary_em.task import TaskConfig, generate_episode


@pytest.mark.parametrize("capacity_mode", ["fixed", "historical"])
def test_endpoint_only_is_the_unique_optimal_schedule(capacity_mode):
    config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episodes = [generate_episode(config, seed=seed) for seed in range(32)]
    results = enumerate_schedules(
        episodes,
        temperature=0.1,
        capacity_mode=capacity_mode,
        fixed_capacity=4,
    )
    endpoint = (False, False, False, True)
    always = (True, True, True, True)
    never = (False, False, False, False)

    assert len(results) == 16
    assert results[0].schedule == endpoint
    assert results[0].mean_reward == pytest.approx(1.0)
    assert results[0].mean_reward > results[1].mean_reward
    assert evaluate_schedule(
        episodes, always, temperature=0.1, capacity_mode=capacity_mode
    ).mean_reward < 0.4
    assert evaluate_schedule(
        episodes, never, temperature=0.1, capacity_mode=capacity_mode
    ).mean_reward == pytest.approx(0.0)


def test_matched_random_one_write_has_a_literal_expected_reward():
    config = TaskConfig(n_features=4, cue_dim=6, query_features=2)
    episodes = [generate_episode(config, seed=seed) for seed in range(8)]
    one_write_rewards = []
    for write_position in range(config.n_features):
        schedule = tuple(index == write_position for index in range(config.n_features))
        result = evaluate_schedule(episodes, schedule, temperature=0.1)
        one_write_rewards.append(result.mean_reward)

    assert sum(one_write_rewards) / len(one_write_rewards) == pytest.approx(0.375)
