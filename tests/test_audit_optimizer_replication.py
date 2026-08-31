from boundary_em.audit_optimizer_replication import (
    audit_optimizer_replication_summary,
)


def _run(seed: int, *, passed: bool) -> dict:
    return {
        "seed": seed,
        "endpoint_probability": 0.99 if passed else 0.79,
        "endpoint_probability_gap": 0.98 if passed else 0.49,
        "a1_endpoint_probability": 0.99 if passed else 0.79,
        "b1_endpoint_probability": 0.99 if passed else 0.79,
        "last_five_checkpoints_meet_selectivity": passed,
        "learned_expected_reward": 0.73,
        "never_pair_reward": 0.60,
        "matched_random_one_reward": 0.63,
        "target_memory_removed_reward": 0.60,
        "distractor_memory_removed_reward": 0.74,
    }


def _summary(passing_seeds: int) -> dict:
    return {
        "experiment": "sampled_hazard_stability_replication_b32",
        "seeds": list(range(980, 990)),
        "runs": [
            _run(seed, passed=index < passing_seeds)
            for index, seed in enumerate(range(980, 990))
        ],
        "success_audit": {"passed": True},
        "provenance": {"git_shas": ["abc123"]},
    }


def test_optimizer_replication_passes_with_eight_stable_seed_passes() -> None:
    result = audit_optimizer_replication_summary(_summary(8))

    assert result["passed"] is True
    assert result["individual_seed_passes"] == 8
    assert result["criteria"]["exactly_ten_declared_seeds"] is True
    assert result["criteria"]["at_least_eight_individual_seed_passes"] is True


def test_optimizer_replication_fails_with_only_seven_stable_seed_passes() -> None:
    result = audit_optimizer_replication_summary(_summary(7))

    assert result["passed"] is False
    assert result["individual_seed_passes"] == 7
    assert result["criteria"]["at_least_eight_individual_seed_passes"] is False
