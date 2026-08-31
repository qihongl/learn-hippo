from boundary_em.compare_optimizer_stability import (
    compare_optimizer_stability_cells,
)


def _cell(
    name: str,
    *,
    batch_size: int,
    schedule: str,
    endpoints: list[float],
) -> dict:
    runs = []
    for seed, endpoint in zip((970, 971, 972), endpoints, strict=True):
        runs.append(
            {
                "seed": seed,
                "endpoint_probability": endpoint,
                "endpoint_probability_gap": endpoint - 0.01,
                "a1_endpoint_probability": endpoint,
                "b1_endpoint_probability": endpoint,
                "learned_expected_reward": 0.72,
                "never_pair_reward": 0.60,
                "matched_random_one_reward": 0.63,
                "target_memory_removed_reward": 0.60,
                "distractor_memory_removed_reward": 0.73,
                "last_five_checkpoints_meet_selectivity": endpoint >= 0.80,
                "largest_post_epoch_200_endpoint_drop": 0.01,
            }
        )
    return {
        "experiment": name,
        "seeds": [970, 971, 972],
        "configuration_values": {
            "training": {
                "free_policy": {
                    "batch_size": batch_size,
                    "learning_rate_schedule": schedule,
                }
            }
        },
        "runs": runs,
        "provenance": {"mode": "measured", "git_shas": ["abc123"]},
    }


def test_optimizer_stability_selection_uses_minimum_seed_then_tie_breaks() -> None:
    comparison = compare_optimizer_stability_cells(
        [
            _cell(
                "cosine-b16",
                batch_size=16,
                schedule="cosine_second_half",
                endpoints=[0.91, 0.92, 0.93],
            ),
            _cell(
                "constant-b32",
                batch_size=32,
                schedule="constant",
                endpoints=[0.94, 0.95, 0.96],
            ),
        ]
    )

    assert comparison["selected_cell"] == "constant-b32"
    assert all(cell["screen_passed"] for cell in comparison["cells"])

    tied = compare_optimizer_stability_cells(
        [
            _cell(
                "cosine-b16",
                batch_size=16,
                schedule="cosine_second_half",
                endpoints=[0.91, 0.92, 0.93],
            ),
            _cell(
                "cosine-b32",
                batch_size=32,
                schedule="cosine_second_half",
                endpoints=[0.92, 0.93, 0.94],
            ),
        ]
    )

    assert tied["selected_cell"] == "cosine-b16"


def test_optimizer_stability_selection_returns_none_when_any_seed_fails() -> None:
    comparison = compare_optimizer_stability_cells(
        [
            _cell(
                "cosine-b16",
                batch_size=16,
                schedule="cosine_second_half",
                endpoints=[0.99, 0.99, 0.20],
            )
        ]
    )

    assert comparison["selected_cell"] is None
    assert comparison["cells"][0]["screen_passed"] is False
