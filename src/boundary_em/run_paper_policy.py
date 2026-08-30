"""Run and preserve one learned-encoding seed on the exact paper task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from boundary_em.paper_policy_training import (
    EncodingStageConfig,
    sample_encoding_episode,
    train_encoding_stage,
)
from boundary_em.paper_rollout import (
    expected_prediction_reward,
    forced_schedule,
    rollout_trial,
)
from boundary_em.paper_task import Condition, PaperTaskConfig, generate_trial
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel

FORCED_SCHEDULES = (
    "endpoint_only",
    "midpoint_only",
    "midpoint_plus_endpoint",
    "dense",
    "never",
)


def _git_sha(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _cell(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n_trials": len(values),
    }


def _trial_reward(probabilities: torch.Tensor, trial: Any) -> float:
    valid = trial.b2.valid_predictions
    reward = expected_prediction_reward(
        probabilities[valid],
        trial.b2.targets[valid],
        trial.b2.inputs[valid, -1],
    ).mean()
    return float(reward.detach().item())


def _evaluate_condition(
    model: StructuredEpisodicPredictionModel,
    task_config: PaperTaskConfig,
    *,
    condition: Condition,
    trial_seed_start: int,
    action_seed_start: int,
    n_trials: int,
    memory_capacity: int,
) -> dict[str, Any]:
    learned_reward: list[float] = []
    retrieval_off_reward: list[float] = []
    endpoint_probability: list[float] = []
    nonendpoint_probability: list[float] = []
    encodings_per_event: list[float] = []
    time_probabilities: list[list[float]] = [[] for _ in range(16)]
    forced_rewards = {name: [] for name in FORCED_SCHEDULES}

    with torch.no_grad():
        for trial_index in range(n_trials):
            trial = generate_trial(
                task_config,
                seed=trial_seed_start + trial_index,
                condition=condition,
                evaluation=True,
            )
            episode = sample_encoding_episode(
                model,
                trial,
                action_generator=torch.Generator().manual_seed(
                    action_seed_start + trial_index
                ),
                memory_capacity=memory_capacity,
                forced_exploration=False,
            )
            learned_reward.append(float(episode.reward.item()))
            a1_length = len(episode.a1_actions)
            endpoint_probability.append(
                float(
                    torch.stack(
                        [
                            episode.probabilities[a1_length - 1],
                            episode.probabilities[-1],
                        ]
                    )
                    .mean()
                    .item()
                )
            )
            nonendpoint_probability.append(
                float(
                    torch.cat(
                        [
                            episode.probabilities[: a1_length - 1],
                            episode.probabilities[a1_length:-1],
                        ]
                    )
                    .mean()
                    .item()
                )
            )
            encodings_per_event.append(float(episode.actions.float().sum().item() / 2))
            for time_index in range(16):
                time_probabilities[time_index].append(
                    float(
                        torch.stack(
                            [
                                episode.probabilities[time_index],
                                episode.probabilities[a1_length + time_index],
                            ]
                        )
                        .mean()
                        .item()
                    )
                )

            retrieval_off = rollout_trial(
                model,
                trial,
                a1_encoding_actions=episode.a1_actions,
                b1_encoding_actions=episode.b1_actions,
                memory_capacity=memory_capacity,
                retrieval_enabled=False,
            )
            retrieval_off_reward.append(
                _trial_reward(retrieval_off.b2.probabilities, trial)
            )
            for schedule in FORCED_SCHEDULES:
                forced = rollout_trial(
                    model,
                    trial,
                    a1_encoding_actions=forced_schedule(trial.a1, schedule),
                    b1_encoding_actions=forced_schedule(trial.b1, schedule),
                    memory_capacity=memory_capacity,
                    retrieval_enabled=True,
                )
                forced_rewards[schedule].append(
                    _trial_reward(forced.b2.probabilities, trial)
                )

    return {
        "learned": {
            "reward": _cell(learned_reward),
            "endpoint_probability": _cell(endpoint_probability),
            "nonendpoint_probability": _cell(nonendpoint_probability),
            "encodings_per_event": _cell(encodings_per_event),
            "time_probabilities": [_cell(values) for values in time_probabilities],
        },
        "retrieval_off_same_actions": {
            "reward": _cell(retrieval_off_reward),
        },
        "forced": {
            schedule: {"reward": _cell(values)}
            for schedule, values in forced_rewards.items()
        },
    }


def run_paper_policy_config(
    config_path: str | Path,
    *,
    seed: int,
    output_directory: str | Path | None = None,
    updates_override: int | None = None,
    batch_size_override: int | None = None,
    evaluation_trials_override: int | None = None,
) -> dict[str, Any]:
    """Train, freeze, evaluate, and save one exact-task model seed."""

    config_path = Path(config_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    config_bytes = config_path.read_bytes()
    raw = yaml.safe_load(config_bytes)
    if seed not in raw["experiment"]["model_seeds"]:
        raise ValueError(f"seed {seed} is not declared in the configuration")

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    task_config = PaperTaskConfig(**raw["task"])
    model_values = dict(raw["model"])
    initial_probability = float(model_values.pop("initial_encoding_probability"))
    model = StructuredEpisodicPredictionModel(
        n_features=task_config.n_features,
        n_values=task_config.n_values,
        **model_values,
    )
    with torch.no_grad():
        model.encoding_actor.bias.fill_(
            math.log(initial_probability / (1 - initial_probability))
        )

    conditions = tuple(raw["training"]["conditions"])
    stage_configs: dict[str, EncodingStageConfig] = {}
    for stage_name in ("forced_value", "free_policy"):
        values = dict(raw["training"][stage_name])
        if updates_override is not None:
            values["updates"] = updates_override
        if batch_size_override is not None:
            values["batch_size"] = batch_size_override
        values["conditions"] = conditions
        stage_configs[stage_name] = EncodingStageConfig(**values)

    forced_result = train_encoding_stage(
        model,
        task_config,
        stage_configs["forced_value"],
        seed=seed,
        forced_exploration=True,
    )
    free_result = train_encoding_stage(
        model,
        task_config,
        stage_configs["free_policy"],
        seed=seed + 100,
        forced_exploration=False,
    )

    evaluation_config = raw["evaluation"]
    n_trials = int(
        evaluation_trials_override or evaluation_config["trials_per_condition"]
    )
    evaluation: dict[str, Any] = {}
    for condition_index, condition in enumerate(evaluation_config["conditions"]):
        evaluation[condition] = _evaluate_condition(
            model,
            task_config,
            condition=condition,
            trial_seed_start=int(evaluation_config["trial_seed_start"])
            + condition_index * n_trials,
            action_seed_start=int(evaluation_config["action_seed_start"])
            + seed * 10_000
            + condition_index * n_trials,
            n_trials=n_trials,
            memory_capacity=stage_configs["free_policy"].memory_capacity,
        )

    dm = evaluation["DM"]
    endpoint_gap = (
        dm["learned"]["endpoint_probability"]["mean"]
        - dm["learned"]["nonendpoint_probability"]["mean"]
    )
    prediction_benefit = (
        dm["learned"]["reward"]["mean"] - dm["forced"]["never"]["reward"]["mean"]
    )
    retrieval_benefit = (
        dm["learned"]["reward"]["mean"]
        - dm["retrieval_off_same_actions"]["reward"]["mean"]
    )

    effective_configuration = {
        "task": raw["task"],
        "model": raw["model"],
        "training": {
            "conditions": list(conditions),
            "forced_value": {
                **raw["training"]["forced_value"],
                "updates": stage_configs["forced_value"].updates,
                "batch_size": stage_configs["forced_value"].batch_size,
            },
            "free_policy": {
                **raw["training"]["free_policy"],
                "updates": stage_configs["free_policy"].updates,
                "batch_size": stage_configs["free_policy"].batch_size,
            },
        },
        "evaluation": {
            **evaluation_config,
            "trials_per_condition": n_trials,
        },
    }
    result: dict[str, Any] = {
        "task": "exact Lu-Hasson-Norman 2022 event-prediction generator",
        "data_kind": "measured synthetic simulation",
        "seed": seed,
        "experiment_status": raw["experiment"]["status"],
        "provenance": {
            "mode": "measured",
            "source": "boundary_em.run_paper_policy",
            "data_contract": "docs/paper_task_encoding/data_contract.md",
            "config": str(config_path.relative_to(repository)),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "git_sha": _git_sha(repository),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": platform.platform(),
            "torch_version": torch.__version__,
        },
        "configuration": effective_configuration,
        "training": {
            "forced_value_history": forced_result.history,
            "free_policy_history": free_result.history,
        },
        "evaluation": evaluation,
        "seed_success_audit": {
            "endpoint_probability_gap": endpoint_gap,
            "dm_prediction_benefit_over_never": prediction_benefit,
            "dm_retrieval_benefit": retrieval_benefit,
            "endpoint_above_nonendpoint": endpoint_gap > 0,
            "dm_reward_above_never": prediction_benefit > 0,
            "retrieval_off_removes_benefit": retrieval_benefit > 0,
            "all_passed": all(
                value > 0
                for value in (endpoint_gap, prediction_benefit, retrieval_benefit)
            ),
        },
        "notes": (
            "Weights were frozen on held-out trials. All mappings and observation "
            "orders were newly sampled. This diagnostic configuration was selected "
            "after exploratory runs and is not labeled confirmatory."
        ),
    }
    output_root = (
        Path(output_directory)
        if output_directory is not None
        else repository / raw["output"]["directory"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{raw['experiment']['name']}_seed{seed}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run_paper_policy_config(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
