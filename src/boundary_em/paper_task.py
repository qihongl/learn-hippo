"""Deterministic implementation of the 2022 event-prediction task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

TaskProfile = Literal["released_code", "paper_text"]
Condition = Literal["RM", "DM", "NM"]
RemovalMode = Literal["released_count", "independent"]


@dataclass(frozen=True)
class PaperTaskConfig:
    """Frozen dimensions and sampling conventions for the published task."""

    profile: TaskProfile = "released_code"
    n_features: int = 16
    n_values: int = 4
    removal_probability: float = 0.3
    condition_probabilities: tuple[float, float, float] = (0.25, 0.25, 0.5)
    maximum_penalty: int = 4
    evaluation_penalty: int = 2
    maximum_event_similarity: float = 0.9

    def __post_init__(self) -> None:
        if self.profile not in ("released_code", "paper_text"):
            raise ValueError(f"unknown task profile: {self.profile}")
        if self.n_features != 16 or self.n_values != 4:
            raise ValueError("the exact paper task requires 16 features and 4 values")
        if not 0 <= self.removal_probability <= 1:
            raise ValueError("removal_probability must fall in [0, 1]")
        if not np.isclose(sum(self.condition_probabilities), 1.0):
            raise ValueError("condition probabilities must sum to one")
        if self.maximum_penalty < 0 or self.evaluation_penalty < 0:
            raise ValueError("penalties cannot be negative")
        if not 0 <= self.maximum_event_similarity <= 1:
            raise ValueError("maximum_event_similarity must fall in [0, 1]")

    @property
    def input_dim(self) -> int:
        """Width of the original observation, query, and penalty vector."""

        return 2 * self.n_features + self.n_values + 1

    @property
    def output_dim(self) -> int:
        """Four specific predictions plus the don't-know response."""

        return self.n_values + 1

    @property
    def max_delay(self) -> int:
        """Maximum inclusive delay under the selected source profile."""

        return 4 if self.profile == "released_code" else 3

    @property
    def removal_mode(self) -> RemovalMode:
        """Observation-removal rule under the selected source profile."""

        return "released_count" if self.profile == "released_code" else "independent"


@dataclass(frozen=True)
class EventSequence:
    """One event's complete situation and presented prediction sequence."""

    situation: Tensor
    inputs: Tensor
    targets: Tensor
    valid_predictions: Tensor
    observation_order: Tensor
    removed_observations: Tensor
    delay: int
    penalty: float
    boundary_index: int


@dataclass(frozen=True)
class PaperTrial:
    """Distractor and two target events under one memory condition."""

    condition: Condition
    a1: EventSequence
    b1: EventSequence
    b2: EventSequence
    reset_working_memory_before_b2: bool
    target_memory_available: bool


def sample_delay(*, max_delay: int, rng: np.random.RandomState) -> int:
    """Sample an inclusive integer prediction delay as in the released task."""

    if max_delay < 0:
        raise ValueError("max_delay cannot be negative")
    return int(rng.randint(low=0, high=max_delay + 1))


def sample_removed_features(
    *,
    n_features: int,
    probability: float,
    mode: RemovalMode,
    rng: np.random.RandomState,
) -> Tensor:
    """Sample which presented observations are replaced by zero vectors."""

    if n_features < 1:
        raise ValueError("n_features must be positive")
    if not 0 <= probability <= 1:
        raise ValueError("probability must fall in [0, 1]")

    removed = np.zeros(n_features, dtype=bool)
    if mode == "released_count":
        maximum_removed = probability * n_features
        count = int(np.round(rng.uniform(high=maximum_removed)))
        indices = rng.choice(np.arange(n_features), size=count, replace=False)
        removed[indices] = True
    elif mode == "independent":
        removed = rng.random_sample(n_features) < probability
    else:
        raise ValueError(f"unknown removal mode: {mode}")
    return torch.from_numpy(removed.copy())


def generate_trial(
    config: PaperTaskConfig,
    *,
    seed: int,
    condition: Condition | None = None,
    evaluation: bool,
) -> PaperTrial:
    """Generate one deterministic a1-b1-b2 trial without a boundary input."""

    rng = np.random.RandomState(seed)
    if condition is None:
        condition = rng.choice(
            np.array(["RM", "DM", "NM"]),
            p=config.condition_probabilities,
        ).item()
    if condition not in ("RM", "DM", "NM"):
        raise ValueError(f"unknown condition: {condition}")

    situations = _sample_trial_situations(config, condition, rng)
    if evaluation:
        delay_a = delay_b = 0
        penalty_a = penalty_b1 = penalty_b2 = float(config.evaluation_penalty)
        remove_a = remove_b1 = False
    else:
        delay_a = sample_delay(max_delay=config.max_delay, rng=rng)
        delay_b = sample_delay(max_delay=config.max_delay, rng=rng)
        penalty_a = penalty_b1 = config.maximum_penalty / 2
        penalty_b2 = float(rng.choice(np.arange(config.maximum_penalty + 1)))
        remove_a = remove_b1 = True

    a1 = _generate_event(
        config,
        situation=situations[0],
        delay=delay_a,
        penalty=penalty_a,
        remove_observations=remove_a,
        rng=rng,
    )
    b1 = _generate_event(
        config,
        situation=situations[1],
        delay=delay_b,
        penalty=penalty_b1,
        remove_observations=remove_b1,
        rng=rng,
    )
    b2 = _generate_event(
        config,
        situation=situations[2],
        delay=delay_b,
        penalty=penalty_b2,
        remove_observations=False,
        rng=rng,
    )
    return PaperTrial(
        condition=condition,
        a1=a1,
        b1=b1,
        b2=b2,
        reset_working_memory_before_b2=condition != "RM",
        target_memory_available=condition != "NM",
    )


def _sample_trial_situations(
    config: PaperTaskConfig,
    condition: Condition,
    rng: np.random.RandomState,
) -> tuple[Tensor, Tensor, Tensor]:
    situations: list[Tensor] = []
    required = 2 if condition in ("RM", "DM") else 3
    while len(situations) < required:
        candidate = torch.tensor(
            [rng.choice(config.n_values) for _ in range(config.n_features)],
            dtype=torch.long,
        )
        recent = situations[-2:]
        if all(
            float((candidate == previous).to(torch.float32).mean())
            <= config.maximum_event_similarity
            for previous in recent
        ):
            situations.append(candidate)

    if condition in ("RM", "DM"):
        return situations[0], situations[1], situations[1].clone()
    return situations[0], situations[1], situations[2]


def _generate_event(
    config: PaperTaskConfig,
    *,
    situation: Tensor,
    delay: int,
    penalty: float,
    remove_observations: bool,
    rng: np.random.RandomState,
) -> EventSequence:
    observation_order = torch.from_numpy(rng.permutation(config.n_features).copy())
    if remove_observations:
        removed = sample_removed_features(
            n_features=config.n_features,
            probability=config.removal_probability,
            mode=config.removal_mode,
            rng=rng,
        )
    else:
        removed = torch.zeros(config.n_features, dtype=torch.bool)

    length = config.n_features + delay
    observed_keys = torch.zeros(length, config.n_features)
    observed_values = torch.zeros(length, config.n_values)
    query_keys = torch.zeros(length, config.n_features)
    targets = torch.full((length,), -1, dtype=torch.long)

    for time, feature_index in enumerate(observation_order.tolist()):
        observed_keys[time, feature_index] = 1
        if not bool(removed[time]):
            observed_values[time] = F.one_hot(
                situation[feature_index],
                num_classes=config.n_values,
            ).to(torch.float32)

    for feature_index in range(config.n_features):
        query_time = delay + feature_index
        query_keys[query_time, feature_index] = 1
        targets[query_time] = situation[feature_index]

    penalty_column = torch.full((length, 1), float(penalty))
    inputs = torch.cat(
        [observed_keys, observed_values, query_keys, penalty_column],
        dim=1,
    )
    return EventSequence(
        situation=situation.clone(),
        inputs=inputs,
        targets=targets,
        valid_predictions=targets >= 0,
        observation_order=observation_order,
        removed_observations=removed,
        delay=delay,
        penalty=float(penalty),
        boundary_index=length - 1,
    )
