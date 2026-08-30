"""Synthetic event task used to test learned episodic writing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TaskConfig:
    """Dimensions of the controlled event-generation process."""

    n_features: int = 4
    cue_dim: int = 6
    query_features: int = 2

    def __post_init__(self) -> None:
        if self.n_features < 2:
            raise ValueError("n_features must be at least two")
        if self.cue_dim < 1:
            raise ValueError("cue_dim must be positive")
        if not 0 < self.query_features < self.n_features:
            raise ValueError("query_features must fall within the event")


@dataclass(frozen=True)
class EventEpisode:
    """One event's study trajectory and delayed partial query."""

    cue: Tensor
    features: Tensor
    reveal_order: Tensor
    states: Tensor
    masks: Tensor
    policy_inputs: Tensor
    query_state: Tensor
    query_mask: Tensor


def generate_episode(
    config: TaskConfig,
    *,
    seed: int,
    null_steps: Sequence[int] = (),
) -> EventEpisode:
    """Generate a deterministic episode without exposing a boundary indicator."""

    invalid_null_steps = [
        progress for progress in null_steps if not 0 <= progress < config.n_features
    ]
    if invalid_null_steps:
        raise ValueError("null steps must occur before semantic completion")
    null_counts = Counter(null_steps)

    generator = torch.Generator().manual_seed(seed)
    cue = F.normalize(torch.randn(config.cue_dim, generator=generator), dim=0)
    features = torch.randint(
        0,
        2,
        (config.n_features,),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)
    features = features.mul(2).sub(1)
    reveal_order = torch.randperm(config.n_features, generator=generator)

    state = torch.zeros(config.n_features)
    mask = torch.zeros(config.n_features)
    states: list[Tensor] = []
    masks: list[Tensor] = []
    semantic_states: list[Tensor] = []
    semantic_masks: list[Tensor] = []
    for _ in range(null_counts[0]):
        states.append(state.clone())
        masks.append(mask.clone())

    for progress, feature_index in enumerate(reveal_order, start=1):
        index = int(feature_index.item())
        state = state.clone()
        mask = mask.clone()
        state[index] = features[index]
        mask[index] = 1
        states.append(state)
        masks.append(mask)
        semantic_states.append(state)
        semantic_masks.append(mask)
        for _ in range(null_counts[progress]):
            states.append(state.clone())
            masks.append(mask.clone())

    state_tensor = torch.stack(states)
    mask_tensor = torch.stack(masks)
    repeated_cue = cue.expand(state_tensor.shape[0], -1)
    policy_inputs = torch.cat([repeated_cue, state_tensor, mask_tensor], dim=1)
    query_index = config.query_features - 1

    return EventEpisode(
        cue=cue,
        features=features,
        reveal_order=reveal_order,
        states=state_tensor,
        masks=mask_tensor,
        policy_inputs=policy_inputs,
        query_state=semantic_states[query_index],
        query_mask=semantic_masks[query_index],
    )
