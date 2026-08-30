"""Differentiable content-addressable episodic memory operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryReadout:
    """Value and diagnostics returned by one episodic-memory query."""

    value: Tensor
    attention: Tensor
    similarities: Tensor


def compose_key(cue: Tensor, state: Tensor, mask: Tensor) -> Tensor:
    """Compose an episodic key from the event cue and accumulated state."""

    if state.shape != mask.shape:
        raise ValueError("state and mask must have identical shapes")
    return torch.cat([cue, state, mask])


def differentiable_read(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    temperature: float,
) -> MemoryReadout:
    """Read an attention-weighted value using cosine similarity."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("keys and values must be rank-two tensors")
    if keys.shape[0] != values.shape[0] or keys.shape[0] == 0:
        raise ValueError("memory must contain matching, non-empty keys and values")
    if query.shape != keys.shape[1:]:
        raise ValueError("query width must match memory-key width")

    similarities = F.cosine_similarity(query.unsqueeze(0), keys, dim=1)
    attention = torch.softmax(similarities / temperature, dim=0)
    value = attention @ values
    return MemoryReadout(value=value, attention=attention, similarities=similarities)
