"""Fixed-schedule evaluation for validating the selective-encoding objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

import torch

from boundary_em.memory import compose_key, differentiable_read
from boundary_em.task import EventEpisode

CapacityMode = Literal["fixed", "historical"]
RetrievalMode = Literal["competitive", "latest"]
Schedule = tuple[bool, ...]


@dataclass(frozen=True)
class ScheduleResult:
    """Held-out performance of one deterministic write schedule."""

    schedule: Schedule
    mean_reward: float
    mean_mse: float
    per_episode_reward: tuple[float, ...]


def evaluate_schedule(
    episodes: Sequence[EventEpisode],
    schedule: Schedule,
    *,
    temperature: float,
    capacity_mode: CapacityMode = "fixed",
    fixed_capacity: int | None = None,
    retrieval_mode: RetrievalMode = "competitive",
) -> ScheduleResult:
    """Evaluate one schedule on paired synthetic episodes."""

    if not episodes:
        raise ValueError("at least one episode is required")
    if len(schedule) != episodes[0].states.shape[0]:
        raise ValueError("schedule length must match the study trajectory")
    if capacity_mode not in {"fixed", "historical"}:
        raise ValueError(f"unknown capacity mode: {capacity_mode}")
    if retrieval_mode not in {"competitive", "latest"}:
        raise ValueError(f"unknown retrieval mode: {retrieval_mode}")

    selected = [index for index, write in enumerate(schedule) if write]
    if capacity_mode == "historical":
        capacity = max(len(selected), 1)
    else:
        capacity = fixed_capacity or len(schedule)
        if capacity < 1:
            raise ValueError("fixed capacity must be positive")
    selected = selected[-capacity:]

    rewards: list[float] = []
    losses: list[float] = []
    for episode in episodes:
        held_out = episode.query_mask == 0
        if selected and retrieval_mode == "competitive":
            all_keys = torch.stack(
                [
                    compose_key(episode.cue, state, mask)
                    for state, mask in zip(
                        episode.states, episode.masks, strict=True
                    )
                ]
            )
            query = compose_key(
                episode.cue, episode.query_state, episode.query_mask
            )
            readout = differentiable_read(
                query,
                all_keys[selected],
                episode.states[selected],
                temperature=temperature,
            ).value
        elif selected and retrieval_mode == "latest":
            readout = episode.states[selected[-1]]
        else:
            readout = torch.zeros_like(episode.features)

        mse = torch.mean((readout[held_out] - episode.features[held_out]) ** 2)
        reward = 1.0 - mse
        losses.append(float(mse.item()))
        rewards.append(float(reward.item()))

    return ScheduleResult(
        schedule=schedule,
        mean_reward=sum(rewards) / len(rewards),
        mean_mse=sum(losses) / len(losses),
        per_episode_reward=tuple(rewards),
    )


def enumerate_schedules(
    episodes: Sequence[EventEpisode],
    *,
    temperature: float,
    capacity_mode: CapacityMode = "fixed",
    fixed_capacity: int | None = None,
    retrieval_mode: RetrievalMode = "competitive",
) -> list[ScheduleResult]:
    """Evaluate all binary write schedules and rank them by mean reward."""

    n_steps = episodes[0].states.shape[0]
    results = [
        evaluate_schedule(
            episodes,
            tuple(schedule),
            temperature=temperature,
            capacity_mode=capacity_mode,
            fixed_capacity=fixed_capacity,
            retrieval_mode=retrieval_mode,
        )
        for schedule in product((False, True), repeat=n_steps)
    ]
    return sorted(results, key=lambda result: result.mean_reward, reverse=True)
