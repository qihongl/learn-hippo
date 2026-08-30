"""Three-event execution and forced encoding schedules for the paper task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from boundary_em.paper_model import EpisodicPredictionModel, RecurrentState
from boundary_em.paper_task import EventSequence, PaperTrial

ScheduleName = Literal[
    "endpoint_only",
    "midpoint_only",
    "midpoint_plus_endpoint",
    "dense",
    "never",
]


@dataclass(frozen=True)
class MemoryTrace:
    """One encoded recurrent state and its trial identity."""

    state: Tensor
    label: str
    time_index: int


@dataclass(frozen=True)
class EventComputation:
    """Observable model behavior over one event."""

    logits: Tensor
    probabilities: Tensor
    states: Tensor
    retrieval_attention: Tensor
    encoding_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrialComputation:
    """Observable model behavior over a1, b1, and b2."""

    a1: EventComputation
    b1: EventComputation
    b2: EventComputation
    memory_labels: tuple[str, ...]
    working_memory_was_reset: bool


def forced_schedule(event: EventSequence, name: ScheduleName) -> Tensor:
    """Return one declared experimenter-imposed encoding schedule."""

    actions = torch.zeros(event.inputs.shape[0], dtype=torch.bool)
    midpoint = event.delay + event.situation.shape[0] // 2 - 1
    if name == "endpoint_only":
        actions[event.boundary_index] = True
    elif name == "midpoint_only":
        actions[midpoint] = True
    elif name == "midpoint_plus_endpoint":
        actions[midpoint] = True
        actions[event.boundary_index] = True
    elif name == "dense":
        actions[:] = True
    elif name != "never":
        raise ValueError(f"unknown forced schedule: {name}")
    return actions


def expected_prediction_reward(
    probabilities: Tensor,
    targets: Tensor,
    penalties: Tensor,
) -> Tensor:
    """Expected +1/error-penalty/zero task reward under response probabilities."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [time, specific + don't know]")
    if targets.shape != probabilities.shape[:1] or penalties.shape != targets.shape:
        raise ValueError("targets and penalties must match the time dimension")

    specific = probabilities[:, :-1]
    valid = targets >= 0
    safe_targets = targets.clamp_min(0)
    correct = specific.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    correct = torch.where(valid, correct, torch.zeros_like(correct))
    incorrect = specific.sum(dim=1) - correct
    return correct - penalties * incorrect


def rollout_trial(
    model: EpisodicPredictionModel,
    trial: PaperTrial,
    *,
    b1_encoding_actions: Tensor,
    memory_capacity: int,
    retrieval_enabled: bool,
    a1_encoding_actions: Tensor | None = None,
) -> TrialComputation:
    """Process the exact three-event trial under declared encoding actions."""

    if memory_capacity < 1:
        raise ValueError("memory_capacity must be positive")
    if a1_encoding_actions is None:
        a1_encoding_actions = forced_schedule(trial.a1, "endpoint_only")

    memories: list[MemoryTrace] = []
    a1, _, memories = _rollout_event(
        model,
        trial.a1,
        state=model.initial_state(),
        memories=memories,
        retrieval_enabled=False,
        encoding_actions=a1_encoding_actions,
        memory_label="lure_a1",
        memory_capacity=memory_capacity,
    )

    b1_actions = b1_encoding_actions
    if not trial.b1_encoding_allowed:
        b1_actions = torch.zeros_like(b1_encoding_actions, dtype=torch.bool)
    b1_memory_label = (
        "target_b1" if trial.target_memory_available else "unrelated_b1"
    )
    b1, b1_state, memories = _rollout_event(
        model,
        trial.b1,
        state=model.initial_state(),
        memories=memories,
        retrieval_enabled=False,
        encoding_actions=b1_actions,
        memory_label=b1_memory_label,
        memory_capacity=memory_capacity,
    )

    b2_state = (
        model.initial_state()
        if trial.reset_working_memory_before_b2
        else b1_state
    )
    b2, _, memories = _rollout_event(
        model,
        trial.b2,
        state=b2_state,
        memories=memories,
        retrieval_enabled=retrieval_enabled,
        encoding_actions=torch.zeros(trial.b2.inputs.shape[0], dtype=torch.bool),
        memory_label="unused_b2",
        memory_capacity=memory_capacity,
    )
    return TrialComputation(
        a1=a1,
        b1=b1,
        b2=b2,
        memory_labels=tuple(memory.label for memory in memories),
        working_memory_was_reset=trial.reset_working_memory_before_b2,
    )


def _rollout_event(
    model: EpisodicPredictionModel,
    event: EventSequence,
    *,
    state: RecurrentState,
    memories: list[MemoryTrace],
    retrieval_enabled: bool,
    encoding_actions: Tensor,
    memory_label: str,
    memory_capacity: int,
) -> tuple[EventComputation, RecurrentState, list[MemoryTrace]]:
    if encoding_actions.shape != (event.inputs.shape[0],):
        raise ValueError("encoding actions must match event duration")

    logits: list[Tensor] = []
    states: list[Tensor] = []
    attention: list[Tensor] = []
    encoded: list[int] = []
    current_memories = list(memories)

    for time_index, model_input in enumerate(event.inputs):
        memory_tensor = (
            torch.stack([memory.state for memory in current_memories])
            if current_memories
            else None
        )
        step = model.predict_step(
            model_input,
            state,
            memories=memory_tensor,
            retrieval_enabled=retrieval_enabled,
        )
        state = step.state
        logits.append(step.logits)
        states.append(state.cell)
        attention.append(step.retrieval_attention)

        if bool(encoding_actions[time_index]):
            current_memories.append(
                MemoryTrace(
                    state=state.cell,
                    label=memory_label,
                    time_index=time_index,
                )
            )
            current_memories = current_memories[-memory_capacity:]
            encoded.append(time_index)

    logits_tensor = torch.stack(logits)
    if attention and attention[0].numel() > 0:
        attention_tensor = torch.stack(attention)
    else:
        attention_tensor = torch.empty(event.inputs.shape[0], 0)
    return (
        EventComputation(
            logits=logits_tensor,
            probabilities=torch.softmax(logits_tensor, dim=1),
            states=torch.stack(states),
            retrieval_attention=attention_tensor,
            encoding_indices=tuple(encoded),
        ),
        state,
        current_memories,
    )
