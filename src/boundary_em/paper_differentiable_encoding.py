"""Continuous episodic encoding trained through differentiable retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from boundary_em.paper_model import RecurrentState
from boundary_em.paper_rollout import (
    expected_prediction_reward,
    forced_schedule,
    rollout_trial,
)
from boundary_em.paper_task import (
    Condition,
    PaperTaskConfig,
    PaperTrial,
    generate_trial,
)
from boundary_em.structured_paper_model import StructuredEpisodicPredictionModel


@dataclass(frozen=True)
class DifferentiableEncodingConfig:
    """Optimization settings for continuous encoding strengths."""

    updates: int
    batch_size: int
    learning_rate: float
    gradient_clip: float
    memory_capacity: int
    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.memory_capacity < 1:
            raise ValueError("memory_capacity must be positive")
        if not self.conditions:
            raise ValueError("at least one condition is required")


@dataclass(frozen=True)
class DifferentiableTrialComputation:
    """Prediction and encoding strengths for one exact paper-task trial."""

    a1_encoding_strengths: Tensor
    b1_encoding_strengths: Tensor
    b2_probabilities: Tensor
    retrieval_attention: Tensor


@dataclass(frozen=True)
class DifferentiableEncodingResult:
    """Complete update history for continuous encoding optimization."""

    history: list[dict[str, float | int]]


def rollout_differentiable_encoding_trial(
    model: StructuredEpisodicPredictionModel,
    trial: PaperTrial,
    *,
    a1_encoding_strengths: Tensor,
    b1_encoding_strengths: Tensor,
    memory_capacity: int,
    retrieval_enabled: bool,
) -> DifferentiableTrialComputation:
    """Run a trial with continuous strengths attached to all encoded situations."""

    if a1_encoding_strengths.shape != (len(trial.a1.inputs),):
        raise ValueError("a1 encoding strengths must match event duration")
    if b1_encoding_strengths.shape != (len(trial.b1.inputs),):
        raise ValueError("b1 encoding strengths must match event duration")
    if memory_capacity < 1:
        raise ValueError("memory_capacity must be positive")

    probe = rollout_trial(
        model,
        trial,
        a1_encoding_actions=forced_schedule(trial.a1, "never"),
        b1_encoding_actions=forced_schedule(trial.b1, "never"),
        memory_capacity=memory_capacity,
        retrieval_enabled=False,
    )
    effective_b1_strengths = b1_encoding_strengths
    if not trial.b1_encoding_allowed:
        effective_b1_strengths = torch.zeros_like(b1_encoding_strengths)
    memories = torch.cat([probe.a1.states, probe.b1.states], dim=0)
    encoding_strengths = torch.cat(
        [a1_encoding_strengths, effective_b1_strengths],
        dim=0,
    )
    memories = memories[-memory_capacity:]
    encoding_strengths = encoding_strengths[-memory_capacity:]

    if trial.reset_working_memory_before_b2:
        state = model.initial_state()
    else:
        final_b1_state = probe.b1.states[-1]
        state = RecurrentState(hidden=final_b1_state, cell=final_b1_state)

    logits: list[Tensor] = []
    attention: list[Tensor] = []
    for model_input in trial.b2.inputs:
        step = model.predict_step(
            model_input,
            state,
            memories=memories,
            encoding_strengths=encoding_strengths,
            retrieval_enabled=retrieval_enabled,
        )
        state = step.state
        logits.append(step.logits)
        attention.append(step.retrieval_attention)
    logits_tensor = torch.stack(logits)
    if attention and attention[0].numel() > 0:
        attention_tensor = torch.stack(attention)
    else:
        attention_tensor = torch.empty(len(trial.b2.inputs), 0)
    return DifferentiableTrialComputation(
        a1_encoding_strengths=a1_encoding_strengths,
        b1_encoding_strengths=effective_b1_strengths,
        b2_probabilities=torch.softmax(logits_tensor, dim=1),
        retrieval_attention=attention_tensor,
    )


def train_differentiable_encoding(
    model: StructuredEpisodicPredictionModel,
    task_config: PaperTaskConfig,
    config: DifferentiableEncodingConfig,
    *,
    seed: int,
) -> DifferentiableEncodingResult:
    """Train only continuous encoding strengths from delayed prediction reward."""

    torch.manual_seed(seed)
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    actor_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_actor = name.startswith("encoding_actor")
        parameter.requires_grad_(is_actor)
        if is_actor:
            actor_parameters.append(parameter)
    optimizer = torch.optim.Adam(actor_parameters, lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for update in range(config.updates):
        rewards: list[Tensor] = []
        endpoint_strengths: list[Tensor] = []
        nonendpoint_strengths: list[Tensor] = []
        for batch_index in range(config.batch_size):
            condition_index = (update * config.batch_size + batch_index) % len(
                config.conditions
            )
            trial = generate_trial(
                task_config,
                seed=seed * 10_000_000 + update * config.batch_size + batch_index,
                condition=config.conditions[condition_index],
                evaluation=False,
            )
            probe = rollout_trial(
                model,
                trial,
                a1_encoding_actions=forced_schedule(trial.a1, "never"),
                b1_encoding_actions=forced_schedule(trial.b1, "never"),
                memory_capacity=config.memory_capacity,
                retrieval_enabled=False,
            )
            a1_strengths = torch.stack(
                [
                    model.encoding_policy(state.detach()).probability
                    for state in probe.a1.states
                ]
            )
            b1_strengths = torch.stack(
                [
                    model.encoding_policy(state.detach()).probability
                    for state in probe.b1.states
                ]
            )
            computation = rollout_differentiable_encoding_trial(
                model,
                trial,
                a1_encoding_strengths=a1_strengths,
                b1_encoding_strengths=b1_strengths,
                memory_capacity=config.memory_capacity,
                retrieval_enabled=True,
            )
            valid = trial.b2.valid_predictions
            rewards.append(
                expected_prediction_reward(
                    computation.b2_probabilities[valid],
                    trial.b2.targets[valid],
                    trial.b2.inputs[valid, -1],
                ).mean()
            )
            endpoint_strengths.extend([a1_strengths[-1], b1_strengths[-1]])
            nonendpoint_strengths.extend([*a1_strengths[:-1], *b1_strengths[:-1]])

        mean_reward = torch.stack(rewards).mean()
        loss = -mean_reward
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(actor_parameters, config.gradient_clip)
        optimizer.step()
        history.append(
            {
                "update": update,
                "loss": float(loss.detach().item()),
                "mean_reward": float(mean_reward.detach().item()),
                "endpoint_strength": float(
                    torch.stack(endpoint_strengths).mean().detach().item()
                ),
                "nonendpoint_strength": float(
                    torch.stack(nonendpoint_strengths).mean().detach().item()
                ),
                "gradient_norm": float(gradient_norm.detach().item()),
            }
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])
    return DifferentiableEncodingResult(history=history)
