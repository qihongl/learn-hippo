"""Structured situation model for isolating episodic encoding policy learning."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from boundary_em.memory import differentiable_read
from boundary_em.paper_model import (
    EncodingPolicyOutput,
    PredictionStep,
    RecurrentState,
    RetrievalOutput,
)


class StructuredEpisodicPredictionModel(nn.Module):
    """Accumulate observed feature values and retrieve encoded situation models."""

    def __init__(
        self,
        *,
        n_features: int,
        n_values: int,
        policy_hidden_dim: int,
        retrieval_temperature: float,
        temporal_context_scale: float,
        content_match_threshold: float = 0.6,
        content_match_sharpness: float = 30.0,
        retrieval_strength: float = 0.2,
        policy_mode: Literal["state_mlp", "progress_monotone"] = "state_mlp",
    ) -> None:
        super().__init__()
        if n_features != 16 or n_values != 4:
            raise ValueError("the exact paper task requires 16 features and 4 values")
        if policy_hidden_dim < 1:
            raise ValueError("policy_hidden_dim must be positive")
        if retrieval_temperature <= 0 or temporal_context_scale <= 0:
            raise ValueError("retrieval settings must be positive")
        if not 0 <= content_match_threshold <= 1:
            raise ValueError("content_match_threshold must fall in [0, 1]")
        if content_match_sharpness <= 0:
            raise ValueError("content_match_sharpness must be positive")
        if not 0 < retrieval_strength < 2:
            raise ValueError("retrieval_strength must fall between 0 and 2")
        if policy_mode not in ("state_mlp", "progress_monotone"):
            raise ValueError(f"unknown policy mode: {policy_mode}")

        self.n_features = n_features
        self.n_values = n_values
        self.situation_dim = n_features * n_values
        self.hidden_dim = self.situation_dim + n_features
        self.input_dim = 2 * n_features + n_values + 1
        self.output_dim = n_values + 1
        self.retrieval_temperature = retrieval_temperature
        self.temporal_context_scale = temporal_context_scale
        self.content_match_threshold = content_match_threshold
        self.content_match_sharpness = content_match_sharpness
        self.policy_mode = policy_mode

        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(8.0)))
        self.dont_know_bias = nn.Parameter(torch.tensor(4.0))
        self.dont_know_suppression_log = nn.Parameter(torch.tensor(math.log(8.0)))
        self.retrieval_strength_logit = nn.Parameter(
            torch.tensor(
                math.log((retrieval_strength / 2) / (1 - retrieval_strength / 2))
            )
        )

        self.encoding_actor_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, policy_hidden_dim),
            nn.Tanh(),
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.Tanh(),
        )
        self.encoding_critic_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, policy_hidden_dim),
            nn.Tanh(),
        )
        self.encoding_actor = nn.Linear(policy_hidden_dim, 1)
        self.encoding_progress_slope_raw = nn.Parameter(torch.tensor(-4.0))
        self.encoding_critic = nn.Linear(policy_hidden_dim, 1)
        self.reset_policy_parameters()

    def reset_policy_parameters(self) -> None:
        """Initialize the encoding system without imposing a temporal policy."""

        for name, parameter in self.named_parameters():
            if not name.startswith("encoding_"):
                continue
            if "weight" in name:
                gain = 0.01 if "encoding_actor" in name else 1.0
                nn.init.orthogonal_(parameter, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def initial_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecurrentState:
        """Return an empty situation model and empty temporal context."""

        reference = next(self.parameters())
        state_device = reference.device if device is None else torch.device(device)
        state_dtype = reference.dtype if dtype is None else dtype
        zeros = torch.zeros(self.hidden_dim, device=state_device, dtype=state_dtype)
        return RecurrentState(hidden=zeros, cell=zeros.clone())

    def retrieve(
        self,
        query: Tensor,
        memories: Tensor,
        encoding_strengths: Tensor | None = None,
    ) -> RetrievalOutput:
        """Retrieve a differentiable competitive average of episodic traces."""

        if query.shape != (self.hidden_dim,):
            raise ValueError("query must have shape [hidden_dim]")
        if memories.ndim != 2 or memories.shape[1] != self.hidden_dim:
            raise ValueError("memories must have shape [items, hidden_dim]")
        readout = differentiable_read(
            query,
            memories,
            memories,
            temperature=self.retrieval_temperature,
            encoding_strengths=encoding_strengths,
        )
        return RetrievalOutput(
            value=readout.value,
            attention=readout.attention,
            similarities=readout.similarities,
        )

    def predict_step(
        self,
        model_input: Tensor,
        state: RecurrentState,
        *,
        memories: Tensor | None = None,
        encoding_strengths: Tensor | None = None,
        retrieval_enabled: bool = False,
    ) -> PredictionStep:
        """Accumulate observations, optionally reinstate memory, and predict."""

        if model_input.shape != (self.input_dim,):
            raise ValueError("model_input must have shape [input_dim]")
        observed_key = model_input[: self.n_features]
        observed_value = model_input[self.n_features : self.n_features + self.n_values]
        query_start = self.n_features + self.n_values
        query_key = model_input[query_start : query_start + self.n_features]

        previous_situation = state.cell[: self.situation_dim].reshape(
            self.n_features,
            self.n_values,
        )
        observation = torch.outer(observed_key, observed_value)
        situation = torch.maximum(previous_situation, observation)
        previous_time = state.cell[self.situation_dim :]
        temporal_context = torch.maximum(
            previous_time,
            query_key * self.temporal_context_scale,
        )
        cell = torch.cat([situation.flatten(), temporal_context])

        attention = torch.empty(0, device=cell.device, dtype=cell.dtype)
        retrieval_gate = torch.zeros((), device=cell.device, dtype=cell.dtype)
        if retrieval_enabled and memories is not None and memories.shape[0] > 0:
            retrieval = self.retrieve(cell, memories, encoding_strengths)
            content_similarities = torch.nn.functional.cosine_similarity(
                cell[: self.situation_dim].unsqueeze(0),
                memories[:, : self.situation_dim],
                dim=1,
            )
            content_match = torch.sigmoid(
                self.content_match_sharpness
                * (content_similarities.max() - self.content_match_threshold)
            )
            retrieval_gate = (
                2.0 * torch.sigmoid(self.retrieval_strength_logit) * content_match
            )
            cell = cell + retrieval_gate * retrieval.value
            attention = retrieval.attention

        reinstated_situation = cell[: self.situation_dim].reshape(
            self.n_features,
            self.n_values,
        )
        queried_values = query_key @ reinstated_situation
        logit_scale = torch.exp(self.logit_scale_log)
        specific_logits = queried_values * logit_scale
        known_strength = queried_values.sum()
        dont_know_logit = (
            self.dont_know_bias
            - torch.exp(self.dont_know_suppression_log) * known_strength
        )
        logits = torch.cat([specific_logits, dont_know_logit.view(1)])
        return PredictionStep(
            logits=logits,
            state=RecurrentState(hidden=cell, cell=cell),
            retrieval_attention=attention,
            retrieval_gate=retrieval_gate,
        )

    def encoding_policy(self, situation_model: Tensor) -> EncodingPolicyOutput:
        """Evaluate the boundary-blind Bernoulli encoding actor and critic."""

        if situation_model.shape != (self.hidden_dim,):
            raise ValueError("situation_model must have shape [hidden_dim]")
        actor_hidden = self.encoding_actor_encoder(situation_model)
        critic_hidden = self.encoding_critic_encoder(situation_model)
        if self.policy_mode == "state_mlp":
            logit = self.encoding_actor(actor_hidden).squeeze()
        else:
            slope = torch.nn.functional.softplus(
                self.encoding_progress_slope_raw
            )
            logit = self.encoding_actor.bias.squeeze() + slope * (
                self.observable_progress(situation_model) - 0.5
            )
        value = self.encoding_critic(critic_hidden).squeeze()
        return EncodingPolicyOutput(
            probability=torch.sigmoid(logit),
            value=value,
            logit=logit,
        )

    def observable_progress(self, situation_model: Tensor) -> Tensor:
        """Measure accumulated feature and query evidence already in the state."""

        if situation_model.shape != (self.hidden_dim,):
            raise ValueError("situation_model must have shape [hidden_dim]")
        situation = situation_model[: self.situation_dim].reshape(
            self.n_features,
            self.n_values,
        )
        observed_fraction = (situation.sum(dim=1) > 0).to(situation.dtype).mean()
        query_context = situation_model[self.situation_dim :]
        queried_fraction = (query_context > 0).to(situation.dtype).mean()
        return 0.5 * (observed_fraction + queried_fraction)
