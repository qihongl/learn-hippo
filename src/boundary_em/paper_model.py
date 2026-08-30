"""Recurrent prediction model with differentiable episodic retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from boundary_em.memory import differentiable_read


@dataclass(frozen=True)
class RecurrentState:
    """Working-memory state carried by the recurrent controller."""

    hidden: Tensor
    cell: Tensor


@dataclass(frozen=True)
class RetrievalOutput:
    """Reinstated state and competitive retrieval diagnostics."""

    value: Tensor
    attention: Tensor
    similarities: Tensor


@dataclass(frozen=True)
class PredictionStep:
    """Prediction and updated working-memory state for one time point."""

    logits: Tensor
    state: RecurrentState
    retrieval_attention: Tensor
    retrieval_gate: Tensor


@dataclass(frozen=True)
class EncodingPolicyOutput:
    """Encoding probability and delayed-return estimate."""

    probability: Tensor
    value: Tensor
    logit: Tensor


class EpisodicPredictionModel(nn.Module):
    """LSTM situation model with competitive episodic reinstatement."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        decision_dim: int,
        retrieval_temperature: float,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim, decision_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        if retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.retrieval_temperature = retrieval_temperature

        self.controller = nn.LSTMCell(input_dim, hidden_dim)
        self.retrieval_gate_layer = nn.Linear(hidden_dim, 1)
        self.decision = nn.Linear(hidden_dim, decision_dim)
        self.response_head = nn.Linear(decision_dim, output_dim)
        self.encoding_actor_encoder = nn.Sequential(
            nn.Linear(hidden_dim, decision_dim),
            nn.Tanh(),
        )
        self.encoding_critic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, decision_dim),
            nn.Tanh(),
        )
        self.encoding_actor = nn.Linear(decision_dim, 1)
        self.encoding_critic = nn.Linear(decision_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable maps without imposing an encoding schedule."""

        for name, parameter in self.named_parameters():
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
        """Return a deterministic empty working-memory state."""

        reference = next(self.parameters())
        state_device = reference.device if device is None else torch.device(device)
        state_dtype = reference.dtype if dtype is None else dtype
        zeros = torch.zeros(self.hidden_dim, device=state_device, dtype=state_dtype)
        return RecurrentState(hidden=zeros, cell=zeros.clone())

    def retrieve(self, query: Tensor, memories: Tensor) -> RetrievalOutput:
        """Retrieve a differentiable attention-weighted recurrent state."""

        if query.shape != (self.hidden_dim,):
            raise ValueError("query must have shape [hidden_dim]")
        if memories.ndim != 2 or memories.shape[1] != self.hidden_dim:
            raise ValueError("memories must have shape [items, hidden_dim]")
        readout = differentiable_read(
            query,
            memories,
            memories,
            temperature=self.retrieval_temperature,
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
        retrieval_enabled: bool = False,
    ) -> PredictionStep:
        """Update the situation model, optionally retrieve, and predict."""

        if model_input.shape != (self.input_dim,):
            raise ValueError("model_input must have shape [input_dim]")
        hidden, cell = self.controller(model_input, (state.hidden, state.cell))
        attention = torch.empty(0, device=cell.device, dtype=cell.dtype)
        gate = torch.zeros((), device=cell.device, dtype=cell.dtype)

        if retrieval_enabled and memories is not None and memories.shape[0] > 0:
            retrieval = self.retrieve(cell, memories)
            gate = torch.sigmoid(self.retrieval_gate_layer(cell)).squeeze()
            cell = cell + gate * retrieval.value
            hidden = torch.tanh(cell)
            attention = retrieval.attention

        decision = F.relu(self.decision(hidden))
        logits = self.response_head(decision)
        return PredictionStep(
            logits=logits,
            state=RecurrentState(hidden=hidden, cell=cell),
            retrieval_attention=attention,
            retrieval_gate=gate,
        )

    def encoding_policy(self, situation_model: Tensor) -> EncodingPolicyOutput:
        """Evaluate the boundary-blind encoding actor and critic."""

        if situation_model.shape != (self.hidden_dim,):
            raise ValueError("situation_model must have shape [hidden_dim]")
        actor_hidden = self.encoding_actor_encoder(situation_model)
        critic_hidden = self.encoding_critic_encoder(situation_model)
        logit = self.encoding_actor(actor_hidden).squeeze()
        value = self.encoding_critic(critic_hidden).squeeze()
        return EncodingPolicyOutput(
            probability=torch.sigmoid(logit),
            value=value,
            logit=logit,
        )
