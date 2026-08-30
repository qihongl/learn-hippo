"""Actor-critic policy for discrete episodic-memory writes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyOutput:
    """Per-study-state write probabilities and return estimates."""

    probabilities: Tensor
    values: Tensor
    logits: Tensor


class WriteActorCritic(nn.Module):
    """Shared state encoder with Bernoulli actor and scalar critic heads."""

    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use stable small policy outputs without imposing a write schedule."""

        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, policy_inputs: Tensor) -> PolicyOutput:
        if policy_inputs.ndim != 2:
            raise ValueError("policy inputs must have shape [time, features]")
        hidden = self.encoder(policy_inputs)
        logits = self.actor(hidden).squeeze(-1)
        values = self.critic(hidden).squeeze(-1)
        return PolicyOutput(
            probabilities=torch.sigmoid(logits),
            values=values,
            logits=logits,
        )
