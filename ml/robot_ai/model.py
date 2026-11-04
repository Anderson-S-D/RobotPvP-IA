"""Red Actor-Crítico de cabezas compartidas (8 → 16 → {4, 1})."""

from __future__ import annotations

import math

import torch
from torch import nn

from . import config as C


class SharedActorCritic(nn.Module):
    """Actor-Crítico con cuerpo compartido, exactamente 229 parámetros.

    Arquitectura:
        x (8) -> fc1 (16) -> tanh -> h
        h -> actor  -> logits (4)   [UP, DOWN, LEFT, RIGHT]
        h -> critic -> value  (1)   [lineal]

    ``forward`` devuelve ``(logits, value)``. El softmax NO se aplica aquí:
    el llamador debe usar ``torch.distributions.Categorical(logits=...)``.

    Convención de pesos compatible con la exportación JSON:
        y = W·x + b, con W de forma [salidas, entradas] (nn.Linear.weight).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(C.INPUT_DIM, C.HIDDEN_DIM)
        self.actor = nn.Linear(C.HIDDEN_DIM, C.ACTOR_OUTPUT_DIM)
        self.critic = nn.Linear(C.HIDDEN_DIM, C.CRITIC_OUTPUT_DIM)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        # Capa oculta: gain sqrt(2) para tanh.
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.fc1.bias)

        # Actor: política casi uniforme al inicio.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Crítico: escala estándar.
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve ``(logits, value)`` con formas ``(N, 4)`` y ``(N, 1)``."""
        hidden = torch.tanh(self.fc1(x))
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

    def count_params(self) -> int:
        """Número total de parámetros entrenables (debe ser 229)."""
        return sum(parameter.numel() for parameter in self.parameters())
