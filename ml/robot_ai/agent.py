"""Construcción de la observación del agente y pasos Actor-Crítico.

Flujo (coincide con la Figura 1 del Informe Técnico T2)::

    s_t = [geom_t, a_{t-1}, δ_{t-1}]
        -> actor elige a_t, crítico da V(s_t)
    entorno -> r_t, geom_{t+1}, done
    s'_{t+1} = [geom_{t+1}, a_t, δ_{t-1}]   (δ provisional, sin circularidad)
    δ_t = r_t + γ·V(s'_{t+1})·(1 − done) − V(s_t)    (con torch.no_grad)
    memory.update(a_t, δ_t) -> s_{t+1} = [geom_{t+1}, a_t, δ_t]

Para entrenar, el paso se divide en dos llamadas porque la acción debe
elegirse antes de avanzar el entorno:

    policy_step = select_action(model, memory, geom_t)
    geom_next, r, terminated, truncated, info = env.step(policy_step.actions)
    discovery_step = update_discovery(
        model, memory, policy_step.actions, geom_next, r,
        terminated | truncated, policy_step.values,
    )

``decision_step`` combina ambas para evaluación/rollouts sin entorno
interactivo. El estado provisional reutiliza ``δ_{t-1}`` para poder evaluar
V(s'_{t+1}) antes de conocer δ_t. ``memory.reset(done)`` reinicia la ventana
t−1 al empezar el siguiente episodio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from torch.distributions import Categorical

from . import config as C
from .discovery import DiscoveryMemory
from .model import SharedActorCritic


def build_obs(
    geom: np.ndarray,
    memory: DiscoveryMemory,
) -> np.ndarray:
    """Concatena geometría (N, 6) + descubrimiento (N, 2) -> (N, 8).

    Orden exacto de AGENTS.md:
        0-5 geometría del entorno
        6   encode(a_{t-1})
        7   tanh(δ_{t-1})
    """
    geom_array = np.asarray(geom, dtype=np.float32)
    if geom_array.ndim != 2 or geom_array.shape[1] != C.OBS_GEOMETRY_DIM:
        raise ValueError(
            f"geom debe tener forma (N, {C.OBS_GEOMETRY_DIM}), "
            f"recibido {geom_array.shape}"
        )
    if geom_array.shape[0] != memory.num_envs:
        raise ValueError(
            f"geom tiene {geom_array.shape[0]} entornos y la memoria "
            f"{memory.num_envs}"
        )

    obs = np.concatenate([geom_array, memory.features()], axis=1)
    return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class PolicyStep:
    """Acción elegida y valores de la política en s_t."""

    actions: np.ndarray        # (N,) int64, índice 0..3
    log_probs: np.ndarray      # (N,) float32
    entropies: np.ndarray      # (N,) float32
    values: np.ndarray         # (N,) float32
    obs_t: np.ndarray          # (N, 8)


@dataclass(frozen=True)
class DiscoveryStep:
    """Actualización de la memoria de descubrimiento tras avanzar el entorno."""

    next_values: np.ndarray    # (N,) V(s'_{t+1}), δ provisional
    deltas: np.ndarray         # (N,) δ_t
    obs_next: np.ndarray       # (N, 8), ya con la memoria actualizada


@dataclass(frozen=True)
class DecisionStep:
    """Resultado completo de un paso de decisión para N entornos."""

    actions: np.ndarray        # (N,) int64, índice 0..3
    log_probs: np.ndarray      # (N,) float32
    entropies: np.ndarray      # (N,) float32
    values: np.ndarray         # (N,) float32
    next_values: np.ndarray    # (N,) float32
    deltas: np.ndarray         # (N,) float32
    obs_t: np.ndarray          # (N, 8)
    obs_next: np.ndarray       # (N, 8), ya con la memoria actualizada


@torch.no_grad()
def select_action(
    model: SharedActorCritic,
    memory: DiscoveryMemory,
    geom_t: np.ndarray,
    *,
    deterministic: bool = False,
) -> PolicyStep:
    """Elige ``a_t`` y evalúa ``V(s_t)`` con la memoria actual."""
    obs_t = build_obs(geom_t, memory)
    device = next(model.parameters()).device
    obs_t_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=device)
    logits, values = model(obs_t_tensor)
    distribution = Categorical(logits=logits)

    if deterministic:
        actions_tensor = torch.argmax(logits, dim=-1)
    else:
        actions_tensor = distribution.sample()

    return PolicyStep(
        actions=actions_tensor.detach().cpu().numpy().astype(
            np.int64, copy=False
        ),
        log_probs=distribution.log_prob(actions_tensor)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        entropies=distribution.entropy()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        values=values.squeeze(-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        obs_t=obs_t,
    )


@torch.no_grad()
def update_discovery(
    model: SharedActorCritic,
    memory: DiscoveryMemory,
    actions: Union[np.ndarray, Sequence[int]],
    geom_next: np.ndarray,
    rewards: Union[np.ndarray, Sequence[float]],
    dones: Union[np.ndarray, Sequence[bool]],
    values: Union[np.ndarray, Sequence[float]],
    *,
    gamma: float = C.GAMMA,
) -> DiscoveryStep:
    """Calcula δ_t y actualiza la memoria de descubrimiento.

    ``values`` debe ser V(s_t) obtenido en ``select_action``. El bootstrap
    usa ``s'_{t+1} = [geom_{t+1}, a_t, δ_{t-1}]`` para evitar la dependencia
    circular. Los entornos con ``done`` no hacen bootstrap y reinician su
    memoria a ``(ninguna, 0)``.
    """
    memory_features = memory.num_envs
    geom_next_array = np.asarray(geom_next, dtype=np.float32)
    if (
        geom_next_array.ndim != 2
        or geom_next_array.shape[1] != C.OBS_GEOMETRY_DIM
    ):
        raise ValueError(
            f"geom_next debe tener forma (N, {C.OBS_GEOMETRY_DIM}), "
            f"recibido {geom_next_array.shape}"
        )
    if geom_next_array.shape[0] != memory_features:
        raise ValueError("geom_next y la memoria tienen distinto N")

    actions_array = np.asarray(actions, dtype=np.int64).reshape(-1)
    rewards_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones_array = np.asarray(dones, dtype=bool).reshape(-1)
    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    for name, array in (
        ("actions", actions_array),
        ("rewards", rewards_array),
        ("dones", dones_array),
        ("values", values_array),
    ):
        if array.shape != (memory_features,):
            raise ValueError(
                f"{name} debe tener forma ({memory_features},), "
                f"recibido {array.shape}"
            )

    # s'_{t+1} = [geom_{t+1}, a_t, δ_{t-1}]
    obs_next_provisional = np.concatenate(
        [geom_next_array, memory.provisional_features(actions_array)],
        axis=1,
    ).astype(np.float32, copy=False)
    obs_next_provisional = np.clip(
        obs_next_provisional, -1.0, 1.0
    ).astype(np.float32, copy=False)

    device = next(model.parameters()).device
    obs_next_tensor = torch.as_tensor(
        obs_next_provisional, dtype=torch.float32, device=device
    )
    _, next_values = model(obs_next_tensor)
    next_values_array = (
        next_values.squeeze(-1).detach().cpu().numpy().astype(
            np.float32, copy=False
        )
    )

    # δ_t = r_t + γ·V(s'_{t+1})·(1 − done) − V(s_t)
    deltas = (
        rewards_array
        + gamma * next_values_array * (1.0 - dones_array.astype(np.float32))
        - values_array
    ).astype(np.float32, copy=False)

    memory.update(actions_array, deltas)
    memory.reset(dones_array)

    # s_{t+1} = [geom_{t+1}, a_t, δ_t] (en done queda (ninguna, 0)).
    obs_next = build_obs(geom_next_array, memory)

    return DiscoveryStep(
        next_values=next_values_array,
        deltas=deltas,
        obs_next=obs_next,
    )


@torch.no_grad()
def decision_step(
    model: SharedActorCritic,
    memory: DiscoveryMemory,
    geom_t: np.ndarray,
    geom_next: np.ndarray,
    rewards: Union[np.ndarray, Sequence[float]],
    dones: Union[np.ndarray, Sequence[bool]],
    *,
    deterministic: bool = False,
) -> DecisionStep:
    """Paso completo (acción + δ + memoria) para evaluación/rollouts.

    Parameters
    ----------
    geom_t:
        Geometría actual ``(N, 6)``.
    geom_next:
        Geometría tras ``action_repeat`` frames ``(N, 6)``.
    rewards:
        Recompensa ``r_t`` por entorno ``(N,)``.
    dones:
        ``terminated | truncated`` por entorno ``(N,)``.

    Notes
    -----
    La función se ejecuta bajo ``torch.no_grad`` porque es un paso de rollout.
    Para entrenar la política se deben recalcular ``log_probs``/entropías en el
    update, o eliminar este decorador y conservar los grafos. δ_t siempre se
    calcula sin gradiente, tal como exige la especificación.
    """
    policy_step = select_action(
        model, memory, geom_t, deterministic=deterministic
    )
    discovery_step = update_discovery(
        model,
        memory,
        policy_step.actions,
        geom_next,
        rewards,
        dones,
        policy_step.values,
    )
    return DecisionStep(
        actions=policy_step.actions,
        log_probs=policy_step.log_probs,
        entropies=policy_step.entropies,
        values=policy_step.values,
        next_values=discovery_step.next_values,
        deltas=discovery_step.deltas,
        obs_t=policy_step.obs_t,
        obs_next=discovery_step.obs_next,
    )
