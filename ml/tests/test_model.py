"""Tests de la red, la memoria de descubrimiento y el paso de decisión."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ai import config as C
from robot_ai.agent import build_obs, decision_step
from robot_ai.discovery import DiscoveryMemory, encode_actions
from robot_ai.model import SharedActorCritic


def test_model_output_shapes_and_parameter_count() -> None:
    model = SharedActorCritic()
    x = torch.randn(5, C.INPUT_DIM)
    logits, value = model(x)

    assert logits.shape == (5, C.ACTOR_OUTPUT_DIM)
    assert value.shape == (5, C.CRITIC_OUTPUT_DIM)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert model.count_params() == 229
    assert C.PARAM_COUNT == 229


def test_action_encoding_matches_config() -> None:
    memory = DiscoveryMemory(num_envs=C.NUM_ACTIONS)
    actions = np.arange(C.NUM_ACTIONS, dtype=np.int64)
    memory.update(actions, np.zeros(C.NUM_ACTIONS, dtype=np.float32))

    expected = np.asarray(
        [C.ACTION_ENCODINGS[name] for name in C.ACTION_NAMES],
        dtype=np.float32,
    )
    features = memory.features()
    np.testing.assert_allclose(features[:, 0], expected)
    np.testing.assert_allclose(
        encode_actions(actions), expected, rtol=0.0, atol=0.0
    )

    # -1 = ninguna acción -> 0.
    np.testing.assert_allclose(
        encode_actions(np.array([-1], dtype=np.int64)), np.array([0.0])
    )


def test_memory_resets_only_done_envs() -> None:
    memory = DiscoveryMemory(num_envs=3)
    memory.update(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0.5, -0.2, 1.0], dtype=np.float32),
    )
    before = memory.features().copy()

    memory.reset(np.array([False, True, False]))
    after = memory.features()

    # Los entornos no terminados conservan (a, δ).
    np.testing.assert_allclose(after[0], before[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(after[2], before[2], rtol=0.0, atol=0.0)
    # El entorno terminado vuelve a (ninguna, 0).
    np.testing.assert_allclose(after[1], np.zeros(2), rtol=0.0, atol=0.0)
    assert memory.actions_prev[1] == -1
    assert memory.delta_prev[1] == 0.0


def test_disabled_discovery_columns_are_zero() -> None:
    rng = np.random.default_rng(0)
    geom = rng.uniform(-1.0, 1.0, size=(4, C.OBS_GEOMETRY_DIM)).astype(
        np.float32
    )
    memory = DiscoveryMemory(num_envs=4, enabled=False)
    memory.update(
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32),
    )

    obs = build_obs(geom, memory)
    assert obs.shape == (4, C.OBS_DIM)
    np.testing.assert_allclose(obs[:, : C.OBS_GEOMETRY_DIM], geom)
    np.testing.assert_allclose(obs[:, C.OBS_GEOMETRY_DIM :], np.zeros((4, 2)))
    np.testing.assert_allclose(
        memory.provisional_features(np.arange(4)),
        np.zeros((4, 2)),
    )


def test_build_obs_order_and_memory_features() -> None:
    rng = np.random.default_rng(1)
    geom = rng.uniform(-1.0, 1.0, size=(2, C.OBS_GEOMETRY_DIM)).astype(
        np.float32
    )
    memory = DiscoveryMemory(num_envs=2)
    memory.update(
        np.array([C.ACTION_RIGHT, C.ACTION_UP], dtype=np.int64),
        np.array([0.3, -0.7], dtype=np.float32),
    )

    obs = build_obs(geom, memory)
    np.testing.assert_allclose(obs[:, : C.OBS_GEOMETRY_DIM], geom)
    assert obs[0, 6] == pytest.approx(C.ACTION_ENCODINGS["RIGHT"])
    assert obs[1, 6] == pytest.approx(C.ACTION_ENCODINGS["UP"])
    np.testing.assert_allclose(
        obs[:, 7], np.tanh(np.array([0.3, -0.7], dtype=np.float32))
    )


def test_decision_step_delta_formula_and_done_reset() -> None:
    torch.manual_seed(0)
    model = SharedActorCritic().eval()
    memory = DiscoveryMemory(num_envs=2, enabled=True)
    geom_t = np.zeros((2, C.OBS_GEOMETRY_DIM), dtype=np.float32)
    geom_next = np.full((2, C.OBS_GEOMETRY_DIM), 0.1, dtype=np.float32)
    rewards = np.array([0.25, -1.0], dtype=np.float32)
    dones = np.array([False, True])

    result = decision_step(
        model,
        memory,
        geom_t,
        geom_next,
        rewards,
        dones,
        deterministic=False,
    )

    expected_deltas = (
        rewards
        + C.GAMMA * result.next_values * (1.0 - dones.astype(np.float32))
        - result.values
    )
    np.testing.assert_allclose(result.deltas, expected_deltas, rtol=1e-6, atol=1e-6)

    # Entorno no terminado: memoria actualizada con (a_t, δ_t).
    assert memory.actions_prev[0] == result.actions[0]
    assert memory.delta_prev[0] == pytest.approx(result.deltas[0])
    # Entorno terminado: memoria reiniciada.
    assert memory.actions_prev[1] == -1
    assert memory.delta_prev[1] == 0.0


def test_decision_step_shapes() -> None:
    model = SharedActorCritic().eval()
    memory = DiscoveryMemory(num_envs=3)
    geom = np.zeros((3, C.OBS_GEOMETRY_DIM), dtype=np.float32)
    result = decision_step(
        model,
        memory,
        geom,
        geom,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=bool),
        deterministic=True,
    )

    assert result.actions.shape == (3,)
    assert result.log_probs.shape == (3,)
    assert result.entropies.shape == (3,)
    assert result.values.shape == (3,)
    assert result.next_values.shape == (3,)
    assert result.deltas.shape == (3,)
    assert result.obs_t.shape == (3, C.OBS_DIM)
    assert result.obs_next.shape == (3, C.OBS_DIM)
    assert np.all(np.isfinite(result.deltas))
