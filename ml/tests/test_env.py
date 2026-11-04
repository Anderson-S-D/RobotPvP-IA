"""Tests del entorno RobotPVP (física, observación, recompensas y determinismo)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_ai import config as C
from robot_ai.env import RobotPvpEnv

FIXED_ARENA = (C.EVAL_ARENA_WIDTH, C.EVAL_ARENA_HEIGHT)


def _greedy_towards_player(env: RobotPvpEnv) -> int:
    """Acción que reduce la coordenada con mayor diferencia hacia el jugador."""
    dx = env.player_x - env.agent_x
    dy = env.player_y - env.agent_y
    if abs(dx) >= abs(dy):
        return C.ACTION_RIGHT if dx > 0 else C.ACTION_LEFT
    return C.ACTION_DOWN if dy > 0 else C.ACTION_UP


def _away_from_player(env: RobotPvpEnv) -> int:
    dx = env.agent_x - env.player_x
    dy = env.agent_y - env.player_y
    if abs(dx) >= abs(dy):
        return C.ACTION_RIGHT if dx > 0 else C.ACTION_LEFT
    return C.ACTION_DOWN if dy > 0 else C.ACTION_UP


def test_observation_shape_and_range() -> None:
    rng = np.random.default_rng(123)
    for policy in C.EVADER_POLICIES:
        env = RobotPvpEnv(
            arena_size=FIXED_ARENA,
            episode_duration_seconds=60.0,
            evader_policy=policy,
        )
        obs, _ = env.reset(seed=7)
        for step in range(120):
            assert obs.shape == (C.OBS_GEOMETRY_DIM,)
            assert obs.dtype == np.float32
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0), (
                policy,
                step,
                obs,
            )
            action = int(rng.integers(0, C.NUM_ACTIONS))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=1000 + step)


@pytest.mark.parametrize("policy", C.EVADER_POLICIES)
def test_entities_never_leave_bounds(policy: str) -> None:
    env = RobotPvpEnv(
        arena_size=FIXED_ARENA,
        episode_duration_seconds=60.0,
        evader_policy=policy,
    )
    rng = np.random.default_rng(456)
    env.reset(seed=11)
    min_x, max_x, min_y, max_y = env.bounds

    for step in range(400):
        env.step(int(rng.integers(0, C.NUM_ACTIONS)))
        px, py = env.player_position
        ax, ay = env.agent_position
        assert min_x <= px <= max_x, (step, px, min_x, max_x)
        assert min_x <= ax <= max_x, (step, ax, min_x, max_x)
        assert min_y <= py <= max_y, (step, py, min_y, max_y)
        assert min_y <= ay <= max_y, (step, ay, min_y, max_y)


def test_greedy_agent_captures_static_player() -> None:
    env = RobotPvpEnv(
        arena_size=FIXED_ARENA,
        episode_duration_seconds=60.0,
        evader_policy=C.EVADER_STATIC,
    )
    env.reset(seed=3)

    decisions = 0
    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    while decisions < 300 and not (terminated or truncated):
        action = _greedy_towards_player(env)
        _, reward, terminated, truncated, info = env.step(action)
        decisions += 1

    assert terminated, "el agente greedy no capturó al jugador estático"
    assert not truncated
    assert info["captured"] is True
    assert decisions < 300
    assert reward == pytest.approx(C.REWARD_CAPTURE)


def _rollout(seed: int, steps: int = 80) -> np.ndarray:
    env = RobotPvpEnv(
        arena_size=None,
        episode_duration_seconds=60.0,
        evader_policy=C.EVADER_MIXED,
    )
    obs, info = env.reset(seed=seed)
    action_rng = np.random.default_rng(999)
    rows = [
        np.concatenate(
            [
                obs,
                [0.0, 0.0, 0.0, info["frames"], info["distance"]],
                [env.agent_x, env.agent_y, env.player_x, env.player_y],
            ]
        )
    ]
    for _ in range(steps):
        action = int(action_rng.integers(0, C.NUM_ACTIONS))
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            np.concatenate(
                [
                    obs,
                    [reward, float(terminated), float(truncated),
                     info["frames"], info["distance"]],
                    [env.agent_x, env.agent_y, env.player_x, env.player_y],
                ]
            )
        )
        if terminated or truncated:
            break
    return np.asarray(rows, dtype=np.float64)


def test_same_seed_same_trajectory() -> None:
    trajectory_a = _rollout(seed=2026)
    trajectory_b = _rollout(seed=2026)
    assert trajectory_a.shape == trajectory_b.shape
    np.testing.assert_allclose(trajectory_a, trajectory_b, rtol=0.0, atol=0.0)


def test_timeout_reward_is_from_config() -> None:
    env = RobotPvpEnv(
        arena_size=FIXED_ARENA,
        episode_duration_seconds=1.0,
        evader_policy=C.EVADER_STATIC,
    )
    env.reset(seed=5)

    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        # Alejarse evita una captura accidental durante el episodio corto.
        _, reward, terminated, truncated, info = env.step(_away_from_player(env))

    assert truncated is True
    assert terminated is False
    assert info["captured"] is False
    assert info["frames"] == env.max_frames
    assert reward == pytest.approx(C.REWARD_TIMEOUT)
