"""Tests de humo de la implementación vectorizada NumPy."""

from __future__ import annotations

import numpy as np

from robot_ai import config as C
from robot_ai.vec_env import VecRobotPvpEnv


def test_vec_env_shapes_bounds_and_autoreset() -> None:
    num_envs = 8
    venv = VecRobotPvpEnv(
        num_envs=num_envs,
        arena_size=(C.EVAL_ARENA_WIDTH, C.EVAL_ARENA_HEIGHT),
        episode_duration_seconds=2.0,
        evader_policy=C.EVADER_MIXED,
        seed=0,
    )
    obs, infos = venv.reset(seed=0)
    assert obs.shape == (num_envs, C.OBS_GEOMETRY_DIM)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert len(infos) == num_envs

    rng = np.random.default_rng(1)
    saw_done = False
    for _ in range(100):
        obs, rewards, terminateds, truncateds, infos = venv.step(
            rng.integers(0, C.NUM_ACTIONS, size=num_envs)
        )
        assert obs.shape == (num_envs, C.OBS_GEOMETRY_DIM)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        assert rewards.shape == terminateds.shape == truncateds.shape == (num_envs,)
        assert np.all(venv.player_x >= venv.min_x) and np.all(
            venv.player_x <= venv.max_x
        )
        assert np.all(venv.agent_x >= venv.min_x) and np.all(
            venv.agent_x <= venv.max_x
        )
        assert np.all(venv.player_y >= venv.min_y) and np.all(
            venv.player_y <= venv.max_y
        )
        assert np.all(venv.agent_y >= venv.min_y) and np.all(
            venv.agent_y <= venv.max_y
        )
        done = terminateds | truncateds
        if done.any():
            saw_done = True
            for index in np.flatnonzero(done):
                assert "final_observation" in infos[int(index)]
    assert saw_done, "con 2 s por episodio y 100 decisiones debe haber autoreset"


def test_vec_env_same_seed_same_trajectory() -> None:
    def rollout(seed: int) -> np.ndarray:
        venv = VecRobotPvpEnv(num_envs=4, evader_policy=C.EVADER_MIXED, seed=seed)
        obs, _ = venv.reset(seed=seed)
        rng = np.random.default_rng(99)
        rows = []
        for _ in range(150):
            obs, rewards, terminateds, truncateds, _ = venv.step(
                rng.integers(0, C.NUM_ACTIONS, size=4)
            )
            rows.append(
                np.concatenate(
                    [obs.ravel(), rewards, terminateds.astype(float),
                     truncateds.astype(float)]
                )
            )
        return np.asarray(rows)

    np.testing.assert_array_equal(rollout(2026), rollout(2026))
