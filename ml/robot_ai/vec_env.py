"""Versión vectorizada en NumPy del entorno RobotPVP.

Mantiene N episodios en paralelo con arrays NumPy y autoreset inmediato:
cuando un entorno termina/trunca, su observación final se guarda en
``info["final_observation"]`` y el entorno se reinicia en el mismo ``step``.

API:
    reset(seed=...) -> (obs[N, 6], infos[list])
    step(actions[N]) -> (obs, rewards, terminateds, truncateds, infos)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from . import config as C

_POLICY_TO_CODE = {
    C.EVADER_FLEE: 0,
    C.EVADER_TILT_RANDOM: 1,
    C.EVADER_STATIC: 2,
}
_CODE_TO_POLICY = {code: policy for policy, code in _POLICY_TO_CODE.items()}


class VecRobotPvpEnv:
    """N copias del entorno en arrays NumPy, con autoreset.

    No usa ``SyncVectorEnv`` porque el bucle Python de N entornos añadía
    overhead; esta implementación vectoriza las operaciones de física,
    evasor, colisión y observación.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int = C.VECTORIZED_DEFAULT_NUM_ENVS,
        *,
        arena_size: Optional[Tuple[float, float]] = None,
        episode_duration_seconds: Optional[float] = None,
        evader_policy: str = C.EVADER_MIXED,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs debe ser >= 1")
        if evader_policy != C.EVADER_MIXED and evader_policy not in C.EVADER_POLICIES:
            raise ValueError(
                f"evader_policy inválida: {evader_policy!r}. "
                f"Usa {C.EVADER_MIXED!r} o una de {C.EVADER_POLICIES}."
            )
        if episode_duration_seconds is not None and episode_duration_seconds <= 0:
            raise ValueError("episode_duration_seconds debe ser positivo")

        self.num_envs = int(num_envs)
        self.fixed_arena_size = arena_size
        self.fixed_duration_seconds = episode_duration_seconds
        self.evader_policy_request = evader_policy
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        n = self.num_envs
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(C.OBS_GEOMETRY_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n, C.OBS_GEOMETRY_DIM), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(C.NUM_ACTIONS)
        self.action_space = spaces.MultiDiscrete(
            np.full(n, C.NUM_ACTIONS, dtype=np.int64)
        )

        # Estado vectorizado.
        self.arena_width = np.zeros(n, dtype=np.float64)
        self.arena_height = np.zeros(n, dtype=np.float64)
        self.min_x = np.zeros(n, dtype=np.float64)
        self.max_x = np.zeros(n, dtype=np.float64)
        self.min_y = np.zeros(n, dtype=np.float64)
        self.max_y = np.zeros(n, dtype=np.float64)
        self.duration_seconds = np.zeros(n, dtype=np.float64)
        self.max_frames = np.ones(n, dtype=np.int64)
        self.frames = np.zeros(n, dtype=np.int64)

        self.player_x = np.zeros(n, dtype=np.float64)
        self.player_y = np.zeros(n, dtype=np.float64)
        self.player_vx = np.zeros(n, dtype=np.float64)
        self.player_vy = np.zeros(n, dtype=np.float64)

        self.agent_x = np.zeros(n, dtype=np.float64)
        self.agent_y = np.zeros(n, dtype=np.float64)
        self.last_agent_move_x = np.zeros(n, dtype=np.float64)
        self.last_agent_move_y = np.zeros(n, dtype=np.float64)

        self.policy_code = np.zeros(n, dtype=np.int8)
        self.policy_names: List[str] = [C.EVADER_STATIC] * n
        self.tilt_accel = np.zeros((n, 2), dtype=np.float64)
        self.tilt_max_magnitude = np.full(n, C.EVADER_ACCEL_MAX, dtype=np.float64)
        self.flee_lateral_remaining = np.zeros(n, dtype=np.int64)
        self.flee_lateral_sign = np.ones(n, dtype=np.float64)

        # Deja el objeto listo para step() aunque el usuario no llame reset().
        self._reset_indices(np.arange(n))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_indices(np.arange(self.num_envs))
        return self._obs(), [
            self._info(i, captured=False) for i in range(self.num_envs)
        ]

    def _reset_indices(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        rng = self.rng

        if self.fixed_arena_size is None:
            self.arena_width[indices] = rng.uniform(
                C.ARENA_WIDTH_MIN, C.ARENA_WIDTH_MAX, size=indices.size
            )
            self.arena_height[indices] = rng.uniform(
                C.ARENA_HEIGHT_MIN, C.ARENA_HEIGHT_MAX, size=indices.size
            )
        else:
            self.arena_width[indices] = float(self.fixed_arena_size[0])
            self.arena_height[indices] = float(self.fixed_arena_size[1])

        self.min_x[indices] = C.BORDER_WIDTH + C.ROBOT_RADIUS
        self.max_x[indices] = self.arena_width[indices] - C.BORDER_WIDTH - C.ROBOT_RADIUS
        self.min_y[indices] = C.BORDER_WIDTH + C.ROBOT_RADIUS
        self.max_y[indices] = self.arena_height[indices] - C.BORDER_WIDTH - C.ROBOT_RADIUS
        if np.any(self.min_x[indices] >= self.max_x[indices]) or np.any(
            self.min_y[indices] >= self.max_y[indices]
        ):
            raise ValueError(
                "arena_size demasiado pequeña para borderWidth + robotRadius"
            )

        if self.fixed_duration_seconds is None:
            choices = np.asarray(C.EPISODE_DURATION_CHOICES, dtype=np.float64)
            self.duration_seconds[indices] = rng.choice(choices, size=indices.size)
        else:
            self.duration_seconds[indices] = float(self.fixed_duration_seconds)
        self.max_frames[indices] = np.maximum(
            1, np.rint(self.duration_seconds[indices] * C.FPS).astype(np.int64)
        )
        self.frames[indices] = 0

        # Política por episodio.
        if self.evader_policy_request == C.EVADER_MIXED:
            probabilities = np.asarray(
                [probability for _, probability in C.EVADER_TRAIN_MIX],
                dtype=np.float64,
            )
            names = [policy for policy, _ in C.EVADER_TRAIN_MIX]
            chosen = rng.choice(
                np.arange(len(names)), size=indices.size, p=probabilities
            )
            for local, env_index in enumerate(indices):
                self.policy_names[int(env_index)] = names[int(chosen[local])]
        else:
            for env_index in indices:
                self.policy_names[int(env_index)] = self.evader_policy_request
        self.policy_code[indices] = np.asarray(
            [_POLICY_TO_CODE[self.policy_names[int(i)]] for i in indices],
            dtype=np.int8,
        )

        # Estado del evasor.
        self.tilt_accel[indices] = 0.0
        self.tilt_max_magnitude[indices] = rng.uniform(
            C.EVADER_ACCEL_MIN, C.EVADER_ACCEL_MAX, size=indices.size
        )
        self.flee_lateral_remaining[indices] = 0
        self.flee_lateral_sign[indices] = np.where(
            rng.random(indices.size) < 0.5, 1.0, -1.0
        )

        # Jugador: (300, 400) con probabilidad 0.5, si no aleatorio.
        fixed_player = rng.random(indices.size) < C.PLAYER_FIXED_START_PROB
        player_x = rng.uniform(
            self.min_x[indices], self.max_x[indices], size=indices.size
        )
        player_y = rng.uniform(
            self.min_y[indices], self.max_y[indices], size=indices.size
        )
        player_x[fixed_player] = np.clip(
            C.PLAYER_START_X,
            self.min_x[indices][fixed_player],
            self.max_x[indices][fixed_player],
        )
        player_y[fixed_player] = np.clip(
            C.PLAYER_START_Y,
            self.min_y[indices][fixed_player],
            self.max_y[indices][fixed_player],
        )
        self.player_x[indices] = player_x
        self.player_y[indices] = player_y
        self.player_vx[indices] = 0.0
        self.player_vy[indices] = 0.0

        # Agente aleatorio con distancia inicial >= 4 radio.
        for env_index in indices:
            i = int(env_index)
            self.agent_x[i], self.agent_y[i] = self._sample_agent_position(i)

        self.last_agent_move_x[indices] = 0.0
        self.last_agent_move_y[indices] = 0.0

    def _sample_agent_position(self, index: int) -> Tuple[float, float]:
        for _ in range(10_000):
            x = float(self.rng.uniform(self.min_x[index], self.max_x[index]))
            y = float(self.rng.uniform(self.min_y[index], self.max_y[index]))
            if (
                math.hypot(
                    x - self.player_x[index], y - self.player_y[index]
                )
                >= C.MIN_INITIAL_DISTANCE
            ):
                return x, y
        corners = (
            (self.min_x[index], self.min_y[index]),
            (self.min_x[index], self.max_y[index]),
            (self.max_x[index], self.min_y[index]),
            (self.max_x[index], self.max_y[index]),
        )
        return max(
            corners,
            key=lambda p: math.hypot(
                p[0] - self.player_x[index], p[1] - self.player_y[index]
            ),
        )

    # ------------------------------------------------------------------
    # Step vectorizado
    # ------------------------------------------------------------------
    def step(
        self, actions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        actions_array = np.asarray(actions, dtype=np.int64)
        if actions_array.shape != (self.num_envs,):
            raise ValueError(
                f"actions debe tener forma ({self.num_envs},), "
                f"recibido {actions_array.shape}"
            )
        if np.any(actions_array < 0) or np.any(actions_array >= C.NUM_ACTIONS):
            raise ValueError("acción fuera de [0, 3]")

        distance_prev = self._distance()
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        captured = np.zeros(self.num_envs, dtype=bool)
        active = np.ones(self.num_envs, dtype=bool)

        for _ in range(C.ACTION_REPEAT):
            if not active.any():
                break
            idx = np.flatnonzero(active)
            self.frames[idx] += 1

            # (1) Jugador.
            accel_x, accel_y = self._evader_accel(idx)
            self.player_vx[idx] += -accel_x * C.PLAYER_ACCEL_SENSITIVITY
            self.player_vy[idx] += accel_y * C.PLAYER_ACCEL_SENSITIVITY
            self.player_vx[idx] *= C.PLAYER_FRICTION
            self.player_vy[idx] *= C.PLAYER_FRICTION

            next_x = self.player_x[idx] + self.player_vx[idx]
            next_y = self.player_y[idx] + self.player_vy[idx]
            low_x = next_x <= self.min_x[idx]
            high_x = next_x >= self.max_x[idx]
            low_y = next_y <= self.min_y[idx]
            high_y = next_y >= self.max_y[idx]
            next_x = np.clip(next_x, self.min_x[idx], self.max_x[idx])
            next_y = np.clip(next_y, self.min_y[idx], self.max_y[idx])
            self.player_vx[idx] = np.where(
                low_x | high_x,
                -self.player_vx[idx] * C.WALL_ELASTICITY,
                self.player_vx[idx],
            )
            self.player_vy[idx] = np.where(
                low_y | high_y,
                -self.player_vy[idx] * C.WALL_ELASTICITY,
                self.player_vy[idx],
            )
            self.player_x[idx] = next_x
            self.player_y[idx] = next_y

            # (2) Agente: 12 px + clamp.
            dirs = np.asarray(C.ACTION_VECTORS, dtype=np.float64)[actions_array[idx]]
            old_x = self.agent_x[idx]
            old_y = self.agent_y[idx]
            new_x = np.clip(old_x + dirs[:, 0], self.min_x[idx], self.max_x[idx])
            new_y = np.clip(old_y + dirs[:, 1], self.min_y[idx], self.max_y[idx])
            moved_x = new_x - old_x
            moved_y = new_y - old_y
            self.agent_x[idx] = new_x
            self.agent_y[idx] = new_y
            moved = np.hypot(moved_x, moved_y) > 1e-9
            if moved.any():
                moved_idx = idx[moved]
                self.last_agent_move_x[moved_idx] = moved_x[moved]
                self.last_agent_move_y[moved_idx] = moved_y[moved]

            # (3) Colisión en este frame.
            distance_now = self._distance()
            newly_captured = distance_now[idx] <= C.CAPTURE_DISTANCE
            if newly_captured.any():
                captured_idx = idx[newly_captured]
                captured[captured_idx] = True
                terminated[captured_idx] = True
                active[captured_idx] = False

            # (4) Tiempo: solo para los que siguen activos y no capturaron.
            active_idx = np.flatnonzero(active)
            timed_out = self.frames[active_idx] >= self.max_frames[active_idx]
            if timed_out.any():
                timeout_idx = active_idx[timed_out]
                truncated[timeout_idx] = True
                active[timeout_idx] = False

        distance_now = self._distance()
        diagonal = np.hypot(self.arena_width, self.arena_height)
        rewards = (
            C.REWARD_DISTANCE_SCALE
            * (distance_prev - distance_now)
            / diagonal
            - C.REWARD_STEP_PENALTY
        )
        rewards[terminated] = C.REWARD_CAPTURE
        rewards[truncated] = C.REWARD_TIMEOUT

        obs = self._obs()
        infos = [
            self._info(i, captured=bool(captured[i]))
            for i in range(self.num_envs)
        ]

        # Autoreset inmediato.
        done = terminated | truncated
        if done.any():
            done_indices = np.flatnonzero(done)
            final_obs = obs[done_indices].copy()
            for local, env_index in enumerate(done_indices):
                infos[int(env_index)]["final_observation"] = final_obs[local]
            self._reset_indices(done_indices)
            obs = self._obs()

        return obs, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Evasor vectorizado
    # ------------------------------------------------------------------
    def _evader_accel(
        self, indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng
        accel_x = np.zeros(indices.size, dtype=np.float64)
        accel_y = np.zeros(indices.size, dtype=np.float64)
        if indices.size == 0:
            return accel_x, accel_y

        codes = self.policy_code[indices]

        # --- tilt_random (OU) -------------------------------------------
        tilt_local = np.flatnonzero(codes == _POLICY_TO_CODE[C.EVADER_TILT_RANDOM])
        if tilt_local.size:
            tilt_idx = indices[tilt_local]
            noise = rng.normal(0.0, 1.0, size=(tilt_local.size, 2))
            self.tilt_accel[tilt_idx] = (
                self.tilt_accel[tilt_idx]
                + C.TILT_OU_THETA * (0.0 - self.tilt_accel[tilt_idx])
                + C.TILT_OU_SIGMA * noise
            )
            magnitudes = np.linalg.norm(self.tilt_accel[tilt_idx], axis=1)
            too_big = magnitudes > self.tilt_max_magnitude[tilt_idx]
            if too_big.any():
                scale = (
                    self.tilt_max_magnitude[tilt_idx[too_big]]
                    / magnitudes[too_big]
                )
                self.tilt_accel[tilt_idx[too_big]] *= scale[:, None]
            accel_x[tilt_local] = self.tilt_accel[tilt_idx, 0]
            accel_y[tilt_local] = self.tilt_accel[tilt_idx, 1]

        # --- flee --------------------------------------------------------
        flee_local = np.flatnonzero(codes == _POLICY_TO_CODE[C.EVADER_FLEE])
        if flee_local.size:
            flee_idx = indices[flee_local]
            away_x = self.player_x[flee_idx] - self.agent_x[flee_idx]
            away_y = self.player_y[flee_idx] - self.agent_y[flee_idx]
            away_norm = np.hypot(away_x, away_y)
            dir_x = np.divide(
                away_x, away_norm, out=np.zeros_like(away_x), where=away_norm > 1e-9
            )
            dir_y = np.divide(
                away_y, away_norm, out=np.zeros_like(away_y), where=away_norm > 1e-9
            )
            no_direction = away_norm <= 1e-9
            if no_direction.any():
                angles = rng.uniform(0.0, 2.0 * math.pi, size=int(no_direction.sum()))
                dir_x[no_direction] = np.cos(angles)
                dir_y[no_direction] = np.sin(angles)

            perpendicular_x = -dir_y
            perpendicular_y = dir_x

            lateral = self.flee_lateral_remaining[flee_idx] > 0
            if lateral.any():
                lateral_local = np.flatnonzero(lateral)
                sign = self.flee_lateral_sign[flee_idx[lateral]]
                dir_x[lateral_local] = perpendicular_x[lateral_local] * sign
                dir_y[lateral_local] = perpendicular_y[lateral_local] * sign
                self.flee_lateral_remaining[flee_idx[lateral]] -= 1

            can_trigger = ~lateral
            trigger = can_trigger & (
                rng.random(flee_local.size) < C.FLEE_LATERAL_PROB
            )
            if trigger.any():
                trigger_local = np.flatnonzero(trigger)
                trigger_idx = flee_idx[trigger_local]
                self.flee_lateral_remaining[trigger_idx] = C.FLEE_LATERAL_FRAMES
                self.flee_lateral_sign[trigger_idx] = np.where(
                    rng.random(trigger_local.size) < 0.5, 1.0, -1.0
                )
                sign = self.flee_lateral_sign[trigger_idx]
                dir_x[trigger_local] = perpendicular_x[trigger_local] * sign
                dir_y[trigger_local] = perpendicular_y[trigger_local] * sign

            magnitude = rng.uniform(
                C.EVADER_ACCEL_MIN, C.EVADER_ACCEL_MAX, size=flee_local.size
            )
            noise = rng.normal(0.0, C.FLEE_NOISE_STD, size=(flee_local.size, 2))
            accel_x[flee_local] = dir_x * magnitude + noise[:, 0]
            accel_y[flee_local] = dir_y * magnitude + noise[:, 1]

        return accel_x, accel_y

    # ------------------------------------------------------------------
    # Observación e info
    # ------------------------------------------------------------------
    def _distance(self) -> np.ndarray:
        return np.hypot(
            self.player_x - self.agent_x,
            self.player_y - self.agent_y,
        )

    def _obs(self) -> np.ndarray:
        agent_move_norm = np.hypot(
            self.last_agent_move_x, self.last_agent_move_y
        )
        agent_theta = np.where(
            agent_move_norm < 1e-9,
            0.0,
            np.arctan2(self.last_agent_move_y, self.last_agent_move_x) / math.pi,
        )
        player_speed = np.hypot(self.player_vx, self.player_vy)
        player_theta = np.where(
            player_speed < C.PLAYER_VELOCITY_EPS,
            0.0,
            np.arctan2(self.player_vy, self.player_vx) / math.pi,
        )
        obs = np.column_stack(
            [
                2.0 * self.agent_x / self.arena_width - 1.0,
                2.0 * self.agent_y / self.arena_height - 1.0,
                agent_theta,
                2.0 * self.player_x / self.arena_width - 1.0,
                2.0 * self.player_y / self.arena_height - 1.0,
                player_theta,
            ]
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)

    def _info(self, index: int, captured: bool) -> Dict[str, Any]:
        return {
            "captured": bool(captured),
            "frames": int(self.frames[index]),
            "distance": float(
                math.hypot(
                    self.player_x[index] - self.agent_x[index],
                    self.player_y[index] - self.agent_y[index],
                )
            ),
            "evader_policy": str(self.policy_names[index]),
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    @property
    def global_frames(self) -> int:
        """Frames simulados acumulados en todos los entornos."""
        return int(self.frames.sum())

    def close(self) -> None:
        """No hay recursos externos que liberar."""
