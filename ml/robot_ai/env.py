"""Entorno Gymnasium que replica la física de ``GameScreen.kt``.

El entorno expone SOLO la geometría de 6 entradas. Las entradas 6-7 del agente
(codificación de la acción anterior y delta TD anterior) las añade el agente,
porque dependen del crítico y no del simulador.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C


class RobotPvpEnv(gym.Env):
    """Duelo 2D: robot verde (evasor) contra robot rojo (agente).

    Parameters
    ----------
    arena_size:
        ``(width, height)`` fijo en píxeles. Si es ``None`` se muestrea un
        tamaño por episodio dentro de los rangos de ``config.py``.
    episode_duration_seconds:
        Duración fija del episodio. Si es ``None`` se elige al azar entre
        ``config.EPISODE_DURATION_CHOICES`` (30/45/60 s).
    evader_policy:
        ``"mixed"`` (mezcla de entrenamiento), ``"flee"``, ``"tilt_random"`` o
        ``"static"``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        arena_size: Optional[Tuple[float, float]] = None,
        episode_duration_seconds: Optional[float] = None,
        evader_policy: str = C.EVADER_MIXED,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if evader_policy != C.EVADER_MIXED and evader_policy not in C.EVADER_POLICIES:
            raise ValueError(
                f"evader_policy inválida: {evader_policy!r}. "
                f"Usa {C.EVADER_MIXED!r} o una de {C.EVADER_POLICIES}."
            )
        if arena_size is not None:
            if arena_size[0] <= 0 or arena_size[1] <= 0:
                raise ValueError("arena_size debe tener dimensiones positivas")
        if episode_duration_seconds is not None and episode_duration_seconds <= 0:
            raise ValueError("episode_duration_seconds debe ser positivo")

        self.fixed_arena_size = arena_size
        self.fixed_duration_seconds = episode_duration_seconds
        self.evader_policy_request = evader_policy
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(C.NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(C.OBS_GEOMETRY_DIM,),
            dtype=np.float32,
        )

        # Estado que se define en reset(); se declara aquí para los type hints.
        self.arena_width = 0.0
        self.arena_height = 0.0
        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.duration_seconds = 0.0
        self.max_frames = 0
        self.frames = 0
        self.player_x = 0.0
        self.player_y = 0.0
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.evader_policy = C.EVADER_STATIC
        self.last_action: Optional[int] = None
        self._last_agent_move_x = 0.0
        self._last_agent_move_y = 0.0
        self._tilt_accel = np.zeros(2, dtype=np.float64)
        self._tilt_max_magnitude = C.EVADER_ACCEL_MAX
        self._flee_lateral_remaining = 0
        self._flee_lateral_sign = 1.0

    # ------------------------------------------------------------------
    # Propiedades de inspección (tests, depuración, evaluación)
    # ------------------------------------------------------------------
    @property
    def player_position(self) -> Tuple[float, float]:
        return self.player_x, self.player_y

    @property
    def agent_position(self) -> Tuple[float, float]:
        return self.agent_x, self.agent_y

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_x, max_x, min_y, max_y)."""
        return self.min_x, self.max_x, self.min_y, self.max_y

    @property
    def distance(self) -> float:
        return self._distance()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        if self.fixed_arena_size is None:
            self.arena_width = float(
                rng.uniform(C.ARENA_WIDTH_MIN, C.ARENA_WIDTH_MAX)
            )
            self.arena_height = float(
                rng.uniform(C.ARENA_HEIGHT_MIN, C.ARENA_HEIGHT_MAX)
            )
        else:
            self.arena_width = float(self.fixed_arena_size[0])
            self.arena_height = float(self.fixed_arena_size[1])

        self.min_x = C.BORDER_WIDTH + C.ROBOT_RADIUS
        self.max_x = self.arena_width - C.BORDER_WIDTH - C.ROBOT_RADIUS
        self.min_y = C.BORDER_WIDTH + C.ROBOT_RADIUS
        self.max_y = self.arena_height - C.BORDER_WIDTH - C.ROBOT_RADIUS
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError(
                "arena_size demasiado pequeña para borderWidth + robotRadius"
            )

        if self.fixed_duration_seconds is None:
            self.duration_seconds = float(
                rng.choice(np.asarray(C.EPISODE_DURATION_CHOICES, dtype=np.float64))
            )
        else:
            self.duration_seconds = float(self.fixed_duration_seconds)
        self.max_frames = max(1, int(round(self.duration_seconds * C.FPS)))
        self.frames = 0

        self.evader_policy = self._select_evader_policy()
        self._reset_evader_state()

        # Jugador: (300, 400) con probabilidad 0.5, si no, posición aleatoria.
        if rng.random() < C.PLAYER_FIXED_START_PROB:
            self.player_x = float(
                np.clip(C.PLAYER_START_X, self.min_x, self.max_x)
            )
            self.player_y = float(
                np.clip(C.PLAYER_START_Y, self.min_y, self.max_y)
            )
        else:
            self.player_x = float(rng.uniform(self.min_x, self.max_x))
            self.player_y = float(rng.uniform(self.min_y, self.max_y))
        self.player_vx = 0.0
        self.player_vy = 0.0

        # Agente: posición aleatoria con distancia inicial >= 4 radio.
        agent_x, agent_y = self._sample_agent_position()
        self.agent_x = agent_x
        self.agent_y = agent_y

        self.last_action = None
        self._last_agent_move_x = 0.0
        self._last_agent_move_y = 0.0

        return self._obs(), self._info(captured=False)

    def _select_evader_policy(self) -> str:
        if self.evader_policy_request != C.EVADER_MIXED:
            return self.evader_policy_request
        u = float(self.np_random.random())
        cumulative = 0.0
        for policy, probability in C.EVADER_TRAIN_MIX:
            cumulative += probability
            if u < cumulative:
                return policy
        return C.EVADER_TRAIN_MIX[-1][0]

    def _reset_evader_state(self) -> None:
        self._tilt_accel = np.zeros(2, dtype=np.float64)
        self._tilt_max_magnitude = float(
            self.np_random.uniform(C.EVADER_ACCEL_MIN, C.EVADER_ACCEL_MAX)
        )
        self._flee_lateral_remaining = 0
        self._flee_lateral_sign = (
            1.0 if self.np_random.random() < 0.5 else -1.0
        )

    def _sample_agent_position(self) -> Tuple[float, float]:
        rng = self.np_random
        for _ in range(10_000):
            x = float(rng.uniform(self.min_x, self.max_x))
            y = float(rng.uniform(self.min_y, self.max_y))
            if math.hypot(x - self.player_x, y - self.player_y) >= C.MIN_INITIAL_DISTANCE:
                return x, y

        # Fallback determinista: la esquina más lejana al jugador.
        corners = (
            (self.min_x, self.min_y),
            (self.min_x, self.max_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y),
        )
        return max(
            corners,
            key=lambda p: math.hypot(p[0] - self.player_x, p[1] - self.player_y),
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"acción inválida: {action}")

        self.last_action = action
        distance_prev = self._distance()
        terminated = False
        truncated = False
        captured = False

        for _ in range(C.ACTION_REPEAT):
            self.frames += 1

            # (1) Jugador con la política de evasor.
            self._update_player()

            # (2) Agente: 12 px en la dirección y clamp a la arena.
            old_agent_x = self.agent_x
            old_agent_y = self.agent_y
            dir_x, dir_y = C.ACTION_VECTORS[action]
            self.agent_x = float(
                np.clip(self.agent_x + dir_x, self.min_x, self.max_x)
            )
            self.agent_y = float(
                np.clip(self.agent_y + dir_y, self.min_y, self.max_y)
            )
            moved_x = self.agent_x - old_agent_x
            moved_y = self.agent_y - old_agent_y
            if math.hypot(moved_x, moved_y) > 1e-9:
                self._last_agent_move_x = moved_x
                self._last_agent_move_y = moved_y

            # (3) Colisión en cada frame.
            distance_now = self._distance()
            if distance_now <= C.CAPTURE_DISTANCE:
                captured = True
                terminated = True
                break

            # (4) Descontar tiempo (1 frame = DT).
            if self.frames >= self.max_frames:
                truncated = True
                break

        distance_now = self._distance()
        if terminated:
            reward = C.REWARD_CAPTURE
        elif truncated:
            reward = C.REWARD_TIMEOUT
        else:
            diagonal = C.arena_diagonal(self.arena_width, self.arena_height)
            reward = (
                C.REWARD_DISTANCE_SCALE
                * (distance_prev - distance_now)
                / diagonal
                - C.REWARD_STEP_PENALTY
            )

        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            self._info(captured=captured),
        )

    def _update_player(self) -> None:
        """Un frame de física del jugador, idéntico a GameScreen.kt."""
        accel_x, accel_y = self._evader_accel()

        self.player_vx += -accel_x * C.PLAYER_ACCEL_SENSITIVITY
        self.player_vy += accel_y * C.PLAYER_ACCEL_SENSITIVITY

        self.player_vx *= C.PLAYER_FRICTION
        self.player_vy *= C.PLAYER_FRICTION

        next_x = self.player_x + self.player_vx
        next_y = self.player_y + self.player_vy

        if next_x <= self.min_x:
            next_x = self.min_x
            self.player_vx = -self.player_vx * C.WALL_ELASTICITY
        elif next_x >= self.max_x:
            next_x = self.max_x
            self.player_vx = -self.player_vx * C.WALL_ELASTICITY

        if next_y <= self.min_y:
            next_y = self.min_y
            self.player_vy = -self.player_vy * C.WALL_ELASTICITY
        elif next_y >= self.max_y:
            next_y = self.max_y
            self.player_vy = -self.player_vy * C.WALL_ELASTICITY

        self.player_x = float(next_x)
        self.player_y = float(next_y)

    def _evader_accel(self) -> Tuple[float, float]:
        """Devuelve (accelX, accelY) en m/s² de la política de evasión."""
        rng = self.np_random

        if self.evader_policy == C.EVADER_STATIC:
            return 0.0, 0.0

        if self.evader_policy == C.EVADER_TILT_RANDOM:
            # Proceso de Ornstein-Uhlenbeck 2D, suavizado y acotado.
            noise = rng.normal(0.0, 1.0, size=2)
            self._tilt_accel = (
                self._tilt_accel
                + C.TILT_OU_THETA * (0.0 - self._tilt_accel)
                + C.TILT_OU_SIGMA * noise
            )
            magnitude = float(np.linalg.norm(self._tilt_accel))
            if magnitude > self._tilt_max_magnitude:
                self._tilt_accel *= self._tilt_max_magnitude / magnitude
            return float(self._tilt_accel[0]), float(self._tilt_accel[1])

        if self.evader_policy == C.EVADER_FLEE:
            away_x = self.player_x - self.agent_x
            away_y = self.player_y - self.agent_y
            away_norm = math.hypot(away_x, away_y)
            if away_norm < 1e-9:
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                dir_x, dir_y = math.cos(angle), math.sin(angle)
            else:
                dir_x, dir_y = away_x / away_norm, away_y / away_norm

            if self._flee_lateral_remaining > 0:
                # Perpendicular a la dirección de huida.
                dir_x, dir_y = -dir_y, dir_x
                dir_x *= self._flee_lateral_sign
                dir_y *= self._flee_lateral_sign
                self._flee_lateral_remaining -= 1
            elif float(rng.random()) < C.FLEE_LATERAL_PROB:
                self._flee_lateral_remaining = C.FLEE_LATERAL_FRAMES
                self._flee_lateral_sign = (
                    1.0 if rng.random() < 0.5 else -1.0
                )
                dir_x, dir_y = -dir_y, dir_x
                dir_x *= self._flee_lateral_sign
                dir_y *= self._flee_lateral_sign

            magnitude = float(
                rng.uniform(C.EVADER_ACCEL_MIN, C.EVADER_ACCEL_MAX)
            )
            noise = rng.normal(0.0, C.FLEE_NOISE_STD, size=2)
            accel_x = dir_x * magnitude + float(noise[0])
            accel_y = dir_y * magnitude + float(noise[1])
            return accel_x, accel_y

        # Política desconocida: no debería ocurrir.
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Observación e info
    # ------------------------------------------------------------------
    def _distance(self) -> float:
        return math.hypot(
            self.player_x - self.agent_x,
            self.player_y - self.agent_y,
        )

    def _obs(self) -> np.ndarray:
        agent_move_norm = math.hypot(
            self._last_agent_move_x, self._last_agent_move_y
        )
        if agent_move_norm < 1e-9:
            agent_theta = 0.0
        else:
            agent_theta = (
                math.atan2(self._last_agent_move_y, self._last_agent_move_x)
                / math.pi
            )

        player_speed = math.hypot(self.player_vx, self.player_vy)
        if player_speed < C.PLAYER_VELOCITY_EPS:
            player_theta = 0.0
        else:
            player_theta = math.atan2(self.player_vy, self.player_vx) / math.pi

        obs = np.array(
            [
                2.0 * self.agent_x / self.arena_width - 1.0,
                2.0 * self.agent_y / self.arena_height - 1.0,
                agent_theta,
                2.0 * self.player_x / self.arena_width - 1.0,
                2.0 * self.player_y / self.arena_height - 1.0,
                player_theta,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)

    def _info(self, captured: bool) -> Dict[str, Any]:
        return {
            "captured": bool(captured),
            "frames": int(self.frames),
            "distance": float(self._distance()),
            "evader_policy": str(self.evader_policy),
        }
