"""Fuente única de verdad de constantes para el entorno y el agente RobotPVP.

Todas las constantes de la especificación (física, recompensa, red, acciones y
rangos de arena) viven aquí. El exporter del modelo debe copiarlas al bloque
``meta`` del JSON; Kotlin debe leerlas desde ese JSON, nunca duplicarlas a mano.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Bucle temporal
# ---------------------------------------------------------------------------
FPS = 60
DT = 1.0 / FPS

# ---------------------------------------------------------------------------
# Arena y geometría
# ---------------------------------------------------------------------------
ROBOT_RADIUS = 60.0
BORDER_WIDTH = 20.0

ARENA_WIDTH_MIN = 900.0
ARENA_WIDTH_MAX = 1200.0
ARENA_HEIGHT_MIN = 1700.0
ARENA_HEIGHT_MAX = 2300.0

# Arena fija recomendada para evaluación / depuración.
EVAL_ARENA_WIDTH = 1080.0
EVAL_ARENA_HEIGHT = 1920.0

CAPTURE_DISTANCE = 2.0 * ROBOT_RADIUS

# ---------------------------------------------------------------------------
# Posiciones iniciales
# ---------------------------------------------------------------------------
PLAYER_START_X = 300.0
PLAYER_START_Y = 400.0
PLAYER_FIXED_START_PROB = 0.5
MIN_INITIAL_DISTANCE_FACTOR = 4.0
MIN_INITIAL_DISTANCE = MIN_INITIAL_DISTANCE_FACTOR * ROBOT_RADIUS

# ---------------------------------------------------------------------------
# Física del jugador (idéntica a GameScreen.kt)
# ---------------------------------------------------------------------------
PLAYER_ACCEL_SENSITIVITY = 0.15
PLAYER_FRICTION = 0.985
WALL_ELASTICITY = 0.75
PLAYER_VELOCITY_EPS = 0.01

# ---------------------------------------------------------------------------
# Agente (robot rojo)
# ---------------------------------------------------------------------------
AGENT_SPEED = 12.0
ACTION_REPEAT = 4
NUM_ACTIONS = 4

# Orden canónico de acciones: [UP, DOWN, LEFT, RIGHT]. Eje y hacia abajo.
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_VECTORS = (
    (0.0, -AGENT_SPEED),   # UP
    (0.0, AGENT_SPEED),    # DOWN
    (-AGENT_SPEED, 0.0),   # LEFT
    (AGENT_SPEED, 0.0),    # RIGHT
)

# Codificación escalar de acciones para la entrada 6 del agente.
ACTION_ENCODING_NONE = 0.0
ACTION_ENCODINGS = {
    "none": ACTION_ENCODING_NONE,
    "UP": -1.0,
    "DOWN": -0.5,
    "LEFT": 0.5,
    "RIGHT": 1.0,
}

# ---------------------------------------------------------------------------
# Episodios
# ---------------------------------------------------------------------------
EPISODE_DURATION_CHOICES = (30, 45, 60)  # segundos

# ---------------------------------------------------------------------------
# Recompensa y descuento
# ---------------------------------------------------------------------------
GAMMA = 0.99
REWARD_DISTANCE_SCALE = 2.0
REWARD_STEP_PENALTY = 0.001
REWARD_CAPTURE = 1.0
REWARD_TIMEOUT = -1.0


def arena_diagonal(width: float, height: float) -> float:
    """Diagonal de la arena usada para normalizar la recompensa por distancia."""
    return math.hypot(width, height)


# ---------------------------------------------------------------------------
# Políticas del evasor (simulan al jugador humano inclinando el teléfono)
# ---------------------------------------------------------------------------
EVADER_FLEE = "flee"
EVADER_TILT_RANDOM = "tilt_random"
EVADER_STATIC = "static"
EVADER_POLICIES = (EVADER_FLEE, EVADER_TILT_RANDOM, EVADER_STATIC)
EVADER_MIXED = "mixed"

# Mezcla por defecto en entrenamiento: 50 % flee, 40 % tilt_random, 10 % static.
EVADER_TRAIN_MIX = (
    (EVADER_FLEE, 0.50),
    (EVADER_TILT_RANDOM, 0.40),
    (EVADER_STATIC, 0.10),
)

EVADER_ACCEL_MIN = 1.0
EVADER_ACCEL_MAX = 4.0

# Ornstein-Uhlenbeck para "tilt_random" (por frame).
TILT_OU_THETA = 0.12
TILT_OU_SIGMA = 0.35

# "flee": ruido y cambio lateral para no quedarse atascado en una esquina.
FLEE_NOISE_STD = 0.30
FLEE_LATERAL_PROB = 0.02
FLEE_LATERAL_FRAMES = 30

# ---------------------------------------------------------------------------
# Observación y red
# ---------------------------------------------------------------------------
OBS_GEOMETRY_DIM = 6       # entradas 0-5 que produce el entorno
OBS_DISCOVERY_DIM = 2      # entradas 6-7 que añade el agente (memoria t-1)
OBS_DIM = OBS_GEOMETRY_DIM + OBS_DISCOVERY_DIM

INPUT_DIM = OBS_DIM
HIDDEN_DIM = 16
ACTOR_OUTPUT_DIM = NUM_ACTIONS
CRITIC_OUTPUT_DIM = 1

# Orden canónico de las 8 entradas del agente (ver AGENTS.md).
INPUT_NAMES = (
    "x_agent",
    "y_agent",
    "theta_agent",
    "x_enemy",
    "y_enemy",
    "theta_enemy",
    "a_prev",
    "delta_prev",
)
PARAM_COUNT = (
    (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM)
    + (HIDDEN_DIM * ACTOR_OUTPUT_DIM + ACTOR_OUTPUT_DIM)
    + (HIDDEN_DIM * CRITIC_OUTPUT_DIM + CRITIC_OUTPUT_DIM)
)

# Clave del bloque de metadatos en el JSON exportado.
MODEL_META_KEY = "meta"

VECTORIZED_DEFAULT_NUM_ENVS = 8
