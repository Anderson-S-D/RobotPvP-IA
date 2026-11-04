#!/usr/bin/env python3
"""Evaluación de los cinco agentes de RobotPVP en un escenario fijo.

Escenario:
    arena fija 1080 x 1920, duraciones 30/45/60 s y políticas de evasor
    ``flee`` y ``tilt_random``. Las semillas de episodio son idénticas para
    todos los agentes, de modo que las posiciones iniciales y el ruido del
    evasor de cada episodio son los mismos.

Agentes:
    random_target  réplica del robot rojo actual (objetivo aleatorio a 12 px/frame)
    random_action  acción uniforme cada ``action_repeat`` frames
    ac_base        actor-crítico sin descubrimiento
    ac_disc        actor-crítico con Componente de Descubrimiento
    greedy_chase   heurística por el eje de mayor diferencia (referencia superior)

Uso:
    python evaluate.py --episodes 300
    python evaluate.py --episodes 30 --disc-checkpoints runs/ac_disc_s1/last.pt
    python evaluate.py --greedy --episodes 300

Salidas por defecto en ``ml/results/``:
    eval_episodes.csv        una fila por episodio y agente
    eval_table.csv           resumen por agente, duración y política
    eval_by_seed.csv         resumen por agente, semilla de entrenamiento, ...
    discovery_samples.csv    muestras (δ_{t-1}, ¿repitió acción?) de ac_disc
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from robot_ai import config as C
from robot_ai.discovery import DiscoveryMemory, encode_actions
from robot_ai.env import RobotPvpEnv
from robot_ai.model import SharedActorCritic

ML_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ML_DIR / "results"

AGENT_RANDOM_TARGET = "random_target"
AGENT_RANDOM_ACTION = "random_action"
AGENT_AC_BASE = "ac_base"
AGENT_AC_DISC = "ac_disc"
AGENT_GREEDY_CHASE = "greedy_chase"

AGENT_ORDER = (
    AGENT_RANDOM_TARGET,
    AGENT_RANDOM_ACTION,
    AGENT_AC_BASE,
    AGENT_AC_DISC,
    AGENT_GREEDY_CHASE,
)

AGENT_LABELS_ES = {
    AGENT_RANDOM_TARGET: "Objetivo aleatorio",
    AGENT_RANDOM_ACTION: "Acción aleatoria",
    AGENT_AC_BASE: "AC sin descubrimiento",
    AGENT_AC_DISC: "AC con descubrimiento",
    AGENT_GREEDY_CHASE: "Persecución voraz",
}

# Flujo de números aleatorios propio de las políticas aleatorias. Se deriva de
# la semilla del episodio para que no dependa de la física del entorno.
_RANDOM_ACTION_STREAM = 0xBADC0DE
_RANDOM_TARGET_STREAM = 0xC0FFEE


@dataclass(frozen=True)
class EpisodeResult:
    agent: str
    checkpoint_seed: int
    duration_s: float
    evader_policy: str
    episode_seed: int
    captured: bool
    capture_time_s: float
    mean_distance_px: float
    decisions: int
    frames: int


@dataclass(frozen=True)
class DiscoverySample:
    delta_prev: float
    repeated: bool
    action: int
    prev_action: int
    episode_seed: int
    checkpoint_seed: int
    duration_s: float
    evader_policy: str


# ---------------------------------------------------------------------------
# Utilidades de entorno y políticas
# ---------------------------------------------------------------------------
def make_env(
    duration_s: float,
    evader_policy: str,
) -> RobotPvpEnv:
    """Entorno de evaluación con arena fija y duración/política fijas."""
    return RobotPvpEnv(
        arena_size=(C.EVAL_ARENA_WIDTH, C.EVAL_ARENA_HEIGHT),
        episode_duration_seconds=float(duration_s),
        evader_policy=evader_policy,
    )


def episode_seeds(base_seed: int, episodes: int) -> List[int]:
    """Lista de semillas idéntica para todos los agentes y configuraciones."""
    return [int(base_seed) + index for index in range(int(episodes))]


def _policy_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stream)])
    )


def _dominant_action(dx: float, dy: float) -> int:
    """Acción cardinal hacia (dx, dy); empate -> eje horizontal."""
    if abs(dx) >= abs(dy):
        return C.ACTION_RIGHT if dx >= 0.0 else C.ACTION_LEFT
    return C.ACTION_DOWN if dy >= 0.0 else C.ACTION_UP


def _random_target(env: RobotPvpEnv, rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(env.min_x, env.max_x),
            rng.uniform(env.min_y, env.max_y),
        ],
        dtype=np.float64,
    )


def _random_target_action(
    env: RobotPvpEnv,
    target: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, np.ndarray]:
    """Acción hacia un objetivo aleatorio, con reobjetivo al acercarse.

    El juego original mueve el robot 12 px por frame hacia el objetivo y lo
    cambia cuando la distancia es menor que 12 px. Aquí la decisión dura
    ``action_repeat`` frames, así que el umbral se escala a 12·4 px.
    """
    dx = float(target[0] - env.agent_x)
    dy = float(target[1] - env.agent_y)
    if math.hypot(dx, dy) <= C.AGENT_SPEED * C.ACTION_REPEAT:
        target = _random_target(env, rng)
        dx = float(target[0] - env.agent_x)
        dy = float(target[1] - env.agent_y)
    return _dominant_action(dx, dy), target


def _greedy_action(env: RobotPvpEnv) -> int:
    return _dominant_action(
        float(env.player_x - env.agent_x),
        float(env.player_y - env.agent_y),
    )


# ---------------------------------------------------------------------------
# Carga de checkpoints e inferencia
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedModel:
    model: SharedActorCritic
    discovery_enabled: bool
    checkpoint_seed: int
    path: Path


def _seed_from_path(path: Path) -> int:
    stem = path.parent.name
    if "_s" in stem:
        suffix = stem.rsplit("_s", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return -1


def load_checkpoint(path: Path, device: torch.device) -> LoadedModel:
    """Carga un checkpoint de ``train.py`` y deduce su semilla."""
    payload = torch.load(path, map_location=device)
    model = SharedActorCritic().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    discovery_enabled = bool(payload.get("discovery_enabled", True))
    checkpoint_seed = payload.get("seed", None)
    if checkpoint_seed is None:
        checkpoint_seed = _seed_from_path(path)
    return LoadedModel(
        model=model,
        discovery_enabled=discovery_enabled,
        checkpoint_seed=int(checkpoint_seed),
        path=path,
    )


def _obs_for_indices(
    geom: np.ndarray,
    memory: DiscoveryMemory,
    indices: np.ndarray,
) -> np.ndarray:
    features = memory.features()[indices]
    obs = np.concatenate([geom[indices], features], axis=1)
    return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)


def _provisional_obs_for_indices(
    geom_next_subset: np.ndarray,
    memory: DiscoveryMemory,
    indices: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    if memory.enabled:
        action_feature = encode_actions(actions).reshape(-1, 1)
        delta_feature = np.tanh(memory.delta_prev[indices]).reshape(-1, 1)
    else:
        # Ablación sin descubrimiento: entradas 6-7 siempre en 0.
        action_feature = np.zeros((len(indices), 1), dtype=np.float32)
        delta_feature = np.zeros((len(indices), 1), dtype=np.float32)
    obs = np.concatenate(
        [geom_next_subset, action_feature, delta_feature], axis=1
    )
    return np.clip(obs, -1.0, 1.0).astype(np.float32, copy=False)


def evaluate_ac_agent(
    agent_name: str,
    loaded: LoadedModel,
    seeds: Sequence[int],
    duration_s: float,
    evader_policy: str,
    *,
    deterministic: bool,
    collect_discovery: bool,
    device: torch.device,
    policy_seed: int,
) -> Tuple[List[EpisodeResult], List[DiscoverySample]]:
    """Evalúa un agente Actor-Crítico en N episodios simultáneos.

    Cada entorno es independiente y se reinicia con su propia semilla, por lo
    que el ruido del evasor no se comparte entre episodios.
    """
    num_envs = len(seeds)
    envs = [make_env(duration_s, evader_policy) for _ in range(num_envs)]
    geom = np.zeros((num_envs, C.OBS_GEOMETRY_DIM), dtype=np.float32)
    for index, seed in enumerate(seeds):
        obs, _ = envs[index].reset(seed=int(seed))
        geom[index] = obs

    memory = DiscoveryMemory(
        num_envs=num_envs,
        enabled=loaded.discovery_enabled,
        record_history=False,
    )
    memory.reset()

    active = np.ones(num_envs, dtype=bool)
    decisions = np.zeros(num_envs, dtype=np.int64)
    distance_sum = np.zeros(num_envs, dtype=np.float64)
    distance_samples = np.ones(num_envs, dtype=np.int64)
    for index in range(num_envs):
        distance_sum[index] = envs[index].distance

    results: List[EpisodeResult] = []
    samples: List[DiscoverySample] = []

    # Mismo flujo de muestreo para todos los modelos evaluados con la misma
    # semilla de política.
    torch.manual_seed(int(policy_seed))

    while active.any():
        indices = np.flatnonzero(active)
        obs = _obs_for_indices(geom, memory, indices)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, values = loaded.model(obs_tensor)
            distribution = Categorical(logits=logits)
            if deterministic:
                actions_tensor = torch.argmax(logits, dim=-1)
            else:
                actions_tensor = distribution.sample()
            values_np = (
                values.squeeze(-1).detach().cpu().numpy().astype(np.float32)
            )

        actions = actions_tensor.detach().cpu().numpy().astype(np.int64)

        if collect_discovery:
            prev_actions = memory.actions_prev[indices].copy()
            delta_prev = memory.delta_prev[indices].copy()
            valid = prev_actions >= 0
            repeats = actions == prev_actions
            for local in np.flatnonzero(valid):
                env_index = int(indices[local])
                samples.append(
                    DiscoverySample(
                        delta_prev=float(delta_prev[local]),
                        repeated=bool(repeats[local]),
                        action=int(actions[local]),
                        prev_action=int(prev_actions[local]),
                        episode_seed=int(seeds[env_index]),
                        checkpoint_seed=loaded.checkpoint_seed,
                        duration_s=float(duration_s),
                        evader_policy=evader_policy,
                    )
                )

        rewards = np.zeros(indices.size, dtype=np.float32)
        dones = np.zeros(indices.size, dtype=bool)
        geom_next = np.zeros(
            (indices.size, C.OBS_GEOMETRY_DIM), dtype=np.float32
        )

        for local, env_index in enumerate(indices):
            env_index = int(env_index)
            obs_next, reward, terminated, truncated, info = envs[env_index].step(
                int(actions[local])
            )
            decisions[env_index] += 1
            distance_sum[env_index] += float(info["distance"])
            distance_samples[env_index] += 1
            rewards[local] = float(reward)
            geom_next[local] = obs_next
            done = bool(terminated or truncated)
            dones[local] = done

            if done:
                captured = bool(info["captured"])
                frames = int(info["frames"])
                results.append(
                    EpisodeResult(
                        agent=agent_name,
                        checkpoint_seed=loaded.checkpoint_seed,
                        duration_s=float(duration_s),
                        evader_policy=evader_policy,
                        episode_seed=int(seeds[env_index]),
                        captured=captured,
                        capture_time_s=(
                            frames / C.FPS if captured else float("nan")
                        ),
                        mean_distance_px=float(
                            distance_sum[env_index]
                            / max(1, distance_samples[env_index])
                        ),
                        decisions=int(decisions[env_index]),
                        frames=frames,
                    )
                )
                active[env_index] = False

        provisional = _provisional_obs_for_indices(
            geom_next, memory, indices, actions
        )
        with torch.no_grad():
            _, next_values = loaded.model(
                torch.as_tensor(provisional, dtype=torch.float32, device=device)
            )
        next_values_np = (
            next_values.squeeze(-1).detach().cpu().numpy().astype(np.float32)
        )
        deltas = (
            rewards
            + C.GAMMA * next_values_np * (1.0 - dones.astype(np.float32))
            - values_np
        ).astype(np.float32)

        full_actions = np.full(num_envs, -1, dtype=np.int64)
        full_actions[indices] = actions
        full_deltas = np.zeros(num_envs, dtype=np.float32)
        full_deltas[indices] = deltas
        memory.update(full_actions, full_deltas)

        done_mask = np.zeros(num_envs, dtype=bool)
        done_mask[indices[dones]] = True
        memory.reset(done_mask)

        geom[indices] = geom_next

    if len(results) != num_envs:
        raise RuntimeError(
            f"{agent_name}: se esperaban {num_envs} episodios y se registraron "
            f"{len(results)}"
        )
    return results, samples


def evaluate_simple_agent(
    agent_name: str,
    seeds: Sequence[int],
    duration_s: float,
    evader_policy: str,
) -> List[EpisodeResult]:
    """Evalúa una política que no usa la red neuronal."""
    num_envs = len(seeds)
    envs = [make_env(duration_s, evader_policy) for _ in range(num_envs)]
    rngs: List[np.random.Generator] = []
    targets = np.zeros((num_envs, 2), dtype=np.float64)

    for index, seed in enumerate(seeds):
        env = envs[index]
        env.reset(seed=int(seed))
        stream = (
            _RANDOM_TARGET_STREAM
            if agent_name == AGENT_RANDOM_TARGET
            else _RANDOM_ACTION_STREAM
        )
        rng = _policy_rng(int(seed), stream)
        rngs.append(rng)
        if agent_name == AGENT_RANDOM_TARGET:
            targets[index] = _random_target(env, rng)

    active = np.ones(num_envs, dtype=bool)
    decisions = np.zeros(num_envs, dtype=np.int64)
    distance_sum = np.zeros(num_envs, dtype=np.float64)
    distance_samples = np.ones(num_envs, dtype=np.int64)
    for index in range(num_envs):
        distance_sum[index] = envs[index].distance

    results: List[EpisodeResult] = []

    while active.any():
        for env_index in np.flatnonzero(active):
            env_index = int(env_index)
            env = envs[env_index]
            rng = rngs[env_index]

            if agent_name == AGENT_RANDOM_TARGET:
                action, targets[env_index] = _random_target_action(
                    env, targets[env_index], rng
                )
            elif agent_name == AGENT_RANDOM_ACTION:
                action = int(rng.integers(0, C.NUM_ACTIONS))
            elif agent_name == AGENT_GREEDY_CHASE:
                action = _greedy_action(env)
            else:
                raise ValueError(f"agente simple desconocido: {agent_name!r}")

            obs_next, reward, terminated, truncated, info = env.step(action)
            decisions[env_index] += 1
            distance_sum[env_index] += float(info["distance"])
            distance_samples[env_index] += 1

            done = bool(terminated or truncated)
            if done:
                captured = bool(info["captured"])
                frames = int(info["frames"])
                results.append(
                    EpisodeResult(
                        agent=agent_name,
                        checkpoint_seed=0,
                        duration_s=float(duration_s),
                        evader_policy=evader_policy,
                        episode_seed=int(seeds[env_index]),
                        captured=captured,
                        capture_time_s=(
                            frames / C.FPS if captured else float("nan")
                        ),
                        mean_distance_px=float(
                            distance_sum[env_index]
                            / max(1, distance_samples[env_index])
                        ),
                        decisions=int(decisions[env_index]),
                        frames=frames,
                    )
                )
                active[env_index] = False

    if len(results) != num_envs:
        raise RuntimeError(
            f"{agent_name}: se esperaban {num_envs} episodios y se registraron "
            f"{len(results)}"
        )
    return results


# ---------------------------------------------------------------------------
# Evaluación completa
# ---------------------------------------------------------------------------
def evaluate_all(
    *,
    episodes: int,
    durations: Sequence[float],
    policies: Sequence[str],
    base_checkpoints: Sequence[Path],
    disc_checkpoints: Sequence[Path],
    base_seed: int,
    greedy: bool,
    device: torch.device,
    agents: Sequence[str] = AGENT_ORDER,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa los agentes seleccionados y devuelve (episodios, descubrimiento)."""
    seeds = episode_seeds(base_seed, episodes)
    selected = set(agents)
    results: List[EpisodeResult] = []
    samples: List[DiscoverySample] = []

    simple_agents = [
        agent
        for agent in (
            AGENT_RANDOM_TARGET,
            AGENT_RANDOM_ACTION,
            AGENT_GREEDY_CHASE,
        )
        if agent in selected
    ]
    for duration in durations:
        for policy in policies:
            for agent in simple_agents:
                partial = evaluate_simple_agent(
                    agent, seeds, float(duration), policy
                )
                results.extend(partial)
                if verbose:
                    _print_progress(partial, agent, duration, policy)

    loaded_models: List[Tuple[str, LoadedModel]] = []
    if AGENT_AC_BASE in selected:
        if not base_checkpoints:
            raise ValueError("No se indicó ningún checkpoint de ac_base")
        for path in base_checkpoints:
            if not Path(path).exists():
                raise FileNotFoundError(f"No existe el checkpoint: {path}")
            loaded_models.append((AGENT_AC_BASE, load_checkpoint(Path(path), device)))
    if AGENT_AC_DISC in selected:
        if not disc_checkpoints:
            raise ValueError("No se indicó ningún checkpoint de ac_disc")
        for path in disc_checkpoints:
            if not Path(path).exists():
                raise FileNotFoundError(f"No existe el checkpoint: {path}")
            loaded_models.append((AGENT_AC_DISC, load_checkpoint(Path(path), device)))

    for agent, loaded in loaded_models:
        for duration in durations:
            for policy in policies:
                partial, discovery = evaluate_ac_agent(
                    agent,
                    loaded,
                    seeds,
                    float(duration),
                    policy,
                    deterministic=greedy,
                    collect_discovery=(agent == AGENT_AC_DISC),
                    device=device,
                    policy_seed=base_seed,
                )
                results.extend(partial)
                samples.extend(discovery)
                if verbose:
                    _print_progress(partial, agent, duration, policy)

    episodes_df = pd.DataFrame([asdict(result) for result in results])
    samples_df = pd.DataFrame(
        [asdict(sample) for sample in samples],
        columns=[
            "delta_prev",
            "repeated",
            "action",
            "prev_action",
            "episode_seed",
            "checkpoint_seed",
            "duration_s",
            "evader_policy",
        ],
    )
    return episodes_df, samples_df


def _print_progress(
    partial: Sequence[EpisodeResult],
    agent: str,
    duration: float,
    policy: str,
) -> None:
    captures = sum(1 for result in partial if result.captured)
    capture_times = [
        result.capture_time_s for result in partial if result.captured
    ]
    mean_time = (
        float(np.mean(capture_times)) if capture_times else float("nan")
    )
    print(
        f"[eval] {agent:<14} dur={duration:>2.0f}s pol={policy:<11} "
        f"eps={len(partial):>4d} cap={captures / max(1, len(partial)):.3f} "
        f"t_cap={mean_time:5.2f}s"
    )


# ---------------------------------------------------------------------------
# Resúmenes y formato
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "agente",
    "duracion_s",
    "politica_evasor",
    "episodios",
    "n_capturas",
    "tasa_captura",
    "tiempo_captura_medio_s",
    "tiempo_captura_mediano_s",
    "tiempo_captura_desv_s",
    "distancia_media_px",
    "decisiones_por_episodio",
    "n_checkpoints",
]


def _summarize_group(group: pd.DataFrame) -> Dict[str, float]:
    captured = group["captured"].astype(bool)
    capture_times = group.loc[captured, "capture_time_s"].dropna()
    return {
        "episodios": int(len(group)),
        "n_capturas": int(captured.sum()),
        "tasa_captura": float(captured.mean()) if len(group) else float("nan"),
        "tiempo_captura_medio_s": (
            float(capture_times.mean()) if len(capture_times) else float("nan")
        ),
        "tiempo_captura_mediano_s": (
            float(capture_times.median()) if len(capture_times) else float("nan")
        ),
        "tiempo_captura_desv_s": (
            float(capture_times.std(ddof=1))
            if len(capture_times) > 1
            else 0.0 if len(capture_times) == 1 else float("nan")
        ),
        "distancia_media_px": float(group["mean_distance_px"].mean()),
        "decisiones_por_episodio": float(group["decisions"].mean()),
        "n_checkpoints": int(group["checkpoint_seed"].nunique()),
    }


def summarize_episodes(
    episodes_df: pd.DataFrame,
    *,
    by_checkpoint: bool = False,
) -> pd.DataFrame:
    """Agrega las métricas por agente, duración y política."""
    if episodes_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    group_columns = ["agent", "duration_s", "evader_policy"]
    if by_checkpoint:
        group_columns = group_columns + ["checkpoint_seed"]

    rows: List[Dict[str, object]] = []
    for keys, group in episodes_df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = {
            "agente": keys[0],
            "duracion_s": float(keys[1]),
            "politica_evasor": keys[2],
        }
        if by_checkpoint:
            row["semilla_entrenamiento"] = int(keys[3])
        row.update(_summarize_group(group))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["agente"] = pd.Categorical(
        summary["agente"], categories=list(AGENT_ORDER), ordered=True
    )
    sort_columns = ["agente", "duracion_s", "politica_evasor"]
    if by_checkpoint:
        sort_columns.append("semilla_entrenamiento")
    summary = summary.sort_values(sort_columns).reset_index(drop=True)
    return summary


def summarize_overall(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega todas las duraciones y políticas por agente."""
    if episodes_df.empty:
        return pd.DataFrame(
            columns=[
                "agente",
                "episodios",
                "n_capturas",
                "tasa_captura",
                "tiempo_captura_medio_s",
                "tiempo_captura_mediano_s",
                "tiempo_captura_desv_s",
                "distancia_media_px",
                "decisiones_por_episodio",
                "n_checkpoints",
            ]
        )

    rows: List[Dict[str, object]] = []
    for agent, group in episodes_df.groupby("agent", dropna=False):
        row: Dict[str, object] = {"agente": agent}
        row.update(_summarize_group(group))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["agente"] = pd.Categorical(
        summary["agente"], categories=list(AGENT_ORDER), ordered=True
    )
    return summary.sort_values("agente").reset_index(drop=True)


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "—"
    return f"{number:.{digits}f}"


def format_percent(value: object, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "—"
    return f"{number * 100:.{digits}f} %"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def print_overall_table(overall: pd.DataFrame) -> None:
    if overall.empty:
        print("(sin resultados)")
        return
    display = overall.copy()
    display["agente"] = display["agente"].astype(str)
    display["tasa_captura"] = display["tasa_captura"].map(format_percent)
    for column in (
        "tiempo_captura_medio_s",
        "tiempo_captura_mediano_s",
        "tiempo_captura_desv_s",
        "distancia_media_px",
        "decisiones_por_episodio",
    ):
        display[column] = display[column].map(format_number)
    display = display.rename(
        columns={
            "agente": "agente",
            "episodios": "episodios",
            "n_capturas": "capturas",
            "tasa_captura": "tasa_captura",
            "tiempo_captura_medio_s": "t_medio_s",
            "tiempo_captura_mediano_s": "t_mediana_s",
            "tiempo_captura_desv_s": "t_desv_s",
            "distancia_media_px": "dist_media_px",
            "decisiones_por_episodio": "dec/eps",
            "n_checkpoints": "checkpoints",
        }
    )
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa los cinco agentes de RobotPVP en arena fija"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=list(C.EPISODE_DURATION_CHOICES),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[C.EVADER_FLEE, C.EVADER_TILT_RANDOM],
        choices=[C.EVADER_FLEE, C.EVADER_TILT_RANDOM],
    )
    parser.add_argument(
        "--base-checkpoints",
        nargs="+",
        default=[str(ML_DIR / "runs" / "ac_base_s1" / "last.pt")],
        help="checkpoints de ac_base (uno por semilla de entrenamiento)",
    )
    parser.add_argument(
        "--disc-checkpoints",
        nargs="+",
        default=[str(ML_DIR / "runs" / "ac_disc_s1" / "last.pt")],
        help="checkpoints de ac_disc (uno por semilla de entrenamiento)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENT_ORDER),
        choices=list(AGENT_ORDER),
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="usa argmax en los agentes Actor-Críticos (por defecto muestreo)",
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "eval_episodes.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "eval_table.csv"),
    )
    parser.add_argument(
        "--summary-by-seed-output",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "eval_by_seed.csv"),
    )
    parser.add_argument(
        "--discovery-output",
        type=str,
        default=str(DEFAULT_RESULTS_DIR / "discovery_samples.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes debe ser >= 1")
    if any(duration <= 0 for duration in args.durations):
        raise SystemExit("--durations debe contener valores positivos")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda solicitado, pero CUDA no está disponible")

    episodes_df, samples_df = evaluate_all(
        episodes=args.episodes,
        durations=args.durations,
        policies=args.policies,
        base_checkpoints=[Path(path) for path in args.base_checkpoints],
        disc_checkpoints=[Path(path) for path in args.disc_checkpoints],
        base_seed=args.base_seed,
        greedy=args.greedy,
        device=device,
        agents=args.agents,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_csv(output_path, index=False)

    summary = summarize_episodes(episodes_df, by_checkpoint=False)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    by_seed = summarize_episodes(episodes_df, by_checkpoint=True)
    by_seed_path = Path(args.summary_by_seed_output)
    by_seed_path.parent.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(by_seed_path, index=False)

    discovery_path = Path(args.discovery_output)
    discovery_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(discovery_path, index=False)

    overall = summarize_overall(episodes_df)
    overall_path = summary_path.with_name("eval_overall.csv")
    overall.to_csv(overall_path, index=False)

    print("\nResumen general (todas las duraciones y políticas):")
    print_overall_table(overall)
    print(f"\n[ok] episodios -> {output_path}")
    print(f"[ok] resumen   -> {summary_path}")
    print(f"[ok] por semilla -> {by_seed_path}")
    print(f"[ok] descubrimiento -> {discovery_path}")
    print(f"[ok] resumen general -> {overall_path}")


if __name__ == "__main__":
    main()
