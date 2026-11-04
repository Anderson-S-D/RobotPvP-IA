#!/usr/bin/env python3
"""A2C síncrono con GAE para RobotPVP.

Recolecta ``n_steps`` decisiones por entorno con ``VecRobotPvpEnv``, calcula
ventajas GAE y hace un update Actor-Crítico. El δ de la memoria de
descubrimiento se calcula online durante la recolección (``agent.py``) y NO es
la ventaja GAE: la GAE solo se usa en la pérdida.

Uso:
    python train.py --total-steps 20000 --run-name smoke
    python train.py --run-name ac_disc_s1 --seed 1
    python train.py --run-name ac_base_s1 --seed 1 --no-discovery
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from robot_ai import config as C
from robot_ai.agent import build_obs, select_action, update_discovery
from robot_ai.discovery import DiscoveryMemory
from robot_ai.model import SharedActorCritic
from robot_ai.vec_env import VecRobotPvpEnv

LOG_INTERVAL_DECISIONS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A2C síncrono con GAE y Componente de Descubrimiento"
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=2_000_000,
        help="decisiones totales (cada decisión = action_repeat frames)",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="ablación: entradas 6-7 siempre en 0",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="directorio raíz de runs (por defecto ml/runs)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().entropy % (2**31 - 1))


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_lr(
    initial_lr: float, steps_done: int, total_steps: int
) -> float:
    progress = min(1.0, max(0.0, steps_done / max(1, total_steps)))
    return initial_lr * (1.0 - progress)


def build_config(
    args: argparse.Namespace, seed: int, run_name: str
) -> Dict[str, Any]:
    args_dict = vars(args).copy()
    args_dict["seed"] = seed
    args_dict["run_name"] = run_name
    return {
        "run_name": run_name,
        "seed": seed,
        "algorithm": "A2C synchronous + GAE",
        "hyperparameters": args_dict,
        "discovery_enabled": not args.no_discovery,
        "architecture": {
            "input_dim": C.INPUT_DIM,
            "hidden_dim": C.HIDDEN_DIM,
            "actor_output_dim": C.ACTOR_OUTPUT_DIM,
            "critic_output_dim": C.CRITIC_OUTPUT_DIM,
            "param_count": C.PARAM_COUNT,
            "hidden_activation": "tanh",
            "actor_output": "logits (softmax outside)",
            "critic_output": "linear",
        },
        "action_repeat": C.ACTION_REPEAT,
        "action_order": list(C.ACTION_NAMES),
        "action_encodings": dict(C.ACTION_ENCODINGS),
        "gamma_spec": C.GAMMA,
        "environment": {
            "fps": C.FPS,
            "robot_radius": C.ROBOT_RADIUS,
            "border_width": C.BORDER_WIDTH,
            "arena_width_range": [C.ARENA_WIDTH_MIN, C.ARENA_WIDTH_MAX],
            "arena_height_range": [C.ARENA_HEIGHT_MIN, C.ARENA_HEIGHT_MAX],
            "episode_duration_choices_s": list(C.EPISODE_DURATION_CHOICES),
            "player_accel_sensitivity": C.PLAYER_ACCEL_SENSITIVITY,
            "player_friction": C.PLAYER_FRICTION,
            "wall_elasticity": C.WALL_ELASTICITY,
            "agent_speed": C.AGENT_SPEED,
            "capture_distance": C.CAPTURE_DISTANCE,
            "reward_distance_scale": C.REWARD_DISTANCE_SCALE,
            "reward_step_penalty": C.REWARD_STEP_PENALTY,
            "reward_capture": C.REWARD_CAPTURE,
            "reward_timeout": C.REWARD_TIMEOUT,
            "evader_train_mix": dict(C.EVADER_TRAIN_MIX),
        },
    }


def save_checkpoint(
    path: Path,
    model: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    seed: int,
    steps: int,
    episodes: int,
    best_capture_rate: float,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "steps": int(steps),
        "episodes": int(episodes),
        "best_capture_rate": float(best_capture_rate),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "args": vars(args),
        "seed": int(seed),
        "discovery_enabled": not args.no_discovery,
        "architecture": {
            "input_dim": C.INPUT_DIM,
            "hidden_dim": C.HIDDEN_DIM,
            "actor_output_dim": C.ACTOR_OUTPUT_DIM,
            "critic_output_dim": C.CRITIC_OUTPUT_DIM,
            "param_count": C.PARAM_COUNT,
        },
    }
    torch.save(payload, path)


def mean_or_nan(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    if args.total_steps < 1:
        raise SystemExit("--total-steps debe ser >= 1")
    if args.num_envs < 1 or args.n_steps < 1:
        raise SystemExit("--num-envs y --n-steps deben ser >= 1")
    if not (0.0 <= args.gamma <= 1.0):
        raise SystemExit("--gamma debe estar en [0, 1]")
    if not (0.0 <= args.gae_lambda <= 1.0):
        raise SystemExit("--gae-lambda debe estar en [0, 1]")

    seed = resolve_seed(args.seed)
    args.seed = seed
    set_global_seeds(seed)
    run_name = args.run_name or f"ac_{datetime.now():%Y%m%d_%H%M%S}"

    ml_dir = Path(__file__).resolve().parent
    runs_root = (
        Path(args.runs_dir).expanduser().resolve()
        if args.runs_dir
        else ml_dir / "runs"
    )
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda solicitado, pero CUDA no está disponible")

    config = build_config(args, seed, run_name)
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False)

    model = SharedActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    venv = VecRobotPvpEnv(
        num_envs=args.num_envs,
        evader_policy=C.EVADER_MIXED,
        seed=seed,
    )
    geom, _ = venv.reset(seed=seed)
    memory = DiscoveryMemory(
        num_envs=args.num_envs,
        enabled=not args.no_discovery,
        record_history=False,
    )

    num_envs = args.num_envs
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    recent_episodes: deque[Tuple[bool, float, float]] = deque(maxlen=100)
    total_episodes = 0
    total_steps = 0
    last_log_steps = 0
    best_capture_rate = -1.0
    best_capture_time = float("inf")
    start_time = time.perf_counter()
    metrics: Dict[str, float] = {
        "entropy": float("nan"),
        "value_loss": float("nan"),
        "policy_loss": float("nan"),
    }

    progress_path = run_dir / "progress.csv"
    progress_file = open(progress_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(progress_file)
    writer.writerow(
        [
            "steps",
            "episodes",
            "capture_rate",
            "mean_capture_time_s",
            "mean_return",
            "entropy",
            "value_loss",
            "policy_loss",
            "fps",
        ]
    )
    progress_file.flush()

    print(
        f"[RobotPVP] run={run_name} seed={seed} device={device} "
        f"discovery={not args.no_discovery}"
    )
    print(
        f"[RobotPVP] entornos={num_envs} n_steps={args.n_steps} "
        f"total_steps={args.total_steps:,} -> {run_dir}"
    )

    try:
        while total_steps < args.total_steps:
            obs_buffer: List[np.ndarray] = []
            actions_buffer: List[np.ndarray] = []
            rewards_buffer: List[np.ndarray] = []
            dones_buffer: List[np.ndarray] = []
            values_buffer: List[np.ndarray] = []

            for _ in range(args.n_steps):
                # Decisión con s_t = [geom_t, a_{t-1}, δ_{t-1}].
                policy_step = select_action(model, memory, geom)

                # El entorno se mantiene 4 frames por decisión.
                geom_next, rewards, terminated, truncated, infos = venv.step(
                    policy_step.actions
                )
                dones = terminated | truncated

                # δ online del Componente de Descubrimiento (no es GAE).
                update_discovery(
                    model,
                    memory,
                    policy_step.actions,
                    geom_next,
                    rewards,
                    dones,
                    policy_step.values,
                    gamma=args.gamma,
                )

                obs_buffer.append(policy_step.obs_t)
                actions_buffer.append(policy_step.actions)
                rewards_buffer.append(rewards.astype(np.float32, copy=False))
                dones_buffer.append(dones.astype(np.float32, copy=False))
                values_buffer.append(policy_step.values)

                # Estadísticas de episodios (el entorno hace autoreset).
                episode_returns += rewards
                if dones.any():
                    for env_index in np.flatnonzero(dones):
                        info = infos[int(env_index)]
                        captured = bool(info["captured"])
                        frames = int(info["frames"])
                        recent_episodes.append(
                            (
                                captured,
                                frames / C.FPS if captured else float("nan"),
                                float(episode_returns[env_index]),
                            )
                        )
                        total_episodes += 1
                        episode_returns[env_index] = 0.0

                geom = geom_next
                total_steps += num_envs
                if total_steps >= args.total_steps:
                    break

            # V(s_{t+1}) del último estado, con la memoria ya actualizada.
            last_obs = build_obs(geom, memory)
            with torch.no_grad():
                _, last_values_tensor = model(
                    torch.as_tensor(last_obs, dtype=torch.float32, device=device)
                )
                last_values = (
                    last_values_tensor.squeeze(-1).detach().cpu().numpy()
                )

            # GAE sobre el rollout recolectado.
            num_collected = len(obs_buffer)
            rewards_array = np.stack(rewards_buffer)
            dones_array = np.stack(dones_buffer)
            values_array = np.stack(values_buffer)
            advantages = np.zeros_like(rewards_array)
            last_gae = np.zeros(num_envs, dtype=np.float32)

            for step in reversed(range(num_collected)):
                if step == num_collected - 1:
                    next_values = last_values
                else:
                    next_values = values_array[step + 1]
                next_nonterminal = 1.0 - dones_array[step]
                delta = (
                    rewards_array[step]
                    + args.gamma * next_values * next_nonterminal
                    - values_array[step]
                )
                last_gae = (
                    delta
                    + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                )
                advantages[step] = last_gae

            returns = advantages + values_array

            obs_flat = np.concatenate(obs_buffer, axis=0)
            actions_flat = np.concatenate(actions_buffer, axis=0)
            advantages_flat = advantages.reshape(-1)
            returns_flat = returns.reshape(-1).astype(np.float32, copy=False)

            # Normalización de ventajas por lote.
            advantage_mean = float(advantages_flat.mean())
            advantage_std = float(advantages_flat.std())
            advantages_normalized = (
                advantages_flat - advantage_mean
            ) / (advantage_std + 1e-8)

            obs_tensor = torch.as_tensor(
                obs_flat, dtype=torch.float32, device=device
            )
            actions_tensor = torch.as_tensor(
                actions_flat, dtype=torch.long, device=device
            )
            returns_tensor = torch.as_tensor(
                returns_flat, dtype=torch.float32, device=device
            )
            advantages_tensor = torch.as_tensor(
                advantages_normalized.astype(np.float32, copy=False),
                dtype=torch.float32,
                device=device,
            )

            # Decaimiento lineal del learning rate.
            learning_rate = linear_lr(args.lr, total_steps, args.total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            model.train()
            logits, values = model(obs_tensor)
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(actions_tensor)
            entropy = distribution.entropy().mean()
            policy_loss = -(log_probs * advantages_tensor).mean()
            value_loss = ((values.squeeze(-1) - returns_tensor) ** 2).mean()
            loss = (
                policy_loss
                + args.vf_coef * value_loss
                - args.ent_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            model.eval()

            metrics = {
                "entropy": float(entropy.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "policy_loss": float(policy_loss.detach().cpu().item()),
            }

            should_log = (
                total_steps - last_log_steps >= LOG_INTERVAL_DECISIONS
                or total_steps >= args.total_steps
            )
            if should_log:
                elapsed = time.perf_counter() - start_time
                fps = (
                    total_steps * C.ACTION_REPEAT / elapsed
                    if elapsed > 0.0
                    else float("nan")
                )
                recent_list = list(recent_episodes)
                capture_rate = (
                    float(np.mean([record[0] for record in recent_list]))
                    if recent_list
                    else 0.0
                )
                capture_times = [
                    record[1] for record in recent_list if record[0]
                ]
                mean_capture_time = mean_or_nan(capture_times)
                mean_return = mean_or_nan(
                    [record[2] for record in recent_list]
                )

                writer.writerow(
                    [
                        total_steps,
                        total_episodes,
                        f"{capture_rate:.6f}",
                        f"{mean_capture_time:.6f}",
                        f"{mean_return:.6f}",
                        f"{metrics['entropy']:.6f}",
                        f"{metrics['value_loss']:.6f}",
                        f"{metrics['policy_loss']:.6f}",
                        f"{fps:.3f}",
                    ]
                )
                progress_file.flush()

                print(
                    f"[{total_steps:>9,d}] eps={total_episodes:<6d} "
                    f"cap={capture_rate:.3f} tcap={mean_capture_time:5.2f}s "
                    f"ret={mean_return:+.3f} ent={metrics['entropy']:.3f} "
                    f"pi={metrics['policy_loss']:+.4f} "
                    f"v={metrics['value_loss']:.4f} fps={fps:,.0f}"
                )

                checkpoint_kwargs = dict(
                    args=args,
                    seed=seed,
                    steps=total_steps,
                    episodes=total_episodes,
                    best_capture_rate=max(best_capture_rate, capture_rate),
                    metrics=metrics,
                )
                # La tasa de captura satura en 1.0, así que a igual tasa se
                # prefiere el checkpoint con menor tiempo medio de captura.
                # Sin este desempate, best.pt se queda con la primera ventana
                # que alcanza 1.0 aunque sea un estado mucho peor.
                faster_tie = (
                    capture_rate == best_capture_rate
                    and not math.isnan(mean_capture_time)
                    and mean_capture_time < best_capture_time
                )
                if capture_rate > best_capture_rate or faster_tie:
                    best_capture_rate = capture_rate
                    if not math.isnan(mean_capture_time):
                        best_capture_time = mean_capture_time
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        **checkpoint_kwargs,
                    )
                save_checkpoint(
                    run_dir / "last.pt",
                    model,
                    optimizer,
                    **checkpoint_kwargs,
                )
                last_log_steps = total_steps
    finally:
        progress_file.close()

    elapsed = time.perf_counter() - start_time
    fps = total_steps * C.ACTION_REPEAT / max(elapsed, 1e-9)
    print(
        f"\n[RobotPVP] Completado: {total_steps:,} decisiones, "
        f"{total_episodes:,} episodios, {elapsed:.1f}s ({fps:,.0f} frames/s)"
    )
    print(f"[RobotPVP] best_capture_rate={max(best_capture_rate, 0.0):.3f}")
    print(f"[RobotPVP] run_dir={run_dir}")


if __name__ == "__main__":
    main()
