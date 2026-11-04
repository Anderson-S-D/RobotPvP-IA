#!/usr/bin/env python3
"""Experimentos completos de RobotPVP: entrenamiento, evaluación y figuras.

Pasos:
1. Entrena ``ac_base`` y ``ac_disc`` con las semillas indicadas (1, 2 y 3 por
   defecto). Con ``--skip-train`` reutiliza los runs que ya existan.
2. Evalúa los cinco agentes con las mismas semillas de episodio.
3. Genera en ``ml/results/``:
   - ``learning_curves.png``
   - ``eval_table.csv`` y ``eval_table.md``
   - ``capture_time_boxplot.png``
   - ``discovery_analysis.png``
   - ``results.md``

Uso:
    python experiments.py
    python experiments.py --skip-train --jobs 3
    python experiments.py --skip-train --seeds 1 --episodes 30 \
        --results-dir results_quick
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import evaluate as eval_mod  # noqa: E402
from robot_ai import config as C  # noqa: E402

ML_DIR = Path(__file__).resolve().parent

METHOD_LABELS = {
    "ac_base": "AC sin descubrimiento",
    "ac_disc": "AC con descubrimiento",
}
METHOD_COLORS = {
    "ac_base": "#1f77b4",
    "ac_disc": "#d62728",
}
POLICY_LABELS = {
    C.EVADER_FLEE: "Evade (flee)",
    C.EVADER_TILT_RANDOM: "Inclinación aleatoria",
}
POLICY_COLORS = {
    C.EVADER_FLEE: "#4c72b0",
    C.EVADER_TILT_RANDOM: "#dd8452",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y compara ac_base vs ac_disc y los baselines"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="semillas de entrenamiento",
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
        "--skip-train",
        action="store_true",
        help="reutiliza runs existentes; entrena solo los que falten",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="entrenamientos en paralelo (subprocesos independientes)",
    )
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(ML_DIR / "runs"),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(ML_DIR / "results"),
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="last.pt",
        choices=["last.pt", "best.pt"],
        help=(
            "checkpoint de cada run para evaluar; last.pt (por defecto) usa "
            "el mismo presupuesto de entrenamiento en todas las semillas"
        ),
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="evalúa ac_base/ac_disc con argmax (por defecto muestreo)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def run_name(method: str, seed: int) -> str:
    return f"ac_{method}_s{seed}"


def train_one(
    method: str,
    seed: int,
    args: argparse.Namespace,
    runs_dir: Path,
) -> str:
    name = run_name(method, seed)
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    config_file = run_dir / "config.json"

    if args.skip_train and checkpoint.exists() and config_file.exists():
        print(f"[train] reutilizando {name}", flush=True)
        return name

    command = [
        sys.executable,
        str(ML_DIR / "train.py"),
        "--run-name",
        name,
        "--seed",
        str(seed),
        "--total-steps",
        str(args.total_steps),
        "--device",
        args.device,
        "--runs-dir",
        str(runs_dir),
    ]
    if method == "base":
        command.append("--no-discovery")

    print(f"[train] lanzando {name}: {' '.join(command)}", flush=True)
    log_path = run_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            cwd=str(ML_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[train] terminado {name} (log: {log_path})", flush=True)
    return name


def train_all(args: argparse.Namespace, runs_dir: Path) -> None:
    tasks = [
        (method, seed)
        for seed in args.seeds
        for method in ("base", "disc")
    ]
    if args.jobs <= 1:
        for method, seed in tasks:
            train_one(method, seed, args, runs_dir)
        return

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(train_one, method, seed, args, runs_dir): (
                method,
                seed,
            )
            for method, seed in tasks
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            future.result()
            print(
                f"[train] completado {run_name(method, seed)}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Curvas de aprendizaje
# ---------------------------------------------------------------------------
def load_progress(runs_dir: Path, seeds: Sequence[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for method in ("base", "disc"):
        for seed in seeds:
            path = runs_dir / run_name(method, seed) / "progress.csv"
            if not path.exists():
                raise FileNotFoundError(f"Falta progress.csv: {path}")
            frame = pd.read_csv(path)
            frame["metodo"] = f"ac_{method}"
            frame["semilla"] = int(seed)
            frames.append(frame[["metodo", "semilla", "steps", "capture_rate"]])
    return pd.concat(frames, ignore_index=True)


def summarize_learning_curves(curves: pd.DataFrame) -> pd.DataFrame:
    summary = (
        curves.groupby(["metodo", "steps"])["capture_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary = summary.rename(
        columns={
            "mean": "tasa_captura_media",
            "std": "tasa_captura_desv",
            "count": "n_semillas",
        }
    )
    return summary


def plot_learning_curves(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for method in ("ac_base", "ac_disc"):
        method_summary = summary[summary["metodo"] == method].sort_values(
            "steps"
        )
        if method_summary.empty:
            continue
        steps = method_summary["steps"].to_numpy()
        mean = method_summary["tasa_captura_media"].to_numpy()
        std = method_summary["tasa_captura_desv"].to_numpy()
        color = METHOD_COLORS[method]
        ax.plot(
            steps,
            mean,
            color=color,
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(
            steps,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.20,
        )

    ax.set_xlabel("Decisiones de entrenamiento")
    ax.set_ylabel("Tasa de captura (ventana de 100 episodios)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="lower right")
    ax.set_title("Curvas de aprendizaje: AC con y sin descubrimiento")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def checkpoints_for(
    runs_dir: Path,
    method: str,
    seeds: Sequence[int],
    checkpoint_name: str = "last.pt",
) -> List[Path]:
    """Checkpoints de cada semilla.

    Por defecto se usa ``last.pt`` (política final tras el mismo presupuesto
    de decisiones para todas las semillas). Se evita así el sesgo de
    ``best.pt``, que puede quedar anclado a una ventana temprana y ruidosa
    cuando la tasa de captura satura en 1.0.
    """
    paths = [
        runs_dir / run_name(method, seed) / checkpoint_name
        for seed in seeds
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan checkpoints:\n  " + "\n  ".join(missing)
        )
    return paths


def run_evaluation(
    args: argparse.Namespace,
    runs_dir: Path,
    results_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_checkpoints = checkpoints_for(
        runs_dir, "base", args.seeds, args.checkpoint_name
    )
    disc_checkpoints = checkpoints_for(
        runs_dir, "disc", args.seeds, args.checkpoint_name
    )

    print(
        f"[eval] {args.episodes} episodios por duración/política; "
        f"semillas de episodio {args.base_seed}.."
        f"{args.base_seed + args.episodes - 1}",
        flush=True,
    )
    episodes_df, samples_df = eval_mod.evaluate_all(
        episodes=args.episodes,
        durations=args.durations,
        policies=args.policies,
        base_checkpoints=base_checkpoints,
        disc_checkpoints=disc_checkpoints,
        base_seed=args.base_seed,
        greedy=args.greedy,
        device=eval_mod.torch.device(args.device),
    )

    episodes_df.to_csv(results_dir / "eval_episodes.csv", index=False)
    samples_df.to_csv(results_dir / "discovery_samples.csv", index=False)

    table = eval_mod.summarize_episodes(episodes_df, by_checkpoint=False)
    table.to_csv(results_dir / "eval_table.csv", index=False)

    by_seed = eval_mod.summarize_episodes(episodes_df, by_checkpoint=True)
    by_seed.to_csv(results_dir / "eval_by_seed.csv", index=False)

    overall = eval_mod.summarize_overall(episodes_df)
    overall.to_csv(results_dir / "eval_overall.csv", index=False)

    return episodes_df, samples_df


# ---------------------------------------------------------------------------
# Figuras y tablas
# ---------------------------------------------------------------------------
def plot_capture_time_boxplot(
    episodes_df: pd.DataFrame,
    output_path: Path,
) -> None:
    captured = episodes_df[episodes_df["captured"].astype(bool)].copy()
    agents = [
        agent
        for agent in eval_mod.AGENT_ORDER
        if agent in set(captured["agent"].unique())
    ]
    if not agents:
        raise RuntimeError("No hay episodios con captura para el boxplot")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    width = 0.34
    legend_handles: List[Patch] = []

    for policy_index, policy in enumerate(
        [C.EVADER_FLEE, C.EVADER_TILT_RANDOM]
    ):
        data: List[np.ndarray] = []
        positions: List[float] = []
        for agent_index, agent in enumerate(agents):
            values = captured[
                (captured["agent"] == agent)
                & (captured["evader_policy"] == policy)
            ]["capture_time_s"].dropna().to_numpy()
            if values.size == 0:
                continue
            data.append(values)
            positions.append(
                agent_index + (policy_index - 0.5) * width * 1.25
            )
        if not data:
            continue
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )
        color = POLICY_COLORS[policy]
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)
        legend_handles.append(
            Patch(
                facecolor=color,
                alpha=0.75,
                label=POLICY_LABELS[policy],
            )
        )

    ax.set_xticks(np.arange(len(agents)))
    ax.set_xticklabels(
        [eval_mod.AGENT_LABELS_ES[agent] for agent in agents],
        rotation=15,
        ha="right",
    )
    ax.set_xlabel("Agente")
    ax.set_ylabel("Tiempo de captura (s)")
    ax.set_title(
        "Distribución del tiempo de captura por agente y política del evasor"
    )
    ax.grid(axis="y", alpha=0.30)
    ax.legend(handles=legend_handles, title="Política del evasor")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def discovery_stats(samples_df: pd.DataFrame) -> Dict[str, float]:
    if samples_df.empty:
        return {
            "n": 0,
            "n_neg": 0,
            "n_pos": 0,
            "n_zero": 0,
            "p_repeat_neg": float("nan"),
            "p_repeat_pos": float("nan"),
            "diferencia": float("nan"),
        }
    delta = samples_df["delta_prev"].astype(float).to_numpy()
    repeated = samples_df["repeated"].astype(bool).to_numpy()
    neg = delta < 0.0
    pos = delta > 0.0
    zero = delta == 0.0
    p_neg = float(repeated[neg].mean()) if neg.any() else float("nan")
    p_pos = float(repeated[pos].mean()) if pos.any() else float("nan")
    return {
        "n": int(delta.size),
        "n_neg": int(neg.sum()),
        "n_pos": int(pos.sum()),
        "n_zero": int(zero.sum()),
        "p_repeat_neg": p_neg,
        "p_repeat_pos": p_pos,
        "diferencia": p_neg - p_pos,
    }


def plot_discovery_analysis(
    samples_df: pd.DataFrame,
    output_path: Path,
    *,
    deterministic: bool = False,
) -> Dict[str, float]:
    stats = discovery_stats(samples_df)
    fig, (ax_hist, ax_bar) = plt.subplots(
        1, 2, figsize=(12.0, 5.0)
    )

    if samples_df.empty:
        ax_hist.text(
            0.5,
            0.5,
            "Sin muestras de descubrimiento",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_hist.set_axis_off()
        ax_bar.text(
            0.5,
            0.5,
            "Sin muestras de descubrimiento",
            ha="center",
            va="center",
            transform=ax_bar.transAxes,
        )
        ax_bar.set_axis_off()
    else:
        delta = samples_df["delta_prev"].astype(float).to_numpy()
        low, high = np.percentile(delta, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.min(delta)) - 1.0
            high = float(np.max(delta)) + 1.0
        ax_hist.hist(
            delta,
            bins=60,
            range=(float(low), float(high)),
            color="#4c72b0",
            alpha=0.78,
            edgecolor="white",
            linewidth=0.3,
        )
        ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax_hist.set_xlabel("δ_{t-1} (error TD del paso anterior)")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.set_title("(a) Histograma de δ_{t-1}")
        ax_hist.grid(axis="y", alpha=0.30)
        ax_hist.text(
            0.02,
            0.98,
            f"N = {stats['n']:,}\n"
            f"rango visual: percentiles 1–99",
            transform=ax_hist.transAxes,
            va="top",
            fontsize=9,
        )

        labels = ["δ_{t-1} < 0", "δ_{t-1} > 0"]
        values = [stats["p_repeat_neg"], stats["p_repeat_pos"]]
        counts = [stats["n_neg"], stats["n_pos"]]
        heights = [
            0.0 if math.isnan(value) else value for value in values
        ]
        bars = ax_bar.bar(
            labels,
            heights,
            color=["#d62728", "#2ca02c"],
            alpha=0.80,
            width=0.55,
        )
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("P(repetir la acción anterior)")
        ax_bar.set_title("(b) Repetición según el signo de δ_{t-1}")
        ax_bar.grid(axis="y", alpha=0.30)
        for bar, value, count in zip(bars, values, counts):
            text = (
                "sin muestras"
                if math.isnan(value)
                else f"{value:.2%}\n(n = {count:,})"
            )
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(bar.get_height(), 0.02) + 0.02,
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_bar.text(
            0.02,
            0.98,
            "δ = 0 se excluye de los grupos",
            transform=ax_bar.transAxes,
            va="top",
            fontsize=9,
        )

    fig.suptitle(
        "Componente de Descubrimiento de ac_disc "
        f"({'evaluación con argmax' if deterministic else 'evaluación con muestreo'})"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# Tablas Markdown
# ---------------------------------------------------------------------------
def _overall_rows(overall: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for _, row in overall.iterrows():
        rows.append(
            [
                eval_mod.AGENT_LABELS_ES.get(str(row["agente"]), str(row["agente"])),
                str(int(row["episodios"])),
                str(int(row["n_capturas"])),
                eval_mod.format_percent(row["tasa_captura"]),
                eval_mod.format_number(row["tiempo_captura_medio_s"]),
                eval_mod.format_number(row["tiempo_captura_mediano_s"]),
                eval_mod.format_number(row["tiempo_captura_desv_s"]),
                eval_mod.format_number(row["distancia_media_px"], 1),
                eval_mod.format_number(row["decisiones_por_episodio"], 1),
            ]
        )
    return rows


def write_eval_table_md(
    table: pd.DataFrame,
    overall: pd.DataFrame,
    output_path: Path,
    *,
    episodes: int,
    base_seed: int,
    greedy: bool,
) -> None:
    overall_headers = [
        "Agente",
        "Episodios",
        "Capturas",
        "Tasa de captura",
        "Tiempo medio de captura (s)",
        "Tiempo mediano de captura (s)",
        "Desv. t. captura (s)",
        "Distancia media (px)",
        "Decisiones por episodio",
    ]
    detailed_headers = [
        "Agente",
        "Duración (s)",
        "Política del evasor",
        "Episodios",
        "Tasa de captura",
        "Tiempo medio de captura (s)",
        "Tiempo mediano de captura (s)",
        "Distancia media (px)",
        "Decisiones por episodio",
    ]

    detailed_rows: List[List[str]] = []
    for _, row in table.iterrows():
        detailed_rows.append(
            [
                eval_mod.AGENT_LABELS_ES.get(str(row["agente"]), str(row["agente"])),
                str(int(row["duracion_s"])),
                str(row["politica_evasor"]),
                str(int(row["episodios"])),
                eval_mod.format_percent(row["tasa_captura"]),
                eval_mod.format_number(row["tiempo_captura_medio_s"]),
                eval_mod.format_number(row["tiempo_captura_mediano_s"]),
                eval_mod.format_number(row["distancia_media_px"], 1),
                eval_mod.format_number(row["decisiones_por_episodio"], 1),
            ]
        )

    lines = [
        "# Tabla de evaluación — RobotPVP",
        "",
        f"- Arena fija: {int(C.EVAL_ARENA_WIDTH)} × {int(C.EVAL_ARENA_HEIGHT)} px.",
        f"- Episodios por duración y política: {episodes}.",
        f"- Semillas de episodio: {base_seed} a {base_seed + episodes - 1} "
        "(idénticas para todos los agentes).",
        f"- Modo de acción AC: {'argmax (--greedy)' if greedy else 'muestreo'}.",
        "",
        "## Resumen general",
        "",
        eval_mod.markdown_table(overall_headers, _overall_rows(overall)),
        "",
        "## Detalle por duración y política",
        "",
        eval_mod.markdown_table(detailed_headers, detailed_rows),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# results.md
# ---------------------------------------------------------------------------
def _aggregate_metrics(
    episodes_df: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in episodes_df.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_columns, keys))
        row.update(eval_mod._summarize_group(group))
        rows.append(row)
    result = pd.DataFrame(rows)
    if "agent" in result.columns and not result.empty:
        result["agent"] = pd.Categorical(
            result["agent"],
            categories=list(eval_mod.AGENT_ORDER),
            ordered=True,
        )
    return result


def write_results_md(
    results_dir: Path,
    *,
    seeds: Sequence[int],
    episodes: int,
    durations: Sequence[int],
    policies: Sequence[str],
    base_seed: int,
    greedy: bool,
    checkpoint_name: str = "last.pt",
) -> None:
    episodes_df = pd.read_csv(results_dir / "eval_episodes.csv")
    overall = pd.read_csv(results_dir / "eval_overall.csv")
    exploration = pd.read_csv(results_dir / "discovery_samples.csv")
    learning_curves = pd.read_csv(results_dir / "learning_curves.csv")
    stats = discovery_stats(exploration)

    by_policy = _aggregate_metrics(
        episodes_df, ["agent", "evader_policy"]
    )
    by_duration = _aggregate_metrics(episodes_df, ["agent", "duration_s"])
    ac_only = episodes_df[
        episodes_df["agent"].isin([eval_mod.AGENT_AC_BASE, eval_mod.AGENT_AC_DISC])
    ]
    by_training_seed = _aggregate_metrics(
        ac_only, ["agent", "checkpoint_seed"]
    )

    overall_headers = [
        "Agente",
        "Tasa de captura",
        "Tiempo medio de captura (s)",
        "Tiempo mediano de captura (s)",
        "Distancia media (px)",
        "Decisiones por episodio",
    ]
    overall_rows = [
        [
            eval_mod.AGENT_LABELS_ES.get(str(row["agente"]), str(row["agente"])),
            eval_mod.format_percent(row["tasa_captura"]),
            eval_mod.format_number(row["tiempo_captura_medio_s"]),
            eval_mod.format_number(row["tiempo_captura_mediano_s"]),
            eval_mod.format_number(row["distancia_media_px"], 1),
            eval_mod.format_number(row["decisiones_por_episodio"], 1),
        ]
        for _, row in overall.iterrows()
    ]

    policy_headers = [
        "Agente",
        "Política del evasor",
        "Tasa de captura",
        "Tiempo medio de captura (s)",
        "Tiempo mediano de captura (s)",
        "Distancia media (px)",
    ]
    policy_rows: List[List[str]] = []
    for _, row in by_policy.sort_values(
        ["agent", "evader_policy"]
    ).iterrows():
        policy_rows.append(
            [
                eval_mod.AGENT_LABELS_ES.get(str(row["agent"]), str(row["agent"])),
                str(row["evader_policy"]),
                eval_mod.format_percent(row["tasa_captura"]),
                eval_mod.format_number(row["tiempo_captura_medio_s"]),
                eval_mod.format_number(row["tiempo_captura_mediano_s"]),
                eval_mod.format_number(row["distancia_media_px"], 1),
            ]
        )

    duration_headers = [
        "Agente",
        "Duración (s)",
        "Tasa de captura",
        "Tiempo medio de captura (s)",
        "Tiempo mediano de captura (s)",
        "Decisiones por episodio",
    ]
    duration_rows: List[List[str]] = []
    for _, row in by_duration.sort_values(
        ["agent", "duration_s"]
    ).iterrows():
        duration_rows.append(
            [
                eval_mod.AGENT_LABELS_ES.get(str(row["agent"]), str(row["agent"])),
                str(int(row["duration_s"])),
                eval_mod.format_percent(row["tasa_captura"]),
                eval_mod.format_number(row["tiempo_captura_medio_s"]),
                eval_mod.format_number(row["tiempo_captura_mediano_s"]),
                eval_mod.format_number(row["decisiones_por_episodio"], 1),
            ]
        )

    seed_headers = [
        "Agente",
        "Semilla de entrenamiento",
        "Tasa de captura",
        "Tiempo medio de captura (s)",
        "Tiempo mediano de captura (s)",
        "Distancia media (px)",
    ]
    seed_rows: List[List[str]] = []
    for _, row in by_training_seed.sort_values(
        ["agent", "checkpoint_seed"]
    ).iterrows():
        seed_rows.append(
            [
                eval_mod.AGENT_LABELS_ES.get(str(row["agent"]), str(row["agent"])),
                str(int(row["checkpoint_seed"])),
                eval_mod.format_percent(row["tasa_captura"]),
                eval_mod.format_number(row["tiempo_captura_medio_s"]),
                eval_mod.format_number(row["tiempo_captura_mediano_s"]),
                eval_mod.format_number(row["distancia_media_px"], 1),
            ]
        )

    final_rows: List[List[str]] = []
    for method in ("ac_base", "ac_disc"):
        method_curves = learning_curves[
            learning_curves["metodo"] == method
        ]
        if method_curves.empty:
            continue
        final_per_seed = (
            method_curves.sort_values("steps")
            .groupby("semilla")
            .tail(1)["capture_rate"]
            .to_numpy()
        )
        if final_per_seed.size == 0:
            continue
        mean_final = float(np.mean(final_per_seed))
        std_final = (
            float(np.std(final_per_seed, ddof=1))
            if final_per_seed.size > 1
            else 0.0
        )
        final_rows.append(
            [
                METHOD_LABELS[method],
                str(final_per_seed.size),
                eval_mod.format_percent(mean_final),
                eval_mod.format_number(std_final * 100.0, 1) + " pp",
            ]
        )

    lines = [
        "# Resultados — RobotPVP",
        "",
        "## Configuración",
        "",
        f"- Arena fija: {int(C.EVAL_ARENA_WIDTH)} × {int(C.EVAL_ARENA_HEIGHT)} px.",
        f"- Duraciones evaluadas: {', '.join(str(int(d)) for d in durations)} s.",
        f"- Políticas de evasor: {', '.join(policies)}.",
        f"- Episodios por duración y política: {episodes}.",
        f"- Semillas de episodio: {base_seed} a "
        f"{base_seed + episodes - 1} (idénticas para todos los agentes).",
        f"- Semillas de entrenamiento de los agentes AC: "
        f"{', '.join(str(seed) for seed in seeds)}.",
        f"- Checkpoint de los agentes AC: `{checkpoint_name}` "
        "(mismo presupuesto de entrenamiento para todas las semillas).",
        f"- Modo de acción de los agentes AC: "
        f"{'argmax (--greedy)' if greedy else 'muestreo de la política'}.",
        "",
        "## Tabla general",
        "",
        "Los agentes AC agregan sus resultados sobre las semillas de "
        "entrenamiento indicadas. El tiempo de captura se calcula solo sobre "
        "episodios capturados.",
        "",
        eval_mod.markdown_table(overall_headers, overall_rows),
        "",
        "## Resultados por política del evasor",
        "",
        eval_mod.markdown_table(policy_headers, policy_rows),
        "",
        "## Resultados por duración",
        "",
        eval_mod.markdown_table(duration_headers, duration_rows),
        "",
        "## Agentes AC por semilla de entrenamiento",
        "",
        eval_mod.markdown_table(seed_headers, seed_rows),
        "",
        "## Componente de Descubrimiento (ac_disc)",
        "",
        f"- Muestras de decisiones analizadas: **{stats['n']:,}**.",
        f"- Muestras con δ_{{t-1}} < 0: **{stats['n_neg']:,}**; "
        f"con δ_{{t-1}} > 0: **{stats['n_pos']:,}**; "
        f"con δ_{{t-1}} = 0: **{stats['n_zero']:,}**.",
        f"- P(repetir la acción anterior | δ_{{t-1}} < 0) = "
        f"**{eval_mod.format_percent(stats['p_repeat_neg'], 2)}**.",
        f"- P(repetir la acción anterior | δ_{{t-1}} > 0) = "
        f"**{eval_mod.format_percent(stats['p_repeat_pos'], 2)}**.",
        f"- Diferencia (δ<0 − δ>0) = "
        f"**{eval_mod.format_number(stats['diferencia'] * 100.0, 2)} pp**.",
        "",
        "Si el agente usa la memoria de descubrimiento, se espera que "
        "P(repetir | δ<0) sea menor que P(repetir | δ>0). El número se "
        "reporta tal como sale de las evaluaciones, aunque no favorezca a la "
        "propuesta.",
        "",
        "## Curvas de aprendizaje",
        "",
        "Tasa de captura final (último punto de `progress.csv`, media ± "
        "desviación entre semillas):",
        "",
        eval_mod.markdown_table(
            ["Método", "Semillas", "Tasa de captura final", "Desviación"],
            final_rows,
        ),
        "",
        "## Notas metodológicas",
        "",
        "- Los agentes AC se evalúan con `last.pt` (política final tras el "
        "mismo presupuesto de 2 000 000 decisiones en todas las semillas). "
        "Así se evita el sesgo de `best.pt`, que con la tasa de captura "
        "saturada en 1.0 podía quedar anclado a la primera ventana que "
        "alcanzaba el máximo. `--checkpoint-name best.pt` permite evaluar "
        "el mejor checkpoint si se desea.",
        "- `random_target` reproduce el comportamiento del robot rojo actual: "
        "objetivo aleatorio y velocidad de 12 px/frame. Como la interfaz de "
        "evaluación decide cada `action_repeat` frames, la dirección se "
        "aproxima con la acción cardinal dominante y el objetivo se renueva "
        "cuando la distancia es menor que 12·`action_repeat` px.",
        "- `random_action` muestrea una acción uniforme en {UP, DOWN, LEFT, "
        "RIGHT} cada `action_repeat` frames.",
        "- `greedy_chase` elige el eje con mayor diferencia de posición y "
        "avanza hacia el jugador; es una referencia superior, no un agente "
        "entrenado.",
        "- La distancia media se muestrea al inicio y al final de cada "
        "decisión (cada `action_repeat` frames), no en cada frame.",
        "- Las semillas de episodio son idénticas para todos los agentes, "
        "pero las trayectorias difieren porque cada agente toma decisiones "
        "distintas.",
        "",
    ]
    (results_dir / "results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes debe ser >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs debe ser >= 1")
    if args.total_steps < 1:
        raise SystemExit("--total-steps debe ser >= 1")

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_all(args, runs_dir)

    curves = load_progress(runs_dir, args.seeds)
    curve_summary = summarize_learning_curves(curves)
    curves.to_csv(results_dir / "learning_curves.csv", index=False)
    curve_summary.to_csv(
        results_dir / "learning_curves_summary.csv", index=False
    )
    plot_learning_curves(
        curves,
        curve_summary,
        results_dir / "learning_curves.png",
    )

    episodes_df, samples_df = run_evaluation(args, runs_dir, results_dir)

    table = pd.read_csv(results_dir / "eval_table.csv")
    overall = pd.read_csv(results_dir / "eval_overall.csv")
    write_eval_table_md(
        table,
        overall,
        results_dir / "eval_table.md",
        episodes=args.episodes,
        base_seed=args.base_seed,
        greedy=args.greedy,
    )

    plot_capture_time_boxplot(
        episodes_df, results_dir / "capture_time_boxplot.png"
    )
    stats = plot_discovery_analysis(
        samples_df,
        results_dir / "discovery_analysis.png",
        deterministic=args.greedy,
    )

    write_results_md(
        results_dir,
        seeds=args.seeds,
        episodes=args.episodes,
        durations=args.durations,
        policies=args.policies,
        base_seed=args.base_seed,
        greedy=args.greedy,
        checkpoint_name=args.checkpoint_name,
    )

    print("\nResumen general:")
    eval_mod.print_overall_table(overall)
    print(
        f"\nDescubrimiento: P(repetir | δ<0)="
        f"{eval_mod.format_percent(stats['p_repeat_neg'], 2)}, "
        f"P(repetir | δ>0)="
        f"{eval_mod.format_percent(stats['p_repeat_pos'], 2)} "
        f"(N={stats['n']:,})"
    )
    print(f"\n[ok] resultados -> {results_dir}")


if __name__ == "__main__":
    main()
