#!/usr/bin/env python3
"""Exporta el Actor-Crítico de RobotPVP a JSON para la inferencia en Kotlin.

El JSON contiene los pesos con la convención ``y = W·x + b`` y ``W`` de forma
``[salidas][entradas]`` (igual que ``nn.Linear.weight``), además de un bloque
``meta`` con todas las constantes compartidas tomadas de ``config.py`` y los
datos del checkpoint.

También genera ``robot_ac_test_vectors.json`` con 32 entradas aleatorias en
``[-1, 1]^8`` y sus salidas esperadas (``probs`` y ``value``) calculadas con
PyTorch en float32, para verificar la inferencia de Kotlin.

Uso:
    python export.py --checkpoint runs/ac_disc_s1/best.pt \
        --out ../app/src/main/assets/robot_ac_model.json

Salidas por defecto:
    ../app/src/main/assets/robot_ac_model.json
    ../app/src/test/resources/robot_ac_test_vectors.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from robot_ai import config as C
from robot_ai.model import SharedActorCritic

ML_DIR = Path(__file__).resolve().parent
REPO_DIR = ML_DIR.parent

DEFAULT_CHECKPOINT = ML_DIR / "runs" / "ac_disc_s1" / "best.pt"
DEFAULT_OUT = REPO_DIR / "app" / "src" / "main" / "assets" / "robot_ac_model.json"
DEFAULT_TEST_VECTORS_OUT = (
    REPO_DIR / "app" / "src" / "test" / "resources"
    / "robot_ac_test_vectors.json"
)

MODEL_VERSION = 1
TEST_VECTORS_VERSION = 1
NUM_TEST_VECTORS = 32
TEST_VECTOR_SEED = 20261103

# Orden exacto de las entradas del agente (fuente única: config.py).
INPUT_ORDER = C.INPUT_NAMES


# ---------------------------------------------------------------------------
# Carga del checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[SharedActorCritic, Dict[str, Any]]:
    """Carga el modelo y el payload del checkpoint en modo evaluación."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in payload:
        raise KeyError(
            f"El checkpoint {checkpoint_path} no contiene 'model_state_dict'"
        )

    model = SharedActorCritic().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    param_count = model.count_params()
    if param_count != C.PARAM_COUNT:
        raise RuntimeError(
            f"El modelo tiene {param_count} parámetros y la especificación "
            f"exige {C.PARAM_COUNT}"
        )
    return model, payload


# ---------------------------------------------------------------------------
# Construcción del JSON
# ---------------------------------------------------------------------------
def _as_float_list(tensor: torch.Tensor) -> List[Any]:
    array = tensor.detach().cpu().numpy().astype(np.float32)
    return array.tolist()


def _created_at() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _reward_meta() -> Dict[str, float]:
    """Recompensa en el formato que consume Kotlin.

    ``config.REWARD_STEP_PENALTY`` se guarda en positivo (0.001) porque el
    entorno le resta el castigo; el JSON lo publica como valor negativo.
    """
    return {
        "approach_coef": float(C.REWARD_DISTANCE_SCALE),
        "step_penalty": float(-C.REWARD_STEP_PENALTY),
        "capture": float(C.REWARD_CAPTURE),
        "timeout": float(C.REWARD_TIMEOUT),
    }


def build_model_dict(
    model: SharedActorCritic,
    payload: Dict[str, Any],
    *,
    created_at: str,
) -> Dict[str, Any]:
    """Construye el diccionario completo que se serializa a JSON."""
    state = model.state_dict()
    weights = {
        "W1": _as_float_list(state["fc1.weight"]),
        "b1": _as_float_list(state["fc1.bias"]),
        "Wa": _as_float_list(state["actor.weight"]),
        "ba": _as_float_list(state["actor.bias"]),
        "Wc": _as_float_list(state["critic.weight"]),
        "bc": _as_float_list(state["critic.bias"]),
    }

    discovery_enabled = bool(payload.get("discovery_enabled", True))
    trained_steps = int(payload.get("steps", 0))
    seed = int(payload.get("seed", -1))
    raw_capture_rate = payload.get("best_capture_rate", None)
    eval_capture_rate = (
        float(raw_capture_rate)
        if raw_capture_rate is not None
        and math.isfinite(float(raw_capture_rate))
        else 0.0
    )

    meta = {
        "actions": list(C.ACTION_NAMES),
        "action_encoding": [
            float(C.ACTION_ENCODINGS[name]) for name in C.ACTION_NAMES
        ],
        "no_action_encoding": float(C.ACTION_ENCODING_NONE),
        "action_repeat": int(C.ACTION_REPEAT),
        "agent_speed": float(C.AGENT_SPEED),
        "gamma": float(C.GAMMA),
        "reward": _reward_meta(),
        "theta_zero_speed": float(C.PLAYER_VELOCITY_EPS),
        "discovery_enabled": discovery_enabled,
        "input_order": list(INPUT_ORDER),
        "trained_steps": trained_steps,
        "seed": seed,
        "eval_capture_rate": eval_capture_rate,
        "created_at": created_at,
    }

    return {
        "version": MODEL_VERSION,
        "architecture": {
            "input": int(C.INPUT_DIM),
            "hidden": int(C.HIDDEN_DIM),
            "actor": int(C.ACTOR_OUTPUT_DIM),
            "critic": int(C.CRITIC_OUTPUT_DIM),
            "hidden_activation": "tanh",
        },
        "weights": weights,
        "meta": meta,
    }


def build_test_vectors(
    model: SharedActorCritic,
    *,
    seed: int = TEST_VECTOR_SEED,
    count: int = NUM_TEST_VECTORS,
    device: torch.device,
) -> Dict[str, Any]:
    """32 entradas uniformes en [-1, 1]^8 con salidas PyTorch float32."""
    rng = np.random.default_rng(int(seed))
    inputs = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(int(count), C.INPUT_DIM),
    ).astype(np.float32)

    with torch.no_grad():
        logits, value = model(
            torch.as_tensor(inputs, dtype=torch.float32, device=device)
        )
        probs = torch.softmax(logits, dim=-1)

    return {
        "version": TEST_VECTORS_VERSION,
        "seed": int(seed),
        "count": int(count),
        "inputs": inputs.tolist(),
        "probs": probs.detach().cpu().numpy().astype(np.float32).tolist(),
        "value": (
            value.squeeze(-1).detach().cpu().numpy().astype(np.float32).tolist()
        ),
    }


# ---------------------------------------------------------------------------
# Serialización y verificación
# ---------------------------------------------------------------------------
def write_json(path: Path, payload: Dict[str, Any]) -> int:
    """Escribe JSON legible y devuelve su tamaño en bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(data)


def numpy_forward(
    weights: Dict[str, List[Any]],
    inputs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward en NumPy puro leyendo los pesos del JSON.

    Se usa en la verificación para comprobar la paridad con PyTorch.
    """
    x = np.asarray(inputs, dtype=np.float32)
    w1 = np.asarray(weights["W1"], dtype=np.float32)
    b1 = np.asarray(weights["b1"], dtype=np.float32)
    wa = np.asarray(weights["Wa"], dtype=np.float32)
    ba = np.asarray(weights["ba"], dtype=np.float32)
    wc = np.asarray(weights["Wc"], dtype=np.float32)
    bc = np.asarray(weights["bc"], dtype=np.float32)

    hidden = np.tanh(x @ w1.T + b1)
    logits = hidden @ wa.T + ba
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    value = hidden @ wc.T + bc
    return probs, value.squeeze(-1)


def verify_parity(
    model_payload: Dict[str, Any],
    vectors_payload: Dict[str, Any],
    *,
    tolerance: float = 1e-5,
) -> Tuple[float, float]:
    """Comprueba NumPy vs las salidas PyTorch de los vectores de prueba."""
    probs, value = numpy_forward(
        model_payload["weights"], vectors_payload["inputs"]
    )
    expected_probs = np.asarray(vectors_payload["probs"], dtype=np.float32)
    expected_value = np.asarray(vectors_payload["value"], dtype=np.float32)

    prob_error = float(np.max(np.abs(probs - expected_probs)))
    value_error = float(np.max(np.abs(value - expected_value)))
    if not math.isfinite(prob_error) or not math.isfinite(value_error):
        raise RuntimeError("La verificación produjo valores no finitos")
    if prob_error >= tolerance or value_error >= tolerance:
        raise RuntimeError(
            "Paridad NumPy/PyTorch fuera de tolerancia: "
            f"probs={prob_error:.3e}, value={value_error:.3e} "
            f"(tolerancia {tolerance:.0e})"
        )
    return prob_error, value_error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta el Actor-Crítico de RobotPVP a JSON"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="checkpoint de train.py (best.pt o last.pt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="ruta del JSON del modelo",
    )
    parser.add_argument(
        "--test-vectors-out",
        type=str,
        default=str(DEFAULT_TEST_VECTORS_OUT),
        help="ruta de los vectores de prueba para Kotlin",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TEST_VECTOR_SEED,
        help="semilla de las 32 entradas aleatorias de prueba",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    vectors_path = Path(args.test_vectors_out).expanduser().resolve()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            "[export] CUDA no disponible; se usa CPU", file=sys.stderr
        )
        device = torch.device("cpu")

    model, payload = load_checkpoint(checkpoint_path, device)
    created_at = _created_at()

    model_payload = build_model_dict(
        model, payload, created_at=created_at
    )
    vectors_payload = build_test_vectors(
        model, seed=args.seed, device=device
    )

    prob_error, value_error = verify_parity(model_payload, vectors_payload)

    model_bytes = write_json(out_path, model_payload)
    vectors_bytes = write_json(vectors_path, vectors_payload)

    if model_bytes >= 20 * 1024:
        raise RuntimeError(
            f"El JSON del modelo pesa {model_bytes} bytes y debe ser < 20 KB"
        )

    print(f"[export] checkpoint      : {checkpoint_path}")
    print(
        f"[export] pasos           : {int(payload.get('steps', 0)):,} "
        f"(semilla {int(payload.get('seed', -1))})"
    )
    print(
        f"[export] descubrimiento  : "
        f"{bool(payload.get('discovery_enabled', True))}"
    )
    print(f"[export] parámetros      : {model.count_params()}")
    print(f"[export] modelo          : {out_path} ({model_bytes:,} bytes)")
    print(f"[export] vectores prueba : {vectors_path} ({vectors_bytes:,} bytes)")
    print(
        f"[export] paridad numpy   : probs={prob_error:.2e}, "
        f"value={value_error:.2e}"
    )


if __name__ == "__main__":
    main()
