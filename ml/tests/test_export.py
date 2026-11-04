"""Tests de la exportación a JSON y de la paridad NumPy/PyTorch.

Verifican el artefacto real (``app/src/main/assets/robot_ac_model.json`` y
``app/src/test/resources/robot_ac_test_vectors.json``) y, si el checkpoint
existe, que el JSON corresponde exactamente a ese checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

import export as E
from robot_ai import config as C
from robot_ai.model import SharedActorCritic

REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_DIR / "app" / "src" / "main" / "assets" / "robot_ac_model.json"
VECTORS_PATH = (
    REPO_DIR / "app" / "src" / "test" / "resources"
    / "robot_ac_test_vectors.json"
)
CHECKPOINT_PATH = REPO_DIR / "ml" / "runs" / "ac_disc_s1" / "best.pt"

TOLERANCE = 1e-5
MAX_MODEL_BYTES = 20 * 1024


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"Falta el artefacto exportado: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _model_from_json(payload: Dict[str, Any]) -> SharedActorCritic:
    weights = payload["weights"]
    model = SharedActorCritic()
    state = {
        "fc1.weight": torch.tensor(weights["W1"], dtype=torch.float32),
        "fc1.bias": torch.tensor(weights["b1"], dtype=torch.float32),
        "actor.weight": torch.tensor(weights["Wa"], dtype=torch.float32),
        "actor.bias": torch.tensor(weights["ba"], dtype=torch.float32),
        "critic.weight": torch.tensor(weights["Wc"], dtype=torch.float32),
        "critic.bias": torch.tensor(weights["bc"], dtype=torch.float32),
    }
    model.load_state_dict(state)
    model.eval()
    return model


def _torch_outputs(
    model: SharedActorCritic,
    inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, value = model(
            torch.as_tensor(np.asarray(inputs, dtype=np.float32))
        )
        probs = torch.softmax(logits, dim=-1)
    return (
        probs.numpy().astype(np.float32),
        value.squeeze(-1).numpy().astype(np.float32),
    )


def _assert_parity(
    actual_probs: np.ndarray,
    actual_value: np.ndarray,
    expected_probs: np.ndarray,
    expected_value: np.ndarray,
) -> None:
    prob_error = float(np.max(np.abs(actual_probs - expected_probs)))
    value_error = float(np.max(np.abs(actual_value - expected_value)))
    assert math.isfinite(prob_error) and math.isfinite(value_error)
    assert prob_error < TOLERANCE, f"Error máximo en probs: {prob_error:.3e}"
    assert value_error < TOLERANCE, f"Error máximo en value: {value_error:.3e}"


# ---------------------------------------------------------------------------
# Artefacto del modelo
# ---------------------------------------------------------------------------
def test_model_json_size_and_top_level_structure() -> None:
    assert MODEL_PATH.exists(), f"Falta {MODEL_PATH}"
    assert MODEL_PATH.stat().st_size < MAX_MODEL_BYTES

    payload = _load_json(MODEL_PATH)
    assert payload["version"] == 1
    assert set(payload) == {"version", "architecture", "weights", "meta"}
    assert payload["architecture"] == {
        "input": C.INPUT_DIM,
        "hidden": C.HIDDEN_DIM,
        "actor": C.ACTOR_OUTPUT_DIM,
        "critic": C.CRITIC_OUTPUT_DIM,
        "hidden_activation": "tanh",
    }


def test_weights_shapes_and_parameter_count() -> None:
    weights = _load_json(MODEL_PATH)["weights"]
    assert set(weights) == {"W1", "b1", "Wa", "ba", "Wc", "bc"}

    expected_shapes = {
        "W1": (C.HIDDEN_DIM, C.INPUT_DIM),
        "b1": (C.HIDDEN_DIM,),
        "Wa": (C.ACTOR_OUTPUT_DIM, C.HIDDEN_DIM),
        "ba": (C.ACTOR_OUTPUT_DIM,),
        "Wc": (C.CRITIC_OUTPUT_DIM, C.HIDDEN_DIM),
        "bc": (C.CRITIC_OUTPUT_DIM,),
    }
    total = 0
    for name, shape in expected_shapes.items():
        array = np.asarray(weights[name], dtype=np.float32)
        assert array.shape == shape, f"{name}: {array.shape} != {shape}"
        assert np.isfinite(array).all()
        total += array.size
    assert total == C.PARAM_COUNT == 229


def test_meta_matches_config_and_checkpoint_fields() -> None:
    meta = _load_json(MODEL_PATH)["meta"]

    assert meta["actions"] == list(C.ACTION_NAMES)
    assert meta["action_encoding"] == [
        C.ACTION_ENCODINGS[name] for name in C.ACTION_NAMES
    ]
    assert meta["no_action_encoding"] == C.ACTION_ENCODING_NONE
    assert meta["action_repeat"] == C.ACTION_REPEAT
    assert meta["agent_speed"] == C.AGENT_SPEED
    assert meta["gamma"] == C.GAMMA
    assert meta["theta_zero_speed"] == C.PLAYER_VELOCITY_EPS
    assert meta["reward"] == {
        "approach_coef": C.REWARD_DISTANCE_SCALE,
        "step_penalty": -C.REWARD_STEP_PENALTY,
        "capture": C.REWARD_CAPTURE,
        "timeout": C.REWARD_TIMEOUT,
    }
    assert meta["input_order"] == list(E.INPUT_ORDER)
    assert isinstance(meta["discovery_enabled"], bool)
    assert int(meta["trained_steps"]) > 0
    assert int(meta["seed"]) >= 0
    assert 0.0 <= float(meta["eval_capture_rate"]) <= 1.0
    assert str(meta["created_at"]).endswith("Z")


def test_numpy_forward_matches_pytorch_from_json() -> None:
    payload = _load_json(MODEL_PATH)
    vectors = _load_json(VECTORS_PATH)
    inputs = np.asarray(vectors["inputs"], dtype=np.float32)

    model = _model_from_json(payload)
    expected_probs, expected_value = _torch_outputs(model, inputs)
    actual_probs, actual_value = E.numpy_forward(payload["weights"], inputs)

    _assert_parity(
        actual_probs, actual_value, expected_probs, expected_value
    )


def test_test_vectors_have_expected_format_and_outputs() -> None:
    payload = _load_json(MODEL_PATH)
    vectors = _load_json(VECTORS_PATH)

    inputs = np.asarray(vectors["inputs"], dtype=np.float32)
    probs = np.asarray(vectors["probs"], dtype=np.float32)
    value = np.asarray(vectors["value"], dtype=np.float32)

    assert vectors["count"] == E.NUM_TEST_VECTORS
    assert inputs.shape == (E.NUM_TEST_VECTORS, C.INPUT_DIM)
    assert probs.shape == (E.NUM_TEST_VECTORS, C.ACTOR_OUTPUT_DIM)
    assert value.shape == (E.NUM_TEST_VECTORS,)
    assert np.all(inputs >= -1.0) and np.all(inputs <= 1.0)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    actual_probs, actual_value = E.numpy_forward(payload["weights"], inputs)
    _assert_parity(actual_probs, actual_value, probs, value)


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="checkpoint de entrenamiento no disponible",
)
def test_export_roundtrip_from_checkpoint(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model, checkpoint = E.load_checkpoint(CHECKPOINT_PATH, device)
    model_payload = E.build_model_dict(
        model, checkpoint, created_at="2026-01-01T00:00:00Z"
    )
    vectors_payload = E.build_test_vectors(model, device=device)

    model_path = tmp_path / "model.json"
    vectors_path = tmp_path / "vectors.json"
    model_bytes = E.write_json(model_path, model_payload)
    E.write_json(vectors_path, vectors_payload)
    assert model_bytes < MAX_MODEL_BYTES

    prob_error, value_error = E.verify_parity(
        model_payload, vectors_payload
    )
    assert prob_error < TOLERANCE
    assert value_error < TOLERANCE

    # Además de comparar contra las salidas guardadas, se compara el forward
    # NumPy del JSON con el modelo PyTorch cargado del checkpoint.
    inputs = np.asarray(vectors_payload["inputs"], dtype=np.float32)
    expected_probs, expected_value = _torch_outputs(model, inputs)
    actual_probs, actual_value = E.numpy_forward(
        model_payload["weights"], inputs
    )
    _assert_parity(
        actual_probs, actual_value, expected_probs, expected_value
    )
