"""Componente de Descubrimiento: memoria episódica de ventana deslizante 1.

Guarda, por entorno, la última acción tomada ``a_{t-1}`` y el último error TD
``δ_{t-1}``. Solo las dos características derivadas (``encode(a_prev)`` y
``tanh(delta_prev)``) entran a la red; el historial es únicamente para
análisis/registro.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from . import config as C

# Código escalar por índice de acción en el orden canónico [UP, DOWN, LEFT, RIGHT].
ACTION_CODE_BY_INDEX = np.asarray(
    [C.ACTION_ENCODINGS[name] for name in C.ACTION_NAMES],
    dtype=np.float32,
)

MaskLike = Union[np.ndarray, Sequence[bool], Sequence[int], None]


def encode_actions(actions: Union[np.ndarray, Sequence[int]]) -> np.ndarray:
    """Codifica índices de acción con la tabla de ``config.ACTION_ENCODINGS``.

    ``-1`` representa "ninguna acción" y se codifica como 0. Los índices
    válidos son ``0..3`` en el orden [UP, DOWN, LEFT, RIGHT].
    """
    actions_array = np.asarray(actions, dtype=np.int64)
    if np.any(actions_array < -1) or np.any(actions_array >= C.NUM_ACTIONS):
        raise ValueError("las acciones deben estar en {-1, 0, 1, 2, 3}")

    safe_actions = np.where(actions_array < 0, 0, actions_array)
    encoded = ACTION_CODE_BY_INDEX[safe_actions]
    encoded = np.where(
        actions_array < 0,
        np.float32(C.ACTION_ENCODING_NONE),
        encoded,
    )
    return encoded.astype(np.float32, copy=False)


class DiscoveryMemory:
    """Memoria episódica vectorizada para N entornos.

    Parameters
    ----------
    num_envs:
        Número de entornos en paralelo.
    enabled:
        Si es ``False``, ``features()`` devuelve ceros (ablación
        "sin descubrimiento": actor-crítico clásico).
    record_history:
        Si es ``True``, guarda ``(acción, δ)`` por entorno para análisis.
        Este historial nunca se usa para construir observaciones.
    """

    def __init__(
        self,
        num_envs: int,
        enabled: bool = True,
        record_history: bool = False,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs debe ser >= 1")
        self.num_envs = int(num_envs)
        self.enabled = bool(enabled)
        self.record_history = bool(record_history)

        # -1 = "ninguna". Se usa int64 para indexar la tabla de codificación.
        self.actions_prev = np.full(self.num_envs, -1, dtype=np.int64)
        self.delta_prev = np.zeros(self.num_envs, dtype=np.float32)

        self.history: Optional[List[List[Tuple[int, float]]]]
        if self.record_history:
            self.history = [[] for _ in range(self.num_envs)]
        else:
            self.history = None

    # ------------------------------------------------------------------
    # Memoria
    # ------------------------------------------------------------------
    def reset(self, mask: MaskLike = None) -> None:
        """Reinicia a ``(ninguna, 0)`` los entornos indicados.

        ``mask`` puede ser un booleano de forma ``(N,)``, una secuencia de
        índices, o ``None`` para reiniciar todos. Se llama con la máscara de
        ``done`` al terminar cada episodio.
        """
        if mask is None:
            indices = np.arange(self.num_envs)
        else:
            mask_array = np.asarray(mask)
            if mask_array.dtype == np.bool_:
                if mask_array.shape != (self.num_envs,):
                    raise ValueError(
                        f"mask booleana debe tener forma ({self.num_envs},)"
                    )
                indices = np.flatnonzero(mask_array)
            else:
                indices = np.asarray(mask, dtype=np.int64).reshape(-1)
                if np.any(indices < 0) or np.any(indices >= self.num_envs):
                    raise IndexError("índice de entorno fuera de rango")

        if indices.size:
            self.actions_prev[indices] = -1
            self.delta_prev[indices] = 0.0

    def features(self) -> np.ndarray:
        """Devuelve ``[encode(a_prev), tanh(delta_prev)]`` de forma ``(N, 2)``.

        Con ``enabled=False`` devuelve siempre ceros.
        """
        if not self.enabled:
            return np.zeros((self.num_envs, 2), dtype=np.float32)

        action_feature = encode_actions(self.actions_prev)
        delta_feature = np.tanh(self.delta_prev).astype(np.float32, copy=False)
        return np.column_stack([action_feature, delta_feature]).astype(
            np.float32, copy=False
        )

    def update(
        self,
        actions: Union[np.ndarray, Sequence[int]],
        deltas: Union[np.ndarray, Sequence[float]],
    ) -> None:
        """Guarda ``a_t`` y ``δ_t`` para el siguiente estado."""
        actions_array = np.asarray(actions, dtype=np.int64).reshape(-1)
        deltas_array = np.asarray(deltas, dtype=np.float32).reshape(-1)
        if actions_array.shape != (self.num_envs,):
            raise ValueError(
                f"actions debe tener forma ({self.num_envs},), "
                f"recibido {actions_array.shape}"
            )
        if deltas_array.shape != (self.num_envs,):
            raise ValueError(
                f"deltas debe tener forma ({self.num_envs},), "
                f"recibido {deltas_array.shape}"
            )
        # Valida también la codificación (no solo el rango del array).
        encode_actions(actions_array)

        self.actions_prev[:] = actions_array
        self.delta_prev[:] = deltas_array

        if self.record_history and self.history is not None:
            for index in range(self.num_envs):
                self.history[index].append(
                    (int(actions_array[index]), float(deltas_array[index]))
                )

    def provisional_features(
        self, actions: Union[np.ndarray, Sequence[int]]
    ) -> np.ndarray:
        """``[encode(a_t), tanh(δ_{t-1})]`` para evaluar V(s'_{t+1}).

        Se usa antes de ``update`` para romper la dependencia circular del
        cálculo de δ_t. Con ``enabled=False`` devuelve ceros (ablación).
        """
        if not self.enabled:
            return np.zeros((self.num_envs, 2), dtype=np.float32)
        features = self.features()
        action_feature = encode_actions(actions).reshape(-1, 1)
        return np.concatenate(
            [action_feature, features[:, 1:2]], axis=1
        ).astype(np.float32, copy=False)

    def clear_history(self) -> None:
        """Vacía el historial de análisis (no afecta a la memoria de la red)."""
        if self.history is not None:
            for entries in self.history:
                entries.clear()
