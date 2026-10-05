"""De la probabilidad por paso a marcas, y su P/R con tolerancia (ADR 0004 §3 y §4).

Sin torch: lo usan el entrenamiento, la evaluación y, en Python, la referencia.
- `decode`: por clase, máximos locales que pasan el umbral, de mayor a menor, y supresión
  de lo que quede a menos de la tolerancia (NMS temporal).
- `match_events`: emparejado uno a uno dentro de la tolerancia, por clase; P, R y F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ftrain.constants import EVENTS_HZ, N3_CLASSES, N3_TOLERANCE_S

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ClassScore", "decode", "match_events"]


def decode(
    probs: np.ndarray,
    thresholds: Sequence[float],
    *,
    hz: float = EVENTS_HZ,
    window_s: float = N3_TOLERANCE_S,
) -> list[tuple[float, str, float]]:
    """Las marcas (segundo, clase, confianza) de `probs` (T, 1 + clases), con la columna 0
    = «nada». Una marca por pico; cada pico calla a su clase ±`window_s`."""
    marcas: list[tuple[float, str, float]] = []
    radio = round(window_s * hz)
    for k, clase in enumerate(N3_CLASSES):
        p = probs[:, k + 1].astype(np.float64).copy()
        while True:
            i = int(np.argmax(p))
            if p.size == 0 or p[i] < thresholds[k]:
                break
            marcas.append((i / hz, clase, float(probs[i, k + 1])))
            p[max(0, i - radio) : i + radio + 1] = -1.0
    return sorted(marcas)


@dataclass(frozen=True)
class ClassScore:
    truths: int
    predictions: int
    hits: int

    @property
    def precision(self) -> float:
        return self.hits / self.predictions if self.predictions else 0.0

    @property
    def recall(self) -> float:
        return self.hits / self.truths if self.truths else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_events(
    truth: Sequence[tuple[float, str]],
    predicted: Sequence[tuple[float, str, float]],
    *,
    tolerance_s: float = N3_TOLERANCE_S,
) -> dict[str, ClassScore]:
    """P/R por clase: cada marca, de la más segura a la menos, se queda con la etiqueta
    libre más cercana de su clase dentro de la tolerancia."""
    salida = {}
    for clase in N3_CLASSES:
        verdad = sorted(t for t, c in truth if c == clase)
        libres = list(verdad)
        aciertos = 0
        mias = sorted((p for p in predicted if p[1] == clase), key=lambda p: -p[2])
        for t, _, _ in mias:
            cerca = min(libres, key=lambda v: abs(v - t), default=None)
            if cerca is not None and abs(cerca - t) <= tolerance_s:
                libres.remove(cerca)
                aciertos += 1
        salida[clase] = ClassScore(len(verdad), len(mias), aciertos)
    return salida
