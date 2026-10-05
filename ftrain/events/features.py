"""La rejilla de ocupación de N3 y su etiqueta por paso (ML-48, ADR 0004 §1).

Una rejilla de `N3_GRID_H`x`N3_GRID_W` celdas por paso, con los canales de `N3_CHANNELS`:
jugadores por celda, su velocidad media (m/s, de la muestra anterior), porteros,
árbitros, el balón y el silbato. Sin equipos, como N0 v1. El campo es el de referencia
(105x68, origen en el centro); lo que cae fuera (un saque de banda pisa la línea) se
lleva a la celda del borde.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ftrain.constants import (
    EVENT_LABEL_DILATION_STEPS,
    EVENTS_HZ,
    N3_CHANNELS,
    N3_CLASSES,
    N3_GRID_H,
    N3_GRID_W,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ftrain.events.skillcorner import Match, Sample

__all__ = ["cell_of", "grids", "label_timeline"]

CH = {nombre: k for k, nombre in enumerate(N3_CHANNELS)}


def cell_of(x: float, y: float) -> tuple[int, int]:
    """(fila, columna) de un punto en metros del campo de referencia."""
    col = int((x + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * N3_GRID_W)
    fila = int((y + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * N3_GRID_H)
    return min(max(fila, 0), N3_GRID_H - 1), min(max(col, 0), N3_GRID_W - 1)


def _grid(s: Sample, prev: Sample | None, goalkeepers: set[int], referees: set[int]) -> np.ndarray:
    g = np.zeros((len(N3_CHANNELS), N3_GRID_H, N3_GRID_W), dtype=np.float32)
    dt = (s.t_s - prev.t_s) if prev is not None else 0.0
    for pid, (x, y) in s.players.items():
        f, c = cell_of(x, y)
        canal = CH["referee"] if pid in referees else CH["players"]
        g[canal, f, c] += 1.0
        if pid in goalkeepers:
            g[CH["goalkeeper"], f, c] += 1.0
        if prev is not None and dt > 0 and pid in prev.players:
            px, py = prev.players[pid]
            g[CH["vx"], f, c] += (x - px) / dt
            g[CH["vy"], f, c] += (y - py) / dt
    con_gente = g[CH["players"]] > 0
    g[CH["vx"]][con_gente] /= g[CH["players"]][con_gente]
    g[CH["vy"]][con_gente] /= g[CH["players"]][con_gente]
    if s.ball is not None:
        f, c = cell_of(*s.ball)
        g[CH["ball"], f, c] = 1.0
    return g


def grids(match: Match, referees: set[int] | None = None) -> np.ndarray:
    """La serie de rejillas del partido: (T, canales, alto, ancho), float32."""
    arbitros = referees or set()
    salida = [
        _grid(s, match.samples[k - 1] if k else None, match.goalkeepers, arbitros)
        for k, s in enumerate(match.samples)
    ]
    if not salida:
        return np.zeros((0, len(N3_CHANNELS), N3_GRID_H, N3_GRID_W), dtype=np.float32)
    return np.stack(salida)


def label_timeline(samples: Sequence[Sample], labels: Sequence[tuple[float, str]]) -> np.ndarray:
    """La clase de cada paso: 0 = nada; la clase k de `N3_CLASSES` va en k + 1, en el
    paso más cercano a su instante y en ±`EVENT_LABEL_DILATION_STEPS`. Si dos eventos se
    pisan, gana el más cercano a su instante."""
    y = np.zeros(len(samples), dtype=np.int64)
    if not samples:
        return y
    t0 = samples[0].t_s
    distancia = np.full(len(samples), np.inf)
    for t, c in labels:
        centro = round((t - t0) * EVENTS_HZ)
        for k in range(
            centro - EVENT_LABEL_DILATION_STEPS, centro + EVENT_LABEL_DILATION_STEPS + 1
        ):
            if 0 <= k < len(samples) and abs(k - centro) < distancia[k]:
                distancia[k] = abs(k - centro)
                y[k] = N3_CLASSES.index(c) + 1
    return y
