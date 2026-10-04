"""Verificación de grabaciones del soporte (ML-07).

Una grabación que no pasa esto no entra en la campaña: mejor enterarse en la banda,
con el partido aún caliente, que en la ingesta una semana después. Comprueba lo que
el dataset necesita: el códec y el formato declarados, la cadencia, el bitrate (una
recompresión por el camino lo hunde), el rigMs legible y monótono —leído con la
REFERENCIA de football-ai, no con una copia— y, entre las dos cámaras, que el solape
de rigMs cubra el partido.

Los umbrales son inyectables (los tests verifican con vídeos sintéticos pequeños);
los de verdad viven en `ftrain/constants.py` con sus unidades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ftrain.constants import (
    RECORDING_FPS,
    RECORDING_FPS_TOL_HZ,
    RECORDING_HEIGHT,
    RECORDING_MIN_BITRATE_BPS,
    RECORDING_MIN_RIG_OVERLAP,
    RECORDING_MIN_TIMECODE_READABLE,
    RECORDING_SAMPLE_INTERVAL_S,
    RECORDING_WIDTH,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = ["RecordingLimits", "RecordingReport", "check_pair_overlap", "check_recording"]


@dataclass(frozen=True, slots=True)
class RecordingLimits:
    """Los umbrales de la verificación, inyectables: los tests verifican sintéticos
    pequeños y la campaña usa los de `ftrain/constants.py` tal cual."""

    codec: str = "hevc"
    width: int = RECORDING_WIDTH
    height: int = RECORDING_HEIGHT
    fps: float = RECORDING_FPS
    fps_tol_hz: float = RECORDING_FPS_TOL_HZ
    min_bitrate_bps: float = RECORDING_MIN_BITRATE_BPS
    sample_interval_s: float = RECORDING_SAMPLE_INTERVAL_S
    min_readable: float = RECORDING_MIN_TIMECODE_READABLE


@dataclass(slots=True)
class RecordingReport:
    """El informe de una grabación: apto para JSON tal cual."""

    path: str
    ok: bool = True
    checks: list[dict[str, object]] = field(default_factory=list)
    rig_ms_first: int | None = None
    rig_ms_last: int | None = None
    duration_s: float = 0.0

    def add(self, name: str, ok: bool, detail: str) -> None:  # noqa: FBT001 — es el veredicto
        self.checks.append({"name": name, "ok": ok, "detail": detail})
        self.ok = self.ok and ok

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "ok": self.ok,
            "checks": self.checks,
            "rig_ms": [self.rig_ms_first, self.rig_ms_last],
            "duration_s": self.duration_s,
        }


def _default_reader() -> Callable[[np.ndarray], int | None]:
    """El lector de la referencia, perezoso: solo se paga si se verifica."""
    from ftrain.ref import read_timecode_ms  # noqa: PLC0415 — perezoso adrede (grupo ref)

    return read_timecode_ms


def check_recording(
    path: Path,
    *,
    limits: RecordingLimits | None = None,
    read_timecode: Callable[[np.ndarray], int | None] | None = None,
) -> RecordingReport:
    """Verifica una grabación y devuelve su informe. No lanza por un vídeo malo:
    el FALLO con su motivo ES el resultado; solo revienta si el fichero no se abre."""
    import av  # noqa: PLC0415 — perezoso adrede (grupo data)

    tope = limits or RecordingLimits()
    lector = read_timecode or _default_reader()
    informe = RecordingReport(path=str(path))

    with av.open(str(path)) as contenedor:
        stream = contenedor.streams.video[0]
        stream.thread_type = "AUTO"

        nombre = stream.codec_context.name
        informe.add("codec", nombre == tope.codec, f"{nombre} (se esperaba {tope.codec})")
        tamano_ok = stream.width == tope.width and stream.height == tope.height
        informe.add(
            "size",
            tamano_ok,
            f"{stream.width}x{stream.height} (se esperaba {tope.width}x{tope.height})",
        )
        cadencia = float(stream.average_rate or 0)
        informe.add(
            "fps",
            abs(cadencia - tope.fps) <= tope.fps_tol_hz,
            f"{cadencia:.3f} fps (se esperaba {tope.fps}±{tope.fps_tol_hz})",
        )

        duracion = float((stream.duration or 0) * stream.time_base) or float(
            (contenedor.duration or 0) / 1_000_000
        )
        informe.duration_s = duracion
        tasa = (path.stat().st_size * 8 / duracion) if duracion > 0 else 0.0
        informe.add(
            "bitrate",
            tasa >= tope.min_bitrate_bps,
            f"{tasa / 1e6:.1f} Mbit/s de media (mínimo {tope.min_bitrate_bps / 1e6:.0f})",
        )

        legibles = 0
        muestras = 0
        monotono = True
        anterior: int | None = None
        proxima_s = 0.0
        for frame in contenedor.decode(stream):
            if frame.pts is None:
                continue
            instante = float(frame.pts * stream.time_base)
            if instante < proxima_s:
                continue
            proxima_s = instante + tope.sample_interval_s
            muestras += 1
            gris = frame.reformat(format="gray").to_ndarray()
            valor = lector(np.ascontiguousarray(gris))
            if valor is None:
                continue
            legibles += 1
            if informe.rig_ms_first is None:
                informe.rig_ms_first = valor
            if anterior is not None and valor <= anterior:
                monotono = False
            anterior = valor
            informe.rig_ms_last = valor

        fraccion = (legibles / muestras) if muestras else 0.0
        informe.add(
            "timecode_readable",
            muestras > 0 and fraccion >= tope.min_readable,
            f"{legibles}/{muestras} legibles ({fraccion:.1%}, mínimo {tope.min_readable:.0%})",
        )
        informe.add(
            "timecode_monotonic",
            monotono and legibles > 1,
            "rigMs estrictamente creciente" if monotono else "rigMs retrocede en la muestra",
        )
    return informe


def check_pair_overlap(
    left: RecordingReport,
    right: RecordingReport,
    *,
    min_overlap: float = RECORDING_MIN_RIG_OVERLAP,
) -> dict[str, object]:
    """El solape de rigMs entre las dos cámaras, sobre la UNIÓN de lo grabado.

    La unión y no el máximo: si una cámara arrancó dos minutos tarde, esos dos
    minutos no tienen pareja y cuentan en contra, que es exactamente lo que se
    quiere cazar."""
    valores = (left.rig_ms_first, left.rig_ms_last, right.rig_ms_first, right.rig_ms_last)
    if any(valor is None for valor in valores):
        return {"name": "rig_overlap", "ok": False, "detail": "falta rigMs en un lado"}
    li, lf = int(left.rig_ms_first or 0), int(left.rig_ms_last or 0)
    ri, rf = int(right.rig_ms_first or 0), int(right.rig_ms_last or 0)
    solape = max(0, min(lf, rf) - max(li, ri))
    union = max(lf, rf) - min(li, ri)
    fraccion = (solape / union) if union > 0 else 0.0
    return {
        "name": "rig_overlap",
        "ok": fraccion >= min_overlap,
        "detail": f"{fraccion:.1%} del partido con las dos cámaras (mínimo {min_overlap:.0%})",
    }
