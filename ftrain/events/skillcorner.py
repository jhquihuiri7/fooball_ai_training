"""SkillCorner opendata a la serie de N3: posiciones, remuestreo y etiquetas (ML-48).

- **Posiciones:** en metros de un campo de 105x68, con el origen en el centro, x a lo
  largo e y a lo ancho, como `PitchModel`. SkillCorner usa el mismo convenio con la
  medida real de cada cancha (`pitch_length`, `pitch_width`), que aquí se escala.
- **Sin equipos:** N0 v1 no los trae, así que N3 no los ve. Sí se sabe quién es portero
  (el rol GK de `match.json`), y eso va en su canal.
- **Remuestreo de 10 a 7,5 Hz:** interpolación lineal entre los dos fotogramas vecinos
  para el jugador que está en los dos; si solo está en uno, se toma de ese.
- **Etiquetas:** de las posesiones de `dynamic_events.csv`:
  - la reanudación que precede a una posesión (`game_interruption_before`) se fecha al
    ACABAR esa posesión, cuando quien reanuda pone el balón en juego: saque de banda,
    falta, saque de puerta y córner; tras un gol, el saque inicial;
  - el gol se fecha al acabar la posesión que lo marca (`game_interruption_after`);
  - además, un saque inicial al empezar cada parte.
  Los penaltis no son una clase de N3 y se dejan fuera.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftrain.constants import (
    EVENT_DEDUP_S,
    EVENTS_HZ,
    N3_CLASSES,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SKILLCORNER_HZ,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

__all__ = [
    "Match",
    "Sample",
    "labels_from_events",
    "load_match",
    "resample",
]

TRACKING_FILE = "tracking_extrapolated.jsonl"
GOALKEEPER_ACRONYM = "GK"

RESTART_LABELS: dict[str, str] = {
    "throw_in": "throw_in",
    "free_kick": "free_kick",
    "goal_kick": "goal_kick",
    "corner": "corner",
    "goal": "kickoff",
}
"""De `game_interruption_before` (sin su `_for`/`_against`) a la clase de N3."""

GOAL_INTERRUPTION = "goal"


@dataclass(frozen=True)
class Sample:
    """Un instante de la serie: jugadores por id (metros) y el balón, si se ve."""

    t_s: float
    players: dict[int, tuple[float, float]]
    ball: tuple[float, float] | None


@dataclass
class Match:
    match_id: int
    samples: list[Sample]
    goalkeepers: set[int]
    labels: list[tuple[float, str]] = field(default_factory=list)


def _kind(interruption: str) -> str:
    """`throw_in_for` → `throw_in`."""
    for sufijo in ("_for", "_against"):
        if interruption.endswith(sufijo):
            return interruption[: -len(sufijo)]
    return interruption


def labels_from_events(
    rows: Iterable[Mapping[str, str]], period_starts: Iterable[int], fps: float = SKILLCORNER_HZ
) -> list[tuple[float, str]]:
    """Las etiquetas (segundos, clase) de las filas de `dynamic_events.csv`."""
    crudas: list[tuple[float, str]] = [(f / fps, "kickoff") for f in period_starts]
    for r in rows:
        if r.get("event_type") != "player_possession":
            continue
        antes = RESTART_LABELS.get(_kind(r.get("game_interruption_before", "")))
        if antes is not None:
            # El balón se pone en juego al acabar la posesión de quien reanuda (mediana de
            # 1 s después de cogerlo, p90 de 3-4 s): ese es el instante físico del evento.
            crudas.append((int(r["frame_end"]) / fps, antes))
        if _kind(r.get("game_interruption_after", "")) == GOAL_INTERRUPTION:
            crudas.append((int(r["frame_end"]) / fps, "goal"))
    salida: list[tuple[float, str]] = []
    for t, c in sorted(crudas):
        if c not in N3_CLASSES:
            continue
        if any(c == c2 and t - t2 < EVENT_DEDUP_S for t2, c2 in salida):
            continue
        salida.append((t, c))
    return salida


def _frame(
    line: Mapping[str, object], sx: float, sy: float
) -> tuple[dict[int, tuple[float, float]], tuple[float, float] | None]:
    jugadores = {
        int(p["player_id"]): (float(p["x"]) * sx, float(p["y"]) * sy)
        for p in line.get("player_data") or []  # type: ignore[union-attr]
        if p.get("x") is not None and p.get("y") is not None
    }
    b = line.get("ball_data") or {}
    balon = (float(b["x"]) * sx, float(b["y"]) * sy) if b.get("x") is not None else None  # type: ignore[union-attr]
    return jugadores, balon


def resample(
    frames: Mapping[int, tuple[dict[int, tuple[float, float]], tuple[float, float] | None]],
    src_hz: float = SKILLCORNER_HZ,
    dst_hz: float = EVENTS_HZ,
) -> list[Sample]:
    """De fotogramas de origen (por índice) a muestras a `dst_hz`, entre el primero y el
    último. Interpola el jugador que está en los dos vecinos; si está en uno, se toma de
    ese. El balón, igual; sin vecinos, el fotograma más cercano."""
    if not frames:
        return []
    primero, ultimo = min(frames), max(frames)
    n = math.floor((ultimo - primero) / src_hz * dst_hz) + 1
    muestras = []
    for k in range(n):
        t = primero / src_hz + k / dst_hz
        pos = t * src_hz
        a, b = math.floor(pos), math.ceil(pos)
        w = pos - a
        fa, fb = frames.get(a), frames.get(b)
        if fa is None or fb is None:
            cerca = frames.get(round(pos)) or fa or fb
            if cerca is None:
                continue
            muestras.append(Sample(t, dict(cerca[0]), cerca[1]))
            continue
        jug = {}
        for pid in fa[0].keys() | fb[0].keys():
            if pid in fa[0] and pid in fb[0]:
                (xa, ya), (xb, yb) = fa[0][pid], fb[0][pid]
                jug[pid] = (xa + (xb - xa) * w, ya + (yb - ya) * w)
            else:
                jug[pid] = fa[0][pid] if pid in fa[0] else fb[0][pid]
        if fa[1] is not None and fb[1] is not None:
            balon: tuple[float, float] | None = (
                fa[1][0] + (fb[1][0] - fa[1][0]) * w,
                fa[1][1] + (fb[1][1] - fa[1][1]) * w,
            )
        else:
            balon = fa[1] if w < 0.5 else fb[1]  # noqa: PLR2004
        muestras.append(Sample(t, jug, balon))
    return muestras


def load_match(directory: Path) -> Match:
    """Un partido de `datasets/skillcorner/<commit>/<id>/`."""
    info = json.loads((directory / "match.json").read_text())
    sx = PITCH_LENGTH_M / float(info["pitch_length"])
    sy = PITCH_WIDTH_M / float(info["pitch_width"])
    porteros = {
        int(p["id"])
        for p in info["players"]
        if isinstance(p.get("player_role"), dict)
        and p["player_role"].get("acronym") == GOALKEEPER_ACRONYM
    }
    fotogramas = {}
    with (directory / TRACKING_FILE).open() as f:
        for linea in f:
            d = json.loads(linea)
            if d.get("period") is None:
                continue
            fotogramas[int(d["frame"])] = _frame(d, sx, sy)
    with (directory / "dynamic_events.csv").open() as f:
        etiquetas = labels_from_events(
            csv.DictReader(f), [int(p["start_frame"]) for p in info["match_periods"]]
        )
    return Match(int(info["id"]), resample(fotogramas), porteros, etiquetas)
