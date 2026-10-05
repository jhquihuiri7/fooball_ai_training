"""Trayectorias del balón para etiquetar (ML-36).

Del detector (o de la preanotación) llegan candidatos por fotograma: centros con su
confianza, entre ellos el balón y los fantasmas (cabezas, balones de la grada, un poste).
Esto los convierte en pistas que una persona revisa:

- un Kalman de velocidad constante en píxeles nativos a 30 fps, independiente en x e y;
- una puerta de `BALL_GATE_PX_PER_FRAME` alrededor de la predicción: lo que cae fuera
  no es esta pista;
- un repaso offline de cada pista contra todos los candidatos, con una cuadrática
  robusta por fotograma: el fantasma que se coló en un hueco se cae, el balón recupera
  su sitio y la pista se prolonga por los extremos;
- asociación voraz por distancia a la predicción, y la puntuación de cada pista (la suma
  de las confianzas de sus detecciones);
- los huecos de hasta `BALL_INTERP_MAX_GAP_FRAMES` se rellenan con una cuadrática ajustada
  a las detecciones de alrededor;
- banderas de ambigüedad: hueco largo, pista rival cerca, confianza media baja o fuera de
  la máscara del campo.

Una pista con bandera no se tira: sale, y la bandera dice por qué hay que mirarla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from ftrain.constants import (
    BALL_GATE_PX_PER_FRAME,
    BALL_INTERP_CONTEXT_POINTS,
    BALL_INTERP_MAX_GAP_FRAMES,
    BALL_KF_ACCEL_STD_PX,
    BALL_KF_MEAS_STD_PX,
    BALL_LOW_MEAN_SCORE,
    BALL_OUTLIER_PX,
    BALL_REFINE_CONTEXT_POINTS,
    BALL_REFINE_PASSES,
    BALL_RIVAL_RADIUS_PX,
    BALL_TRACK_MAX_COAST_FRAMES,
    BALL_TRACK_MIN_DETECTIONS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "Candidate",
    "Flag",
    "Track",
    "TrackPoint",
    "build_tracks",
    "fill_gaps",
    "refine",
]

QUADRATIC_DEGREE = 2
"""Grado del ajuste de los huecos: una parábola, la del balón en el aire."""


class Flag(StrEnum):
    """Por qué una pista necesita que la mire una persona."""

    LONG_GAP = "long_gap"
    RIVAL_NEARBY = "rival_nearby"
    LOW_CONFIDENCE = "low_confidence"
    OUTSIDE_MASK = "outside_mask"


@dataclass(frozen=True)
class Candidate:
    """Un centro candidato a balón en un fotograma, en píxeles nativos."""

    frame: int
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    x: float
    y: float
    interpolated: bool
    """True si lo puso la cuadrática; False si es una detección."""
    score: float
    """La confianza de la detección; 0 en los interpolados."""


@dataclass
class Track:
    id: int
    points: list[TrackPoint]
    flags: set[Flag] = field(default_factory=set)

    @property
    def detections(self) -> list[TrackPoint]:
        return [p for p in self.points if not p.interpolated]

    @property
    def score(self) -> float:
        """La puntuación de la pista: la suma de las confianzas de sus detecciones."""
        return float(sum(p.score for p in self.detections))

    @property
    def mean_score(self) -> float:
        d = self.detections
        return self.score / len(d) if d else 0.0


class _Axis:
    """Un Kalman 1-D de velocidad constante: posición y velocidad por fotograma."""

    def __init__(self, position: float) -> None:
        self.state = np.array([position, 0.0])
        # Sin velocidad conocida: la incertidumbre de la velocidad inicial es la puerta.
        self.cov = np.diag([BALL_KF_MEAS_STD_PX**2, BALL_GATE_PX_PER_FRAME**2])

    def predict(self, frames: int) -> tuple[np.ndarray, np.ndarray]:
        f = np.array([[1.0, frames], [0.0, 1.0]])
        # Aceleración blanca a trozos (el modelo discreto de siempre).
        g = np.array([[0.5 * frames**2], [float(frames)]])
        q = g @ g.T * BALL_KF_ACCEL_STD_PX**2
        return f @ self.state, f @ self.cov @ f.T + q

    def update(self, frames: int, measured: float) -> None:
        estado, cov = self.predict(frames)
        r = BALL_KF_MEAS_STD_PX**2
        innov = measured - estado[0]
        s = cov[0, 0] + r
        k = cov[:, 0] / s
        self.state = estado + k * innov
        self.cov = cov - np.outer(k, cov[0, :])


@dataclass
class _Live:
    last_frame: int
    ax: _Axis
    ay: _Axis
    found: list[Candidate]

    def predicted(self, frame: int) -> tuple[float, float]:
        dt = frame - self.last_frame
        return float(self.ax.predict(dt)[0][0]), float(self.ay.predict(dt)[0][0])

    def accept(self, c: Candidate) -> None:
        dt = c.frame - self.last_frame
        self.ax.update(dt, c.x)
        self.ay.update(dt, c.y)
        self.last_frame = c.frame
        self.found.append(c)


def _associate(
    vivas: list[_Live], candidatos: list[Candidate], frame: int
) -> tuple[list[tuple[_Live, Candidate]], list[Candidate]]:
    """Parejas pista↔candidato dentro de la puerta, de la más cercana a la más lejana."""
    pares: list[tuple[float, int, int]] = []
    for i, v in enumerate(vivas):
        px, py = v.predicted(frame)
        # Con una sola detección no hay velocidad: la puerta es lo que puede recorrer un
        # balón en los fotogramas transcurridos. Después, fija alrededor de la predicción.
        dt = frame - v.last_frame
        puerta = BALL_GATE_PX_PER_FRAME * (dt if len(v.found) == 1 else 1)
        for j, c in enumerate(candidatos):
            d = float(np.hypot(c.x - px, c.y - py))
            if d <= puerta:
                pares.append((d, i, j))
    pares.sort()
    usadas: set[int] = set()
    tomados: set[int] = set()
    elegidos: list[tuple[_Live, Candidate]] = []
    for _, i, j in pares:
        if i in usadas or j in tomados:
            continue
        usadas.add(i)
        tomados.add(j)
        elegidos.append((vivas[i], candidatos[j]))
    libres = [c for j, c in enumerate(candidatos) if j not in tomados]
    return elegidos, libres


def _quadratic_at(context: Sequence[Candidate], frames: Iterable[int]) -> list[tuple[float, float]]:
    t = np.array([c.frame for c in context], dtype=np.float64)
    grado = min(QUADRATIC_DEGREE, len(context) - 1)
    cx = np.polyfit(t, [c.x for c in context], grado)
    cy = np.polyfit(t, [c.y for c in context], grado)
    return [(float(np.polyval(cx, f)), float(np.polyval(cy, f))) for f in frames]


def _robust_at(points: Sequence[Candidate], frame: int) -> tuple[float, float] | None:
    """Dónde va el balón en `frame` según una cuadrática ajustada a `points`, quitando la
    detección que peor cuadra mientras alguna se aleje más de `BALL_OUTLIER_PX`. None si
    no quedan puntos para una parábola."""
    pts = list(points)
    while len(pts) > QUADRATIC_DEGREE:
        t = np.array([c.frame for c in pts], dtype=np.float64)
        cx = np.polyfit(t, [c.x for c in pts], QUADRATIC_DEGREE)
        cy = np.polyfit(t, [c.y for c in pts], QUADRATIC_DEGREE)
        res = np.hypot(
            np.polyval(cx, t) - [c.x for c in pts], np.polyval(cy, t) - [c.y for c in pts]
        )
        peor = int(np.argmax(res))
        if res[peor] <= BALL_OUTLIER_PX:
            return float(np.polyval(cx, frame)), float(np.polyval(cy, frame))
        pts.pop(peor)
    return None


def _choose(
    frame: int, pista: dict[int, Candidate], pool: dict[int, list[Candidate]]
) -> Candidate | None:
    """El candidato de `frame` que cuadra con las vecinas de la pista, o None.

    Se prueban la cuadrática de las vecinas de los dos lados y la de cada lado por separado:
    en un bote no cuadra la de los dos, pero sí la de antes o la de después. Un fantasma no
    cuadra con ninguna. Sin vecinas para una parábola, se queda lo que hubiera."""
    otras = [c for f, c in pista.items() if f != frame]
    mitad = BALL_REFINE_CONTEXT_POINTS // 2
    grupos = (
        sorted(otras, key=lambda c: abs(c.frame - frame))[:BALL_REFINE_CONTEXT_POINTS],
        sorted((c for c in otras if c.frame < frame), key=lambda c: -c.frame)[:mitad],
        sorted((c for c in otras if c.frame > frame), key=lambda c: c.frame)[:mitad],
    )
    donde = [d for d in (_robust_at(g, frame) for g in grupos) if d is not None]
    if not donde:
        return pista.get(frame)
    cerca = [
        (min(float(np.hypot(c.x - x, c.y - y)) for x, y in donde), k)
        for k, c in enumerate(pool.get(frame, []))
    ]
    cerca = [(d, k) for d, k in cerca if d <= BALL_OUTLIER_PX]
    return pool[frame][min(cerca)[1]] if cerca else None


def _extend_ends(pista: dict[int, Candidate], pool: dict[int, list[Candidate]]) -> None:
    """Prolonga la pista por los dos extremos mientras no pase más de
    `BALL_TRACK_MAX_COAST_FRAMES` sin ver nada."""
    for sentido in (-1, 1):
        sin_ver = 0
        frame = (min(pista) if sentido < 0 else max(pista)) + sentido
        while sin_ver <= BALL_TRACK_MAX_COAST_FRAMES:
            c = _choose(frame, pista, pool)
            if c is None:
                sin_ver += 1
            else:
                pista[frame] = c
                sin_ver = 0
            frame += sentido


def refine(seed: Sequence[Candidate], pool: dict[int, list[Candidate]]) -> list[Candidate]:
    """Rehace una pista en bruto contra TODOS los candidatos de `pool`: en cada fotograma
    toma el que cuadra con sus vecinas (`_choose`), y la prolonga por los extremos. Un
    fantasma que se coló en el seguimiento en línea le devuelve el sitio al balón, y el
    arranque perdido vuelve."""
    pista = {c.frame: c for c in seed}
    for _ in range(BALL_REFINE_PASSES):
        if not pista:
            return []
        nueva = {
            f: c
            for f in range(min(pista), max(pista) + 1)
            if (c := _choose(f, pista, pool)) is not None
        }
        if nueva:
            _extend_ends(nueva, pool)
        if nueva == pista:
            break
        pista = nueva
    return [pista[f] for f in sorted(pista)]


def fill_gaps(found: Sequence[Candidate]) -> tuple[list[TrackPoint], bool]:
    """Las detecciones de una pista con sus huecos cortos rellenos. Devuelve también si
    quedó algún hueco largo (sin rellenar)."""
    puntos: list[TrackPoint] = []
    largo = False
    for k, c in enumerate(found):
        if k > 0:
            hueco = c.frame - found[k - 1].frame - 1
            if 0 < hueco <= BALL_INTERP_MAX_GAP_FRAMES:
                n = BALL_INTERP_CONTEXT_POINTS
                contexto = list(found[max(0, k - n) : k]) + list(found[k : k + n])
                faltan = range(found[k - 1].frame + 1, c.frame)
                puntos.extend(
                    TrackPoint(f, x, y, interpolated=True, score=0.0)
                    for f, (x, y) in zip(faltan, _quadratic_at(contexto, faltan), strict=True)
                )
            elif hueco > BALL_INTERP_MAX_GAP_FRAMES:
                largo = True
        puntos.append(TrackPoint(c.frame, c.x, c.y, interpolated=False, score=c.score))
    return puntos, largo


def _online_tracks(por_frame: dict[int, list[Candidate]]) -> list[_Live]:
    """El seguimiento en línea: Kalman, puerta y asociación, fotograma a fotograma."""
    vivas: list[_Live] = []
    acabadas: list[_Live] = []
    for frame in sorted(por_frame):
        # Las que llevan demasiado sin verse se cierran antes de asociar: lo que llegue
        # ahora empieza otra pista (un corte).
        for v in [v for v in vivas if frame - v.last_frame > BALL_TRACK_MAX_COAST_FRAMES]:
            vivas.remove(v)
            acabadas.append(v)
        elegidos, libres = _associate(vivas, por_frame[frame], frame)
        for v, c in elegidos:
            v.accept(c)
        vivas.extend(_Live(c.frame, _Axis(c.x), _Axis(c.y), [c]) for c in libres)
    return acabadas + vivas


def _flagged(
    track_id: int, found: list[Candidate], inside_mask: Callable[[float, float], bool] | None
) -> Track:
    puntos, largo = fill_gaps(found)
    t = Track(track_id, puntos)
    if largo:
        t.flags.add(Flag.LONG_GAP)
    if t.mean_score < BALL_LOW_MEAN_SCORE:
        t.flags.add(Flag.LOW_CONFIDENCE)
    if inside_mask is not None and not all(inside_mask(p.x, p.y) for p in t.detections):
        t.flags.add(Flag.OUTSIDE_MASK)
    return t


def build_tracks(
    candidates: Iterable[Candidate],
    *,
    inside_mask: Callable[[float, float], bool] | None = None,
) -> list[Track]:
    """Las pistas de una secuencia de candidatos, con huecos rellenos y banderas, de
    la de más puntuación a la de menos. `inside_mask(x, y)` dice si un punto cae en el
    campo; sin máscara no se mira."""
    por_frame: dict[int, list[Candidate]] = {}
    for c in candidates:
        por_frame.setdefault(c.frame, []).append(c)

    # Cada pista en bruto se rehace contra los candidatos que no se haya quedado ya una
    # pista mejor (la más larga primero): los trozos de la misma pelota se quedan sin
    # detecciones y no salen.
    brutas = sorted(
        (v for v in _online_tracks(por_frame) if len(v.found) >= BALL_TRACK_MIN_DETECTIONS),
        key=lambda v: len(v.found),
        reverse=True,
    )
    pool = {f: list(lista) for f, lista in por_frame.items()}
    pistas: list[Track] = []
    for v in brutas:
        limpias = refine([c for c in v.found if c in pool[c.frame]], pool)
        if len(limpias) < BALL_TRACK_MIN_DETECTIONS:
            continue
        for c in limpias:
            pool[c.frame].remove(c)
        pistas.append(_flagged(len(pistas), limpias, inside_mask))
    _flag_rivals(pistas)
    pistas.sort(key=lambda t: t.score, reverse=True)
    return pistas


def _flag_rivals(pistas: list[Track]) -> None:
    donde = [{p.frame: (p.x, p.y) for p in t.points} for t in pistas]
    for i in range(len(pistas)):
        for j in range(i + 1, len(pistas)):
            comunes = donde[i].keys() & donde[j].keys()
            if any(
                np.hypot(donde[i][f][0] - donde[j][f][0], donde[i][f][1] - donde[j][f][1])
                < BALL_RIVAL_RADIUS_PX
                for f in comunes
            ):
                pistas[i].flags.add(Flag.RIVAL_NEARBY)
                pistas[j].flags.add(Flag.RIVAL_NEARBY)
