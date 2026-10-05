"""Evaluación de detecciones por banda de distancia, luz y partido (ML-26: M4, R-P7-1).

Un recall global esconde justo al jugador de 80 m que decide si la franja sirve. Aquí
todo sale por grupo:

- el emparejado, por clase y sin clase: IoU ≥ `EVAL_IOU_MATCH`; y para las cajas de verdad
  de menos de `EVAL_SMALL_BOX_PX` de ancho, distancia de centros ≤ `EVAL_SMALL_CENTER_PX`
  (en el balón lejano, un píxel de error hunde el IoU);
- recall y precisión a `EVAL_SCORE_THRESHOLD`, y AP50 (interpolación de todos los puntos);
- intervalos de confianza por bootstrap SOBRE PARTIDOS: los frames de un partido no son
  independientes, y remuestrear frames daría intervalos de mentira.

Cada caja de verdad trae su banda (`band`, de `ftrain.bands.band_of`), su luz y su partido.
Las predicciones heredan la banda de la caja con la que emparejan; las que no emparejan
con nada se reparten por la banda de su centro con `band_of_prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ftrain.constants import (
    EVAL_BOOTSTRAP_ROUNDS,
    EVAL_BOOTSTRAP_SEED,
    EVAL_CONFIDENCE,
    EVAL_IOU_MATCH,
    EVAL_SCORE_THRESHOLD,
    EVAL_SMALL_BOX_PX,
    EVAL_SMALL_CENTER_PX,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "Detection",
    "GroupMetrics",
    "Truth",
    "average_precision",
    "bootstrap_recall",
    "evaluate",
    "match_image",
]

ANY_CLASS = "*"
"""La clase del emparejado sin clase."""


@dataclass(frozen=True)
class Truth:
    image: str
    match: str
    cls: str
    box: tuple[float, float, float, float]
    band: int
    light: str = "day"


@dataclass(frozen=True)
class Detection:
    image: str
    cls: str
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class GroupMetrics:
    truths: int
    predictions: int
    true_positives: int
    recall: float
    precision: float
    ap50: float


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _matches(t: Truth, d: Detection) -> bool:
    if t.box[2] - t.box[0] < EVAL_SMALL_BOX_PX:
        ct = ((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2)
        cd = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
        return float(np.hypot(ct[0] - cd[0], ct[1] - cd[1])) <= EVAL_SMALL_CENTER_PX
    return _iou(t.box, d.box) >= EVAL_IOU_MATCH


def match_image(
    truths: Sequence[Truth], detections: Sequence[Detection], *, by_class: bool = True
) -> list[tuple[Detection, Truth | None]]:
    """Empareja las detecciones de UNA imagen, de la más segura a la menos, cada una con
    la mejor caja de verdad libre (por IoU; en las pequeñas, por centro). Devuelve cada
    detección con su caja o None."""
    libres = list(truths)
    salida: list[tuple[Detection, Truth | None]] = []
    for d in sorted(detections, key=lambda x: -x.score):
        candidatas = [t for t in libres if (not by_class or t.cls == d.cls) and _matches(t, d)]
        mejor = max(candidatas, key=lambda t: _iou(t.box, d.box), default=None)
        if mejor is not None:
            libres.remove(mejor)
        salida.append((d, mejor))
    return salida


def average_precision(scored_hits: Sequence[tuple[float, bool]], n_truths: int) -> float:
    """AP con interpolación de todos los puntos (la de COCO/VOC 2010)."""
    if n_truths == 0:
        return 0.0
    orden = sorted(scored_hits, key=lambda x: -x[0])
    aciertos = np.cumsum([1.0 if h else 0.0 for _, h in orden])
    total = np.arange(1, len(orden) + 1, dtype=np.float64)
    recall = np.concatenate([[0.0], aciertos / n_truths]) if orden else np.array([0.0])
    precision = np.concatenate([[1.0], aciertos / total]) if orden else np.array([1.0])
    # La envolvente: la precisión, la mejor a la derecha de cada punto.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


@dataclass(frozen=True)
class _Scored:
    """Una detección emparejada (o no) y el grupo al que cuenta."""

    score: float
    hit: bool
    key: tuple[str, ...]


def _scored(
    truths: Sequence[Truth],
    detections: Sequence[Detection],
    group: Callable[[Truth], tuple[str, ...]],
    band_of_prediction: Callable[[Detection], int] | None,
    *,
    by_class: bool,
) -> list[_Scored]:
    por_imagen: dict[str, tuple[list[Truth], list[Detection]]] = {}
    for t in truths:
        por_imagen.setdefault(t.image, ([], []))[0].append(t)
    for d in detections:
        por_imagen.setdefault(d.image, ([], []))[1].append(d)
    salida: list[_Scored] = []
    for gts, dets in por_imagen.values():
        partido = gts[0].match if gts else ""
        luz = gts[0].light if gts else ""
        for d, t in match_image(gts, dets, by_class=by_class):
            if t is not None:
                salida.append(_Scored(d.score, hit=True, key=group(t)))
            elif band_of_prediction is not None:
                falsa = Truth(d.image, partido, d.cls, d.box, band_of_prediction(d), luz)
                salida.append(_Scored(d.score, hit=False, key=group(falsa)))
    return salida


def evaluate(
    truths: Sequence[Truth],
    detections: Sequence[Detection],
    *,
    group: Callable[[Truth], tuple[str, ...]] = lambda t: (str(t.band),),
    band_of_prediction: Callable[[Detection], int] | None = None,
    by_class: bool = True,
) -> dict[tuple[str, ...], GroupMetrics]:
    """Las métricas por grupo (por defecto, por banda). Sin `band_of_prediction`, los
    falsos positivos no tienen grupo y la precisión se calcula solo con los aciertos."""
    puntos = _scored(truths, detections, group, band_of_prediction, by_class=by_class)
    gt_por_grupo: dict[tuple[str, ...], int] = {}
    for t in truths:
        gt_por_grupo[group(t)] = gt_por_grupo.get(group(t), 0) + 1
    salida: dict[tuple[str, ...], GroupMetrics] = {}
    for clave in sorted(set(gt_por_grupo) | {p.key for p in puntos}):
        mios = [p for p in puntos if p.key == clave]
        n = gt_por_grupo.get(clave, 0)
        sobre_umbral = [p for p in mios if p.score >= EVAL_SCORE_THRESHOLD]
        aciertos = sum(p.hit for p in sobre_umbral)
        salida[clave] = GroupMetrics(
            truths=n,
            predictions=len(sobre_umbral),
            true_positives=aciertos,
            recall=aciertos / n if n else 0.0,
            precision=aciertos / len(sobre_umbral) if sobre_umbral else 0.0,
            ap50=average_precision([(p.score, p.hit) for p in mios], n),
        )
    return salida


def bootstrap_recall(
    truths: Sequence[Truth],
    detections: Sequence[Detection],
    *,
    group: Callable[[Truth], tuple[str, ...]] = lambda t: (str(t.band),),
    rounds: int = EVAL_BOOTSTRAP_ROUNDS,
    by_class: bool = True,
) -> dict[tuple[str, ...], tuple[float, float]]:
    """El intervalo de confianza del recall por grupo, remuestreando PARTIDOS con
    reposición. Determinista: la semilla es `EVAL_BOOTSTRAP_SEED`."""
    partidos = sorted({t.match for t in truths})
    # Por partido y grupo: cajas de verdad y aciertos sobre el umbral.
    gts: dict[tuple[str, tuple[str, ...]], int] = {}
    tps: dict[tuple[str, tuple[str, ...]], int] = {}
    for t in truths:
        gts[(t.match, group(t))] = gts.get((t.match, group(t)), 0) + 1
    for clave in _hit_matches(truths, detections, group, by_class=by_class):
        tps[clave] = tps.get(clave, 0) + 1
    rng = np.random.default_rng(EVAL_BOOTSTRAP_SEED)
    grupos = sorted({k for _, k in gts})
    muestras: dict[tuple[str, ...], list[float]] = {g: [] for g in grupos}
    for _ in range(rounds):
        elegidos = rng.choice(len(partidos), size=len(partidos), replace=True)
        for g in grupos:
            n = sum(gts.get((partidos[i], g), 0) for i in elegidos)
            if n:
                muestras[g].append(sum(tps.get((partidos[i], g), 0) for i in elegidos) / n)
    cola = (1.0 - EVAL_CONFIDENCE) / 2.0
    return {
        g: (float(np.quantile(v, cola)), float(np.quantile(v, 1.0 - cola))) if v else (0.0, 0.0)
        for g, v in muestras.items()
    }


def _hit_matches(
    truths: Sequence[Truth],
    detections: Sequence[Detection],
    group: Callable[[Truth], tuple[str, ...]],
    *,
    by_class: bool,
) -> Iterable[tuple[str, tuple[str, ...]]]:
    """(partido, grupo) de cada caja de verdad encontrada sobre el umbral."""
    por_imagen: dict[str, tuple[list[Truth], list[Detection]]] = {}
    for t in truths:
        por_imagen.setdefault(t.image, ([], []))[0].append(t)
    for d in detections:
        if d.score >= EVAL_SCORE_THRESHOLD:
            por_imagen.setdefault(d.image, ([], []))[1].append(d)
    for gts, dets in por_imagen.values():
        for _, t in match_image(gts, dets, by_class=by_class):
            if t is not None:
                yield (t.match, group(t))
