"""El objetivo y los datos de CenterNet-MNv4, en numpy (plan B de jugadores, ADR 0020).

El contrato de las tres salidas de `plan_b.CenterNetMnv4`, que es lo que la app tendrá que
decodificar, sale de aquí:

- `heatmap [C,H/4,W/4]`: la probabilidad de que en la celda esté el centro de una caja de
  cada clase de `PLAYER_CLASSES`;
- `size [2,H/4,W/4]`: ancho y alto de la caja, **en celdas** (x `CENTERNET_STRIDE` son
  píxeles de entrada);
- `offset [2,H/4,W/4]`: dónde cae el centro dentro de su celda, x e y en [0,1) de celda.

Un centro `(cx, cy)` en píxeles de entrada cae en la celda `floor(c / 4)` y su
desplazamiento es la parte fraccionaria. Se decodifica con `(celda + offset) * 4`.

Para el preentreno con COCO (sin datos propios: ML-17 no existe aún) los ejemplos son un
**mosaico en fila**: imágenes puestas una al lado de otra hasta llenar un lienzo con la forma
de la franja, cada una a una escala al azar. Así las personas salen a tamaños de jugador
lejano y cercano, y el lienzo no se gasta en relleno como con un letterbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ftrain.constants import (
    CENTERNET_GAUSS_ALPHA,
    CENTERNET_MIN_BOX_PX,
    CENTERNET_MIN_SIGMA_CELLS,
    CENTERNET_MIN_VISIBLE,
    CENTERNET_STRIDE,
    LETTERBOX_FILL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "MosaicConfig",
    "Targets",
    "draw_targets",
    "hsv_jitter",
    "letterbox",
    "row_mosaic",
    "xywh_to_xyxy",
]

GAUSS_SPAN_SIGMAS = 3.0
"""Hasta cuántas sigmas se pinta el gaussiano: más allá vale menos de 0,012."""


@dataclass(frozen=True, slots=True)
class Targets:
    """Los mapas de un ejemplo, a la resolución del heatmap (H/4, W/4)."""

    heatmap: np.ndarray
    """[C,h,w] float32: 1 en el centro de cada caja, gaussiano alrededor."""
    size: np.ndarray
    """[2,h,w] float32: ancho y alto en celdas, solo en las celdas centrales."""
    offset: np.ndarray
    """[2,h,w] float32: el centro dentro de su celda, solo en las celdas centrales."""
    mask: np.ndarray
    """[1,h,w] float32: 1 en las celdas centrales (donde cuentan size y offset)."""
    ignore: np.ndarray
    """[C,h,w] float32: 1 donde el heatmap no se penaliza (multitudes, cajas cortadas)."""


@dataclass(frozen=True, slots=True)
class MosaicConfig:
    """Cómo se arma el mosaico en fila del preentreno."""

    height: int
    width: int
    min_rel_height: float = 0.25
    """Alto mínimo de cada imagen en el lienzo, relativo al alto del lienzo."""
    max_rel_height: float = 1.5
    """Alto máximo: por encima de 1 la imagen se recorta en vertical (zoom)."""
    flip_p: float = 0.5
    """Probabilidad de voltear cada imagen en horizontal."""


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


def _gaussian(heat: np.ndarray, cx: float, cy: float, sigma_x: float, sigma_y: float) -> None:
    """Pinta un gaussiano con pico 1 en la celda (cx, cy) por máximo, recortado al mapa."""
    alto, ancho = heat.shape
    rx = math.ceil(GAUSS_SPAN_SIGMAS * sigma_x)
    ry = math.ceil(GAUSS_SPAN_SIGMAS * sigma_y)
    x0, x1 = max(0, int(cx) - rx), min(ancho, int(cx) + rx + 1)
    y0, y1 = max(0, int(cy) - ry), min(alto, int(cy) + ry + 1)
    xs = np.arange(x0, x1, dtype=np.float32) - cx
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    g = np.exp(-(xs[None, :] ** 2) / (2 * sigma_x**2) - (ys[:, None] ** 2) / (2 * sigma_y**2))
    np.maximum(heat[y0:y1, x0:x1], g, out=heat[y0:y1, x0:x1])


def draw_targets(  # noqa: PLR0913 - las cajas, las ignoradas y la forma, todo con nombre
    boxes: np.ndarray,
    classes: np.ndarray,
    *,
    input_hw: tuple[int, int],
    n_classes: int,
    ignore_boxes: np.ndarray | None = None,
    ignore_classes: np.ndarray | None = None,
) -> Targets:
    """Los mapas de CenterNet para cajas xyxy en píxeles de entrada.

    Las cajas se pintan de la más grande a la más pequeña, para que si dos comparten celda
    central se quede la pequeña, que es la difícil. Las `ignore_boxes` (multitudes, cajas
    cortadas por el borde) ponen a 1 el `ignore` de su clase en todo su rectángulo.
    """
    alto, ancho = input_hw[0] // CENTERNET_STRIDE, input_hw[1] // CENTERNET_STRIDE
    heat = np.zeros((n_classes, alto, ancho), np.float32)
    size = np.zeros((2, alto, ancho), np.float32)
    offset = np.zeros((2, alto, ancho), np.float32)
    mask = np.zeros((1, alto, ancho), np.float32)
    ignore = np.zeros((n_classes, alto, ancho), np.float32)

    if ignore_boxes is not None and ignore_classes is not None:
        for (x1, y1, x2, y2), c in zip(
            np.asarray(ignore_boxes).reshape(-1, 4) / CENTERNET_STRIDE,
            np.asarray(ignore_classes).reshape(-1),
            strict=True,
        ):
            ignore[
                int(c),
                max(0, int(y1)) : max(0, math.ceil(y2)),
                max(0, int(x1)) : max(0, math.ceil(x2)),
            ] = 1.0

    cajas = np.asarray(boxes, np.float64).reshape(-1, 4) / CENTERNET_STRIDE
    clases = np.asarray(classes).reshape(-1)
    areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    for k in np.argsort(-areas, kind="stable"):
        x1, y1, x2, y2 = cajas[k]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < ancho and 0 <= iy < alto):
            continue
        sx = max(CENTERNET_GAUSS_ALPHA * w / 6, CENTERNET_MIN_SIGMA_CELLS)
        sy = max(CENTERNET_GAUSS_ALPHA * h / 6, CENTERNET_MIN_SIGMA_CELLS)
        c = int(clases[k])
        _gaussian(heat[c], float(ix), float(iy), sx, sy)
        heat[c, iy, ix] = 1.0
        ignore[c, iy, ix] = 0.0
        size[:, iy, ix] = (w, h)
        offset[:, iy, ix] = (cx - ix, cy - iy)
        mask[0, iy, ix] = 1.0
    return Targets(heatmap=heat, size=size, offset=offset, mask=mask, ignore=ignore)


def letterbox(image: np.ndarray, height: int, width: int) -> tuple[np.ndarray, float]:
    """La imagen escalada sin deformar y pegada en (0,0) sobre gris. Y su escala.

    El protocolo de `tools/dfine_dap.py` (ML-16), para que el AP se compare con D-FINE-N.
    """
    import cv2  # noqa: PLC0415 - grupo train

    escala = min(width / image.shape[1], height / image.shape[0])
    nuevo = (max(1, round(image.shape[1] * escala)), max(1, round(image.shape[0] * escala)))
    lienzo = np.full((height, width, 3), LETTERBOX_FILL, np.uint8)
    lienzo[: nuevo[1], : nuevo[0]] = cv2.resize(image, nuevo, interpolation=cv2.INTER_LINEAR)
    return lienzo, escala


def _clip_boxes(
    boxes: np.ndarray, region: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta cajas xyxy a `region` y devuelve (recortadas, fracción visible del área)."""
    x0, y0, x1, y1 = region
    r = boxes.copy()
    r[:, [0, 2]] = np.clip(r[:, [0, 2]], x0, x1)
    r[:, [1, 3]] = np.clip(r[:, [1, 3]], y0, y1)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    visible = (r[:, 2] - r[:, 0]).clip(0) * (r[:, 3] - r[:, 1]).clip(0) / area
    return r, visible


Sample = tuple[np.ndarray, np.ndarray, np.ndarray]
"""(imagen RGB uint8 HxWx3, cajas xywh, cajas de multitud xywh) en píxeles de la imagen."""


def row_mosaic(
    draw: Callable[[], Sample],
    config: MosaicConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Llena un lienzo con imágenes en fila, cada una a una escala al azar.

    `draw()` da la siguiente imagen. Devuelve el lienzo RGB uint8, las cajas xyxy que cuentan
    y las que se ignoran (multitudes y cajas que quedan visibles en menos de
    `CENTERNET_MIN_VISIBLE` o por debajo de `CENTERNET_MIN_BOX_PX`), en píxeles del lienzo.
    """
    import cv2  # noqa: PLC0415 - grupo train

    alto, ancho = config.height, config.width
    lienzo = np.full((alto, ancho, 3), LETTERBOX_FILL, np.uint8)
    buenas, ignoradas = [], []
    x = 0
    while x < ancho:
        imagen, cajas, multitud = draw()
        rel = math.exp(
            rng.uniform(math.log(config.min_rel_height), math.log(config.max_rel_height))
        )
        escala = alto * rel / imagen.shape[0]
        nw = max(1, round(imagen.shape[1] * escala))
        nh = max(1, round(imagen.shape[0] * escala))
        img = cv2.resize(imagen, (nw, nh), interpolation=cv2.INTER_LINEAR)
        todas = np.concatenate([xywh_to_xyxy(cajas), xywh_to_xyxy(multitud)]) * escala
        n_buenas = len(np.asarray(cajas).reshape(-1, 4))
        if rng.random() < config.flip_p:
            img = img[:, ::-1]
            todas[:, [0, 2]] = nw - todas[:, [2, 0]]
        # Ventana de la imagen que cabe: en vertical, al azar; en horizontal, lo que quede.
        sy = int(rng.integers(0, nh - alto + 1)) if nh > alto else 0
        dy = 0 if nh > alto else int(rng.integers(0, alto - nh + 1))
        vh = min(nh, alto)
        vw = min(nw, ancho - x)
        lienzo[dy : dy + vh, x : x + vw] = img[sy : sy + vh, :vw]
        todas += np.array([x, dy - sy, x, dy - sy], np.float64)
        region = (float(x), float(dy), float(x + vw), float(dy + vh))
        recortadas, visible = _clip_boxes(todas, region)
        lados = np.minimum(recortadas[:, 2] - recortadas[:, 0], recortadas[:, 3] - recortadas[:, 1])
        dentro = visible > 0
        vale = (visible >= CENTERNET_MIN_VISIBLE) & (lados >= CENTERNET_MIN_BOX_PX)
        es_buena = np.arange(len(todas)) < n_buenas
        buenas.append(recortadas[es_buena & vale])
        ignoradas.append(recortadas[dentro & ~(es_buena & vale)])
        x += vw
    return (
        lienzo,
        np.concatenate(buenas).astype(np.float32).reshape(-1, 4),
        np.concatenate(ignoradas).astype(np.float32).reshape(-1, 4),
    )


def hsv_jitter(
    image: np.ndarray, rng: np.random.Generator, gains: tuple[float, float, float]
) -> np.ndarray:
    """Tono, saturación y brillo con ganancias al azar en [1-g, 1+g] (el de YOLO)."""
    import cv2  # noqa: PLC0415 - grupo train

    r = rng.uniform(-1, 1, 3) * np.asarray(gains) + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    lut = np.arange(256, dtype=np.float64)
    h = cv2.LUT(h, ((lut * r[0]) % 180).astype(np.uint8))
    s = cv2.LUT(s, np.clip(lut * r[1], 0, 255).astype(np.uint8))
    v = cv2.LUT(v, np.clip(lut * r[2], 0, 255).astype(np.uint8))
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
