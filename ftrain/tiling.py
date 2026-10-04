"""Teselado nativo, TTA y fusión para el maestro de autoanotación (ML-24).

El detector ve teselas del lado de SU entrada, recortadas de la banda nativa
con solape; las cajas vuelven a coordenadas nativas y se funden por clase. El
solape garantiza la propiedad que importa: un objeto más pequeño que el solape
cabe ENTERO en al menos una tesela, y la fusión lo deja una sola vez.

El TTA es geometría con inversa exacta: volteo horizontal, y una ampliación
opcional de las filas lejanas (las de arriba de la banda, donde el jugador mide
pocos píxeles) que se deshace dividiendo. Todo numpy; solo la ampliación de la
IMAGEN usa OpenCV, en perezoso (grupo train).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FUSE_IOU",
    "TILE_OVERLAP_PX",
    "Tile",
    "TilingError",
    "crop",
    "flip_boxes",
    "flip_image",
    "fuse",
    "tile_grid",
    "tiles_covering",
    "to_native",
    "unzoom_far_boxes",
    "zoom_far_rows",
]

TILE_OVERLAP_PX: Final = 256
"""Solape entre teselas, en píxeles NATIVOS. Tiene que superar al objeto más
alto que importe (el jugador cercano en 4K): es lo que garantiza que ningún
objeto quede partido en todas las teselas a la vez."""

FUSE_IOU: Final = 0.6
"""IoU desde el que dos cajas de la misma clase son el mismo objeto al fundir.
Más bajo fundiría vecinos pegados; más alto dejaría duplicados en la costura."""


class TilingError(ValueError):
    """La rejilla o las cajas no cuadran. El mensaje dice qué."""


@dataclass(frozen=True, slots=True)
class Tile:
    """Un recorte de la banda nativa: origen y tamaño, en píxeles nativos."""

    x0: int
    y0: int
    width: int
    height: int


def tile_grid(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    overlap: int = TILE_OVERLAP_PX,
) -> tuple[Tile, ...]:
    """La rejilla que cubre (width, height) con teselas fijas y solape mínimo.

    La última tesela de cada eje se clava al borde: el solape real puede crecer
    ahí, nunca encogerse. Si la imagen es menor que la tesela en un eje, sale
    una sola fila/columna anclada a 0 (el relleno es cosa de quien recorta)."""
    if overlap < 0 or overlap >= min(tile_width, tile_height):
        msg = f"el solape ({overlap}) tiene que ser ≥0 y menor que la tesela"
        raise TilingError(msg)

    def ejes(total: int, lado: int) -> list[int]:
        if total <= lado:
            return [0]
        paso = lado - overlap
        origenes = list(range(0, total - lado, paso))
        origenes.append(total - lado)  # la última, al borde
        return origenes

    return tuple(
        Tile(x0=x, y0=y, width=min(tile_width, width), height=min(tile_height, height))
        for y in ejes(height, tile_height)
        for x in ejes(width, tile_width)
    )


def crop(image: np.ndarray, tile: Tile) -> np.ndarray:
    """La vista de la tesela sobre la imagen nativa (sin copiar)."""
    return image[tile.y0 : tile.y0 + tile.height, tile.x0 : tile.x0 + tile.width]


def to_native(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    """Cajas xyxy en píxeles de la tesela → píxeles nativos."""
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float64)
    desplazo = np.array([tile.x0, tile.y0, tile.x0, tile.y0], dtype=np.float64)
    return boxes.astype(np.float64) + desplazo


# --------------------------------------------------------------------------- #
# Fusión por clase (NMS en numpy: conserva la caja de confianza máxima)
# --------------------------------------------------------------------------- #
def _iou(caja: np.ndarray, resto: np.ndarray) -> np.ndarray:
    x1 = np.maximum(caja[0], resto[:, 0])
    y1 = np.maximum(caja[1], resto[:, 1])
    x2 = np.minimum(caja[2], resto[:, 2])
    y2 = np.minimum(caja[3], resto[:, 3])
    interseccion = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (caja[2] - caja[0]) * (caja[3] - caja[1])
    areas = (resto[:, 2] - resto[:, 0]) * (resto[:, 3] - resto[:, 1])
    return interseccion / np.maximum(area + areas - interseccion, 1e-9)


def fuse(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou: float = FUSE_IOU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NMS por clase sobre las cajas ya en nativo, de todas las teselas y TTA.

    Se queda con la caja de confianza MÁXIMA de cada grupo (por eso NMS y no un
    promedio: la aceptación es que la confianza máxima se conserva)."""
    if not (len(boxes) == len(scores) == len(classes)):
        msg = f"tamaños distintos: {len(boxes)} cajas, {len(scores)} scores, {len(classes)} clases"
        raise TilingError(msg)
    if len(boxes) == 0:
        return boxes.reshape(0, 4), scores, classes

    cajas = boxes.astype(np.float64)
    quedan: list[int] = []
    for clase in np.unique(classes):
        indices = np.flatnonzero(classes == clase)
        indices = indices[np.argsort(-scores[indices])]
        while indices.size:
            mejor = indices[0]
            quedan.append(int(mejor))
            resto = indices[1:]
            if not resto.size:
                break
            indices = resto[_iou(cajas[mejor], cajas[resto]) < iou]
    orden = np.array(sorted(quedan, key=lambda i: -scores[i]), dtype=np.int64)
    return cajas[orden], scores[orden], classes[orden]


# --------------------------------------------------------------------------- #
# TTA: volteo horizontal y ampliación de las filas lejanas, con inversa exacta
# --------------------------------------------------------------------------- #
def flip_image(image: np.ndarray) -> np.ndarray:
    return np.fliplr(image)


def flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    """La inversa del volteo horizontal. Es su propia inversa, y es EXACTA
    sobre cajas float32 (lo que emite un detector): la resta `width - x` de un
    float32 contra un ancho entero cabe entera en los 53 bits del float64, así
    que voltear dos veces devuelve el bit a bit original."""
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float64)
    # A float64 ANTES de restar: con el array en float32, numpy restaría en
    # float32 (promoción débil del escalar) y la exactitud se perdería ahí.
    flotantes = boxes.astype(np.float64)
    volteadas = flotantes.copy()
    volteadas[:, 0] = width - flotantes[:, 2]
    volteadas[:, 2] = width - flotantes[:, 0]
    return volteadas


def zoom_far_rows(image: np.ndarray, rows: int, factor: float) -> np.ndarray:
    """Las `rows` filas de arriba (las lejanas), ampliadas por `factor`."""
    import cv2  # noqa: PLC0415 — perezoso adrede (grupo train)

    if rows <= 0 or rows > image.shape[0]:
        msg = f"rows tiene que estar en (0, {image.shape[0]}] y es {rows}"
        raise TilingError(msg)
    if factor <= 1.0:
        msg = f"la ampliación tiene que ser >1 y es {factor}"
        raise TilingError(msg)
    lejanas = image[:rows]
    return cv2.resize(  # type: ignore[no-any-return]
        lejanas,
        (round(lejanas.shape[1] * factor), round(rows * factor)),
        interpolation=cv2.INTER_LINEAR,
    )


def unzoom_far_boxes(boxes: np.ndarray, factor: float) -> np.ndarray:
    """La inversa de la ampliación: las cajas detectadas en el recorte ampliado
    vuelven a píxeles nativos (el recorte empieza en (0,0): basta dividir)."""
    if factor <= 1.0:
        msg = f"la ampliación tiene que ser >1 y es {factor}"
        raise TilingError(msg)
    return boxes.astype(np.float64) / factor


def tiles_covering(tiles: Sequence[Tile], width: int, height: int) -> bool:
    """Si la rejilla cubre el rectángulo entero (para tests y asserts)."""
    cubierto = np.zeros((height, width), dtype=bool)
    for tesela in tiles:
        cubierto[tesela.y0 : tesela.y0 + tesela.height, tesela.x0 : tesela.x0 + tesela.width] = True
    return bool(cubierto.all())
