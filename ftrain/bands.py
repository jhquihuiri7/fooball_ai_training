"""Bandas de distancia y paso de etiquetas nativo↔lienzo (ML-19).

Las métricas del proyecto van POR BANDA DE DISTANCIA (`DISTANCE_BANDS_M`): un recall
global esconde justo al jugador de 80 m que decide si la franja sirve. Esto resuelve
las dos cuentas que toda la campaña repite:

- **a qué banda pertenece una etiqueta**: por homografía si hay `pitch.json`
  (la referencia `PitchModel.image_to_pitch` y la posición del soporte), o con la
  tabla fila→metros que `band.json` trae de REF-29;
- **dónde cae en el lienzo 1920x576 (o el mosaico)** una caja anotada en nativo, y
  la vuelta — usando la MISMA `BandGeometry` de la referencia, nunca una copia, y
  teniendo en cuenta la rotación de 180° del móvil invertido y la franja del código
  de tiempo, que está enmascarada y no puede aportar etiquetas.

Una etiqueta problemática SE MARCA (`status`), no se pierde en silencio: el dataset
tiene que poder auditar cuántas cayeron fuera y por qué.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ftrain.constants import DISTANCE_BANDS_M, RECORDING_HEIGHT, RECORDING_WIDTH

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

__all__ = [
    "BandsError",
    "MappedPoint",
    "SideBand",
    "band_of",
    "load_band_spec",
]

ROTATION_INVERTED_DEG = 180
"""El montaje del móvil invertido (ADR 0012): girado 180° respecto al otro."""

STATUS_OK = "ok"
STATUS_OUTSIDE = "fuera"
STATUS_TIMECODE = "timecode"


class BandsError(ValueError):
    """El spec de la franja no sirve. El mensaje dice qué falta."""


def band_of(distance_m: float) -> int:
    """El índice de la banda de `DISTANCE_BANDS_M` a la que pertenece la distancia."""
    if distance_m < 0:
        msg = f"una distancia negativa no es de este campo: {distance_m}"
        raise BandsError(msg)
    indice = bisect_right(DISTANCE_BANDS_M, distance_m, 1, len(DISTANCE_BANDS_M) - 1)
    return indice - 1


@dataclass(frozen=True, slots=True)
class MappedPoint:
    """Un punto llevado al lienzo (o de vuelta), con su veredicto."""

    x: float
    y: float
    region: int
    status: str


@dataclass(frozen=True, slots=True)
class SideBand:
    """La franja de un lado, lista para mapear etiquetas de ese móvil."""

    side: str
    geometry: Any  # la BandGeometry de la referencia (fachada ftrain.ref)
    row_to_m: tuple[tuple[float, float], ...]
    rotation_deg: int
    frame_w: int
    frame_h: int

    # ------------------------------------------------------------------ #
    def upright(self, x: float, y: float) -> tuple[float, float]:
        """Del frame TAL CUAL SE ALMACENÓ (donde anota CVAT) al enderezado."""
        if self.rotation_deg == ROTATION_INVERTED_DEG:
            return (self.frame_w - x, self.frame_h - y)
        return (x, y)

    def stored(self, x: float, y: float) -> tuple[float, float]:
        """La vuelta: del enderezado al almacenado. Girar 180° es su propia inversa."""
        return self.upright(x, y)

    def in_timecode_strip(self, x_up: float, y_up: float) -> bool:
        """Si el punto (ya enderezado) cae en la franja enmascarada del código."""
        from ftrain.ref import (  # noqa: PLC0415 — perezoso adrede (grupo ref)
            TIMECODE_BITS,
            strip_height,
        )

        lado = strip_height(self.frame_w)
        ancho = lado * TIMECODE_BITS
        if self.rotation_deg == ROTATION_INVERTED_DEG:
            # El código se pintó en el (0,0) del capturado: tras enderezar el frame
            # invertido queda abajo a la derecha, que es donde el panel lo tapa.
            return x_up >= self.frame_w - ancho and y_up >= self.frame_h - lado
        return x_up < ancho and y_up < lado

    # ------------------------------------------------------------------ #
    def distance_at_row(self, row_up: float) -> float | None:
        """Metros al pie del soporte por la tabla fila→metros, o `None` sin tabla.

        Interpola entre anclas y SUJETA en los extremos: fuera de la franja no hay
        más información que la del borde."""
        if not self.row_to_m:
            return None
        filas = [fila for fila, _ in self.row_to_m]
        if row_up <= filas[0]:
            return self.row_to_m[0][1]
        if row_up >= filas[-1]:
            return self.row_to_m[-1][1]
        alto = bisect_left(filas, row_up)
        fila0, metros0 = self.row_to_m[alto - 1]
        fila1, metros1 = self.row_to_m[alto]
        peso = (row_up - fila0) / (fila1 - fila0)
        return metros0 + peso * (metros1 - metros0)

    # ------------------------------------------------------------------ #
    def to_canvas(self, x_stored: float, y_stored: float) -> MappedPoint:
        """Un punto anotado en el frame almacenado, llevado al lienzo del detector."""
        x_up, y_up = self.upright(x_stored, y_stored)
        layout = self.geometry.layout
        region = None
        for indice, candidata in enumerate(layout.regions):
            if (
                candidata.src_x <= x_up < candidata.src_x + candidata.src_w
                and candidata.src_y <= y_up < candidata.src_y + candidata.src_h
            ):
                region = indice
                break
        dentro = region is not None
        if region is None:
            # Fuera de la franja: se mapea con la región más cercana por filas para
            # no perderlo, pero queda MARCADO.
            region = min(
                range(len(layout.regions)),
                key=lambda i: min(
                    abs(y_up - layout.regions[i].src_y),
                    abs(y_up - (layout.regions[i].src_y + layout.regions[i].src_h)),
                ),
            )
        # El código enmascarado manda sobre todo: una etiqueta ahí no entrena.
        if self.in_timecode_strip(x_up, y_up):
            estado = STATUS_TIMECODE
        else:
            estado = STATUS_OK if dentro else STATUS_OUTSIDE
        elegida = layout.regions[region]
        x_in, y_in = elegida.to_input(x_up, y_up)
        return MappedPoint(x=x_in, y=y_in, region=region, status=estado)

    def to_native(self, x_canvas: float, y_canvas: float) -> MappedPoint:
        """Un punto del lienzo, de vuelta al frame almacenado."""
        layout = self.geometry.layout
        region = layout.region_of(x_canvas, y_canvas)
        indice = layout.regions.index(region)
        x_up, y_up = region.to_native(x_canvas, y_canvas)
        x_st, y_st = self.stored(x_up, y_up)
        estado = STATUS_TIMECODE if self.in_timecode_strip(x_up, y_up) else STATUS_OK
        return MappedPoint(x=x_st, y=y_st, region=indice, status=estado)

    def map_box(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[MappedPoint, MappedPoint]:
        """Una caja al lienzo. La región la decide el CENTRO (la convención de
        REF-21) y las dos esquinas viajan con esa misma región: una caja no se
        parte en la junta del mosaico."""
        centro = self.to_canvas((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        region = self.geometry.layout.regions[centro.region]
        a_up = self.upright(x1, y1)
        b_up = self.upright(x2, y2)
        esquina_a = region.to_input(*a_up)
        esquina_b = region.to_input(*b_up)
        izquierda = MappedPoint(
            x=min(esquina_a[0], esquina_b[0]),
            y=min(esquina_a[1], esquina_b[1]),
            region=centro.region,
            status=centro.status,
        )
        derecha = MappedPoint(
            x=max(esquina_a[0], esquina_b[0]),
            y=max(esquina_a[1], esquina_b[1]),
            region=centro.region,
            status=centro.status,
        )
        return (izquierda, derecha)


def load_band_spec(
    path: Path,
    *,
    rotations: Mapping[str, int] | None = None,
    frame_w: int = RECORDING_WIDTH,
    frame_h: int = RECORDING_HEIGHT,
) -> dict[str, SideBand]:
    """El band.json de REF-29 (los dos lados), listo para mapear etiquetas.

    `rotations` viene del manifiesto del partido (ML-06): cómo iba montado cada
    móvil. Sin él, se asume el montaje nominal: el izquierdo invertido."""
    from ftrain.ref import BandGeometry  # noqa: PLC0415 — perezoso adrede (grupo ref)

    try:
        crudo = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"no se pudo leer el band spec {path}: {exc}"
        raise BandsError(msg) from exc
    if not isinstance(crudo, dict) or not {"left", "right"} <= set(crudo):
        msg = f"el band spec {path} tiene que llevar left y right"
        raise BandsError(msg)
    giro = {"left": 180, "right": 0} if rotations is None else dict(rotations)

    lados: dict[str, SideBand] = {}
    for side in ("left", "right"):
        datos = crudo[side]
        tabla = datos.get("row_to_m", [])
        if not isinstance(tabla, list):
            msg = f"{path}: `row_to_m` de {side} tiene que ser una lista"
            raise BandsError(msg)
        lados[side] = SideBand(
            side=side,
            geometry=BandGeometry.from_dict(datos),
            row_to_m=tuple((float(fila), float(metros)) for fila, metros in tabla),
            rotation_deg=int(giro.get(side, 0)),
            frame_w=frame_w,
            frame_h=frame_h,
        )
    return lados
