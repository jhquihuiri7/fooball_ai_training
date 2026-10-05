"""El formato canónico de etiquetas y sus conversores (ML-22, ADR 0003).

El canónico es un COCO extendido en coordenadas NATIVAS (las del frame tal como se
almacenó, que es donde anota CVAT):

- clases: `MASTER_CLASSES`, con ids 0..4 en ese orden;
- por anotación, `occluded`, `truncated` y `blurred` siempre presentes, más la
  fuente (`human`, `master` o `auto`) y la confianza (null para las humanas);
- por imagen, `unusable`, su `sha256` y la ruta relativa a `frames/<versión>/`.

El balón es una caja, nunca un punto (ADR 0003 §3): su centro y su diámetro se sacan de
la caja. La tarjeta de ML-22 habla de un `BallPoint`; manda el ADR.

Conversores:
- CVAT XML 1.1 de imágenes y de pistas de vídeo, de ida y vuelta. Las pistas se
  interpolan linealmente entre fotogramas clave, como hace CVAT;
- al COCO de DEIM: solo jugadores, ids 0..2 en `PLAYER_CLASSES` y coordenadas del
  lienzo, con la misma `SideBand` de la referencia (ML-19);
- al COCO de RF-DETR, por teselas nativas (ML-24).
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from ftrain.constants import (
    LABEL_TILE_MIN_VISIBLE,
    MASTER_CLASSES,
    PLAYER_CLASSES,
    PLAYER_INPUT_H,
    PLAYER_INPUT_W,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from ftrain.bands import MappedPoint
    from ftrain.tiling import Tile

__all__ = [
    "ATTRIBUTES",
    "Box",
    "FrameRef",
    "LabelSet",
    "LabelsError",
    "Source",
    "from_coco",
    "from_cvat_xml",
    "to_coco",
    "to_cvat_xml",
    "to_deim_coco",
    "to_rfdetr_tiles",
]

ATTRIBUTES = ("occluded", "truncated", "blurred")
"""Los atributos booleanos de cada caja, siempre presentes (ADR 0003 §2)."""

CVAT_VERSION = "1.1"
CVAT_TRUE = "true"
STATUS_OK = "ok"
"""El veredicto de `SideBand` para un punto que cae dentro del lienzo."""


class LabelsError(ValueError):
    """Un fichero de etiquetas que no cumple el formato. El mensaje dice qué falta."""


class Source(StrEnum):
    HUMAN = "human"
    MASTER = "master"
    AUTO = "auto"


@dataclass(frozen=True)
class FrameRef:
    """Un frame del lote: ruta relativa a `frames/<versión>/`, tamaño nativo y sha256."""

    file: str
    width: int
    height: int
    sha256: str = ""
    unusable: bool = False


@dataclass(frozen=True)
class Box:
    """Una caja en píxeles nativos, con su clase, sus atributos y de dónde sale."""

    file: str
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    occluded: bool = False
    truncated: bool = False
    blurred: bool = False
    source: Source = Source.HUMAN
    confidence: float | None = None
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def diameter(self) -> float:
        """El diámetro de un balón: la media del ancho y el alto de su caja."""
        return ((self.x2 - self.x1) + (self.y2 - self.y1)) / 2.0


@dataclass
class LabelSet:
    frames: list[FrameRef] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    def frame(self, file: str) -> FrameRef:
        for f in self.frames:
            if f.file == file:
                return f
        msg = f"no hay frame {file}"
        raise LabelsError(msg)


# --------------------------------------------------------------------------- #
# COCO extendido (el canónico)
# --------------------------------------------------------------------------- #


def to_coco(labels: LabelSet) -> dict[str, Any]:
    ids = {f.file: i for i, f in enumerate(labels.frames)}
    return {
        "categories": [{"id": i, "name": n} for i, n in enumerate(MASTER_CLASSES)],
        "images": [
            {
                "id": ids[f.file],
                "file_name": f.file,
                "width": f.width,
                "height": f.height,
                "sha256": f.sha256,
                "unusable": f.unusable,
            }
            for f in labels.frames
        ],
        "annotations": [
            {
                "id": k,
                "image_id": ids[b.file],
                "category_id": MASTER_CLASSES.index(b.cls),
                "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],
                "attributes": {a: getattr(b, a) for a in ATTRIBUTES},
                "source": b.source.value,
                "confidence": b.confidence,
                "track_id": b.track_id,
            }
            for k, b in enumerate(labels.boxes)
        ],
    }


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        msg = f"{where}: falta `{key}`"
        raise LabelsError(msg)
    return data[key]


def from_coco(data: dict[str, Any]) -> LabelSet:
    nombres = [
        c["name"] for c in sorted(_require(data, "categories", "coco"), key=lambda c: c["id"])
    ]
    if tuple(nombres) != MASTER_CLASSES:
        msg = f"las clases tienen que ser {MASTER_CLASSES} en ese orden, no {tuple(nombres)}"
        raise LabelsError(msg)
    frames = {}
    for im in _require(data, "images", "coco"):
        frames[im["id"]] = FrameRef(
            file=im["file_name"],
            width=int(im["width"]),
            height=int(im["height"]),
            sha256=im.get("sha256", ""),
            unusable=bool(_require(im, "unusable", f"imagen {im['file_name']}")),
        )
    cajas = []
    for an in _require(data, "annotations", "coco"):
        donde = f"anotación {an.get('id')}"
        atributos = _require(an, "attributes", donde)
        for a in ATTRIBUTES:
            _require(atributos, a, donde)
        x, y, w, h = an["bbox"]
        cajas.append(
            Box(
                file=frames[an["image_id"]].file,
                cls=MASTER_CLASSES[an["category_id"]],
                x1=float(x),
                y1=float(y),
                x2=float(x) + float(w),
                y2=float(y) + float(h),
                **{a: bool(atributos[a]) for a in ATTRIBUTES},
                source=Source(_require(an, "source", donde)),
                confidence=an.get("confidence"),
                track_id=an.get("track_id"),
            )
        )
    return LabelSet(list(frames.values()), cajas)


# --------------------------------------------------------------------------- #
# CVAT XML 1.1
# --------------------------------------------------------------------------- #


def _box_xml(parent: ET.Element, b: Box, extra: dict[str, str]) -> None:
    e = ET.SubElement(
        parent,
        "box",
        {
            **extra,
            "xtl": repr(b.x1),
            "ytl": repr(b.y1),
            "xbr": repr(b.x2),
            "ybr": repr(b.y2),
            "occluded": "1" if b.occluded else "0",
        },
    )
    for nombre, valor in (
        ("truncated", str(b.truncated).lower()),
        ("blurred", str(b.blurred).lower()),
        ("source", b.source.value),
        ("confidence", "" if b.confidence is None else repr(b.confidence)),
    ):
        ET.SubElement(e, "attribute", {"name": nombre}).text = valor


def _tracks_xml(raiz: ET.Element, labels: LabelSet, indice: dict[str, int]) -> None:
    pistas: dict[int, list[Box]] = {}
    for b in labels.boxes:
        if b.track_id is not None:
            pistas.setdefault(b.track_id, []).append(b)
    for tid, cajas in sorted(pistas.items()):
        t = ET.SubElement(raiz, "track", {"id": str(tid), "label": cajas[0].cls})
        for b in sorted(cajas, key=lambda c: indice[c.file]):
            _box_xml(t, b, {"frame": str(indice[b.file]), "keyframe": "1", "outside": "0"})


def to_cvat_xml(labels: LabelSet, *, tracks: bool = False) -> str:
    """El lote como CVAT 1.1: de imágenes, o de vídeo con las cajas con `track_id` en
    pistas (el fotograma es el índice del frame en el lote)."""
    raiz = ET.Element("annotations")
    ET.SubElement(raiz, "version").text = CVAT_VERSION
    indice = {f.file: i for i, f in enumerate(labels.frames)}
    sueltas = [b for b in labels.boxes if not (tracks and b.track_id is not None)]
    for i, f in enumerate(labels.frames):
        im = ET.SubElement(
            raiz,
            "image",
            {"id": str(i), "name": f.file, "width": str(f.width), "height": str(f.height)},
        )
        if f.sha256:
            im.set("sha256", f.sha256)
        if f.unusable:
            ET.SubElement(im, "tag", {"label": "unusable"})
        for b in sueltas:
            if b.file == f.file:
                _box_xml(im, b, {"label": b.cls})
    if tracks:
        _tracks_xml(raiz, labels, indice)
    return ET.tostring(raiz, encoding="unicode")


def _box_from_xml(e: ET.Element, file: str, cls: str, track_id: int | None) -> Box:
    attrs = {a.get("name"): (a.text or "") for a in e.findall("attribute")}
    conf = attrs.get("confidence", "")
    return Box(
        file=file,
        cls=cls,
        x1=float(e.get("xtl", "0")),
        y1=float(e.get("ytl", "0")),
        x2=float(e.get("xbr", "0")),
        y2=float(e.get("ybr", "0")),
        occluded=e.get("occluded") == "1",
        truncated=attrs.get("truncated") == CVAT_TRUE,
        blurred=attrs.get("blurred") == CVAT_TRUE,
        source=Source(attrs.get("source", Source.HUMAN.value)),
        confidence=float(conf) if conf else None,
        track_id=track_id,
    )


def _interpolate(a: Box, b: Box, file: str, t: float) -> Box:
    """La caja entre dos fotogramas clave de una pista, en la fracción `t`."""

    def mezcla(p: float, q: float) -> float:
        return p + (q - p) * t

    return replace(
        a,
        file=file,
        x1=mezcla(a.x1, b.x1),
        y1=mezcla(a.y1, b.y1),
        x2=mezcla(a.x2, b.x2),
        y2=mezcla(a.y2, b.y2),
        source=Source.AUTO,
    )


def from_cvat_xml(text: str) -> LabelSet:
    raiz = ET.fromstring(text)  # noqa: S314 - el XML lo exporta nuestro CVAT
    frames: list[FrameRef] = []
    cajas: list[Box] = []
    for im in raiz.findall("image"):
        nombre = im.get("name", "")
        frames.append(
            FrameRef(
                file=nombre,
                width=int(im.get("width", "0")),
                height=int(im.get("height", "0")),
                sha256=im.get("sha256", ""),
                unusable=any(t.get("label") == "unusable" for t in im.findall("tag")),
            )
        )
        cajas.extend(_box_from_xml(e, nombre, e.get("label", ""), None) for e in im.findall("box"))
    for tr in raiz.findall("track"):
        tid = int(tr.get("id", "0"))
        clase = tr.get("label", "")
        claves = sorted(
            (
                int(e.get("frame", "0")),
                _box_from_xml(e, frames[int(e.get("frame", "0"))].file, clase, tid),
            )
            for e in tr.findall("box")
            if e.get("outside") != "1"
        )
        for (fa, a), (fb, b) in itertools.pairwise(claves):
            cajas.append(a)
            cajas.extend(
                _interpolate(a, b, frames[f].file, (f - fa) / (fb - fa)) for f in range(fa + 1, fb)
            )
        if claves:
            cajas.append(claves[-1][1])
    desconocidas = {b.cls for b in cajas} - set(MASTER_CLASSES)
    if desconocidas:
        msg = f"clases que no son de MASTER_CLASSES: {sorted(desconocidas)}"
        raise LabelsError(msg)
    return LabelSet(frames, cajas)


# --------------------------------------------------------------------------- #
# Exportadores de entrenamiento
# --------------------------------------------------------------------------- #


class BoxMapper(Protocol):
    """Lo que hace falta de `SideBand` (ML-19): una caja nativa al lienzo."""

    def map_box(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[MappedPoint, MappedPoint]: ...


def to_deim_coco(labels: LabelSet, mapper_for: Callable[[FrameRef], BoxMapper]) -> dict[str, Any]:
    """El COCO de DEIM: solo jugadores, ids 0..2 en `PLAYER_CLASSES`, en el lienzo de
    1920x576. Se quedan fuera los frames inservibles y las cajas que no caen en el
    lienzo (o en la franja del código de tiempo); se cuentan en `skipped`."""
    imagenes, anotaciones, fuera = [], [], 0
    for i, f in enumerate(labels.frames):
        if f.unusable:
            continue
        imagenes.append(
            {"id": i, "file_name": f.file, "width": PLAYER_INPUT_W, "height": PLAYER_INPUT_H}
        )
        mapa = mapper_for(f)
        for b in labels.boxes:
            if b.file != f.file or b.cls not in PLAYER_CLASSES:
                continue
            a, z = mapa.map_box(b.x1, b.y1, b.x2, b.y2)
            if a.status != STATUS_OK:
                fuera += 1
                continue
            anotaciones.append(
                {
                    "id": len(anotaciones),
                    "image_id": i,
                    "category_id": PLAYER_CLASSES.index(b.cls),
                    "bbox": [a.x, a.y, z.x - a.x, z.y - a.y],
                    "area": (z.x - a.x) * (z.y - a.y),
                    "iscrowd": 0,
                }
            )
    return {
        "categories": [{"id": i, "name": n} for i, n in enumerate(PLAYER_CLASSES)],
        "images": imagenes,
        "annotations": anotaciones,
        "skipped": fuera,
    }


def _clip(b: Box, t: Tile) -> tuple[float, float, float, float] | None:
    x1, y1 = max(b.x1, t.x0), max(b.y1, t.y0)
    x2, y2 = min(b.x2, t.x0 + t.width), min(b.y2, t.y0 + t.height)
    area = (b.x2 - b.x1) * (b.y2 - b.y1)
    if x2 <= x1 or y2 <= y1 or area <= 0 or (x2 - x1) * (y2 - y1) / area < LABEL_TILE_MIN_VISIBLE:
        return None
    return x1 - t.x0, y1 - t.y0, x2 - t.x0, y2 - t.y0


def to_rfdetr_tiles(labels: LabelSet, tiles: Iterable[Tile] | Sequence[Tile]) -> dict[str, Any]:
    """El COCO de RF-DETR por teselas nativas: una imagen por (frame, tesela) con su
    origen, y las cajas recortadas a la tesela si se ve al menos `LABEL_TILE_MIN_VISIBLE`
    de su área. Clases e ids, los de `MASTER_CLASSES`."""
    teselas = list(tiles)
    imagenes, anotaciones = [], []
    for f in labels.frames:
        if f.unusable:
            continue
        for t in teselas:
            iid = len(imagenes)
            imagenes.append(
                {
                    "id": iid,
                    "file_name": f.file,
                    "tile": [t.x0, t.y0, t.width, t.height],
                    "width": t.width,
                    "height": t.height,
                }
            )
            for b in labels.boxes:
                if b.file != f.file or (c := _clip(b, t)) is None:
                    continue
                anotaciones.append(
                    {
                        "id": len(anotaciones),
                        "image_id": iid,
                        "category_id": MASTER_CLASSES.index(b.cls),
                        "bbox": [c[0], c[1], c[2] - c[0], c[3] - c[1]],
                        "area": (c[2] - c[0]) * (c[3] - c[1]),
                        "iscrowd": 0,
                    }
                )
    return {
        "categories": [{"id": i, "name": n} for i, n in enumerate(MASTER_CLASSES)],
        "images": imagenes,
        "annotations": anotaciones,
    }
