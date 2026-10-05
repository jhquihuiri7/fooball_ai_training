"""Formato canónico de etiquetas y conversores (ML-22)."""

from __future__ import annotations

import pytest

from ftrain.bands import MappedPoint
from ftrain.constants import MASTER_CLASSES, PLAYER_CLASSES
from ftrain.labels import (
    Box,
    FrameRef,
    LabelsError,
    LabelSet,
    Source,
    from_coco,
    from_cvat_xml,
    to_coco,
    to_cvat_xml,
    to_deim_coco,
    to_rfdetr_tiles,
)
from ftrain.tiling import Tile


def lote() -> LabelSet:
    frames = [
        FrameRef("left/000001.jpg", 3840, 2160, sha256="a" * 64),
        FrameRef("left/000002.jpg", 3840, 2160, sha256="b" * 64, unusable=True),
    ]
    cajas = [
        Box("left/000001.jpg", "player", 100.123, 200.456, 160.789, 330.001, occluded=True),
        Box("left/000001.jpg", "goalkeeper", 3000.0, 900.0, 3060.5, 1040.25, truncated=True),
        Box(
            "left/000001.jpg",
            "ball",
            1500.0,
            700.0,
            1512.0,
            711.0,
            blurred=True,
            source=Source.MASTER,
            confidence=0.83,
        ),
        Box("left/000001.jpg", "referee", 2000.0, 800.0, 2050.0, 950.0),
        Box(
            "left/000002.jpg",
            "distractor_ball",
            10.0,
            10.0,
            20.0,
            20.0,
            source=Source.AUTO,
            confidence=0.4,
        ),
    ]
    return LabelSet(frames, cajas)


def iguales(a: LabelSet, b: LabelSet, tol: float = 0.01) -> None:
    assert [f.file for f in a.frames] == [f.file for f in b.frames]
    assert [f.unusable for f in a.frames] == [f.unusable for f in b.frames]
    assert len(a.boxes) == len(b.boxes)
    for x, y in zip(a.boxes, b.boxes, strict=True):
        assert (x.file, x.cls, x.occluded, x.truncated, x.blurred, x.source, x.confidence) == (
            y.file,
            y.cls,
            y.occluded,
            y.truncated,
            y.blurred,
            y.source,
            y.confidence,
        )
        for p, q in ((x.x1, y.x1), (x.y1, y.y1), (x.x2, y.x2), (x.y2, y.y2)):
            assert abs(p - q) < tol


def test_coco_cvat_coco_conserva_cajas_atributos_y_clases():
    original = lote()
    coco = to_coco(original)
    assert [c["name"] for c in coco["categories"]] == list(MASTER_CLASSES)
    vuelta = from_coco(to_coco(from_cvat_xml(to_cvat_xml(from_coco(coco)))))
    iguales(original, vuelta)
    assert vuelta.frames[0].sha256 == "a" * 64


def test_coco_sin_atributos_o_con_otras_clases_no_se_acepta():
    coco = to_coco(lote())
    del coco["annotations"][0]["attributes"]["blurred"]
    with pytest.raises(LabelsError, match="blurred"):
        from_coco(coco)
    coco = to_coco(lote())
    coco["categories"][0]["name"] = "pelota"
    with pytest.raises(LabelsError, match="orden"):
        from_coco(coco)


def test_el_balon_es_una_caja_con_centro_y_diametro():
    balon = lote().boxes[2]
    assert balon.center == (1506.0, 705.5)
    assert balon.diameter == pytest.approx(11.5)


def test_las_pistas_de_video_van_y_vuelven_e_interpolan_entre_claves():
    frames = [FrameRef(f"v/{i:06d}.jpg", 3840, 2160) for i in range(5)]
    claves = [
        Box("v/000000.jpg", "ball", 100.0, 100.0, 110.0, 110.0, track_id=7),
        Box("v/000004.jpg", "ball", 140.0, 60.0, 150.0, 70.0, track_id=7),
    ]
    xml = to_cvat_xml(LabelSet(frames, claves), tracks=True)
    assert "<track" in xml

    vuelta = from_cvat_xml(xml)

    pista = sorted((b for b in vuelta.boxes if b.track_id == 7), key=lambda b: b.file)
    assert [b.file for b in pista] == [f.file for f in frames]
    assert pista[2].x1 == pytest.approx(120.0)
    assert pista[2].y1 == pytest.approx(80.0)
    assert pista[2].source == Source.AUTO, "lo interpolado no es humano"
    assert pista[0].source == Source.HUMAN


def test_cvat_con_una_clase_desconocida_no_se_acepta():
    xml = to_cvat_xml(lote()).replace('label="player"', 'label="aficionado"')
    with pytest.raises(LabelsError, match="aficionado"):
        from_cvat_xml(xml)


class _Mitad:
    """Un lienzo de prueba: la mitad de la escala nativa, y fuera lo de x > 3500."""

    def map_box(self, x1, y1, x2, y2):
        estado = "fuera" if x2 > 3500 else "ok"
        return (MappedPoint(x1 / 2, y1 / 2, 0, estado), MappedPoint(x2 / 2, y2 / 2, 0, estado))


def test_deim_solo_jugadores_ids_0_a_2_en_el_lienzo():
    coco = to_deim_coco(lote(), lambda _f: _Mitad())

    assert [(c["id"], c["name"]) for c in coco["categories"]] == list(enumerate(PLAYER_CLASSES))
    assert [im["file_name"] for im in coco["images"]] == ["left/000001.jpg"], "sin las inservibles"
    clases = sorted(a["category_id"] for a in coco["annotations"])
    assert clases == [0, 1, 2], "portero, jugador y árbitro; el balón no va a DEIM"
    assert coco["skipped"] == 0
    jugador = next(
        a for a in coco["annotations"] if a["category_id"] == PLAYER_CLASSES.index("player")
    )
    assert jugador["bbox"][0] == pytest.approx(100.123 / 2)


def test_deim_cuenta_lo_que_cae_fuera_del_lienzo():
    ls = lote()
    ls.boxes.append(Box("left/000001.jpg", "player", 3600.0, 10.0, 3700.0, 100.0))
    assert to_deim_coco(ls, lambda _f: _Mitad())["skipped"] == 1


def test_rfdetr_por_teselas_recorta_y_descarta_lo_poco_visible():
    teselas = [Tile(0, 0, 1600, 1600), Tile(1400, 0, 1600, 1600)]
    ls = LabelSet(
        [FrameRef("f.jpg", 3840, 2160)],
        [
            Box("f.jpg", "player", 1550.0, 100.0, 1650.0, 300.0),  # mitad y mitad
            Box("f.jpg", "ball", 1590.0, 500.0, 1610.0, 520.0),  # mitad y mitad
            Box("f.jpg", "referee", 1580.0, 900.0, 1680.0, 1000.0),  # 20 % en la primera
        ],
    )

    coco = to_rfdetr_tiles(ls, teselas)

    assert [im["tile"] for im in coco["images"]] == [[0, 0, 1600, 1600], [1400, 0, 1600, 1600]]
    por_tesela = {0: [], 1: []}
    for a in coco["annotations"]:
        por_tesela[a["image_id"]].append(MASTER_CLASSES[a["category_id"]])
    assert sorted(por_tesela[0]) == ["ball", "player"]
    assert sorted(por_tesela[1]) == ["ball", "player", "referee"]
    arbitro = next(a for a in coco["annotations"] if a["image_id"] == 1 and a["category_id"] == 4)
    assert arbitro["bbox"] == [180.0, 900.0, 100.0, 100.0]
