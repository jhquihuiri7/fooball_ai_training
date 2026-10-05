"""Evaluación por bandas (ML-26): recall conocido por banda y bootstrap determinista."""

from __future__ import annotations

import pytest

from ftrain.eval.detection import (
    Detection,
    Truth,
    average_precision,
    bootstrap_recall,
    evaluate,
    match_image,
)


def caja(x: float, y: float, w: float = 40.0, h: float = 80.0) -> tuple[float, float, float, float]:
    return (x, y, x + w, y + h)


def escena(partidos: int = 4):
    """Por partido, una imagen: 5 jugadores en la banda 0 (los 5 detectados) y 5 en la
    banda 3 (2 detectados), más un falso positivo lejos de todo."""
    gts, dets = [], []
    for p in range(partidos):
        img = f"p{p}/f0"
        for k in range(5):
            gts.append(Truth(img, f"p{p}", "player", caja(100.0 * k, 100.0), band=0))
            dets.append(Detection(img, "player", caja(100.0 * k + 2, 101.0), 0.9))
        for k in range(5):
            gts.append(Truth(img, f"p{p}", "player", caja(100.0 * k, 900.0), band=3))
            if k < 2:
                dets.append(Detection(img, "player", caja(100.0 * k + 1, 902.0), 0.8))
        dets.append(Detection(img, "player", caja(3000.0, 1500.0), 0.7))
    return gts, dets


def test_recall_y_precision_por_banda():
    gts, dets = escena()

    m = evaluate(gts, dets, band_of_prediction=lambda d: 0 if d.box[1] < 500 else 4)

    assert m[("0",)].recall == 1.0
    assert m[("3",)].recall == pytest.approx(0.4)
    assert m[("0",)].precision == 1.0
    assert m[("4",)].precision == 0.0, "el falso positivo cuenta en su banda"
    assert m[("0",)].ap50 == pytest.approx(1.0)
    assert m[("3",)].ap50 == pytest.approx(0.4)


def test_por_luz_y_partido():
    gts, dets = escena(partidos=2)
    gts = [
        Truth(t.image, t.match, t.cls, t.box, t.band, "night" if t.match == "p1" else "day")
        for t in gts
    ]

    m = evaluate(gts, dets, group=lambda t: (t.light, t.match))

    assert set(m) == {("day", "p0"), ("night", "p1")}
    assert m[("night", "p1")].recall == pytest.approx(0.7)


def test_las_cajas_pequenas_emparejan_por_centro():
    balon = Truth("f", "p", "ball", (100.0, 100.0, 106.0, 106.0), band=4)
    cerca = Detection("f", "ball", (102.0, 102.5, 108.0, 108.5), 0.9)  # IoU < 0,5
    lejos = Detection("f", "ball", (110.0, 110.0, 116.0, 116.0), 0.9)

    assert match_image([balon], [cerca])[0][1] == balon
    assert match_image([balon], [lejos])[0][1] is None


def test_sin_clase_empareja_aunque_la_clase_falle():
    t = Truth("f", "p", "goalkeeper", caja(0.0, 0.0), band=1)
    d = Detection("f", "player", caja(1.0, 1.0), 0.9)

    assert match_image([t], [d])[0][1] is None
    assert match_image([t], [d], by_class=False)[0][1] == t


def test_cada_caja_de_verdad_se_empareja_una_sola_vez():
    t = Truth("f", "p", "player", caja(0.0, 0.0), band=0)
    dos = [
        Detection("f", "player", caja(1.0, 0.0), 0.9),
        Detection("f", "player", caja(0.0, 1.0), 0.8),
    ]

    pares = match_image([t], dos)

    assert [x[1] is not None for x in pares] == [True, False], "la duplicada es un falso positivo"


def test_ap_con_una_lista_conocida():
    # Aciertos en las posiciones 1, 3 y 4 de 5, con 4 de verdad.
    ap = average_precision([(0.9, True), (0.8, False), (0.7, True), (0.6, True), (0.5, False)], 4)
    # Envolvente: R 0.25→P 1, R 0.5→P 0.75, R 0.75→P 0.75.
    assert ap == pytest.approx(0.25 * 1.0 + 0.25 * 0.75 + 0.25 * 0.75)
    assert average_precision([], 3) == 0.0


def test_bootstrap_por_partido_es_determinista_y_contiene_el_recall():
    gts, dets = escena(partidos=6)
    # Un partido flojo en la banda 0: rompe la igualdad entre partidos.
    dets = [d for d in dets if not (d.image == "p5/f0" and d.box[1] < 500 and d.box[0] < 250)]

    a = bootstrap_recall(gts, dets, rounds=300)
    b = bootstrap_recall(gts, dets, rounds=300)

    assert a == b
    bajo, alto = a[("0",)]
    recall = evaluate(gts, dets)[("0",)].recall
    assert bajo <= recall <= alto
    assert bajo < alto
    assert a[("3",)] == (pytest.approx(0.4), pytest.approx(0.4)), (
        "todos los partidos iguales: sin anchura"
    )
