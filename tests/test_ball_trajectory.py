"""Trayectorias del balón para etiquetar (ML-36)."""

from __future__ import annotations

import numpy as np
import pytest

from ftrain.ball.trajectory import Candidate, Flag, build_tracks, fill_gaps
from ftrain.constants import (
    BALL_INTERP_MAX_GAP_FRAMES,
    BALL_TRACK_MAX_COAST_FRAMES,
    RECORDING_HEIGHT,
    RECORDING_WIDTH,
)

TRUE_SCORE = 0.8
GHOST_SCORE = 0.6


def parabola(frames: int, *, start=(400.0, 1500.0), vel=(30.0, -28.0), g=0.6):
    """Un pase en el aire: velocidad constante en x, gravedad en y (px/fotograma²)."""
    (x0, y0), (vx, vy) = start, vel
    return {t: (x0 + vx * t, y0 + vy * t + 0.5 * g * t * t) for t in range(frames)}


def candidates_with_gaps_and_ghosts(truth, rng, *, gap_frac=0.2, ghosts_per_frame=2):
    frames = sorted(truth)
    # Los extremos se ven: un hueco al final no es un hueco, es el fin de la pista.
    quitados = set(rng.choice(frames[1:-1], size=int(gap_frac * len(frames)), replace=False))
    out: list[Candidate] = []
    fantasmas: set[tuple[int, float, float]] = set()
    for f in frames:
        if f not in quitados:
            out.append(Candidate(f, *truth[f], TRUE_SCORE))
        for _ in range(ghosts_per_frame):
            x, y = rng.uniform(0, RECORDING_WIDTH), rng.uniform(0, RECORDING_HEIGHT)
            fantasmas.add((f, x, y))
            out.append(Candidate(f, x, y, GHOST_SCORE))
    return out, quitados, fantasmas


@pytest.mark.parametrize("seed", range(10))
def test_parabola_con_huecos_y_fantasmas(seed):
    rng = np.random.default_rng(seed)
    truth = parabola(120)
    cands, quitados, fantasmas = candidates_with_gaps_and_ghosts(truth, rng)

    pistas = build_tracks(cands)

    assert len(pistas) == 1, "los fantasmas no forman pistas"
    pista = pistas[0]
    for p in pista.detections:
        assert (p.frame, p.x, p.y) not in fantasmas, "ningún fantasma aceptado"
        assert (p.x, p.y) == truth[p.frame]
    rellenos = [p for p in pista.points if p.interpolated]
    huecos_cortos = {p.frame for p in rellenos}
    assert huecos_cortos <= quitados
    for p in rellenos:
        tx, ty = truth[p.frame]
        assert np.hypot(p.x - tx, p.y - ty) <= 1.0, "≤1 px en lo interpolado"
    if Flag.LONG_GAP not in pista.flags:
        assert len(pista.points) == len(truth)
    assert Flag.RIVAL_NEARBY not in pista.flags
    assert Flag.LOW_CONFIDENCE not in pista.flags


def test_un_fantasma_cerca_y_a_destiempo_no_entra():
    truth = parabola(40)
    cands = [Candidate(f, *xy, TRUE_SCORE) for f, xy in truth.items() if f != 20]
    # En el hueco, un fantasma a 150 px de donde va el balón: fuera de la puerta.
    tx, ty = truth[20]
    cands.append(Candidate(20, tx + 150.0, ty, GHOST_SCORE))

    (pista,) = build_tracks(cands)

    assert next(p for p in pista.points if p.frame == 20).interpolated


def test_hueco_largo_se_marca_y_no_se_rellena():
    truth = parabola(60)
    hueco = range(20, 20 + BALL_INTERP_MAX_GAP_FRAMES + 3)
    cands = [Candidate(f, *xy, TRUE_SCORE) for f, xy in truth.items() if f not in hueco]

    (pista,) = build_tracks(cands)

    assert Flag.LONG_GAP in pista.flags
    assert not {p.frame for p in pista.points} & set(hueco)


def test_un_corte_largo_parte_la_pista_en_dos():
    truth = parabola(80)
    hueco = range(30, 30 + BALL_TRACK_MAX_COAST_FRAMES + 5)
    cands = [Candidate(f, *xy, TRUE_SCORE) for f, xy in truth.items() if f not in hueco]

    pistas = build_tracks(cands)

    assert len(pistas) == 2
    assert {min(p.frame for p in t.points) for t in pistas} == {0, hueco.stop}


def test_pistas_rivales_cerca_se_marcan_las_dos():
    a = parabola(30)
    b = {f: (x + 60.0, y + 40.0) for f, (x, y) in a.items()}
    cands = [Candidate(f, *xy, TRUE_SCORE) for f, xy in a.items()]
    cands += [Candidate(f, *xy, TRUE_SCORE) for f, xy in b.items()]

    pistas = build_tracks(cands)

    assert len(pistas) == 2
    assert all(Flag.RIVAL_NEARBY in t.flags for t in pistas)


def test_confianza_baja_y_fuera_de_la_mascara():
    truth = parabola(30)
    cands = [Candidate(f, *xy, 0.2) for f, xy in truth.items()]

    (pista,) = build_tracks(cands, inside_mask=lambda x, _y: x < 1000.0)

    assert pista.flags == {Flag.LOW_CONFIDENCE, Flag.OUTSIDE_MASK}
    assert pista.score == pytest.approx(0.2 * 30)


def test_las_pistas_cortas_no_salen_y_salen_por_puntuacion():
    larga = parabola(30)
    corta = parabola(3, start=(3000.0, 300.0), vel=(-20.0, -28.0))
    media = parabola(10, start=(2000.0, 1000.0), vel=(-25.0, -28.0))
    cands = [Candidate(f, *xy, TRUE_SCORE) for d in (larga, corta, media) for f, xy in d.items()]

    pistas = build_tracks(cands)

    assert [len(t.detections) for t in pistas] == [30, 10]


def test_fill_gaps_con_pocas_detecciones_usa_una_recta():
    a, b = Candidate(0, 0.0, 0.0, 1.0), Candidate(4, 40.0, 20.0, 1.0)

    puntos, largo = fill_gaps([a, b])

    assert not largo
    assert [p.frame for p in puntos] == [0, 1, 2, 3, 4]
    assert [p.x for p in puntos] == pytest.approx([0.0, 10.0, 20.0, 30.0, 40.0])
    assert [p.y for p in puntos] == pytest.approx([0.0, 5.0, 10.0, 15.0, 20.0])


def test_un_bote_no_parte_la_pista_ni_pierde_detecciones():
    # Cae y bota en el suelo (y = 1500) con pérdida: la cuadrática de los dos lados no
    # cuadra en el bote, pero la de antes o la de después sí.
    pts, x, y, vy = {}, 400.0, 1000.0, 5.0
    for f in range(70):
        pts[f] = (x, y)
        x, vy = x + 25.0, vy + 0.8
        y += vy
        if y > 1500.0:
            y, vy = 3000.0 - y, -0.7 * vy
    (pista,) = build_tracks([Candidate(f, *xy, TRUE_SCORE) for f, xy in pts.items()])

    assert len(pista.detections) == 70
    assert pista.flags == set()
