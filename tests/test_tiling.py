"""Teselado, TTA y fusión (ML-24): un caso sintético por propiedad."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ftrain.tiling import (
    Tile,
    TilingError,
    crop,
    flip_boxes,
    flip_image,
    fuse,
    tile_grid,
    tiles_covering,
    to_native,
    unzoom_far_boxes,
)


def test_la_rejilla_cubre_todo_y_la_ultima_va_al_borde():
    teselas = tile_grid(1000, 500, 384, 384, overlap=64)

    assert tiles_covering(teselas, 1000, 500)
    assert max(t.x0 + t.width for t in teselas) == 1000
    assert max(t.y0 + t.height for t in teselas) == 500
    # El paso nunca supera lado - solape: el solape mínimo se respeta.
    xs = sorted({t.x0 for t in teselas})
    assert all(b - a <= 384 - 64 for a, b in pairwise(xs))


def test_una_imagen_menor_que_la_tesela_da_una_sola():
    teselas = tile_grid(300, 200, 384, 384)
    assert teselas == (Tile(0, 0, 300, 200),)


def test_un_solape_imposible_se_rechaza():
    with pytest.raises(TilingError, match="solape"):
        tile_grid(1000, 500, 384, 384, overlap=384)


def test_el_recorte_es_una_vista_y_vuelve_a_nativo():
    imagen = np.arange(20 * 30).reshape(20, 30)
    tesela = Tile(x0=10, y0=5, width=8, height=6)

    recorte = crop(imagen, tesela)
    assert recorte.shape == (6, 8)
    assert recorte[0, 0] == imagen[5, 10]

    nativas = to_native(np.array([[1.0, 2.0, 3.0, 4.0]]), tesela)
    assert nativas.tolist() == [[11.0, 7.0, 13.0, 9.0]]


def test_un_objeto_sobre_la_costura_sale_una_sola_vez():
    # Dos teselas de 384 con solape 128: la costura visible está entre x=256 y 384.
    teselas = tile_grid(640, 384, 384, 384, overlap=128)
    assert len(teselas) == 2
    objeto = (300.0, 100.0, 360.0, 200.0)  # 60 px de ancho, cruza x=320, cabe en ambas

    cajas, puntuaciones, clases = [], [], []
    for tesela in teselas:
        x1 = objeto[0] - tesela.x0
        x2 = objeto[2] - tesela.x0
        if x1 >= 0 and x2 <= tesela.width:  # el detector lo ve si cabe ENTERO
            local = np.array([[x1, objeto[1], x2, objeto[3]]])
            cajas.append(to_native(local, tesela))
            puntuaciones.append(0.8 + 0.1 * len(puntuaciones))  # confianzas distintas
            clases.append(0)

    assert len(cajas) == 2  # el solape garantiza que lo vieron las dos
    fundidas, scores, _ = fuse(np.concatenate(cajas), np.array(puntuaciones), np.array(clases))
    assert len(fundidas) == 1  # la aceptación: una sola vez
    assert fundidas[0].tolist() == list(objeto)
    assert scores[0] == pytest.approx(0.9)  # y con la confianza máxima


def test_la_fusion_conserva_la_confianza_maxima_y_separa_clases():
    cajas = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float64)
    puntuaciones = np.array([0.7, 0.9, 0.95])
    clases = np.array([1, 1, 2])

    fundidas, scores, cuales = fuse(cajas, puntuaciones, clases)

    assert len(fundidas) == 2  # una por clase
    assert scores.tolist() == [0.95, 0.9]  # la máxima de cada grupo, nunca un promedio
    assert sorted(cuales.tolist()) == [1, 2]


def test_la_inversa_del_volteo_es_exacta():
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 900, size=(50, 1))
    y1 = rng.uniform(0, 500, size=(50, 1))
    # float32, como las emite un detector: es lo que hace exacta la involución.
    cajas = np.hstack([x1, y1, x1 + rng.uniform(1, 90, (50, 1)), y1 + 10]).astype(np.float32)

    ida_y_vuelta = flip_boxes(flip_boxes(cajas, 1000), 1000)
    np.testing.assert_array_equal(ida_y_vuelta, cajas)  # exacta, no approx

    imagen = rng.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(flip_image(flip_image(imagen)), imagen)


def test_la_inversa_de_la_ampliacion_es_exacta():
    cajas = np.array([[10.0, 4.0, 50.0, 24.0]])
    ampliadas = cajas * 2.0  # lo que vería el detector sobre el recorte x2

    np.testing.assert_array_equal(unzoom_far_boxes(ampliadas, 2.0), cajas)
    with pytest.raises(TilingError, match="ampliación"):
        unzoom_far_boxes(cajas, 1.0)


def test_fuse_valida_los_tamanos_y_tolera_vacio():
    vacias, _scores, _clases = fuse(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    assert len(vacias) == 0

    with pytest.raises(TilingError, match="tamaños"):
        fuse(np.zeros((2, 4)), np.zeros(1), np.zeros(2, dtype=int))
