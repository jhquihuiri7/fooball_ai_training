"""Invariantes de `ftrain/constants.py` (ML-02).

Ninguna de estas constantes da error si está mal: una banda desordenada o una clase
repetida solo cambian lo que se mide o qué es cada caja. Por eso se comprueban aquí.
"""

from __future__ import annotations

import itertools
import math

from ftrain import constants as c


def test_las_bandas_de_distancia_empiezan_en_cero_crecen_y_acaban_en_infinito() -> None:
    bandas = c.DISTANCE_BANDS_M

    assert bandas == (0.0, 20.0, 40.0, 60.0, 80.0, math.inf)
    assert all(a < b for a, b in itertools.pairwise(bandas))


def test_las_clases_de_jugadores_no_se_repiten_y_van_en_el_orden_del_export() -> None:
    assert c.PLAYER_CLASSES == ("goalkeeper", "player", "referee")
    assert len(set(c.PLAYER_CLASSES)) == len(c.PLAYER_CLASSES)


def test_las_clases_del_maestro_no_se_repiten_y_contienen_las_de_jugadores() -> None:
    assert c.MASTER_CLASSES == ("ball", "distractor_ball", "goalkeeper", "player", "referee")
    assert len(set(c.MASTER_CLASSES)) == len(c.MASTER_CLASSES)
    # En el mismo orden relativo: así pasar de una lista a la otra es un filtro.
    assert tuple(k for k in c.MASTER_CLASSES if k in c.PLAYER_CLASSES) == c.PLAYER_CLASSES


def test_la_entrada_de_jugadores_es_la_franja_4k_por_su_escala() -> None:
    # 3840 columnas de una cámara 4K, y hasta 1152 filas de franja sin mosaico (ADR 0020 §1).
    assert c.PLAYER_INPUT_W == 3840 * c.BAND_SCALE
    assert c.PLAYER_INPUT_H == 1152 * c.BAND_SCALE


def test_las_rois_del_balon_caben_enteras_en_la_rejilla_del_heatmap() -> None:
    assert c.BALL_ROI_SIDES_PX == (256,)
    assert all(lado % c.BALL_HEATMAP_STRIDE == 0 for lado in c.BALL_ROI_SIDES_PX)


def test_la_puerta_del_balon_cubre_un_balon_rapido_cerca_con_margen() -> None:
    # 58,5 px por fotograma: un balón a 25 m/s visto a 20 m (ML-36).
    assert 58.5 < c.BALL_GATE_PX_PER_FRAME <= 58.5 * 1.25
    assert c.BALL_OUTLIER_PX < c.BALL_GATE_PX_PER_FRAME


def test_los_huecos_que_se_rellenan_caben_en_la_vida_de_una_pista() -> None:
    assert c.BALL_INTERP_MAX_GAP_FRAMES < c.BALL_TRACK_MAX_COAST_FRAMES
    assert c.BALL_REFINE_CONTEXT_POINTS >= 2 * (2 + 1)  # una parábola por cada lado
