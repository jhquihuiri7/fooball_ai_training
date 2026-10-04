"""Bandas de distancia y nativo↔lienzo (ML-19), contra la referencia de verdad."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytest.importorskip("libs.vision", reason="el grupo `ref` no está instalado")

from libs.vision.band import band_from_pitch, recover_camera
from libs.vision.pitch import PitchModel
from libs.vision.rig import CameraIntrinsics

from ftrain.bands import MappedPoint, band_of, load_band_spec
from ftrain.constants import DISTANCE_BANDS_M

W, H = 3840, 2160
INTR = CameraIntrinsics.from_hfov(W, H, math.radians(106.0))
ALTURA_M = 5.0
RETRANQUEO_M = 10.0


def _modelo() -> PitchModel:
    """La misma nominal que test_band de la referencia: 5 m, perpendicular, al centro."""
    centro_cam = np.array([0.0, -(34.0 + RETRANQUEO_M), ALTURA_M])
    adelante = -centro_cam / np.linalg.norm(centro_cam)
    derecha = np.array([1.0, 0.0, 0.0])
    abajo = np.cross(adelante, derecha)
    abajo /= np.linalg.norm(abajo)
    rotacion = np.vstack([derecha, abajo, adelante])
    traslacion = -rotacion @ centro_cam
    k = np.array([[INTR.fx, 0, INTR.cx], [0, INTR.fy, INTR.cy], [0, 0, 1.0]])
    return PitchModel(k @ np.column_stack([rotacion[:, 0], rotacion[:, 1], traslacion]))


def _spec(tmp_path, rotations=None):
    """Un band.json como el de REF-29, con la tabla fila→metros de la misma pose."""
    modelo = _modelo()
    geometria = band_from_pitch(modelo, INTR, side="left")
    camara = recover_camera(modelo.homography, INTR)
    cx_m, cy_m = camara.ground_xy_m
    tabla = []
    distancia = 5.0
    distancias = []
    while distancia <= 121.0:
        distancias.append(distancia)
        distancia *= 1.23
    for distancia in distancias:
        punto = camara.project(cx_m, cy_m + distancia, 0.0)
        if punto is not None and geometria.row_top - 1 <= punto[1] <= geometria.row_bottom + 1:
            tabla.append([round(punto[1], 2), round(float(distancia), 3)])
    tabla.sort()
    lado = {**geometria.to_dict(), "row_to_m": tabla}
    ruta = tmp_path / "band.json"
    ruta.write_text(json.dumps({"left": lado, "right": lado}), encoding="utf-8")
    return load_band_spec(ruta, rotations=rotations), modelo, (cx_m, cy_m)


def test_band_of_reparte_por_los_umbrales():
    assert band_of(0.0) == 0
    assert band_of(19.9) == 0
    assert band_of(20.0) == 1
    assert band_of(55.0) == 2
    assert band_of(79.9) == 3
    assert band_of(500.0) == len(DISTANCE_BANDS_M) - 2


def test_la_ida_y_vuelta_queda_bajo_medio_pixel(tmp_path):
    lados, _, _ = _spec(tmp_path)
    banda = lados["right"]  # rotación 0: el viaje es solo la geometría

    for punto in ((1200.0, 1100.0), (3000.0, 1500.0), (300.0, float(banda.geometry.row_top) + 5)):
        al_lienzo = banda.to_canvas(*punto)
        vuelta = banda.to_native(al_lienzo.x, al_lienzo.y)
        assert abs(vuelta.x - punto[0]) < 0.5
        assert abs(vuelta.y - punto[1]) < 0.5


def test_la_rotacion_de_180_se_deshace_sola(tmp_path):
    lados, _, _ = _spec(tmp_path)
    invertida = lados["left"]  # montada cabeza abajo
    punto = (1200.0, 1200.0)

    al_lienzo = invertida.to_canvas(*punto)
    vuelta = invertida.to_native(al_lienzo.x, al_lienzo.y)

    assert abs(vuelta.x - punto[0]) < 0.5
    assert abs(vuelta.y - punto[1]) < 0.5
    # Y el mismo punto SIN rotación cae en otro sitio del lienzo: la rotación cuenta.
    derecho = lados["right"].to_canvas(*punto)
    assert abs(derecho.x - al_lienzo.x) > 1.0


def test_una_caja_no_se_parte_en_la_junta_del_mosaico(tmp_path):
    lados, _, _ = _spec(tmp_path)
    banda = lados["right"]
    esquina_a, esquina_b = banda.map_box(1800.0, 1200.0, 1900.0, 1400.0)

    assert esquina_a.region == esquina_b.region
    assert esquina_a.x < esquina_b.x
    assert esquina_a.y < esquina_b.y


def test_el_codigo_de_tiempo_se_marca_no_se_pierde(tmp_path):
    lados, _, _ = _spec(tmp_path)
    derecho = lados["right"]
    # El código vive en el (0,0) del frame derecho (sin rotación).
    marcado = derecho.to_canvas(10.0, 4.0)
    assert marcado.status == "timecode"

    # También en el invertido: el código se pinta en el (0,0) de lo CAPTURADO, que
    # es el (0,0) del fichero almacenado en los dos montajes.
    invertido = lados["left"]
    marcado = invertido.to_canvas(10.0, 4.0)
    assert marcado.status == "timecode"


def test_lo_de_fuera_de_la_franja_queda_marcado(tmp_path):
    lados, _, _ = _spec(tmp_path)
    banda = lados["right"]
    arriba = banda.to_canvas(2000.0, 5.0)  # el cielo: fuera de la franja (y lejos del codigo)

    assert isinstance(arriba, MappedPoint)
    assert arriba.status == "fuera"


def test_homografia_y_tabla_coinciden_en_dos_metros(tmp_path):
    lados, modelo, (cx_m, cy_m) = _spec(tmp_path)
    banda = lados["right"]

    for fila in range(int(banda.geometry.row_top) + 20, int(banda.geometry.row_bottom) - 5, 97):
        por_tabla = banda.distance_at_row(float(fila))
        assert por_tabla is not None
        x_m, y_m = modelo.image_to_pitch((W / 2.0, float(fila)))
        por_homografia = math.hypot(x_m - cx_m, y_m - cy_m)
        if por_homografia > 125.0:
            continue  # más allá del ancla final la tabla sujeta: no es comparable
        assert por_tabla == pytest.approx(por_homografia, abs=2.0)
