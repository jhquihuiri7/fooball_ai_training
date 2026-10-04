"""El manifiesto de partido (ML-06): la plantilla valida y cada campo se defiende."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ftrain.manifest import ManifestError, MatchManifest

PLANTILLA = Path(__file__).resolve().parents[1] / "matches" / "_plantilla.yaml"


def _base() -> dict:
    return yaml.safe_load(PLANTILLA.read_text(encoding="utf-8"))


def test_la_plantilla_valida():
    manifiesto = MatchManifest.load(PLANTILLA)

    assert manifiesto.match_id == "m_20260703_2030_a1b2"
    assert manifiesto.light == "floodlight"
    assert manifiesto.setback_m == 10.0
    assert manifiesto.sides["left"].rotation_deg == 180
    assert manifiesto.sides["right"].nv12_uri is None
    assert manifiesto.rig_uri.startswith("gs://")
    assert manifiesto.split_hint == "train"


@pytest.mark.parametrize(
    "campo",
    ["match_id", "venue_id", "date", "light", "setback_m", "height_m", "sides", "split_hint"],
)
def test_cada_campo_obligatorio_falla_nombrandose(campo):
    datos = _base()
    del datos[campo]

    with pytest.raises(ManifestError, match=campo):
        MatchManifest.from_dict(datos)


def test_una_luz_desconocida_se_rechaza():
    datos = _base()
    datos["light"] = "antorchas"

    with pytest.raises(ManifestError, match="light"):
        MatchManifest.from_dict(datos)


def test_un_retranqueo_negativo_se_rechaza():
    datos = _base()
    datos["setback_m"] = -2.0

    with pytest.raises(ManifestError, match="setback_m"):
        MatchManifest.from_dict(datos)


def test_sin_rotacion_falla_nombrando_el_lado():
    datos = _base()
    del datos["sides"]["left"]["rotation_deg"]

    with pytest.raises(ManifestError, match="rotation_deg") as fallo:
        MatchManifest.from_dict(datos)
    assert "left" in str(fallo.value)


def test_una_rotacion_rara_se_rechaza():
    datos = _base()
    datos["sides"]["right"]["rotation_deg"] = 90

    with pytest.raises(ManifestError, match="rotation_deg"):
        MatchManifest.from_dict(datos)


def test_una_uri_que_no_es_gs_se_rechaza():
    datos = _base()
    datos["sides"]["left"]["recording_uri"] = "https://drive.google.com/x.mov"

    with pytest.raises(ManifestError, match="recording_uri"):
        MatchManifest.from_dict(datos)


def test_un_split_desconocido_se_rechaza():
    datos = _base()
    datos["split_hint"] = "test"

    with pytest.raises(ManifestError, match="split_hint"):
        MatchManifest.from_dict(datos)


def test_faltar_un_lado_se_rechaza():
    datos = _base()
    del datos["sides"]["right"]

    with pytest.raises(ManifestError, match="sides"):
        MatchManifest.from_dict(datos)


def test_sin_calibracion_se_rechaza():
    datos = _base()
    del datos["calibration"]

    with pytest.raises(ManifestError, match="calibration"):
        MatchManifest.from_dict(datos)
