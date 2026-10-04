"""El export a Core ML (ML-09): lo puro en cualquier SO, la conversión donde se pueda."""

from __future__ import annotations

import json
import os
import zipfile
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from ftrain.export.coreml import (
    ExportError,
    ExportSpec,
    build_metadata,
    deterministic_zip,
    load_spec,
    normalize_manifest,
    sha256_of,
)

SPEC = {
    "model_name": "demo",
    "inputs": [{"name": "image", "kind": "image", "shape": [1, 3, 32, 32], "color": "RGB"}],
    "outputs": ["heatmap"],
    "metadata": {"nota": "de prueba"},
}


# --------------------------------------------------------------------------- #
# Puro: spec, metadatos y zip (corren también en Windows)
# --------------------------------------------------------------------------- #
def test_el_spec_valida_y_nombra_lo_que_falta(tmp_path):
    ruta = tmp_path / "spec.yaml"
    ruta.write_text(yaml.safe_dump(SPEC), encoding="utf-8")
    spec = load_spec(ruta)
    assert spec.model_name == "demo"
    assert spec.inputs[0].color == "RGB"

    for campo, roto in (
        ("inputs", {**SPEC, "inputs": []}),
        ("outputs", {**SPEC, "outputs": []}),
        ("model_name", {k: v for k, v in SPEC.items() if k != "model_name"}),
        ("kind", {**SPEC, "inputs": [{"name": "x", "kind": "audio", "shape": [1]}]}),
        ("shape", {**SPEC, "inputs": [{"name": "x", "kind": "tensor", "shape": [0]}]}),
        (
            "color",
            {**SPEC, "inputs": [{"name": "x", "kind": "image", "shape": [1], "color": "HSV"}]},
        ),
    ):
        with pytest.raises(ExportError, match=campo):
            ExportSpec.from_dict(roto)


def test_los_metadatos_llevan_el_contrato_en_texto():
    spec = ExportSpec.from_dict(SPEC)
    metadatos = build_metadata(
        spec,
        classes=["ball"],
        dataset_version="jugadores/v1",
        commit="abc1234",
        region="roi",
        frames=3,
    )
    assert metadatos["classes"] == "ball"
    assert metadatos["color"] == "RGB"
    assert metadatos["frames"] == "3"
    assert metadatos["dataset_version"] == "jugadores/v1"
    assert metadatos["nota"] == "de prueba"  # lo del spec viaja también
    assert all(isinstance(v, str) for v in metadatos.values())


def _paquete_falso(raiz: Path, *, contenido: bytes = b"pesos") -> Path:
    paquete = raiz / "demo.mlpackage"
    (paquete / "Data" / "com.apple.CoreML").mkdir(parents=True)
    (paquete / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"modelo")
    (paquete / "Data" / "com.apple.CoreML" / "weights").mkdir()
    (paquete / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(contenido)
    manifiesto = {
        "fileFormatVersion": "1.0.0",
        "rootModelIdentifier": "AAAA-1111",
        "itemInfoEntries": {
            "AAAA-1111": {"path": "com.apple.CoreML/model.mlmodel"},
            "BBBB-2222": {"path": "com.apple.CoreML/weights/weight.bin"},
        },
    }
    (paquete / "Manifest.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return paquete


def test_el_zip_es_determinista_con_uuids_y_mtimes_distintos(tmp_path):
    a = _paquete_falso(tmp_path / "a")
    b = _paquete_falso(tmp_path / "b")
    # Al de B le cambian los UUIDs del manifiesto y los mtimes, como entre dos saves.
    manifiesto = json.loads((b / "Manifest.json").read_text(encoding="utf-8"))
    manifiesto["rootModelIdentifier"] = "CCCC-3333"
    manifiesto["itemInfoEntries"] = {
        "CCCC-3333": {"path": "com.apple.CoreML/model.mlmodel"},
        "DDDD-4444": {"path": "com.apple.CoreML/weights/weight.bin"},
    }
    (b / "Manifest.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    for ruta in b.rglob("*"):
        os.utime(ruta, (0, 0))

    normalize_manifest(a)
    normalize_manifest(b)
    sha_a = deterministic_zip(a, tmp_path / "a.mlpackage.zip")
    sha_b = deterministic_zip(b, tmp_path / "b.mlpackage.zip")

    assert sha_a == sha_b
    with zipfile.ZipFile(tmp_path / "a.mlpackage.zip") as zf:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in zf.infolist())


def test_contenido_distinto_da_sha_distinto(tmp_path):
    a = _paquete_falso(tmp_path / "a")
    b = _paquete_falso(tmp_path / "b", contenido=b"otros pesos")
    normalize_manifest(a)
    normalize_manifest(b)

    assert deterministic_zip(a, tmp_path / "a.zip") != deterministic_zip(b, tmp_path / "b.zip")


# --------------------------------------------------------------------------- #
# La conversión de verdad (donde hay torch y coremltools)
# --------------------------------------------------------------------------- #
def test_states_y_functions_avisan_que_no_estan():
    pytest.importorskip("torch", reason="el grupo `train` no está instalado")
    pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")
    from ftrain.export.coreml import convert  # noqa: PLC0415

    spec = ExportSpec.from_dict({**SPEC, "states": ["h"]})
    with pytest.raises(ExportError, match="states"):
        convert(object(), spec, {})
    spec = ExportSpec.from_dict({**SPEC, "functions": {"global": "x"}})
    with pytest.raises(ExportError, match="export_multifunction"):
        convert(object(), spec, {})


def test_dos_exports_del_mismo_modelo_dan_el_mismo_sha(tmp_path):
    torch = pytest.importorskip("torch", reason="el grupo `train` no está instalado")
    ct = pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")
    from ftrain.export.coreml import convert  # noqa: PLC0415

    torch.manual_seed(0)
    red = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    )
    spec = ExportSpec.from_dict(SPEC)
    metadatos = build_metadata(
        spec, classes=["ball"], dataset_version="v0", commit="abc", region="roi", frames=1
    )

    shas = []
    for intento in ("a", "b"):
        modelo = convert(red, spec, metadatos)
        paquete = tmp_path / intento / "demo.mlpackage"
        paquete.parent.mkdir()
        modelo.save(str(paquete))
        normalize_manifest(paquete)
        shas.append(deterministic_zip(paquete, tmp_path / f"{intento}.mlpackage.zip"))

    assert shas[0] == shas[1]
    assert sha256_of(tmp_path / "a.mlpackage.zip") == shas[0]

    # El paquete abre y declara exactamente el contrato.
    abierto = ct.models.MLModel(str(tmp_path / "a" / "demo.mlpackage"))
    descripcion = abierto.get_spec().description
    assert [e.name for e in descripcion.input] == ["image"]
    assert [s.name for s in descripcion.output] == ["heatmap"]
    assert abierto.user_defined_metadata["classes"] == "ball"
