"""El bundle dorado (ML-12): ida y vuelta, sha estable, tolerancias y endianness."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ftrain.export.golden import (
    GoldenArray,
    GoldenError,
    pack_bundle,
    read_bundle,
    to_bgra8,
    write_bundle,
)

TOLERANCIAS = {"y_000": {"ort_fp32": 1e-3, "coreml_fp16": 5e-2}}


def _arrays():
    rng = np.random.default_rng(0)
    entrada = GoldenArray(
        "image_000", rng.integers(0, 256, size=(4, 6, 4), dtype=np.uint8), "BGRA8"
    )
    salida = rng.standard_normal((1, 1, 4, 6)).astype(np.float32)
    salidas = {
        "torch_fp32": [GoldenArray("y_000", salida, "NCHW")],
        "ort_fp32": [GoldenArray("y_000", salida + 1e-5, "NCHW")],
        "coreml_fp16": [GoldenArray("y_000", salida.astype(np.float16), "NCHW")],
    }
    return [entrada], salidas


def test_el_bundle_se_relee_identico(tmp_path):
    entradas, salidas = _arrays()
    bundle = write_bundle(
        tmp_path,
        model="demo",
        version="v1",
        inputs=entradas,
        outputs=salidas,
        tolerances=TOLERANCIAS,
        detections=[[{"score": 0.9}]],
    )
    assert bundle == tmp_path / "demo-v1"

    manifiesto, arrays = read_bundle(bundle)
    np.testing.assert_array_equal(arrays["inputs"]["image_000"], entradas[0].array)
    np.testing.assert_array_equal(
        arrays["outputs"]["torch_fp32"]["y_000"], salidas["torch_fp32"][0].array
    )
    np.testing.assert_array_equal(
        arrays["outputs"]["coreml_fp16"]["y_000"], salidas["coreml_fp16"][0].array
    )
    assert manifiesto["inputs"][0]["dtype"] == "|u1"
    assert manifiesto["inputs"][0]["layout"] == "BGRA8"
    assert manifiesto["tolerances"]["y_000"]["coreml_fp16"] == 5e-2
    detecciones = json.loads((bundle / "detecciones.json").read_text(encoding="utf-8"))
    assert detecciones == [[{"score": 0.9}]]
    assert "Swift" in (bundle / "README.md").read_text(encoding="utf-8")


def test_el_sha_es_estable_y_el_contenido_manda(tmp_path):
    entradas, salidas = _arrays()
    shas = []
    for nombre in ("a", "b"):
        bundle = write_bundle(
            tmp_path / nombre,
            model="demo",
            version="v1",
            inputs=entradas,
            outputs=salidas,
            tolerances=TOLERANCIAS,
        )
        for ruta in bundle.rglob("*"):
            os.utime(ruta, (0, 0) if nombre == "b" else None)
        shas.append(pack_bundle(bundle, tmp_path / f"{nombre}.zip"))
    assert shas[0] == shas[1]

    otras = {
        **salidas,
        "torch_fp32": [GoldenArray("y_000", salidas["torch_fp32"][0].array + 1.0, "NCHW")],
    }
    bundle = write_bundle(
        tmp_path / "c",
        model="demo",
        version="v1",
        inputs=entradas,
        outputs=otras,
        tolerances=TOLERANCIAS,
    )
    assert pack_bundle(bundle, tmp_path / "c.zip") != shas[0]


def test_las_tolerancias_son_obligatorias(tmp_path):
    entradas, salidas = _arrays()
    with pytest.raises(GoldenError, match=r"y_000.*coreml_fp16"):
        write_bundle(
            tmp_path,
            model="demo",
            version="v1",
            inputs=entradas,
            outputs=salidas,
            tolerances={"y_000": {"ort_fp32": 1e-3}},  # falta la de Core ML
        )


def test_sin_torch_no_hay_dorado(tmp_path):
    entradas, salidas = _arrays()
    del salidas["torch_fp32"]
    with pytest.raises(GoldenError, match="torch_fp32"):
        write_bundle(
            tmp_path,
            model="demo",
            version="v1",
            inputs=entradas,
            outputs=salidas,
            tolerances=TOLERANCIAS,
        )


def test_little_endian_en_el_disco(tmp_path):
    entradas, _ = _arrays()
    salidas = {"torch_fp32": [GoldenArray("y_000", np.array([[1, 2]], dtype=np.int32), "NC")]}
    bundle = write_bundle(
        tmp_path,
        model="demo",
        version="v1",
        inputs=entradas,
        outputs=salidas,
        tolerances={},
    )
    crudo = (bundle / "salidas" / "torch_fp32" / "y_000.bin").read_bytes()
    assert crudo == b"\x01\x00\x00\x00\x02\x00\x00\x00"  # little-endian, orden C


def test_i64_no_entra_en_el_formato(tmp_path):
    entradas, _ = _arrays()
    salidas = {"torch_fp32": [GoldenArray("y_000", np.array([1], dtype=np.int64), "N")]}
    with pytest.raises(GoldenError, match="dtype"):
        write_bundle(
            tmp_path,
            model="demo",
            version="v1",
            inputs=entradas,
            outputs=salidas,
            tolerances={},
        )


def test_la_version_del_manifiesto_se_comprueba(tmp_path):
    entradas, salidas = _arrays()
    bundle = write_bundle(
        tmp_path,
        model="demo",
        version="v1",
        inputs=entradas,
        outputs=salidas,
        tolerances=TOLERANCIAS,
    )
    manifiesto = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifiesto["version"] = 99
    (bundle / "manifest.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    with pytest.raises(GoldenError, match="99"):
        read_bundle(bundle)


def test_to_bgra8_cruza_los_canales():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    bgra = to_bgra8(rgb)
    assert bgra[0, 0].tolist() == [30, 20, 10, 255]

    with pytest.raises(GoldenError, match="HWC RGB"):
        to_bgra8(np.zeros((1, 1, 4), dtype=np.uint8))
