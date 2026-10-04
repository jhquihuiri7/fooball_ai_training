"""Paridad (ML-11): comparadores y decodificadores puros; ORT contra numpy;
las patas de torch y de Core ML con importorskip."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from ftrain.export.coreml import ExportSpec
from ftrain.export.parity import (
    DEFAULT_COREML_ATOL,
    TORCH_ORT_ATOL,
    ParityError,
    aggregate_metrics,
    compare_lotes,
    compare_outputs,
    detector_metrics,
    evaluate,
    heatmap_metrics,
    run_coreml,
    run_ort,
    run_torch,
    seeded_batch,
    spotter_metrics,
)

SPEC_TENSOR = ExportSpec.from_dict(
    {
        "model_name": "demo",
        "inputs": [{"name": "x", "kind": "tensor", "shape": [1, 4]}],
        "outputs": ["y"],
    }
)

SPEC_IMAGEN = ExportSpec.from_dict(
    {
        "model_name": "demo",
        "inputs": [{"name": "image", "kind": "image", "shape": [1, 3, 16, 16], "color": "RGB"}],
        "outputs": ["heatmap"],
    }
)


# --------------------------------------------------------------------------- #
# Comparadores
# --------------------------------------------------------------------------- #
def test_los_deltas_absoluto_y_relativo():
    ref = {"y": np.array([1.0, 2.0], dtype=np.float32)}
    otro = {"y": np.array([1.001, 2.0], dtype=np.float32)}
    delta = compare_outputs(ref, otro)["y"]
    assert delta.max_abs == pytest.approx(1e-3, rel=1e-3)
    assert delta.max_rel == pytest.approx(1e-3, rel=1e-3)

    with pytest.raises(ParityError, match="salidas distintas"):
        compare_outputs(ref, {"z": otro["y"]})
    with pytest.raises(ParityError, match="formas distintas"):
        compare_outputs(ref, {"y": np.zeros(3, dtype=np.float32)})


def test_el_lote_se_queda_con_el_peor_delta():
    refs = [{"y": np.zeros(2, dtype=np.float32)}, {"y": np.zeros(2, dtype=np.float32)}]
    otros = [
        {"y": np.array([0.1, 0.0], dtype=np.float32)},
        {"y": np.array([0.0, 0.3], dtype=np.float32)},
    ]
    assert compare_lotes(refs, otros)["y"].max_abs == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Decodificadores
# --------------------------------------------------------------------------- #
def _salidas_detector(centros: list[tuple[float, float]], vivas: list[bool]):
    n = len(centros)
    logits = np.full((1, n, 2), -5.0, dtype=np.float32)
    boxes = np.zeros((1, n, 4), dtype=np.float32)
    for indice, ((cx, cy), viva) in enumerate(zip(centros, vivas, strict=True)):
        logits[0, indice, 0] = 5.0 if viva else -5.0
        boxes[0, indice, :2] = (cx, cy)
        boxes[0, indice, 2:] = 0.05
    return {"logits": logits, "boxes": boxes}


def test_el_detector_mide_recall_por_banda_y_desplazamiento():
    def band_of(cx: float, cy: float) -> str:  # noqa: ARG001 — la firma es el contrato
        return "cerca" if cy < 0.5 else "lejos"

    ref = _salidas_detector([(0.2, 0.2), (0.5, 0.8)], vivas=[True, True])
    # La primera se mueve 0.005 (casa); la segunda desaparece en el otro backend.
    otro = _salidas_detector([(0.205, 0.2), (0.5, 0.8)], vivas=[True, False])

    metricas = detector_metrics(ref, otro, band_of=band_of)
    assert metricas["recall_delta_por_banda"] == {"cerca": 0.0, "lejos": 1.0}
    assert metricas["desplazamiento_centro"] == pytest.approx(0.005, rel=1e-6)


def test_el_heatmap_mide_el_pico():
    plano_ref = np.zeros((1, 1, 8, 8), dtype=np.float32)
    plano_ref[0, 0, 2, 3] = 1.0
    plano_otro = np.zeros((1, 1, 8, 8), dtype=np.float32)
    plano_otro[0, 0, 4, 3] = 0.9

    metricas = heatmap_metrics({"heatmap": plano_ref}, {"heatmap": plano_otro})
    assert metricas["heatmap"]["peak_shift_px"] == pytest.approx(2.0)
    assert metricas["heatmap"]["peak_value_delta"] == pytest.approx(0.1)


def test_el_spotter_mide_los_logits():
    ref = {"logits": np.array([0.0, 1.0], dtype=np.float32)}
    otro = {"logits": np.array([0.0, 1.2], dtype=np.float32)}
    assert spotter_metrics(ref, otro)["logits"] == pytest.approx(0.2)


def test_la_media_de_metricas_anidadas():
    muestras = [
        {"a": 1.0, "bandas": {"cerca": 0.0}},
        {"a": 3.0, "bandas": {"cerca": 1.0}},
    ]
    assert aggregate_metrics(muestras) == {"a": 2.0, "bandas": {"cerca": 0.5}}


# --------------------------------------------------------------------------- #
# ORT contra una referencia en numpy, sobre un .onnx sintético
# --------------------------------------------------------------------------- #
def _onnx_sintetico(ruta):
    from onnx import TensorProto, helper  # noqa: PLC0415 — junto a su único consumidor

    pesos = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    grafo = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["x", "W"], ["xw"]),
            helper.make_node("Relu", ["xw"], ["y"]),
        ],
        name="demo",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, pesos.shape, pesos.flatten())],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 10  # el onnx instalado escribe un IR más nuevo que el que ORT lee
    ruta.write_bytes(modelo.SerializeToString())
    return pesos


def test_ort_contra_numpy_en_un_onnx_sintetico(tmp_path):
    ruta = tmp_path / "demo.onnx"
    pesos = _onnx_sintetico(ruta)

    lote = seeded_batch(SPEC_TENSOR, 5)
    refs = [{"y": np.maximum(muestra["x"] @ pesos, 0.0)} for muestra in lote]
    salidas = run_ort(ruta, SPEC_TENSOR, lote)

    assert compare_lotes(refs, salidas)["y"].max_abs < 1e-6


def test_el_informe_y_los_umbrales(tmp_path):
    refs = [{"y": np.zeros(2, dtype=np.float32)}]
    otros = {"torch_ort": [{"y": np.array([0.01, 0.0], dtype=np.float32)}]}

    informe, ok = evaluate(refs, otros, {"torch_ort": 1e-3})
    assert not ok
    assert informe["violations"]
    assert "torch_ort/y" in informe["violations"][0]

    informe, ok = evaluate(refs, otros, {"torch_ort": 0.1})
    assert ok
    ruta = tmp_path / "informe.json"
    ruta.write_text(json.dumps(informe), encoding="utf-8")  # serializable: el CLI hace esto
    assert json.loads(ruta.read_text(encoding="utf-8"))["pairs"]["torch_ort"]["atol"] == 0.1

    with pytest.raises(ParityError, match="umbral"):
        evaluate(refs, otros, {})


# --------------------------------------------------------------------------- #
# Las patas de torch y de Core ML
# --------------------------------------------------------------------------- #
def _red_pequena(torch):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    )


def test_torch_contra_ort_cumple_la_regla_del_repo(tmp_path):
    torch = pytest.importorskip("torch", reason="el grupo `train` no está instalado")
    red = _red_pequena(torch).eval()
    ruta = tmp_path / "demo.onnx"
    torch.onnx.export(
        red,
        (torch.zeros(1, 3, 16, 16),),
        str(ruta),
        input_names=["image"],
        output_names=["heatmap"],
    )

    lote = seeded_batch(SPEC_IMAGEN, 4)
    refs = run_torch(red, SPEC_IMAGEN, lote)
    informe, ok = evaluate(
        refs, {"torch_ort": run_ort(ruta, SPEC_IMAGEN, lote)}, {"torch_ort": TORCH_ORT_ATOL}
    )
    assert ok, informe["violations"]


@pytest.mark.skipif(sys.platform != "darwin", reason="predecir con Core ML pide macOS")
def test_en_el_mac_aparece_la_columna_de_coreml(tmp_path):
    torch = pytest.importorskip("torch", reason="el grupo `train` no está instalado")
    pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")
    from ftrain.export.coreml import build_metadata, convert  # noqa: PLC0415

    red = _red_pequena(torch)
    metadatos = build_metadata(
        SPEC_IMAGEN, classes=["ball"], dataset_version="v0", commit="abc", region="roi", frames=1
    )
    paquete = tmp_path / "demo.mlpackage"
    convert(red, SPEC_IMAGEN, metadatos).save(str(paquete))

    lote = seeded_batch(SPEC_IMAGEN, 2)
    refs = run_torch(red, SPEC_IMAGEN, lote)
    informe, ok = evaluate(
        refs,
        {"torch_coreml_cpu": run_coreml(paquete, SPEC_IMAGEN, lote, "cpu_only")},
        {"torch_coreml_cpu": DEFAULT_COREML_ATOL},
    )
    assert ok, informe["violations"]
    assert "heatmap" in informe["pairs"]["torch_coreml_cpu"]["outputs"]
