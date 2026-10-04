"""D-FINE-N envuelto (ML-16): formas del contrato, el grafo ONNX sin postproceso
y el decode del ΔAP. Lo pesado se salta donde no está DEIM o torch."""

from __future__ import annotations

import numpy as np
import pytest

from ftrain.players.dfine import DEIM_DIR

torch = pytest.importorskip("torch", reason="el grupo `train` no está instalado")
pytestmark = pytest.mark.skipif(
    not DEIM_DIR.is_dir(), reason="DEIM no está: `uv run python tools/fetch_deim.py`"
)

TEST_SIZE = (192, 320)
"""(alto, ancho) chico adrede: trazar a 576x1920 tarda minutos. No puede bajar
más: con zancadas 16 y 32 da (12x20)+(6x10) = 300 anchors, justo las queries."""


@pytest.fixture(scope="module")
def red():
    from ftrain.players.dfine import build  # noqa: PLC0415

    return build(None, size=TEST_SIZE)


def test_las_salidas_son_el_contrato(red):
    alto, ancho = TEST_SIZE
    with torch.no_grad():
        logits, boxes = red(torch.rand(1, 3, alto, ancho))

    assert logits.shape == (1, 300, 80)  # COCO hasta los pesos propios (ADR 0020)
    assert boxes.shape == (1, 300, 4)
    assert bool((boxes >= 0).all())  # cxcywh normalizado
    assert bool((boxes <= 1).all())


def test_el_onnx_es_estatico_opset_17_y_sin_postproceso(red, tmp_path):
    import onnx  # noqa: PLC0415

    from ftrain.players.dfine import export_onnx  # noqa: PLC0415

    ruta = export_onnx(red, TEST_SIZE, tmp_path / "dfine.onnx")
    modelo = onnx.load(str(ruta), load_external_data=False)

    tipos = [n.op_type for n in modelo.graph.node]
    # Sin PostProcessor: ni NMS ni el topk de detecciones finales. Los DOS TopK
    # que quedan son arquitectura del decoder (selector de queries y stats del
    # LQE), no postproceso: D-FINE no existe sin ellos.
    assert "NonMaxSuppression" not in tipos
    assert tipos.count("TopK") == 2

    assert [i.version for i in modelo.opset_import if i.domain in ("", "ai.onnx")] == [17]
    salidas = {
        s.name: [d.dim_value for d in s.type.tensor_type.shape.dim] for s in modelo.graph.output
    }
    assert salidas == {"logits": [1, 300, 80], "boxes": [1, 300, 4]}


def test_el_decode_del_dap_vuelve_a_pixeles_originales():
    from tools.dfine_dap import decode  # noqa: PLC0415

    logits = np.full((2, 3), -10.0, dtype=np.float32)
    logits[1, 2] = 8.0  # una detección clara: consulta 1, clase 2
    boxes = np.array([[0.1, 0.1, 0.1, 0.1], [0.25, 0.5, 0.5, 1.0]], dtype=np.float32)

    # Imagen original 200x100, letterbox a 192x96 con escala 0.96 pegado en (0,0).
    detecciones = decode(
        logits,
        boxes,
        escala=0.96,
        alto=96,
        ancho=192,
        orig_w=200,
        orig_h=100,
        label2category={0: 1, 1: 2, 2: 3},
    )

    mejor = detecciones[0]
    assert mejor["category_id"] == 3
    assert mejor["score"] == pytest.approx(1.0 / (1.0 + np.exp(-8.0)), rel=1e-6)
    # cx 0.25*192=48, w 0.5*192=96 -> x1 0 px; cy 0.5*96=48, h 96 -> y1 0 px; /0.96
    assert mejor["bbox"] == pytest.approx([0.0, 0.0, 100.0, 100.0], abs=1e-6)
