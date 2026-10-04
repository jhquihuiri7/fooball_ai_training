"""La ficha v2 (ML-13): construcción y coherencias puras, el .onnx sintético, y
el contrato con la referencia (load_registry del commit fijado la carga)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from ftrain.export.coreml import sha256_of
from ftrain.export.ficha import (
    CoremlArtifact,
    Ficha,
    FichaError,
    InputBlock,
    OnnxArtifact,
    OutputBlock,
    ParityBlock,
    onnx_facts,
    render_registry,
)


def _ficha_detr(**cambios) -> Ficha:
    base = {
        "name": "dfine-n-band",
        "version": "0.1.0",
        "postprocess": "detr",
        "box_format": "cxcywh_norm",
        "input": InputBlock(name="image", shape=(1, 3, 576, 1920), region="playable_band"),
        "outputs": (
            OutputBlock("boxes", (1, 300, 4), "boxes_cxcywh_norm"),
            OutputBlock("logits", (1, 300, 3), "logits"),
        ),
        "classes": ("player", "goalkeeper", "referee"),
        "license": "Apache-2.0",
        "onnx": OnnxArtifact("onnx/dfine-n-band.onnx", "aaa111", 18),
        "coreml": CoremlArtifact("coreml/dfine-n-band.mlpackage.zip", "bbb222", 18, 98.5),
        "parity": ParityBlock("golden/dfine-n-band-0.1.0.golden.zip", 0.02),
        "dataset": "jugadores/v1",
        "commit": "abc1234",
        "exported": "2026-10-03",
    }
    return Ficha(**{**base, **cambios})


def _ficha_heatmap() -> Ficha:
    return Ficha(
        name="ball-roilite",
        version="0.1.0",
        postprocess="heatmap",
        box_format="heatmap_stride",
        heatmap_stride=2,
        input=InputBlock(
            name="rois", shape=(2, 3, 256, 256), color="gray_temporal", region="roi", frames=3
        ),
        outputs=(OutputBlock("heatmap", (2, 1, 128, 128), "heatmap"),),
        classes=("ball",),
        license="propio",
        onnx=OnnxArtifact("onnx/ball-roilite.onnx", "ccc333", 18),
        exported="2026-10-03",
    )


def test_la_entrada_lleva_todo_lo_que_pide_el_lector():
    entrada = _ficha_detr().to_entry()
    assert entrada["postprocess"] == "detr"
    assert entrada["precision"] == "fp16"
    assert entrada["artifacts"]["onnx"]["opset"] == 18
    assert entrada["artifacts"]["coreml"]["compute_units"] == "cpu_and_ne"
    assert entrada["artifacts"]["coreml"]["ops_off_ane"] == 0
    assert entrada["parity"]["atol"] == 0.02
    assert entrada["input"]["region"] == "playable_band"
    assert entrada["dataset"] == "jugadores/v1"
    assert entrada["commit"] == "abc1234"

    documento = yaml.safe_load(render_registry({"dfine-n-band": _ficha_detr()}))
    assert documento["version"] == 2
    assert "dfine-n-band" in documento["models"]


def test_las_coherencias_del_lector_saltan_al_construir():
    with pytest.raises(FichaError, match="postprocess"):
        _ficha_detr(postprocess="magia").to_entry()
    with pytest.raises(FichaError, match="box_format"):
        _ficha_detr(box_format="heatmap_stride").to_entry()
    with pytest.raises(FichaError, match="heatmap_stride"):
        _ficha_detr(heatmap_stride=2).to_entry()
    with pytest.raises(FichaError, match="nms_iou"):
        _ficha_detr(nms_iou=0.5).to_entry()
    with pytest.raises(FichaError, match="region"):
        _ficha_detr(
            input=InputBlock(name="image", shape=(1, 3, 576, 1920), region="analysis_zone")
        ).to_entry()
    with pytest.raises(FichaError, match="gray_temporal"):
        _ficha_detr(input=InputBlock(name="image", shape=(1, 3, 576, 1920), frames=3)).to_entry()
    with pytest.raises(FichaError, match="compute_units"):
        _ficha_detr(coreml=CoremlArtifact("c.zip", "bbb", 18, 98.5, compute_units="all")).to_entry()


def _onnx_sintetico(ruta):
    from onnx import TensorProto, helper  # noqa: PLC0415 — junto a su único consumidor

    pesos = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    grafo = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["image", "W"], ["xw"]),
            helper.make_node("Relu", ["xw"], ["logits"]),
        ],
        name="demo",
        inputs=[helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, pesos.shape, pesos.flatten())],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    ruta.write_bytes(modelo.SerializeToString())


def test_onnx_facts_lee_del_fichero_y_no_de_ninguna_config(tmp_path):
    ruta = tmp_path / "demo.onnx"
    _onnx_sintetico(ruta)

    datos = onnx_facts(ruta)
    assert datos.input_name == "image"
    assert datos.input_shape == (1, 4)
    assert datos.outputs == (("logits", (1, 3)),)
    assert datos.opset == 17
    assert datos.sha256 == sha256_of(ruta)


def test_load_registry_de_la_referencia_carga_la_ficha(tmp_path):
    pytest.importorskip("libs.vision.registry", reason="el grupo `ref` no está instalado")
    from ftrain import ref  # noqa: PLC0415

    ruta = tmp_path / "registry.yaml"
    ruta.write_text(
        render_registry({"dfine-n-band": _ficha_detr(), "ball-roilite": _ficha_heatmap()}),
        encoding="utf-8",
    )
    modelos = ref.load_registry(ruta)

    detector = modelos["dfine-n-band"]
    assert detector.postprocess == "detr"
    assert sorted(a.kind for a in detector.artifacts) == ["coreml", "onnx"]
    assert detector.parity is not None
    assert detector.parity.atol == 0.02
    assert detector.input.region == "playable_band"

    balon = modelos["ball-roilite"]
    assert balon.postprocess == "heatmap"
    assert balon.input.frames == 3


def test_coreml_facts_lee_target_y_precision(tmp_path):
    ct = pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")
    from coremltools.converters.mil import Builder as mb  # noqa: N813, PLC0415

    from ftrain.export.ficha import coreml_facts  # noqa: PLC0415

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 16, 8, 8))], opset_version=ct.target.iOS18)
    def programa(x):
        return mb.relu(x=x)

    modelo = ct.convert(
        programa,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    paquete = tmp_path / "demo.mlpackage"
    modelo.save(str(paquete))

    datos = coreml_facts(paquete)
    assert datos.min_ios == 18
    assert datos.precision == "fp16"
    assert datos.inputs == ("x",)
    assert len(datos.outputs) == 1
