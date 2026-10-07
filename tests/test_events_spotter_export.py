"""El modo paso de N4 para Core ML (SPK-53): las dos variantes del estado, sus specs y
la secuencia dorada."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ftrain.constants import N4_INPUT_H, N4_INPUT_W, N4_STATE_ROW, N4_STEP_COREML_ATOL
from ftrain.export.coreml import ExportError, ExportSpec, load_spec

torch = pytest.importorskip("torch", reason="el grupo `train` no está instalado")
pytest.importorskip("timm")

from ftrain.events.spotter import N4Spotter  # noqa: E402
from ftrain.events.spotter_export import (  # noqa: E402
    TINY_INPUT_H,
    TINY_INPUT_W,
    N4StepExplicit,
    N4StepStateful,
    build_step_explicit,
    build_step_stateful,
    build_tiny,
    build_tiny_explicit,
    build_tiny_stateful,
    frames_to_input,
    golden_sequence,
    packed_shape,
    run_coreml_sequence,
    seeded_frames,
    state_names,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "export"
ALTO, ANCHO = 128, 224  # N4 a la mitad, para que el test sea rápido
PASOS = 4


def test_empaquetar_llena_filas_de_32_sin_cambiar_el_orden():
    assert packed_shape((1, 4, 64, 112)) == (1, 4, 224, N4_STATE_ROW)
    assert packed_shape((1, 12, 8, 14)) == (1, 1, 42, N4_STATE_ROW)  # 112 por canal
    assert packed_shape((1, 64, 1, 1)) == (1, 1, 2, N4_STATE_ROW)
    with pytest.raises(ValueError, match="filas"):
        packed_shape((1, 3, 5, 7))
    assert state_names(3) == ["tsm0", "tsm1", "tsm2", "h"]


def _pasos(red, clip, estado):
    salidas = []
    with torch.no_grad():
        for k in range(clip.shape[0]):
            y, estado = red.step(clip[k : k + 1], estado)
            salidas.append(y)
    return torch.cat(salidas)


def test_las_dos_variantes_son_el_mismo_paso_y_el_reinicio_vuelve_a_cero():
    torch.manual_seed(0)
    red = N4Spotter().eval()
    clip = torch.rand(PASOS, 3, ALTO, ANCHO)
    esperado = _pasos(red, clip, red.initial_state(1, ALTO, ANCHO))

    con_estado = N4StepStateful(red, ALTO, ANCHO).eval()
    explicito = N4StepExplicit(red, ALTO, ANCHO).eval()
    with torch.no_grad():
        visto = torch.cat([con_estado(clip[k : k + 1]) for k in range(PASOS)])
        assert (visto - esperado).abs().max().item() < 1e-5
        # Reiniciar es volver a empezar: sin él, el primer paso ya no sería el mismo.
        assert (con_estado(clip[:1]) - esperado[:1]).abs().max().item() > 1e-4
        con_estado.reset()
        assert (con_estado(clip[:1]) - esperado[:1]).abs().max().item() < 1e-5

        estado = [torch.zeros(forma) for forma in explicito.packed]
        salidas = []
        for k in range(PASOS):
            y, *estado = explicito(clip[k : k + 1], *estado)
            salidas.append(y)
        assert (torch.cat(salidas) - esperado).abs().max().item() < 1e-5


@pytest.mark.parametrize(
    ("spec", "builder", "alto", "ancho"),
    [
        ("n4-step-mlstate.yaml", build_step_stateful, N4_INPUT_H, N4_INPUT_W),
        ("n4-step-explicit.yaml", build_step_explicit, N4_INPUT_H, N4_INPUT_W),
        ("n4-tiny-mlstate.yaml", build_tiny_stateful, TINY_INPUT_H, TINY_INPUT_W),
        ("n4-tiny-explicit.yaml", build_tiny_explicit, TINY_INPUT_H, TINY_INPUT_W),
    ],
)
def test_los_specs_casan_con_el_modulo(spec, builder, alto, ancho):
    """Las formas del YAML son el contrato con la app: se leen del módulo, no de memoria."""
    s = load_spec(CONFIGS / spec)
    m = builder(None)
    entradas = {e.name: e.shape for e in s.inputs}
    assert entradas["frame"] == (1, 3, alto, ancho)
    if s.states:
        assert list(s.states) == m.names
        assert len(s.inputs) == 1
        assert s.output_names == ("logits",)
    else:
        assert [entradas[n] for n in m.names] == m.packed
        assert s.output_names == ("logits", *(f"{n}_out" for n in m.names))
        assert set(s.fp16_outputs) == {f"{n}_out" for n in m.names}


def test_fp16_outputs_tiene_que_ser_parte_del_contrato():
    datos = {
        "model_name": "x",
        "inputs": [{"name": "frame", "kind": "tensor", "shape": [1, 3, 8, 8]}],
        "outputs": ["logits"],
        "fp16_outputs": ["otra"],
    }
    with pytest.raises(ExportError, match="fp16_outputs"):
        ExportSpec.from_dict(datos)


def test_la_secuencia_dorada_parte_de_cero_y_redondea_la_entrada_a_fp16():
    frames = seeded_frames(PASOS, TINY_INPUT_H, TINY_INPUT_W)
    assert frames.dtype == np.uint8
    assert frames.shape == (PASOS, 3, TINY_INPUT_H, TINY_INPUT_W)
    entrada = frames_to_input(frames)
    assert torch.equal(entrada, entrada.half().float())  # ya viene redondeada
    ref = golden_sequence(build_tiny(None), frames)
    assert ref["logits"].shape == (PASOS, 4)
    assert ref["h"].shape == (PASOS, 32)
    assert np.array_equal(golden_sequence(build_tiny(None), frames)["logits"], ref["logits"])


def test_el_bundle_de_la_secuencia_lo_lee_el_formato_de_ml12(tmp_path):
    from ftrain.export.golden import read_bundle  # noqa: PLC0415
    from tools.golden_sequence import build_bundle  # noqa: PLC0415

    frames = seeded_frames(PASOS, TINY_INPUT_H, TINY_INPUT_W)
    bundle, ref = build_bundle(
        build_tiny(None),
        frames=frames,
        name="tiny",
        version="v0",
        out=tmp_path,
        atol=N4_STEP_COREML_ATOL,
    )
    manifiesto, arrays = read_bundle(bundle)
    assert len(manifiesto["inputs"]) == PASOS
    assert np.array_equal(arrays["inputs"]["frame_003"], frames[3:4])
    assert np.array_equal(arrays["outputs"]["torch_fp32"]["h_002"], ref["h"][2:3])
    assert manifiesto["tolerances"]["logits_000"]["coreml_fp16"] == N4_STEP_COREML_ATOL["logits"]
    secuencia = json.loads((bundle / "sequence.json").read_text(encoding="utf-8"))
    assert secuencia["steps"] == PASOS
    assert secuencia["frame_input"] == "frame"


@pytest.mark.skipif(sys.platform != "darwin", reason="predecir con Core ML pide macOS")
@pytest.mark.parametrize("variante", ["mlstate", "explicit"])
def test_core_ml_recorre_la_secuencia_dentro_de_la_tolerancia(tmp_path, variante):
    pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")
    from ftrain.export.coreml import build_metadata, convert  # noqa: PLC0415

    spec = load_spec(CONFIGS / f"n4-tiny-{variante}.yaml")
    builder = build_tiny_stateful if variante == "mlstate" else build_tiny_explicit
    metadatos = build_metadata(
        spec, classes=[], dataset_version="-", commit="-", region="-", frames=1
    )
    modelo = convert(builder(None), spec, metadatos)
    paquete = tmp_path / f"{variante}.mlpackage"
    modelo.save(str(paquete))
    estados = [e.name for e in modelo.get_spec().description.state]
    assert estados == (["tsm0", "h"] if variante == "mlstate" else [])

    frames = seeded_frames(PASOS * 3, TINY_INPUT_H, TINY_INPUT_W)
    ref = golden_sequence(build_tiny(None), frames)
    visto = run_coreml_sequence(paquete, frames, "cpu_only")
    for nombre, atol in N4_STEP_COREML_ATOL.items():
        assert np.abs(visto[nombre] - ref[nombre]).max() < atol, nombre
