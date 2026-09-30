"""Tests de `tools/mac_smoke.py` (ML-05).

Cada parte se prueba donde se puede:
- la red de prueba, donde haya torch;
- convertir y guardar, donde haya `coremltools` (Linux y el Mac; en Windows se salta);
- predecir, solo en macOS;
- el aviso de que falta `coremltools`, solo donde falta.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from tools.mac_smoke import FP16_SMOKE_TOL, SMOKE_INPUT, build_net, convert, main, predict_diffs

SIN_COREMLTOOLS = importlib.util.find_spec("coremltools") is None
MOTIVO_SIN_COREMLTOOLS = "coremltools no tiene ruedas para Windows: se prueba en Linux o en el Mac"


def test_la_red_de_prueba_son_tres_convs_con_pesos_fijos():
    torch = pytest.importorskip("torch")
    x = torch.zeros(SMOKE_INPUT)

    una, otra = build_net(), build_net()

    assert sum(isinstance(m, torch.nn.Conv2d) for m in una.modules()) == 3
    assert torch.equal(una(x), otra(x))


@pytest.mark.skipif(not SIN_COREMLTOOLS, reason="solo donde falta coremltools")
def test_sin_coremltools_lo_dice_y_sale_con_error(tmp_path, capsys):
    assert main(["--out", str(tmp_path / "smoke.mlpackage")]) == 1
    assert "coremltools" in capsys.readouterr().err


@pytest.mark.skipif(SIN_COREMLTOOLS, reason=MOTIVO_SIN_COREMLTOOLS)
def test_convierte_a_mlprogram_fp16_para_ios_18_y_lo_guarda(tmp_path):
    pytest.importorskip("torch")
    import coremltools as ct  # noqa: PLC0415

    paquete = convert(tmp_path / "smoke.mlpackage")

    spec = ct.utils.load_spec(str(paquete))
    assert paquete.is_dir()
    assert spec.WhichOneof("Type") == "mlProgram"
    assert spec.specificationVersion >= ct.target.iOS18.value
    assert list(spec.description.input[0].type.multiArrayType.shape) == list(SMOKE_INPUT)


@pytest.mark.skipif(sys.platform != "darwin", reason="predecir con Core ML necesita macOS")
def test_en_el_mac_el_fp16_queda_cerca_de_fp32(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("coremltools")

    diferencias = predict_diffs(convert(tmp_path / "smoke.mlpackage"))

    assert set(diferencias) == {"CPU_AND_NE", "CPU_ONLY"}
    assert max(diferencias.values()) < FP16_SMOKE_TOL
