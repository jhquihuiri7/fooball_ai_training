"""Tests del modelo ROI-lite y del contador de MAC (ML-14).

Necesitan torch (`uv sync --group train`); sin él se saltan. Lo que se comprueba es lo que
decide si el modelo cabe en el ANE: las formas, el presupuesto de MAC, que no haya capas que
el ANE no ejecuta y que los canales vayan de 16 en 16.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from ftrain.ball.model import RoiLite, RoiLiteConfig  # noqa: E402
from ftrain.constants import ANE_CHANNEL_QUANTUM, BALL_HEATMAP_STRIDE  # noqa: E402
from ftrain.flops import count_macs  # noqa: E402

ROI: tuple[int, int, int, int] = (2, 3, 256, 256)
"""El lote de las dos ROIs nativas (ADR 0020 §2)."""

MOSAICO: tuple[int, int, int, int] = (1, 3, 896, 1920)
"""El mosaico previsto para el retranqueo de 10 m. La altura la fija SPK-52."""

MAX_MACS_ROI = 1.0e9
"""Presupuesto a [1,3,256,256], en MAC (ADR 0020 §2)."""

MAX_MACS_MOSAICO = 17.0e9
"""Presupuesto al mosaico [1,3,896,1920], en MAC (ML-14)."""

PROHIBIDAS = (nn.Conv3d, nn.GRU, nn.LSTM, nn.RNN)
"""Capas que no van: el tiempo entra por los canales, no por una dimensión aparte."""


@pytest.fixture(scope="module")
def modelo() -> RoiLite:
    return RoiLite().eval()


# --------------------------------------------------------------------------- #
# El modelo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("forma", [ROI, MOSAICO])
def test_heatmap_y_offset_salen_a_stride_2(modelo: RoiLite, forma):
    n, _, h, w = forma
    celdas = (h // BALL_HEATMAP_STRIDE, w // BALL_HEATMAP_STRIDE)

    with torch.no_grad():
        heatmap, offset = modelo(torch.zeros(forma))

    assert tuple(heatmap.shape) == (n, 1, *celdas)
    assert tuple(offset.shape) == (n, 2, *celdas)


def test_cabe_en_el_presupuesto_de_una_roi(modelo: RoiLite):
    assert count_macs(modelo, torch.zeros(1, *ROI[1:])) <= MAX_MACS_ROI


def test_cabe_en_el_presupuesto_del_mosaico(modelo: RoiLite):
    assert count_macs(modelo, torch.zeros(MOSAICO)) <= MAX_MACS_MOSAICO


def test_no_lleva_capas_que_el_ane_no_ejecuta(modelo: RoiLite):
    assert not [m for m in modelo.modules() if isinstance(m, PROHIBIDAS)]


def test_las_convs_son_densas(modelo: RoiLite):
    assert all(m.groups == 1 for m in modelo.modules() if isinstance(m, nn.Conv2d))


def test_los_canales_van_de_16_en_16_salvo_la_entrada_y_las_cabezas(modelo: RoiLite):
    cabezas = {id(modelo.heatmap), id(modelo.offset)}
    for nombre, capa in modelo.named_modules():
        if not isinstance(capa, nn.Conv2d):
            continue
        if capa is not modelo.stem[0]:
            assert capa.in_channels % ANE_CHANNEL_QUANTUM == 0, nombre
        if id(capa) not in cabezas:
            assert capa.out_channels % ANE_CHANNEL_QUANTUM == 0, nombre


@pytest.mark.parametrize("width", [0.25, 0.5, 1.5, 2.0])
def test_el_ancho_se_redondea_a_multiplos_de_16(width: float):
    canales = RoiLiteConfig(width=width).channels()

    assert all(c % ANE_CHANNEL_QUANTUM == 0 and c >= ANE_CHANNEL_QUANTUM for c in canales)


def test_mas_ancho_cuesta_mas():
    x = torch.zeros(1, *ROI[1:])

    estrecho = count_macs(RoiLite(RoiLiteConfig(width=0.5)), x)
    ancho = count_macs(RoiLite(RoiLiteConfig(width=2.0)), x)

    assert estrecho < count_macs(RoiLite(), x) < ancho


# --------------------------------------------------------------------------- #
# El contador
# --------------------------------------------------------------------------- #


def test_una_conv_cuenta_salidas_por_entradas_por_kernel():
    conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)

    assert count_macs(conv, torch.zeros(1, 3, 256, 256)) == 16 * 128 * 128 * 3 * 9


def test_una_conv_agrupada_cuenta_solo_su_grupo():
    conv = nn.Conv2d(16, 16, 3, padding=1, groups=16)

    assert count_macs(conv, torch.zeros(1, 16, 8, 8)) == 16 * 8 * 8 * 1 * 9


def test_una_lineal_y_una_transpuesta():
    assert count_macs(nn.Linear(10, 5), torch.zeros(4, 10)) == 4 * 5 * 10
    transpuesta = nn.ConvTranspose2d(8, 4, 2, stride=2)
    assert count_macs(transpuesta, torch.zeros(1, 8, 16, 16)) == 8 * 16 * 16 * 4 * 4


def test_deja_el_modelo_como_estaba():
    modelo = RoiLite().train()

    count_macs(modelo, torch.zeros(1, *ROI[1:]))

    assert modelo.training
    assert not any(m._forward_hooks for m in modelo.modules())  # noqa: SLF001
