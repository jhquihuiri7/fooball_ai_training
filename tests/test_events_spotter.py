"""N4 en streaming (ML-51): modo clip y modo paso coinciden, sin tipos prohibidos."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ftrain.constants import ANE_CHANNEL_QUANTUM, N4_CLASSES, N4_HIDDEN  # noqa: E402
from ftrain.events.spotter import N4Spotter  # noqa: E402

ALTO, ANCHO = 128, 224  # más pequeño que 256x448 para que el test sea rápido


def test_clip_y_pasos_coinciden_en_fp32():
    torch.manual_seed(0)
    m = N4Spotter().eval()
    clip = torch.randn(2, 5, 3, ALTO, ANCHO)
    with torch.no_grad():
        y_clip = m(clip)
        estado = m.initial_state(2, ALTO, ANCHO)
        pasos = []
        for k in range(5):
            y, estado = m.step(clip[:, k], estado)
            pasos.append(y)
    assert tuple(y_clip.shape) == (2, 5, 1 + len(N4_CLASSES) + 1)
    assert (y_clip - torch.stack(pasos, dim=1)).abs().max().item() < 1e-5


def test_es_causal():
    torch.manual_seed(1)
    m = N4Spotter().eval()
    clip = torch.randn(1, 6, 3, ALTO, ANCHO)
    otro = clip.clone()
    otro[:, 4:] = torch.randn_like(otro[:, 4:])
    with torch.no_grad():
        a, b = m(clip), m(otro)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_sin_tipos_prohibidos_y_canales_multiplo_de_16():
    m = N4Spotter()
    prohibidos = (torch.nn.Conv3d, torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)
    assert not [n for n, x in m.named_modules() if isinstance(x, prohibidos)]
    assert N4_HIDDEN % ANE_CHANNEL_QUANTUM == 0
    for n, x in m.named_modules():
        if isinstance(x, torch.nn.Conv2d) and not n.startswith("head"):
            assert x.out_channels % ANE_CHANNEL_QUANTUM == 0, n
