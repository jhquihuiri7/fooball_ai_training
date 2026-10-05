"""Los dos candidatos del plan B de jugadores (ADR 0020): formas y canales."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ftrain.constants import ANE_CHANNEL_QUANTUM, PLAYER_CLASSES  # noqa: E402
from ftrain.players.plan_b import build_centernet, build_yolox  # noqa: E402

ALTO, ANCHO = 192, 640  # la franja a escala 1/3, para que el test sea rápido


def test_centernet_heatmap_a_paso_4_y_canales_para_el_ane():
    m = build_centernet(None)
    with torch.no_grad():
        heat, size, offset = m(torch.rand(1, 3, ALTO, ANCHO))
    assert tuple(heat.shape) == (1, len(PLAYER_CLASSES), ALTO // 4, ANCHO // 4)
    assert tuple(size.shape) == tuple(offset.shape) == (1, 2, ALTO // 4, ANCHO // 4)
    assert float(heat.min()) >= 0.0
    assert float(heat.max()) <= 1.0
    for n, c in m.named_modules():
        if isinstance(c, torch.nn.Conv2d) and not n.endswith(".2"):
            assert c.out_channels % ANE_CHANNEL_QUANTUM == 0, n


def test_yolox_una_fila_por_ancla_de_los_tres_niveles():
    m = build_yolox(None)
    with torch.no_grad():
        scores, boxes, obj = m(torch.rand(1, 3, ALTO, ANCHO))
    anclas = sum((ALTO // s) * (ANCHO // s) for s in (8, 16, 32))
    assert tuple(scores.shape) == (1, len(PLAYER_CLASSES), anclas)
    assert tuple(boxes.shape) == (1, 4, anclas)
    assert tuple(obj.shape) == (1, 1, anclas)


def test_los_builders_son_deterministas():
    a, b = build_centernet(None), build_centernet(None)
    assert all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )
