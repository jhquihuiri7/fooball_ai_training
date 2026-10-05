"""N3: la TCN causal (ML-50) y la decodificación con su P/R (ADR 0004 §3 y §4)."""

from __future__ import annotations

import numpy as np
import pytest

from ftrain.constants import EVENTS_HZ, N3_CHANNELS, N3_CLASSES, N3_GRID_H, N3_GRID_W, N3_MAX_PARAMS
from ftrain.events.spotting import decode, match_events


def test_decode_un_pico_por_clase_y_supresion():
    t = 100
    p = np.zeros((t, 1 + len(N3_CLASSES)))
    p[:, 0] = 1.0
    k = 1 + N3_CLASSES.index("throw_in")
    p[40, k], p[42, k], p[80, k] = 0.9, 0.8, 0.7  # el 42 está dentro de los ±2 s del 40
    p[60, 1 + N3_CLASSES.index("corner")] = 0.4  # bajo su umbral

    marcas = decode(p, [0.5] * len(N3_CLASSES))

    assert [(round(s * EVENTS_HZ), c) for s, c, _ in marcas] == [(40, "throw_in"), (80, "throw_in")]


def test_match_events_uno_a_uno_dentro_de_la_tolerancia():
    verdad = [(10.0, "corner"), (30.0, "corner"), (50.0, "goal")]
    pred = [
        (11.5, "corner", 0.9),
        (10.5, "corner", 0.8),
        (33.0, "corner", 0.7),
        (50.2, "goal", 0.6),
    ]

    s = match_events(verdad, pred)

    assert (s["corner"].hits, s["corner"].predictions, s["corner"].truths) == (1, 3, 2)
    assert s["corner"].recall == 0.5
    assert s["goal"].f1 == 1.0
    assert s["throw_in"].f1 == 0.0


def test_la_tcn_es_causal_y_cabe_en_100k():
    torch = pytest.importorskip("torch")
    from ftrain.events.tcn import N3Tcn  # noqa: PLC0415 - solo con el grupo train

    m = N3Tcn().eval()
    assert sum(p.numel() for p in m.parameters()) <= N3_MAX_PARAMS
    torch.manual_seed(0)
    x = torch.randn(1, 160, len(N3_CHANNELS), N3_GRID_H, N3_GRID_W)
    y = m(x)
    assert tuple(y.shape) == (1, 160, 1 + len(N3_CLASSES))
    x2 = x.clone()
    x2[:, 100:] = torch.randn_like(x2[:, 100:])
    y2 = m(x2)
    assert torch.allclose(y[:, :100], y2[:, :100], atol=1e-5), "la salida en t no ve después de t"
    assert not torch.allclose(y[:, 100:], y2[:, 100:])
