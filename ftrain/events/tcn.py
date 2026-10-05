"""La TCN causal de N3 (ML-50, ADR 0004 §1), toda en conv2d para que el export sea limpio.

- Por paso, un codificador espacial de la rejilla (canales de `N3_CHANNELS`, 10x16) a un
  vector de `N3_WIDTH`: dos conv 3x3 (la segunda con paso 2) y una conv que cubre el resto
  de la rejilla.
- En el tiempo, la serie va como (lote, canales, 1, T) y cada bloque es una conv (1, k)
  dilatada con relleno solo a la izquierda: la salida en t no ve nada posterior a t.
- Una cabeza 1x1 a 1 + len(N3_CLASSES) logits por paso (la 0 es «nada»).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f

from ftrain.constants import (
    N3_CHANNELS,
    N3_CLASSES,
    N3_DILATIONS,
    N3_GRID_H,
    N3_GRID_W,
    N3_TEMPORAL_KERNEL,
    N3_WIDTH,
)

__all__ = ["N3Tcn"]

SPATIAL_HIDDEN = 16
"""Canales del codificador espacial: múltiplo de 16 (ANE_CHANNEL_QUANTUM)."""


class _Block(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.pad = (N3_TEMPORAL_KERNEL - 1) * dilation
        self.conv = nn.Conv2d(width, width, (1, N3_TEMPORAL_KERNEL), dilation=(1, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return f.relu(x + self.conv(f.pad(x, (self.pad, 0, 0, 0))))


class N3Tcn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c = len(N3_CHANNELS)
        self.spatial = nn.Sequential(
            nn.Conv2d(c, SPATIAL_HIDDEN, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(SPATIAL_HIDDEN, SPATIAL_HIDDEN, 3, stride=2, padding=1),
            nn.ReLU(),
            # Lo que queda de la rejilla (5x8) en una sola conv: un vector por paso.
            nn.Conv2d(SPATIAL_HIDDEN, N3_WIDTH, ((N3_GRID_H + 1) // 2, (N3_GRID_W + 1) // 2)),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[_Block(N3_WIDTH, d) for d in N3_DILATIONS])
        self.head = nn.Conv2d(N3_WIDTH, 1 + len(N3_CLASSES), 1)

    @property
    def receptive_field(self) -> int:
        return 1 + (N3_TEMPORAL_KERNEL - 1) * sum(N3_DILATIONS)

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """(lote, T, canales, alto, ancho) → logits (lote, T, 1 + clases)."""
        b, t = grids.shape[:2]
        z = self.spatial(grids.reshape(b * t, *grids.shape[2:]))  # (b·t, ancho, 1, 1)
        z = z.reshape(b, t, -1).permute(0, 2, 1).unsqueeze(2)  # (b, ancho, 1, t)
        return self.head(self.blocks(z)).squeeze(2).permute(0, 2, 1)
