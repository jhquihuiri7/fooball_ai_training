"""ROI-lite: el detector propio del balón, un heatmap sobre 3 frames en gris (ML-14).

Diseño propio, con las ideas de FootAndBall y WASB (MIT) y sin copiar su código ni sus
pesos (ADR 0020 §2):

- **Entrada**: 3 canales, que son la luma Y de t-2, t-1 y t, en 0-255. Lo que distingue un
  balón de 3-5 px de una cabeza es que se mueve.
- **Tallo**: una conv 3x3 de paso 2. Todo el modelo vive a partir de stride 2, que es donde
  sale el heatmap.
- **Cuerpo**: 4 etapas a strides 2, 4, 8 y 16, con convs densas 3x3 y 1x1, BatchNorm y
  ReLU. Nada de Conv3d ni de GRU: el tiempo va en los canales de entrada, y así todo cae en
  el ANE.
- **Decoder**: de arriba abajo hasta stride 2, sumando en cada nivel lo que vio el cuerpo.
- **Cabezas**: un heatmap de 1 canal y un offset de 2 (x e y, en fracciones de celda), las
  dos a stride 2.

El heatmap sale en **logits**. La sigmoide la pone el export (ML-42), porque la pérdida
focal de ML-39 la quiere fuera y `heatmap_peaks` del repo de detección espera 0-1. Los picos
y el offset se decodifican fuera del grafo: `x = (columna + off_x) · 2`.

Las formas de entrada son fijas en el export: una ROI de 256 y el mosaico de la franja, con
lados múltiplos de 16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ftrain.constants import ANE_CHANNEL_QUANTUM, BALL_TEMPORAL_FRAMES, PIXEL_SCALE


@dataclass(frozen=True, slots=True)
class RoiLiteConfig:
    """La forma del modelo. Los valores por defecto son los de la tarjeta ML-14."""

    stage_channels: tuple[int, int, int, int] = (16, 32, 64, 128)
    """Canales de las etapas a strides 2, 4, 8 y 16, antes de aplicar `width`."""

    stage_blocks: tuple[int, int, int, int] = (1, 2, 2, 2)
    """Bloques 3x3 + 1x1 por etapa. La primera, a stride 2, es la más cara por píxel: lleva
    uno solo."""

    width: float = 1.0
    """Multiplicador de ancho. Cada etapa se redondea a múltiplos de `ANE_CHANNEL_QUANTUM`,
    con ese mínimo."""

    def channels(self) -> tuple[int, ...]:
        cuanto = ANE_CHANNEL_QUANTUM
        return tuple(
            max(cuanto, round(c * self.width / cuanto) * cuanto) for c in self.stage_channels
        )


def _conv(entrada: int, salida: int, kernel: int, stride: int = 1) -> nn.Sequential:
    """Conv densa, sin sesgo (lo pone BatchNorm), más BatchNorm y ReLU."""
    return nn.Sequential(
        nn.Conv2d(entrada, salida, kernel, stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
    )


class _Block(nn.Module):
    """3x3 y 1x1 con un atajo residual: el mismo ancho a la entrada y a la salida."""

    def __init__(self, canales: int) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv(canales, canales, 3), _conv(canales, canales, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def _stage(entrada: int, salida: int, bloques: int, *, reduce: bool) -> nn.Sequential:
    capas: list[nn.Module] = []
    if reduce:
        capas.append(_conv(entrada, salida, 3, stride=2))
    capas.extend(_Block(salida) for _ in range(bloques))
    return nn.Sequential(*capas)


class _Up(nn.Module):
    """Un nivel del decoder: reduce canales, sube x2, suma el atajo y lo mezcla con 3x3."""

    def __init__(self, arriba: int, abajo: int) -> None:
        super().__init__()
        self.reduce = _conv(arriba, abajo, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.mix = _conv(abajo, abajo, 3)

    def forward(self, x: torch.Tensor, atajo: torch.Tensor) -> torch.Tensor:
        return self.mix(self.up(self.reduce(x)) + atajo)


class RoiLite(nn.Module):
    """El modelo del balón.

    `forward(x[N,3,H,W])` devuelve `(heatmap[N,1,H/2,W/2], offset[N,2,H/2,W/2])`.
    """

    def __init__(self, config: RoiLiteConfig | None = None) -> None:
        super().__init__()
        self.config = config or RoiLiteConfig()
        c = self.config.channels()
        b = self.config.stage_blocks
        self.stem = _conv(BALL_TEMPORAL_FRAMES, c[0], 3, stride=2)
        self.stages = nn.ModuleList(
            [
                _stage(c[0], c[0], b[0], reduce=False),
                _stage(c[0], c[1], b[1], reduce=True),
                _stage(c[1], c[2], b[2], reduce=True),
                _stage(c[2], c[3], b[3], reduce=True),
            ]
        )
        self.ups = nn.ModuleList([_Up(c[3], c[2]), _Up(c[2], c[1]), _Up(c[1], c[0])])
        self.head = _conv(c[0], c[0], 3)
        self.heatmap = nn.Conv2d(c[0], 1, 1)
        self.offset = nn.Conv2d(c[0], 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x * PIXEL_SCALE)
        niveles = []
        for etapa in self.stages:
            x = etapa(x)
            niveles.append(x)
        for up, atajo in zip(self.ups, reversed(niveles[:-1]), strict=True):
            x = up(x, atajo)
        x = self.head(x)
        return self.heatmap(x), self.offset(x)
