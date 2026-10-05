"""N4: el spotter de píxeles en streaming, por cámara (ML-51, ADR 0004 §2).

MobileNetV4-Conv-S (timm) con un desplazamiento temporal (TSM) causal en la entrada de
algunas etapas, una GRU causal y una cabeza por fotograma: fondo, las clases de
`N4_CLASSES` y el desplazamiento al instante del evento (en fotogramas).

Todo con lo que exporta limpio al ANE:
- el TSM lleva su estado explícito y se hace con slice y concat: 1/`N4_TSM_FRACTION` de
  los canales de cada fotograma viene del fotograma anterior (nunca del siguiente);
- la GRU va desenrollada con conv 1x1, sigmoid y tanh: ni GRU nativa, ni Conv3d, ni
  scatter.

Dos modos con los mismos pesos: `forward` (clip de T fotogramas, para entrenar) y `step`
(un fotograma y su estado, para exportar). En evaluación coinciden (ML-51).
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from ftrain.constants import N4_BACKBONE, N4_CLASSES, N4_HIDDEN, N4_TSM_FRACTION, N4_TSM_STAGES

__all__ = ["N4Spotter", "SpotterState"]

SpotterState = tuple[list[torch.Tensor], torch.Tensor]
"""(lo que cada TSM guarda del fotograma anterior, el estado de la GRU)."""


class _ConvGru(nn.Module):
    """Una GRU con conv 1x1 sobre mapas de 1x1, desenrollada a mano."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(inputs + hidden, 2 * hidden, 1)
        self.cand_x = nn.Conv2d(inputs, hidden, 1)
        self.cand_h = nn.Conv2d(hidden, hidden, 1)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        z, r = zr[:, : self.hidden], zr[:, self.hidden :]
        n = torch.tanh(self.cand_x(x) + r * self.cand_h(h))
        return (1 - z) * n + z * h


class N4Spotter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        red = timm.create_model(N4_BACKBONE, pretrained=False)
        self.stem = nn.Sequential(red.conv_stem, red.bn1)
        self.stages = red.blocks
        canales = [self._out_channels(s) for s in self.stages]
        entradas = [red.conv_stem.out_channels, *canales[:-1]]
        self.shift = {k: entradas[k] // N4_TSM_FRACTION for k in N4_TSM_STAGES}
        self.gru = _ConvGru(canales[-1], N4_HIDDEN)
        # Fondo + clases, y el desplazamiento al instante del evento.
        self.head = nn.Conv2d(N4_HIDDEN, 1 + len(N4_CLASSES) + 1, 1)

    @staticmethod
    def _out_channels(stage: nn.Module) -> int:
        convs = [m for m in stage.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1].out_channels

    def initial_state(self, batch: int, height: int, width: int) -> SpotterState:
        """El estado antes del primer fotograma: ceros (el pasado no existe)."""
        dev = next(self.parameters()).device
        x = torch.zeros(batch, 3, height, width, device=dev)
        estado: list[torch.Tensor] = []
        with torch.no_grad():
            x = self.stem(x)
            for k, etapa in enumerate(self.stages):
                if k in self.shift:
                    estado.append(torch.zeros_like(x[:, : self.shift[k]]))
                x = etapa(x)
        return estado, torch.zeros(batch, N4_HIDDEN, 1, 1, device=dev)

    def step(self, frame: torch.Tensor, state: SpotterState) -> tuple[torch.Tensor, SpotterState]:
        """Un fotograma (lote, 3, alto, ancho) y su estado → (salida, estado nuevo)."""
        anteriores, h = state
        nuevos: list[torch.Tensor] = []
        x = self.stem(frame)
        i = 0
        for k, etapa in enumerate(self.stages):
            if k in self.shift:
                c = self.shift[k]
                nuevos.append(x[:, :c])
                x = torch.cat([anteriores[i], x[:, c:]], dim=1)
                i += 1
            x = etapa(x)
        x = x.mean(dim=(2, 3), keepdim=True)
        h = self.gru(x, h)
        return self.head(h).flatten(1), (nuevos, h)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """Un clip (lote, T, 3, alto, ancho) → (lote, T, 1 + clases + 1), fotograma a
        fotograma con el estado: el mismo cálculo que `step`, sin ver el futuro."""
        b, t = clip.shape[:2]
        estado = self.initial_state(b, clip.shape[3], clip.shape[4])
        salidas = []
        for k in range(t):
            y, estado = self.step(clip[:, k], estado)
            salidas.append(y)
        return torch.stack(salidas, dim=1)
