"""El plan B CNN del detector de jugadores (ADR 0020, decisión 5; para REF-33).

SPK-51 midió que D-FINE-N no entra al ANE (0 %: la atención deformable lo echa entero) y
la puerta del plan B se abrió: YOLOX-Tiny o CenterNet sobre MobileNetV4, ANE-limpios por
construcción. Aquí están los dos, con pesos sembrados, para medirlos en el mismo banco
del iPhone (SPK-50) antes de entrenar ninguno:

- `CenterNetMnv4`: MobileNetV4-Conv-S (timm, Apache-2.0), un cuello FPN ligero hasta paso
  4 y cabezas de heatmap por clase, tamaño y desplazamiento. Todo conv, relu, suma y
  vecino más cercano; canales múltiplos de 16.
- `YoloxTiny`: la arquitectura de YOLOX-Tiny (Apache-2.0; profundidad 0,33, anchura 0,375)
  reimplementada aquí: Focus, CSPDarknet, SPP, PAFPN y cabeza desacoplada, con SiLU. El
  NMS va fuera del grafo (ADR 0020). Sus anchuras (24, 48, 96…) NO son múltiplos de 16:
  es lo que trae YOLOX, y el banco dirá cuánto cuesta.

Los dos ven la franja de 1920x576 en RGB 0-1: el ImageType del export escala los 0-255
de Metal (ML-09), como en D-FINE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import timm
import torch
from torch import nn
from torch.nn import functional as f

from ftrain.constants import PLAYER_CLASSES

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["CenterNetMnv4", "YoloxTiny", "build_centernet", "build_yolox"]

CENTERNET_BACKBONE = "mobilenetv4_conv_small"
CENTERNET_NECK = 64
"""Canales del cuello y las cabezas de CenterNet (múltiplo de 16)."""
CENTERNET_STAGES = (0, 1, 2, 3)
"""Etapas de MNv4 que entran al cuello: pasos 4, 8, 16 y 32."""

YOLOX_DEPTH = 0.33
YOLOX_WIDTH = 0.375
YOLOX_BASE_CHANNELS = (64, 128, 256, 512, 1024)
YOLOX_BASE_DEPTHS = (3, 9, 9, 3)


# --------------------------------------------------------------------------- #
# CenterNet sobre MobileNetV4
# --------------------------------------------------------------------------- #


class CenterNetMnv4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        red = timm.create_model(CENTERNET_BACKBONE, pretrained=False)
        self.stem = nn.Sequential(red.conv_stem, red.bn1)
        self.stages = nn.ModuleList([red.blocks[k] for k in CENTERNET_STAGES])
        canales = [
            [m for m in s.modules() if isinstance(m, nn.Conv2d)][-1].out_channels
            for s in self.stages
        ]
        self.lateral = nn.ModuleList([nn.Conv2d(c, CENTERNET_NECK, 1) for c in canales])
        self.smooth = nn.Sequential(
            nn.Conv2d(CENTERNET_NECK, CENTERNET_NECK, 3, padding=1), nn.ReLU()
        )

        def cabeza(salidas: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(CENTERNET_NECK, CENTERNET_NECK, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(CENTERNET_NECK, salidas, 1),
            )

        self.heatmap = cabeza(len(PLAYER_CLASSES))
        self.size = cabeza(2)
        self.offset = cabeza(2)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(image)
        rasgos = []
        for etapa in self.stages:
            x = etapa(x)
            rasgos.append(x)
        y = self.lateral[-1](rasgos[-1])
        for k in range(len(rasgos) - 2, -1, -1):
            y = f.interpolate(y, scale_factor=2.0, mode="nearest") + self.lateral[k](rasgos[k])
        y = self.smooth(y)
        return torch.sigmoid(self.heatmap(y)), self.size(y), self.offset(y)


# --------------------------------------------------------------------------- #
# YOLOX-Tiny
# --------------------------------------------------------------------------- #


def _ch(c: int) -> int:
    return int(c * YOLOX_WIDTH)


class _Conv(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU()
        )


class _Bottleneck(nn.Module):
    def __init__(self, c: int, *, shortcut: bool) -> None:
        super().__init__()
        self.a, self.b, self.shortcut = _Conv(c, c, 1), _Conv(c, c, 3), shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b(self.a(x))
        return x + y if self.shortcut else y


class _Csp(nn.Module):
    def __init__(self, cin: int, cout: int, n: int, *, shortcut: bool = True) -> None:
        super().__init__()
        mitad = cout // 2
        self.a, self.b = _Conv(cin, mitad), _Conv(cin, mitad)
        self.m = nn.Sequential(*[_Bottleneck(mitad, shortcut=shortcut) for _ in range(n)])
        self.out = _Conv(2 * mitad, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([self.m(self.a(x)), self.b(x)], dim=1))


class _Spp(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        mitad = cin // 2
        self.a = _Conv(cin, mitad)
        self.pools = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in (5, 9, 13)])
        self.b = _Conv(mitad * 4, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a(x)
        return self.b(torch.cat([x, *[p(x) for p in self.pools]], dim=1))


class _Focus(nn.Module):
    """De espacio a canales con slices (paso 2), y una conv."""

    def __init__(self, cout: int) -> None:
        super().__init__()
        self.conv = _Conv(12, cout, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(
            torch.cat(
                [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1
            )
        )


class YoloxTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c = [_ch(x) for x in YOLOX_BASE_CHANNELS]
        n = [max(round(d * YOLOX_DEPTH), 1) for d in YOLOX_BASE_DEPTHS]
        self.stem = _Focus(c[0])
        self.dark2 = nn.Sequential(_Conv(c[0], c[1], 3, 2), _Csp(c[1], c[1], n[0]))
        self.dark3 = nn.Sequential(_Conv(c[1], c[2], 3, 2), _Csp(c[2], c[2], n[1]))
        self.dark4 = nn.Sequential(_Conv(c[2], c[3], 3, 2), _Csp(c[3], c[3], n[2]))
        self.dark5 = nn.Sequential(
            _Conv(c[3], c[4], 3, 2), _Spp(c[4], c[4]), _Csp(c[4], c[4], n[3], shortcut=False)
        )
        # PAFPN
        self.lat0 = _Conv(c[4], c[3])
        self.c3_p4 = _Csp(2 * c[3], c[3], n[0], shortcut=False)
        self.red1 = _Conv(c[3], c[2])
        self.c3_p3 = _Csp(2 * c[2], c[2], n[0], shortcut=False)
        self.bu2 = _Conv(c[2], c[2], 3, 2)
        self.c3_n3 = _Csp(2 * c[2], c[3], n[0], shortcut=False)
        self.bu1 = _Conv(c[3], c[3], 3, 2)
        self.c3_n4 = _Csp(2 * c[3], c[4], n[0], shortcut=False)
        # Cabeza desacoplada por nivel: clases, caja (4) y objeto (1).
        ancho = _ch(256)
        self.heads = nn.ModuleList()
        for cin in (c[2], c[3], c[4]):
            self.heads.append(
                nn.ModuleDict(
                    {
                        "stem": _Conv(cin, ancho),
                        "cls": nn.Sequential(
                            _Conv(ancho, ancho, 3),
                            _Conv(ancho, ancho, 3),
                            nn.Conv2d(ancho, len(PLAYER_CLASSES), 1),
                        ),
                        "reg": nn.Sequential(_Conv(ancho, ancho, 3), _Conv(ancho, ancho, 3)),
                        "box": nn.Conv2d(ancho, 4, 1),
                        "obj": nn.Conv2d(ancho, 1, 1),
                    }
                )
            )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x2 = self.dark2(self.stem(image))
        x3 = self.dark3(x2)
        x4 = self.dark4(x3)
        x5 = self.dark5(x4)
        f0 = self.lat0(x5)
        p4 = self.c3_p4(torch.cat([f.interpolate(f0, scale_factor=2.0, mode="nearest"), x4], 1))
        f1 = self.red1(p4)
        p3 = self.c3_p3(torch.cat([f.interpolate(f1, scale_factor=2.0, mode="nearest"), x3], 1))
        n3 = self.c3_n3(torch.cat([self.bu2(p3), f1], 1))
        n4 = self.c3_n4(torch.cat([self.bu1(n3), f0], 1))
        clases, cajas, objetos = [], [], []
        for nivel, h in zip((p3, n3, n4), self.heads, strict=True):
            s = h["stem"](nivel)
            r = h["reg"](s)
            clases.append(torch.sigmoid(h["cls"](s)).flatten(2))
            cajas.append(h["box"](r).flatten(2))
            objetos.append(torch.sigmoid(h["obj"](r)).flatten(2))
        # Por ancla (las celdas de los tres niveles, en fila): sin NMS (ADR 0020).
        return torch.cat(clases, 2), torch.cat(cajas, 2), torch.cat(objetos, 2)


def build_centernet(checkpoint: Path | str | None) -> Any:
    """El builder de los CLI: pesos sembrados si no hay checkpoint."""
    torch.manual_seed(0)
    m = CenterNetMnv4()
    if checkpoint is not None:
        m.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return m.eval()


def build_yolox(checkpoint: Path | str | None) -> Any:
    torch.manual_seed(0)
    m = YoloxTiny()
    if checkpoint is not None:
        m.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return m.eval()
