"""Entrenar CenterNet-MNv4: pérdida, decodificación, EMA y el dataset de COCO person.

El modelo es `plan_b.CenterNetMnv4` tal cual, con sus tres clases, para que el checkpoint
lo cargue `plan_b.build_centernet` y lo exporte la spec que ya existe
(`configs/export/players-centernet-mnv4.yaml`). El contrato de las salidas está en
`centernet_data`.

El preentreno con COCO solo sabe de personas: van a la clase `player`, y `goalkeeper` y
`referee` se entrenan como negativos puros, para que en el iPhone no den cajas al azar.
Separarlas es del afinado con datos propios (ML-32 en adelante, sobre CenterNet si REF-33 lo
elige).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import Dataset

from ftrain.constants import (
    CENTERNET_FOCAL_ALPHA,
    CENTERNET_FOCAL_BETA,
    CENTERNET_OFFSET_WEIGHT,
    CENTERNET_PEAK_KERNEL,
    CENTERNET_PRIOR,
    CENTERNET_PROB_EPS,
    CENTERNET_SIZE_WEIGHT,
    CENTERNET_STRIDE,
    CENTERNET_TOPK,
    PLAYER_CLASSES,
)
from ftrain.players.centernet_data import (
    MosaicConfig,
    draw_targets,
    hsv_jitter,
    row_mosaic,
)
from ftrain.players.plan_b import CENTERNET_BACKBONE, CENTERNET_STAGES, CenterNetMnv4

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "PERSON_CLASS",
    "CocoPersonMosaic",
    "Ema",
    "centernet_loss",
    "decode",
    "focal_loss",
    "init_for_training",
    "load_imagenet_backbone",
]

PERSON_CLASS = PLAYER_CLASSES.index("player")
"""La clase a la que van las personas de COCO."""

HSV_GAINS = (0.015, 0.7, 0.4)
"""Ganancias máximas de tono, saturación y brillo del aumento de color (las de YOLOv5):
césped bajo focos, al sol y nublado."""


def focal_loss(pred: torch.Tensor, gt: torch.Tensor, ignore: torch.Tensor) -> torch.Tensor:
    """La focal de CenterNet con los negativos rebajados cerca de un centro.

    Se normaliza por el número de centros; las celdas con `ignore` no penalizan.
    """
    p = pred.float().clamp(CENTERNET_PROB_EPS, 1 - CENTERNET_PROB_EPS)
    pos = gt.eq(1).float()
    neg = (1 - pos) * (1 - ignore)
    pos_loss = torch.log(p) * (1 - p) ** CENTERNET_FOCAL_ALPHA * pos
    neg_loss = torch.log(1 - p) * p**CENTERNET_FOCAL_ALPHA * (1 - gt) ** CENTERNET_FOCAL_BETA * neg
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def centernet_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """La suma ponderada de focal, L1 del tamaño y L1 del desplazamiento, y sus partes.

    Las partes salen como tensores sin gradiente: pasarlas a float obliga a esperar a la GPU,
    y eso se hace solo al escribir el log.
    """
    heat, size, offset = outputs
    mask = targets["mask"]
    n = (mask.sum() * 2).clamp(min=1)
    l_heat = focal_loss(heat, targets["heatmap"], targets["ignore"])
    l_size = (f.l1_loss(size.float(), targets["size"], reduction="none") * mask).sum() / n
    l_off = (f.l1_loss(offset.float(), targets["offset"], reduction="none") * mask).sum() / n
    total = l_heat + CENTERNET_SIZE_WEIGHT * l_size + CENTERNET_OFFSET_WEIGHT * l_off
    return total, {
        "heatmap": l_heat.detach(),
        "size": l_size.detach(),
        "offset": l_off.detach(),
    }


def decode(
    heat: torch.Tensor, size: torch.Tensor, offset: torch.Tensor, k: int = CENTERNET_TOPK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Los k mejores picos: cajas xyxy en píxeles de entrada, puntuación y clase."""
    b, c, h, w = heat.shape
    pico = f.max_pool2d(heat, CENTERNET_PEAK_KERNEL, 1, CENTERNET_PEAK_KERNEL // 2)
    picos = heat * (pico == heat)
    scores, idx = picos.reshape(b, -1).topk(min(k, c * h * w))
    clases = idx // (h * w)
    celda = idx % (h * w)
    ys, xs = (celda // w).float(), (celda % w).float()
    tam = size.reshape(b, 2, -1).gather(2, celda[:, None].expand(-1, 2, -1))
    off = offset.reshape(b, 2, -1).gather(2, celda[:, None].expand(-1, 2, -1))
    cx = (xs + off[:, 0]) * CENTERNET_STRIDE
    cy = (ys + off[:, 1]) * CENTERNET_STRIDE
    bw, bh = tam[:, 0] * CENTERNET_STRIDE, tam[:, 1] * CENTERNET_STRIDE
    cajas = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
    return cajas, scores, clases


def init_for_training(model: CenterNetMnv4) -> None:
    """El sesgo del heatmap a la probabilidad inicial `CENTERNET_PRIOR`."""
    ultima = model.heatmap[-1]
    assert isinstance(ultima, nn.Conv2d)  # noqa: S101 - la forma de plan_b
    assert ultima.bias is not None  # noqa: S101
    nn.init.constant_(ultima.bias, math.log(CENTERNET_PRIOR / (1 - CENTERNET_PRIOR)))


def load_imagenet_backbone(model: CenterNetMnv4, weights: Path) -> None:
    """Copia los pesos de MobileNetV4-Conv-S (safetensors de timm) al tronco del modelo.

    Solo el tronco que usa CenterNet (stem y etapas de `CENTERNET_STAGES`): la cabeza de
    clasificación de ImageNet se queda fuera. Estricto: una clave que no cuadre falla.
    """
    import timm  # noqa: PLC0415 - grupo train
    from safetensors.torch import load_file  # noqa: PLC0415 - llega con timm

    red = timm.create_model(CENTERNET_BACKBONE, pretrained=False)
    red.load_state_dict(load_file(str(weights)))
    model.stem[0].load_state_dict(red.conv_stem.state_dict())
    model.stem[1].load_state_dict(red.bn1.state_dict())
    for etapa, k in zip(model.stages, CENTERNET_STAGES, strict=True):
        etapa.load_state_dict(red.blocks[k].state_dict())


class Ema:
    """Media móvil exponencial de los pesos, con rampa al principio (la de YOLOv5)."""

    def __init__(self, model: nn.Module, decay: float, ramp_steps: float) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(requires_grad=False)
        self.decay, self.ramp, self.updates = decay, ramp_steps, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.ramp))
        ema, cur = self.module.state_dict(), model.state_dict()
        flotantes = [k for k, v in ema.items() if v.dtype.is_floating_point]
        torch._foreach_mul_([ema[k] for k in flotantes], d)
        torch._foreach_add_(
            [ema[k] for k in flotantes], [cur[k].detach() for k in flotantes], alpha=1 - d
        )
        for k, v in ema.items():
            if not v.dtype.is_floating_point:
                v.copy_(cur[k])


class CocoPersonMosaic(Dataset):
    """Mosaicos en fila de imágenes de COCO con personas, con sus mapas de CenterNet.

    `entries` son las del índice de `tools/fetch_coco_person.py`. El primer elemento de
    cada mosaico es el del índice pedido; el resto se sortea entre todas.
    """

    def __init__(self, entries: list[dict[str, Any]], images: Path, config: MosaicConfig) -> None:
        self.entries, self.images, self.config = entries, images, config

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2  # noqa: PLC0415 - grupo train

        e = self.entries[i]
        bgr = cv2.imread(str(self.images / e["file"]), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f"no se puede leer {self.images / e['file']}"
            raise OSError(msg)
        return (
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            np.asarray(e["boxes"], np.float64).reshape(-1, 4),
            np.asarray(e["crowd"], np.float64).reshape(-1, 4),
        )

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rng = np.random.default_rng()
        siguiente = iter([i])

        def draw() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return self._load(next(siguiente, int(rng.integers(len(self.entries)))))

        lienzo, cajas, ignoradas = row_mosaic(draw, self.config, rng)
        lienzo = hsv_jitter(lienzo, rng, HSV_GAINS)
        t = draw_targets(
            cajas,
            np.full(len(cajas), PERSON_CLASS),
            input_hw=(self.config.height, self.config.width),
            n_classes=len(PLAYER_CLASSES),
            ignore_boxes=ignoradas,
            ignore_classes=np.full(len(ignoradas), PERSON_CLASS),
        )
        imagen = torch.from_numpy(np.ascontiguousarray(lienzo.transpose(2, 0, 1)))
        return imagen, {
            "heatmap": torch.from_numpy(t.heatmap),
            "size": torch.from_numpy(t.size),
            "offset": torch.from_numpy(t.offset),
            "mask": torch.from_numpy(t.mask),
            "ignore": torch.from_numpy(t.ignore),
        }
