"""Contador de MAC por hooks, sin dependencias aparte de torch (ML-14).

Cuenta las multiplicaciones-acumulaciones de las capas que cuestan: `Conv2d`,
`ConvTranspose2d` y `Linear`. El resto (BatchNorm, que el export pliega en la conv; ReLU;
sumas; upsample) es ruido al lado y no entra. Es la cifra con la que se comparan los
presupuestos del ADR 0020, por ejemplo ≤1,0 GMAC para una ROI del balón.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


def _macs(modulo: nn.Module, entrada: torch.Tensor, salida: torch.Tensor) -> int:
    if isinstance(modulo, nn.Conv2d):
        por_salida = (modulo.in_channels // modulo.groups) * math.prod(modulo.kernel_size)
        return salida.numel() * por_salida
    if isinstance(modulo, nn.ConvTranspose2d):
        por_entrada = (modulo.out_channels // modulo.groups) * math.prod(modulo.kernel_size)
        return entrada.numel() * por_entrada
    if isinstance(modulo, nn.Linear):
        return salida.numel() * modulo.in_features
    return 0


def count_macs(model: nn.Module, *inputs: torch.Tensor) -> int:
    """MAC de una pasada hacia delante de `model` con `inputs`, para todo el lote.

    Pasa en modo evaluación y sin gradiente, y deja el modelo como estaba.
    """
    total = 0

    def hook(modulo: nn.Module, args: tuple[torch.Tensor, ...], salida: torch.Tensor) -> None:
        nonlocal total
        total += _macs(modulo, args[0], salida)

    capas = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    handles: list[RemovableHandle] = [
        m.register_forward_hook(hook) for m in model.modules() if isinstance(m, capas)
    ]
    entrenando = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(*inputs)
    finally:
        model.train(entrenando)
        for handle in handles:
            handle.remove()
    return total
