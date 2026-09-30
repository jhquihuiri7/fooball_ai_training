"""Prueba de humo de Core ML: convertir en cualquier sitio, predecir en el Mac (ML-05).

Convierte una red de tres convs a mlprogram en fp16, con target iOS 18 y formas fijas, y
guarda el `.mlpackage`. Es lo mismo que hará `export_coreml` (ML-09) con los modelos de
verdad, reducido a lo mínimo para saber si la máquina sirve.

- **En Linux** convierte y guarda: `coremltools` convierte en Linux, pero no predice.
- **En macOS**, además, predice con `CPU_AND_NE` y con `CPU_ONLY` y compara cada salida con
  la de torch en fp32. Si alguna se aleja más de `FP16_SMOKE_TOL`, sale con error.
- **En Windows** no hay ruedas de `coremltools`: la herramienta lo dice y sale.

    uv sync --group train --group apple
    uv run python tools/mac_smoke.py

Qué más se hace en el Mac y cómo prepararlo: `docs/MAC.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

SMOKE_INPUT: Final = (1, 3, 64, 64)
"""Forma fija de la entrada de la red de prueba, NCHW. Pequeña: esto mide la máquina."""

SMOKE_SEED: Final = 0
"""Semilla de los pesos y de la entrada, para que dos pasadas den lo mismo."""

FP16_SMOKE_TOL: Final = 1e-2
"""Diferencia absoluta máxima entre Core ML en fp16 y torch en fp32 (ML-05). Con entradas y
pesos del orden de 1, el fp16 pierde unos 1e-3; más de 1e-2 es que algo va mal."""

COMPUTE_UNITS: Final = ("CPU_AND_NE", "CPU_ONLY")
"""Con qué se predice en el Mac: el ANE, que es donde irá el modelo en el iPhone, y la CPU
como referencia."""

DEFAULT_OUT: Final = Path("runs/mac_smoke/smoke.mlpackage")


class SmokeError(RuntimeError):
    """La máquina no sirve para Core ML tal como está. El mensaje dice qué falta."""


def build_net() -> Any:
    """La red de prueba: tres convs con ReLU, pesos fijados por `SMOKE_SEED`."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    torch.manual_seed(SMOKE_SEED)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 4, 1),
    ).eval()


def _coremltools() -> Any:
    try:
        import coremltools as ct  # noqa: PLC0415
    except ImportError as exc:
        if sys.platform == "win32":
            msg = "coremltools no tiene ruedas para Windows: esto se corre en Linux o en el Mac"
        else:
            msg = "falta coremltools: uv sync --group train --group apple"
        raise SmokeError(msg) from exc
    return ct


def convert(out: Path = DEFAULT_OUT) -> Path:
    """Convierte la red de prueba a mlprogram fp16 para iOS 18 y la guarda en `out`."""
    ct = _coremltools()
    import torch  # noqa: PLC0415

    red = build_net()
    ejemplo = torch.rand(SMOKE_INPUT, generator=torch.Generator().manual_seed(SMOKE_SEED))
    trazada = torch.jit.trace(red, ejemplo)
    modelo = ct.convert(
        trazada,
        inputs=[ct.TensorType(name="x", shape=SMOKE_INPUT)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(str(out))
    return out


def predict_diffs(package: Path) -> dict[str, float]:
    """En macOS: diferencia máxima de cada `COMPUTE_UNITS` frente a torch en fp32."""
    if sys.platform != "darwin":
        msg = "predecir con Core ML necesita macOS"
        raise SmokeError(msg)
    ct = _coremltools()
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    x = torch.rand(SMOKE_INPUT, generator=torch.Generator().manual_seed(SMOKE_SEED))
    with torch.no_grad():
        referencia = build_net()(x).numpy()
    diferencias = {}
    for unidades in COMPUTE_UNITS:
        modelo = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, unidades))
        salida = next(iter(modelo.predict({"x": x.numpy()}).values()))
        diferencias[unidades] = float(np.max(np.abs(salida - referencia)))
    return diferencias


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dónde guardar el paquete")
    args = parser.parse_args(argv)
    try:
        paquete = convert(args.out)
        print(f"convertido: {paquete} (mlprogram, fp16, iOS 18, {SMOKE_INPUT})")
        if sys.platform != "darwin":
            print("predecir   : solo en macOS; aquí se ha convertido y guardado")
            return 0
        diferencias = predict_diffs(paquete)
    except SmokeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for unidades, delta in diferencias.items():
        print(f"{unidades:<11}: max|fp16 - fp32| = {delta:.2e}")
    if max(diferencias.values()) >= FP16_SMOKE_TOL:
        print(f"error: el fp16 se aleja más de {FP16_SMOKE_TOL:g} de fp32", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
