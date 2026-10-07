"""La secuencia dorada de un modo paso con estado (SPK-53).

    uv sync --group train --group apple
    uv run python tools/golden_sequence.py \\
        --spec configs/export/n4-step-mlstate.yaml \\
        --builder ftrain.events.spotter_export:build_spotter \\
        --name n4-step --version v1 --out dist/golden \\
        --package dist/n4-step-mlstate.mlpackage \\
        --package dist/n4-step-explicit.mlpackage

Corre N fotogramas con semilla por torch fp32, uno tras otro desde el estado cero, y
escribe el bundle de ML-12 (golden/<nombre>-<versión>/): las entradas `frame_NNN`
(uint8 planar, como el recorte antes de escalarlo) y, por paso, `logits_NNN` y `h_NNN`
(el estado de la GRU) por la ruta torch_fp32, con la tolerancia fp16 por salida y paso.
Al lado deja `sequence.json`, lo que el arnés de Swift necesita para recorrerla: los
pasos van EN ORDEN y con un solo estado.

Con `--package` (solo macOS) corre además cada .mlpackage por Core ML con la misma
secuencia —MLState o el estado reciclado, lo que lleve el modelo— y termina con
código 1 si algún paso se sale de la tolerancia: es la puerta antes del iPhone.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.constants import N4_GOLDEN_STEPS, N4_STEP_COREML_ATOL
from ftrain.events.spotter_export import (
    FRAME_SCALE,
    golden_sequence,
    run_coreml_sequence,
    seeded_frames,
)
from ftrain.export.coreml import load_spec
from ftrain.export.golden import GoldenArray, pack_bundle, write_bundle

if TYPE_CHECKING:
    from collections.abc import Mapping

SEQUENCE_VERSION = 1
"""La versión de sequence.json; Swift la comprueba antes de leer nada."""

FRAME_INPUT = "frame"
"""El nombre de la entrada del fotograma en los specs del modo paso."""

OUTPUTS = ("logits", "h")
"""Lo que se compara por paso: la salida del modelo y el estado de la GRU."""


def build_bundle(  # noqa: PLR0913 — el contrato del bundle entero, todo con nombre
    red: object,
    *,
    frames: np.ndarray,
    name: str,
    version: str,
    out: Path,
    atol: Mapping[str, float],
) -> tuple[Path, dict[str, np.ndarray]]:
    """Escribe el bundle y sequence.json. Devuelve su ruta y la referencia torch."""
    ref = golden_sequence(red, frames)  # type: ignore[arg-type]
    pasos = frames.shape[0]
    entradas = [
        GoldenArray(f"{FRAME_INPUT}_{k:03d}", frames[k : k + 1], "NCHW_u8") for k in range(pasos)
    ]
    salidas = [
        GoldenArray(f"{nombre}_{k:03d}", ref[nombre][k : k + 1], "NC")
        for k in range(pasos)
        for nombre in OUTPUTS
    ]
    tolerancias = {
        f"{nombre}_{k:03d}": {"coreml_fp16": float(atol[nombre])}
        for k in range(pasos)
        for nombre in OUTPUTS
    }
    bundle = write_bundle(
        out,
        model=name,
        version=version,
        inputs=entradas,
        outputs={"torch_fp32": salidas},
        tolerances=tolerancias,
    )
    secuencia = {
        "version": SEQUENCE_VERSION,
        "steps": pasos,
        "frame_input": FRAME_INPUT,
        "frame_scale": FRAME_SCALE,
        "frame_dtype": "fp16",
        "outputs": list(OUTPUTS),
        "reference_route": "torch_fp32",
        "tolerance_route": "coreml_fp16",
        "explicit_suffix": "_out",
    }
    (bundle / "sequence.json").write_text(
        json.dumps(secuencia, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle, ref


def check_package(
    package: Path,
    frames: np.ndarray,
    ref: Mapping[str, np.ndarray],
    atol: Mapping[str, float],
    units: str,
) -> dict[str, object]:
    """La misma secuencia por Core ML: el peor delta y los pasos fuera, por salida."""
    visto = run_coreml_sequence(package, frames, units)
    informe: dict[str, object] = {"package": str(package), "compute_units": units}
    for nombre in OUTPUTS:
        delta = np.abs(visto[nombre] - ref[nombre]).max(axis=1)
        informe[nombre] = {
            "max_abs": float(delta.max()),
            "atol": float(atol[nombre]),
            "steps_over_atol": int((delta > atol[nombre]).sum()),
        }
    return informe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="un spec del modo paso")
    parser.add_argument("--builder", required=True, help="modulo:funcion del modelo base")
    parser.add_argument("--checkpoint", type=Path, help="los pesos; sin él, sembrados")
    parser.add_argument("--name", required=True, help="el nombre del bundle (p. ej. n4-step)")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path, required=True, help="directorio golden/")
    parser.add_argument("--steps", type=int, default=N4_GOLDEN_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--package", type=Path, action="append", default=[])
    parser.add_argument("--compute-units", default="cpu_and_ne")
    opciones = parser.parse_args(argv)

    spec = load_spec(opciones.spec)
    marco = next((e for e in spec.inputs if e.name == FRAME_INPUT), None)
    if marco is None:
        print(f"el spec no tiene la entrada {FRAME_INPUT!r}", file=sys.stderr)
        return 1
    modulo, _, funcion = opciones.builder.partition(":")
    if not funcion:
        print("--builder tiene que ser modulo:funcion", file=sys.stderr)
        return 1
    red = getattr(import_module(modulo), funcion)(opciones.checkpoint)

    _, _, alto, ancho = marco.shape
    frames = seeded_frames(opciones.steps, alto, ancho, opciones.seed)
    bundle, ref = build_bundle(
        red,
        frames=frames,
        name=opciones.name,
        version=opciones.version,
        out=opciones.out,
        atol=N4_STEP_COREML_ATOL,
    )
    zip_destino = opciones.out / f"{opciones.name}-{opciones.version}.golden.zip"
    sha = pack_bundle(bundle, zip_destino)
    informe: dict[str, object] = {"bundle": str(bundle), "zip": str(zip_destino), "sha256": sha}

    fuera = 0
    if opciones.package and platform.system() != "Darwin":
        print("sin macOS no se comprueba ningún paquete", file=sys.stderr)
    elif opciones.package:
        comprobados = []
        for paquete in opciones.package:
            resultado = check_package(
                paquete, frames, ref, N4_STEP_COREML_ATOL, opciones.compute_units
            )
            comprobados.append(resultado)
            fuera += sum(int(resultado[n]["steps_over_atol"]) for n in OUTPUTS)  # type: ignore[index]
        informe["packages"] = comprobados
    print(json.dumps(informe, ensure_ascii=False, indent=2))
    return 1 if fuera else 0


if __name__ == "__main__":
    raise SystemExit(main())
