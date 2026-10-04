"""Fusiona varios exports del MISMO modelo en un .mlpackage multifunción (SPK-52).

    uv sync --group train --group apple
    uv run python tools/export_multifunction.py \\
        --builder ftrain.ball.model:build \\
        --function global=specs/roilite-global.yaml \\
        --function roi=specs/roilite-roi.yaml \\
        --out dist/

Cada función es el mismo nn.Module convertido con su propia forma (ML-09); la
fusión (`ct.utils.save_multifunction`) DEDUPLICA los pesos compartidos, que es
la gracia: el paquete global/roi debe pesar <=1,1x uno suelto (la aceptación de
SPK-52 lo mide). El zip sale determinista con el sha de siempre.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.export.coreml import (
    ExportError,
    build_metadata,
    convert,
    deterministic_zip,
    load_spec,
    normalize_manifest,
)


def export_multifunction(  # noqa: PLR0913 — el contrato del CLI entero, con nombre
    builder: str,
    functions: dict[str, Path],
    out_dir: Path,
    *,
    checkpoint: Path | None = None,
    name: str | None = None,
    classes: list[str] | None = None,
    dataset_version: str = "sin-dataset",
    commit: str = "sin-git",
) -> tuple[Path, str]:
    """Convierte cada función y las fusiona. Devuelve (paquete, sha256 del zip)."""
    import coremltools as ct  # noqa: PLC0415 — perezoso adrede (grupo apple)

    modulo, _, funcion = builder.partition(":")
    if not funcion:
        msg = "--builder tiene que ser modulo:funcion"
        raise ExportError(msg)
    construir = getattr(import_module(modulo), funcion)

    especificaciones = {nombre: load_spec(ruta) for nombre, ruta in functions.items()}
    nombres_modelo = {spec.model_name for spec in especificaciones.values()}
    paquete_nombre = name or min(nombres_modelo)

    with tempfile.TemporaryDirectory() as temporal:
        descriptor = ct.utils.MultiFunctionDescriptor()
        for nombre_funcion, spec in especificaciones.items():
            red = construir(checkpoint)
            metadatos = build_metadata(
                spec,
                classes=classes or [],
                dataset_version=dataset_version,
                commit=commit,
                region="roi",
                frames=3,
            )
            modelo = convert(red, spec, metadatos)
            suelto = Path(temporal) / f"{nombre_funcion}.mlpackage"
            modelo.save(str(suelto))
            descriptor.add_function(
                str(suelto), src_function_name="main", target_function_name=nombre_funcion
            )
        descriptor.default_function_name = min(especificaciones)

        fusionado = Path(temporal) / f"{paquete_nombre}.mlpackage"
        ct.utils.save_multifunction(descriptor, str(fusionado))
        normalize_manifest(fusionado)

        destino = out_dir / f"{paquete_nombre}.mlpackage"
        out_dir.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            shutil.rmtree(destino)
        shutil.copytree(fusionado, destino)
        sha = deterministic_zip(destino, out_dir / f"{paquete_nombre}.mlpackage.zip")
    return destino, sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", required=True, help="modulo:funcion que construye el modelo")
    parser.add_argument("--checkpoint", type=Path, help="los pesos; sin él, el builder decide")
    parser.add_argument(
        "--function",
        action="append",
        required=True,
        metavar="NOMBRE=SPEC.yaml",
        help="una función del paquete; repetible (global=..., roi=...)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", help="nombre del paquete; sin él, el del primer spec")
    parser.add_argument("--classes", default="", help="clases, separadas por comas")
    opciones = parser.parse_args(argv)

    funciones: dict[str, Path] = {}
    for cruda in opciones.function:
        nombre, _, ruta = cruda.partition("=")
        if not ruta:
            print(f"--function tiene que ser NOMBRE=SPEC.yaml y es {cruda!r}", file=sys.stderr)
            return 1
        funciones[nombre] = Path(ruta)

    paquete, sha = export_multifunction(
        opciones.builder,
        funciones,
        opciones.out,
        checkpoint=opciones.checkpoint,
        name=opciones.name,
        classes=[c for c in opciones.classes.split(",") if c],
    )
    print(json.dumps({"package": str(paquete), "sha256": sha, "functions": sorted(funciones)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
