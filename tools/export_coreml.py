"""Exporta un modelo a Core ML según su spec YAML (ML-09).

    uv sync --group train --group apple
    uv run python tools/export_coreml.py \\
        --spec specs/ball-roilite.yaml \\
        --builder ftrain.ball.model:build \\
        --checkpoint runs/ball/best.pt \\
        --out dist/

Deja `dist/<model_name>.mlpackage.zip` (zip determinista: el sha256 es estable entre
exports idénticos) e imprime el sha para la ficha v2 del registro. `--builder` es
`modulo:funcion`; la función recibe la ruta del checkpoint (o None) y devuelve el
nn.Module listo para exportar. Con `--keep-package` deja además el `.mlpackage` sin
zipear al lado, que es lo que copia el banco del iPhone (SPK-53).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.export.coreml import (
    build_metadata,
    convert,
    deterministic_zip,
    load_spec,
    normalize_manifest,
)


def _commit() -> str:
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return salida.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "sin-git"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="el YAML del export")
    parser.add_argument(
        "--builder", required=True, help="modulo:funcion que construye el nn.Module"
    )
    parser.add_argument("--checkpoint", type=Path, help="los pesos; sin él, el builder decide")
    parser.add_argument("--out", type=Path, required=True, help="directorio de salida")
    parser.add_argument("--classes", default="", help="clases, separadas por comas")
    parser.add_argument("--dataset-version", default="sin-dataset")
    parser.add_argument("--region", default="full_frame")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument(
        "--keep-package", action="store_true", help="deja también el .mlpackage sin zipear"
    )
    opciones = parser.parse_args(argv)

    spec = load_spec(opciones.spec)
    modulo, _, funcion = opciones.builder.partition(":")
    if not funcion:
        print("--builder tiene que ser modulo:funcion", file=sys.stderr)
        return 1
    builder = getattr(import_module(modulo), funcion)
    red = builder(opciones.checkpoint)

    metadatos = build_metadata(
        spec,
        classes=[c for c in opciones.classes.split(",") if c],
        dataset_version=opciones.dataset_version,
        commit=_commit(),
        region=opciones.region,
        frames=opciones.frames,
    )
    modelo = convert(red, spec, metadatos)

    with tempfile.TemporaryDirectory() as temporal:
        paquete = Path(temporal) / f"{spec.model_name}.mlpackage"
        modelo.save(str(paquete))
        normalize_manifest(paquete)
        destino = opciones.out / f"{spec.model_name}.mlpackage.zip"
        sha = deterministic_zip(paquete, destino)
        informe = {"package": str(destino), "sha256": sha}
        if opciones.keep_package:
            suelto = opciones.out / paquete.name
            shutil.rmtree(suelto, ignore_errors=True)
            shutil.copytree(paquete, suelto)
            informe["mlpackage"] = str(suelto)

    print(json.dumps(informe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
