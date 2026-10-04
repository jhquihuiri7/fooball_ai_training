"""Genera el bundle dorado de un modelo exportado (ML-12).

    uv sync --group train --group apple --group ref
    uv run python tools/golden.py \\
        --spec specs/ball-roilite.yaml \\
        --builder ftrain.ball.model:build \\
        --checkpoint runs/ball/best.pt \\
        --onnx dist/ball.onnx \\
        --package dist/ball.mlpackage \\
        --version v1 --decode heatmap --out golden/

Corre el lote con semilla por torch fp32, ORT fp32 y, en macOS con `--package`,
Core ML fp16; guarda las entradas como las entrega la app (BGRA uint8 las
imágenes, fp16 los tensores), las salidas de cada ruta, las detecciones que
decide la referencia (ftrain/ref) y el manifest con las tolerancias. Deja el
zip determinista al lado y imprime su sha256.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.export.coreml import ExportSpec, load_spec
from ftrain.export.golden import GoldenArray, pack_bundle, to_bgra8, write_bundle
from ftrain.export.parity import (
    DEFAULT_COREML_ATOL,
    DETECTOR_SCORE,
    TORCH_ORT_ATOL,
    run_coreml,
    run_ort,
    run_torch,
    seeded_batch,
)

GOLDEN_SAMPLES = 2
"""Muestras del bundle: pocas adrede, que XCTest las corre en un iPhone."""

GOLDEN_PEAKS_K = 5
"""Cuántos picos del heatmap guarda la referencia en detecciones.json."""


def _app_inputs(spec: ExportSpec, lote: list[dict[str, np.ndarray]]) -> list[GoldenArray]:
    """Las entradas tal como las entrega la app: BGRA8 la imagen, fp16 el tensor."""
    arrays: list[GoldenArray] = []
    for indice, muestra in enumerate(lote):
        for entrada in spec.inputs:
            cruda = muestra[entrada.name]
            nombre = f"{entrada.name}_{indice:03d}"
            if entrada.kind == "image":
                arrays.append(GoldenArray(nombre, to_bgra8(cruda), "BGRA8"))
            else:
                arrays.append(GoldenArray(nombre, cruda.astype(np.float16), "fp16_planar"))
    return arrays


def _route_outputs(
    spec: ExportSpec, lote_salidas: list[dict[str, np.ndarray]], dtype: type
) -> list[GoldenArray]:
    return [
        GoldenArray(f"{nombre}_{indice:03d}", salidas[nombre].astype(dtype), "NCHW")
        for indice, salidas in enumerate(lote_salidas)
        for nombre in spec.output_names
    ]


def _decode(spec: ExportSpec, decode: str, refs: list[dict[str, np.ndarray]]) -> Any:
    """Lo que decide la referencia sobre las salidas de torch, por muestra."""
    from ftrain import ref  # noqa: PLC0415 — perezoso adrede (grupo ref)

    if decode == "detr":
        _, _, alto, ancho = spec.inputs[0].shape
        muestras = []
        for salidas in refs:
            puntuaciones = ref.sigmoid(salidas["logits"][0])
            mejores = puntuaciones.max(axis=-1)
            clases = puntuaciones.argmax(axis=-1)
            x1, y1, x2, y2 = ref.decode_boxes_to_corners(
                salidas["boxes"][0], width=float(ancho), height=float(alto)
            )
            quedan = np.flatnonzero(mejores >= DETECTOR_SCORE)
            muestras.append(
                [
                    {
                        "score": float(mejores[i]),
                        "class": int(clases[i]),
                        "box": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    }
                    for i in quedan
                ]
            )
        return muestras
    muestras = []
    for salidas in refs:
        mapa = salidas[spec.output_names[0]][0]
        clases, ys, xs, puntuaciones = ref.heatmap_peaks(mapa, k=GOLDEN_PEAKS_K)
        muestras.append(
            [
                {"class": int(c), "y": int(y), "x": int(x), "score": float(s)}
                for c, y, x, s in zip(clases, ys, xs, puntuaciones, strict=True)
            ]
        )
    return muestras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="el YAML del export (ML-09)")
    parser.add_argument("--builder", required=True, help="modulo:funcion que construye el modelo")
    parser.add_argument("--checkpoint", type=Path, help="los pesos; sin él, el builder decide")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--package", type=Path, help="el .mlpackage (solo cuenta en macOS)")
    parser.add_argument("--version", required=True, help="la versión del modelo en el bundle")
    parser.add_argument("--out", type=Path, required=True, help="directorio golden/")
    parser.add_argument("--n", type=int, default=GOLDEN_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode", choices=["detr", "heatmap", "none"], default="none")
    parser.add_argument("--atol-ort", type=float, default=TORCH_ORT_ATOL)
    parser.add_argument("--atol-coreml", type=float, default=DEFAULT_COREML_ATOL)
    opciones = parser.parse_args(argv)

    spec = load_spec(opciones.spec)
    modulo, _, funcion = opciones.builder.partition(":")
    if not funcion:
        print("--builder tiene que ser modulo:funcion", file=sys.stderr)
        return 1
    from importlib import import_module  # noqa: PLC0415 — junto a su único uso

    red = getattr(import_module(modulo), funcion)(opciones.checkpoint)

    lote = seeded_batch(spec, opciones.n, opciones.seed)
    refs = run_torch(red, spec, lote)
    salidas: dict[str, list[dict[str, np.ndarray]]] = {
        "torch_fp32": refs,
        "ort_fp32": run_ort(opciones.onnx, spec, lote),
    }
    if opciones.package and platform.system() == "Darwin":
        salidas["coreml_fp16"] = run_coreml(opciones.package, spec, lote, "cpu_and_ne")
    elif opciones.package:
        print("sin macOS no hay ruta coreml_fp16 en el bundle", file=sys.stderr)

    rutas_dtype = {"torch_fp32": np.float32, "ort_fp32": np.float32, "coreml_fp16": np.float16}
    tolerancias = {
        f"{nombre}_{indice:03d}": {
            "ort_fp32": float(opciones.atol_ort),
            "coreml_fp16": float(opciones.atol_coreml),
        }
        for indice in range(len(lote))
        for nombre in spec.output_names
    }
    bundle = write_bundle(
        opciones.out,
        model=spec.model_name,
        version=opciones.version,
        inputs=_app_inputs(spec, lote),
        outputs={
            ruta: _route_outputs(spec, lote_salidas, rutas_dtype[ruta])
            for ruta, lote_salidas in salidas.items()
        },
        tolerances=tolerancias,
        detections=_decode(spec, opciones.decode, refs) if opciones.decode != "none" else None,
    )
    zip_destino = opciones.out / f"{spec.model_name}-{opciones.version}.golden.zip"
    sha = pack_bundle(bundle, zip_destino)
    print(json.dumps({"bundle": str(bundle), "zip": str(zip_destino), "sha256": sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
