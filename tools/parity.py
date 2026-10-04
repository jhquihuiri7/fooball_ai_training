"""Paridad torch ↔ ONNX ↔ Core ML de un modelo exportado (ML-11).

    uv sync --group train --group apple
    uv run python tools/parity.py \\
        --spec specs/ball-roilite.yaml \\
        --builder ftrain.ball.model:build \\
        --checkpoint runs/ball/best.pt \\
        --onnx dist/ball.onnx \\
        --package dist/ball.mlpackage \\
        --decoder heatmap --out dist/parity.json

Corre el mismo lote (las entradas propias de `--inputs` más ruido con semilla
hasta `--n`) por torch fp32, onnxruntime fp32 y, en macOS y con `--package`,
Core ML fp16 con CPU_AND_NE y con CPU_ONLY. Escribe el informe JSON y termina
con código 1 si algún par supera su umbral. torch↔ORT es la regla del repo
(1e-3); el de Core ML sale de `parity.atol` de la ficha (`--ficha`) o de
`--atol-coreml`.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.export.coreml import ExportSpec, load_spec
from ftrain.export.parity import (
    DECODERS,
    DEFAULT_COREML_ATOL,
    TORCH_ORT_ATOL,
    aggregate_metrics,
    detector_metrics,
    evaluate,
    fallback_band_of,
    heatmap_metrics,
    run_coreml,
    run_ort,
    run_torch,
    seeded_batch,
    spotter_metrics,
)

Lote = list[dict[str, np.ndarray]]


def _atol_coreml(opciones: argparse.Namespace) -> float:
    if opciones.atol_coreml is not None:
        return float(opciones.atol_coreml)
    if opciones.ficha:
        ficha = json.loads(opciones.ficha.read_text(encoding="utf-8"))
        atol = ficha.get("parity", {}).get("atol")
        if atol is not None:
            return float(atol)
    return DEFAULT_COREML_ATOL


def _propias(directorio: Path, spec: ExportSpec) -> list[dict[str, np.ndarray]]:
    """Las entradas del split de test: un .npz por muestra, claves = entradas."""
    lote = []
    for ruta in sorted(directorio.glob("*.npz")):
        with np.load(ruta) as datos:
            lote.append({entrada.name: datos[entrada.name] for entrada in spec.inputs})
    return lote


def _metricas(decoder: str, refs: Lote, others: dict[str, Lote]) -> dict:
    por_muestra = {
        "detector": lambda r, o: detector_metrics(r, o, band_of=fallback_band_of),
        "heatmap": heatmap_metrics,
        "spotter": spotter_metrics,
    }[decoder]
    return {
        par: aggregate_metrics([por_muestra(r, o) for r, o in zip(refs, lote, strict=True)])
        for par, lote in others.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="el YAML del export (ML-09)")
    parser.add_argument("--builder", required=True, help="modulo:funcion que construye el modelo")
    parser.add_argument("--checkpoint", type=Path, help="los pesos; sin él, el builder decide")
    parser.add_argument("--onnx", type=Path, required=True, help="el .onnx a comparar")
    parser.add_argument("--package", type=Path, help="el .mlpackage (solo cuenta en macOS)")
    parser.add_argument("--inputs", type=Path, help="directorio de .npz del split de test")
    parser.add_argument("--n", type=int, default=200, help="tamaño total del lote")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder", choices=[*DECODERS, "none"], default="none")
    parser.add_argument("--atol-ort", type=float, default=TORCH_ORT_ATOL)
    parser.add_argument("--atol-coreml", type=float, default=None)
    parser.add_argument("--ficha", type=Path, help="ficha v2: de ahí sale parity.atol")
    parser.add_argument("--out", type=Path, required=True, help="dónde escribir el informe JSON")
    opciones = parser.parse_args(argv)

    spec = load_spec(opciones.spec)
    modulo, _, funcion = opciones.builder.partition(":")
    if not funcion:
        print("--builder tiene que ser modulo:funcion", file=sys.stderr)
        return 1
    red = getattr(import_module(modulo), funcion)(opciones.checkpoint)

    lote = _propias(opciones.inputs, spec) if opciones.inputs else []
    if len(lote) < opciones.n:
        lote += seeded_batch(spec, opciones.n - len(lote), opciones.seed)

    refs = run_torch(red, spec, lote)
    others = {"torch_ort": run_ort(opciones.onnx, spec, lote)}
    atol = {"torch_ort": float(opciones.atol_ort)}

    if opciones.package and platform.system() == "Darwin":
        for par, unidades in (("torch_coreml_ne", "cpu_and_ne"), ("torch_coreml_cpu", "cpu_only")):
            others[par] = run_coreml(opciones.package, spec, lote, unidades)
            atol[par] = _atol_coreml(opciones)
    elif opciones.package:
        print("sin macOS no hay pata de Core ML: se informa solo torch↔ORT", file=sys.stderr)

    metricas = _metricas(opciones.decoder, refs, others) if opciones.decoder != "none" else None
    informe, ok = evaluate(refs, others, atol, metricas)

    opciones.out.parent.mkdir(parents=True, exist_ok=True)
    opciones.out.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(informe, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
