"""Evaluación de un detector de jugadores por banda de distancia (ML-26: M4, R-P7-1).

    uv run python tools/eval_players.py --truth labels/v1/instances.json \\
        --pred preds.npz --out informes/players-v0

La verdad es el COCO canónico (ML-22) con, por imagen, `match` y `light`, y por caja su
`band` (la de `ftrain.bands.band_of`). Las predicciones son un .npz de torch o de Core
ML con `image_id`, `category` (índice en PLAYER_CLASSES), `boxes` (Nx4, xyxy en las
mismas coordenadas que la verdad) y `scores`. Escribe <out>.json y <out>.md: recall,
precisión y AP50 por banda, por luz y por partido, y el intervalo del recall por
bootstrap sobre partidos. La entrada directa de un .onnx por onnxruntime queda para cuando
exista el preproceso de la franja en este repo.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.constants import DISTANCE_BANDS_M, MASTER_CLASSES, PLAYER_CLASSES
from ftrain.eval.detection import Detection, Truth, bootstrap_recall, evaluate


def load_truth(path: Path) -> list[Truth]:
    coco = json.loads(path.read_text())
    imagenes = {im["id"]: im for im in coco["images"]}
    verdad = []
    for an in coco["annotations"]:
        clase = MASTER_CLASSES[an["category_id"]]
        if clase not in PLAYER_CLASSES:
            continue
        im = imagenes[an["image_id"]]
        if im.get("unusable"):
            continue
        for clave, donde in (("match", im), ("light", im), ("band", an)):
            if clave not in donde:
                sys.exit(f"falta `{clave}` en {path}: la evaluación es por partido, luz y banda")
        x, y, w, h = an["bbox"]
        verdad.append(
            Truth(
                im["file_name"],
                im["match"],
                clase,
                (x, y, x + w, y + h),
                int(an["band"]),
                im["light"],
            )
        )
    return verdad


def load_predictions(path: Path, coco_path: Path) -> list[Detection]:
    nombres = {im["id"]: im["file_name"] for im in json.loads(coco_path.read_text())["images"]}
    datos = np.load(path)
    return [
        Detection(nombres[int(i)], PLAYER_CLASSES[int(c)], tuple(float(v) for v in b), float(s))
        for i, c, b, s in zip(
            datos["image_id"], datos["category"], datos["boxes"], datos["scores"], strict=True
        )
    ]


def bandas() -> list[str]:
    return [
        f"{a:g}-{b:g} m" if b != float("inf") else f">{a:g} m"
        for a, b in itertools.pairwise(DISTANCE_BANDS_M)
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    o = p.parse_args(argv)

    verdad = load_truth(o.truth)
    preds = load_predictions(o.pred, o.truth)
    por_banda = evaluate(verdad, preds)
    sin_clase = evaluate(verdad, preds, by_class=False)
    ic = bootstrap_recall(verdad, preds)
    por_luz = evaluate(verdad, preds, group=lambda t: (t.light,))
    por_partido = evaluate(verdad, preds, group=lambda t: (t.match,))
    nombres = bandas()

    informe = {
        "truth": str(o.truth),
        "pred": str(o.pred),
        "bands": {
            nombres[int(k[0])]: {
                **vars(m),
                "recall_ci": ic.get(k),
                "recall_any_class": sin_clase[k].recall,
            }
            for k, m in por_banda.items()
        },
        "light": {k[0]: vars(m) for k, m in por_luz.items()},
        "match": {k[0]: vars(m) for k, m in por_partido.items()},
    }
    o.out.parent.mkdir(parents=True, exist_ok=True)
    o.out.with_suffix(".json").write_text(json.dumps(informe, indent=2, ensure_ascii=False))
    filas = [
        "| Banda | Verdad | Recall (IC 95 %) | Recall sin clase | Precisión | AP50 |",
        "|---|---|---|---|---|---|",
    ]
    for k, m in por_banda.items():
        bajo, alto = ic.get(k, (0.0, 0.0))
        filas.append(
            f"| {nombres[int(k[0])]} | {m.truths} | {m.recall:.3f} ({bajo:.3f}-{alto:.3f}) | "
            f"{sin_clase[k].recall:.3f} | {m.precision:.3f} | {m.ap50:.3f} |"
        )
    o.out.with_suffix(".md").write_text("\n".join(filas) + "\n")
    print("\n".join(filas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
