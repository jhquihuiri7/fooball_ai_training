"""El balón en píxeles a cada distancia (ML-26: M5).

    uv run python tools/ball_size_table.py --truth labels/v1/instances.json

Del COCO canónico (ML-22) con la `band` de cada caja: el diámetro del balón (la media del
ancho y el alto de su caja, ADR 0003 §3) en p5/p50/p95 por banda de distancia. Escribe la
tabla en Markdown por la salida estándar y, con --out, en JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.constants import DISTANCE_BANDS_M, MASTER_CLASSES

PERCENTILES = (5, 50, 95)


def table(coco: dict) -> dict[int, dict[str, float]]:
    diametros: dict[int, list[float]] = {}
    for an in coco["annotations"]:
        if MASTER_CLASSES[an["category_id"]] != "ball" or "band" not in an:
            continue
        _, _, w, h = an["bbox"]
        diametros.setdefault(int(an["band"]), []).append((w + h) / 2.0)
    return {
        b: {"n": len(v), **{f"p{q}": float(np.percentile(v, q)) for q in PERCENTILES}}
        for b, v in sorted(diametros.items())
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path)
    o = p.parse_args(argv)
    t = table(json.loads(o.truth.read_text()))
    bordes = list(itertools.pairwise(DISTANCE_BANDS_M))
    print("| Banda | Balones | p5 px | p50 px | p95 px |\n|---|---|---|---|---|")
    for b, f in t.items():
        a, z = bordes[b]
        print(f"| {a:g}-{z:g} m | {f['n']} | {f['p5']:.1f} | {f['p50']:.1f} | {f['p95']:.1f} |")
    if o.out:
        o.out.write_text(json.dumps(t, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
