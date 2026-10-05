"""Los partidos de SkillCorner a la serie de N3 (ML-48).

    uv run python tools/fetch_skillcorner.py
    uv run python tools/convert_skillcorner.py

Por partido, `datasets/n3/skillcorner/<commit>/<id>.npz` con `grids` (T, canales, alto,
ancho, float32), `labels` (T, int64: 0 = nada, k + 1 = N3_CLASSES[k]) y los eventos tal
cual (`event_t` en segundos desde el primer paso, `event_cls` índice en N3_CLASSES).
Imprime el conteo
de etiquetas por clase, que va a PROGRESS.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.constants import N3_CLASSES
from ftrain.events.features import grids, label_timeline
from ftrain.events.skillcorner import load_match
from tools.fetch_skillcorner import COMMIT, ROOT

OUT = Path(__file__).resolve().parent.parent / "datasets" / "n3" / "skillcorner" / COMMIT


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total: collections.Counter[str] = collections.Counter()
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        m = load_match(d)
        g = grids(m)
        y = label_timeline(m.samples, m.labels)
        t0 = m.samples[0].t_s if m.samples else 0.0
        np.savez_compressed(
            OUT / f"{m.match_id}.npz",
            grids=g,
            labels=y,
            # Los eventos tal cual, en segundos desde el primer paso: el P/R se mide contra
            # ellos, no contra la etiqueta dilatada.
            event_t=np.array([t - t0 for t, _ in m.labels], dtype=np.float64),
            event_cls=np.array([N3_CLASSES.index(c) for _, c in m.labels], dtype=np.int64),
        )
        cuenta = collections.Counter(c for _, c in m.labels)
        total.update(cuenta)
        horas = len(m.samples) / 7.5 / 3600
        print(
            f"{m.match_id}: {len(m.samples)} pasos ({horas:.2f} h), "
            + ", ".join(f"{c} {cuenta[c]}" for c in N3_CLASSES)
        )
    print("total: " + ", ".join(f"{c} {total[c]}" for c in N3_CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
