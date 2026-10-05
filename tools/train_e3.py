"""Entrena N3, la TCN de eventos sobre posiciones (ML-50, ADR 0004).

    uv sync --group train
    uv run python tools/convert_skillcorner.py
    uv run python tools/train_e3.py --data datasets/n3/skillcorner/<commit> --out runs/n3-sc

Hoy, el preentreno con SkillCorner. El afinado con partidos propios (ML-49) usa el mismo
guion con otra carpeta. División por partidos completos, nunca por clips: los últimos
`--val` partidos (por id) dan la cifra y los `--tune` anteriores eligen los umbrales de
cada clase; el resto entrena, con espejos del campo como aumento. La cifra va sobre
partidos enteros y con ±2 s.

Escribe en --out el checkpoint (`n3.pt`) y `report.json`: P, R y F1 por clase.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as f

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.constants import N3_CLASSES
from ftrain.events.features import mirror
from ftrain.events.spotting import decode, match_events
from ftrain.events.tcn import N3Tcn

if TYPE_CHECKING:
    from collections.abc import Iterator

WINDOW = 256
"""Pasos por ventana de entrenamiento (34 s): dos campos receptivos."""
BATCH = 32
EVENT_CENTERED = 0.5
"""Fracción de ventanas centradas en un evento: los eventos son el 1 % de los pasos."""
MIRROR_P = 0.5
"""Probabilidad de cada espejo del campo (a lo largo y a lo ancho) por ventana."""
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.1, 0.95, 0.05))
SEED = 2026


def load(folder: Path) -> list[tuple[str, np.ndarray, np.ndarray, list[tuple[float, str]]]]:
    """(id, rejillas, etiqueta por paso, eventos) de cada partido convertido."""
    salida = []
    for p in sorted(folder.glob("*.npz")):
        d = np.load(p)
        eventos = [
            (float(t), N3_CLASSES[int(c)])
            for t, c in zip(d["event_t"], d["event_cls"], strict=True)
        ]
        salida.append((p.stem, d["grids"], d["labels"], eventos))
    return salida


def batches(
    data: list[tuple[str, np.ndarray, np.ndarray, list]], rng: np.random.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    eventos = [(i, int(k)) for i, (_, _, y, _) in enumerate(data) for k in np.flatnonzero(y)]
    while True:
        xs, ys = [], []
        for _ in range(BATCH):
            if rng.random() < EVENT_CENTERED:
                i, k = eventos[rng.integers(len(eventos))]
                inicio = k - int(rng.integers(WINDOW // 4, 3 * WINDOW // 4))
            else:
                i = int(rng.integers(len(data)))
                inicio = int(rng.integers(0, max(1, len(data[i][2]) - WINDOW)))
            _, g, y, _ = data[i]
            inicio = int(np.clip(inicio, 0, max(0, len(y) - WINDOW)))
            ventana = g[inicio : inicio + WINDOW]
            xs.append(
                mirror(ventana, along=rng.random() < MIRROR_P, across=rng.random() < MIRROR_P)
            )
            ys.append(y[inicio : inicio + WINDOW])
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


@torch.no_grad()
def probabilities(model: N3Tcn, grids: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(grids).unsqueeze(0).to(device))
    model.train()
    return torch.softmax(logits[0], dim=-1).cpu().numpy()


def evaluate(
    probs: list[np.ndarray], truths: list[list[tuple[float, str]]], thr: list[float]
) -> dict:
    total = {c: [0, 0, 0] for c in N3_CLASSES}
    for p, t in zip(probs, truths, strict=True):
        for c, s in match_events(t, decode(p, thr)).items():
            total[c] = [total[c][0] + s.truths, total[c][1] + s.predictions, total[c][2] + s.hits]
    salida = {}
    for c, (n, m, h) in total.items():
        pr, rc = (h / m if m else 0.0), (h / n if n else 0.0)
        salida[c] = {
            "truths": n,
            "predictions": m,
            "hits": h,
            "precision": pr,
            "recall": rc,
            "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
        }
    return salida


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=int, default=4, help="partidos para la cifra")
    ap.add_argument("--tune", type=int, default=4, help="partidos para elegir los umbrales")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    o = ap.parse_args(argv)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = load(o.data)
    train, tune, val = data[: -o.val - o.tune], data[-o.val - o.tune : -o.val], data[-o.val :]
    # Pesos de la pérdida: raíz del inverso de la frecuencia de cada clase en entrenamiento.
    cuenta = np.bincount(np.concatenate([y for _, _, y, _ in train]), minlength=1 + len(N3_CLASSES))
    pesos = torch.tensor(np.sqrt(cuenta.sum() / np.maximum(cuenta, 1)), dtype=torch.float32)
    pesos = (pesos / pesos[0]).to(device)

    model = N3Tcn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=o.lr, total_steps=o.steps)
    t0 = time.time()
    for paso, (x, y) in zip(range(o.steps), batches(train, rng), strict=False):
        logits = model(x.to(device))
        loss = f.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1), weight=pesos
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if paso % 250 == 0:
            print(f"paso {paso}: pérdida {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)

    p_train = [probabilities(model, g, device) for _, g, _, _ in train]
    t_train = [e for _, _, _, e in train]
    p_tune = [probabilities(model, g, device) for _, g, _, _ in tune]
    t_tune = [e for _, _, _, e in tune]
    umbrales = []
    for k, c in enumerate(N3_CLASSES):
        mejor = max(
            THRESHOLDS,
            key=lambda u, k=k, c=c: evaluate(
                p_train, t_train, [u if j == k else 1.1 for j in range(len(N3_CLASSES))]
            )[c]["f1"],
        )
        umbrales.append(mejor)
    p_val = [probabilities(model, g, device) for _, g, _, _ in val]
    t_val = [e for _, _, _, e in val]
    informe = {
        "data": str(o.data),
        "train_matches": [n for n, _, _, _ in train],
        "tune_matches": [n for n, _, _, _ in tune],
        "val_matches": [n for n, _, _, _ in val],
        "steps": o.steps,
        "params": sum(p.numel() for p in model.parameters()),
        "thresholds": dict(zip(N3_CLASSES, umbrales, strict=True)),
        "train": evaluate(p_train, t_train, umbrales),
        "tune": evaluate(p_tune, t_tune, umbrales),
        "val": evaluate(p_val, t_val, umbrales),
        "seconds": round(time.time() - t0),
    }
    o.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), o.out / "n3.pt")
    (o.out / "report.json").write_text(json.dumps(informe, indent=2))
    print("| Clase | Verdad | P | R | F1 |\n|---|---|---|---|---|")
    for c, s in informe["val"].items():
        print(f"| {c} | {s['truths']} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
