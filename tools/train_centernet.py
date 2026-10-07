"""Preentrena CenterNet-MNv4 con las personas de COCO (plan B de jugadores, ADR 0020).

    uv sync --group train
    uv run python tools/fetch_coco_person.py
    uv run python tools/train_centernet.py --data datasets/coco2017 --out runs/centernet-coco \\
        --hours 12.5

Sin datos propios todavía (ML-17), los pesos reales de partida salen de aquí: tronco
MobileNetV4-Conv-S de ImageNet (timm, Apache-2.0, fijado por sha256) y COCO person en
mosaicos en fila con la forma de la franja. Las personas van a la clase `player`; portero y
árbitro se entrenan como negativos.

Reanudable: si `--out` ya tiene `last.pt`, sigue desde ahí con el mismo comando. El
calendario del lr es por TIEMPO (coseno sobre `--hours`), así que acaba cuando se le dice
aunque el Mac vaya más lento de lo medido. Escribe en `--out`:

- `last.pt` cada `--ckpt-min` minutos (modelo, EMA, optimizador, paso y tiempo gastado);
- `best.pt`: el state_dict de la EMA con mejor AP de persona, el que carga
  `ftrain.players.plan_b:build_centernet` y exporta `tools/export_coreml.py`;
- `metrics.jsonl`: una línea por evaluación.

La evaluación es la de `tools/dfine_dap.py` (ML-16): las primeras `--eval-n` imágenes de
val2017 por id, con letterbox a 576x1920, AP COCO de `person`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.constants import PIXEL_SCALE, PLAYER_INPUT_H, PLAYER_INPUT_W
from ftrain.players.centernet_data import MosaicConfig, letterbox
from ftrain.players.centernet_train import (
    CocoPersonMosaic,
    Ema,
    centernet_loss,
    decode,
    init_for_training,
    load_imagenet_backbone,
)
from ftrain.players.plan_b import CenterNetMnv4
from tools.fetch_deim import Weights, download

MNV4_IN1K = Weights(
    name="mobilenetv4_conv_small.e2400_r224_in1k.safetensors",
    url=(
        "https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/resolve/"
        "331fb803779522b685cf942e15f914fb6741c1eb/model.safetensors"
    ),
    sha256="7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4",
)
"""MobileNetV4-Conv-S de timm en ImageNet-1k (Ross Wightman), Apache-2.0 según su ficha en
el commit fijado. ImageNet en el preentreno es el riesgo residual que acepta el ADR 0002."""

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models" / "pretrained"
PERSON_CATEGORY = 1
"""La categoría COCO de `person`."""

WARMUP_STEPS = 1000
"""Pasos de subida lineal del lr: la focal con la cabeza recién puesta pega tirones."""
LR_FLOOR = 0.05
"""El lr al final del coseno, como fracción del inicial."""
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
"""Norma máxima del gradiente: corta los tirones de la focal al principio."""
EMA_DECAY = 0.9998
EMA_RAMP = 2000.0
"""Pasos de la rampa de la EMA: al principio sigue al modelo de cerca."""
LOG_EVERY = 50
EVAL_BATCH = 4


def _letterbox_batch(
    rutas: list[Path], alto: int, ancho: int
) -> tuple[np.ndarray, list[float], list[tuple[int, int]]]:
    import cv2  # noqa: PLC0415 - grupo train

    lienzos, escalas, formas = [], [], []
    for r in rutas:
        bgr = cv2.imread(str(r), cv2.IMREAD_COLOR)
        lienzo, escala = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alto, ancho)
        lienzos.append(lienzo)
        escalas.append(escala)
        formas.append((bgr.shape[1], bgr.shape[0]))
    return np.stack(lienzos), escalas, formas


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    gt: Any,
    images: Path,
    ids: list[int],
    device: torch.device,
) -> dict[str, float]:
    """AP y AP50 COCO de `person` con letterbox a la franja (el protocolo de ML-16)."""
    from faster_coco_eval import COCOeval_faster  # noqa: PLC0415 - grupo train

    model.eval()
    resultados: list[dict[str, Any]] = []
    for i in range(0, len(ids), EVAL_BATCH):
        lote = ids[i : i + EVAL_BATCH]
        info = gt.loadImgs(lote)
        lienzos, escalas, formas = _letterbox_batch(
            [images / x["file_name"] for x in info], PLAYER_INPUT_H, PLAYER_INPUT_W
        )
        x = torch.from_numpy(lienzos).permute(0, 3, 1, 2).to(device).float() * PIXEL_SCALE
        cajas, scores, _ = decode(*model(x))
        for b, imagen_id in enumerate(lote):
            w, h = formas[b]
            c = cajas[b].cpu().numpy().astype(np.float64) / escalas[b]
            c[:, [0, 2]] = c[:, [0, 2]].clip(0, w)
            c[:, [1, 3]] = c[:, [1, 3]].clip(0, h)
            resultados.extend(
                {
                    "image_id": imagen_id,
                    "category_id": PERSON_CATEGORY,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                }
                for (x1, y1, x2, y2), s in zip(c.tolist(), scores[b].cpu().tolist(), strict=True)
                if x2 > x1 and y2 > y1
            )
    if not resultados:
        return {"ap": 0.0, "ap50": 0.0}
    ev = COCOeval_faster(gt, gt.loadRes(resultados), "bbox")
    ev.params.imgIds = ids
    ev.params.catIds = [PERSON_CATEGORY]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"ap": float(ev.stats[0]), "ap50": float(ev.stats[1])}


def lr_at(step: int, progress: float, base: float) -> float:
    """Subida lineal `WARMUP_STEPS` pasos y coseno por tiempo hasta `LR_FLOOR`."""
    calentamiento = min(1.0, (step + 1) / WARMUP_STEPS)
    coseno = LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base * calentamiento * coseno


def _worker_init(_: int) -> None:
    import cv2  # noqa: PLC0415 - grupo train

    cv2.setNumThreads(0)


def _save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("datasets/coco2017"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hours", type=float, required=True, help="presupuesto de reloj total")
    p.add_argument("--max-steps", type=int, default=0, help="tope de pasos (0: sin tope)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--canvas", default="384x1152", help="alto x ancho del mosaico")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2000, help="pasos entre evaluaciones")
    p.add_argument("--eval-n", type=int, default=500)
    p.add_argument("--ckpt-min", type=float, default=15.0)
    p.add_argument("--licencias", default="", help="ids Flickr admitidos, p. ej. 4,5,6,7,8")
    p.add_argument("--no-imagenet", action="store_true", help="tronco sin preentrenar")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915 - el bucle entero, a la vista
    o = build_parser().parse_args(argv)
    from faster_coco_eval import COCO  # noqa: PLC0415 - grupo train

    o.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(o.device)
    alto, ancho = (int(v) for v in o.canvas.split("x"))
    indice = json.loads((o.data / "person_train2017.json").read_text())
    if o.licencias:
        admitidas = {int(v) for v in o.licencias.split(",")}
        indice = [e for e in indice if e["license"] in admitidas]
    gt = COCO(str(o.data / "annotations" / "instances_val2017.json"))
    ids_val = sorted(gt.getImgIds())[: o.eval_n]

    model = CenterNetMnv4()
    ultimo = o.out / "last.pt"
    estado = torch.load(ultimo, map_location="cpu") if ultimo.exists() else None
    if estado is None:
        if not o.no_imagenet:
            load_imagenet_backbone(model, download(MNV4_IN1K, WEIGHTS_DIR))
        init_for_training(model)
    else:
        model.load_state_dict(estado["model"])
    model.to(device).train()
    ema = Ema(model, EMA_DECAY, EMA_RAMP)
    opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=WEIGHT_DECAY)
    paso, gastado, mejor = 0, 0.0, -1.0
    if estado is not None:
        ema.module.load_state_dict(estado["ema"])
        ema.updates = estado["ema_updates"]
        opt.load_state_dict(estado["opt"])
        paso, gastado, mejor = estado["step"], estado["elapsed"], estado["best_ap"]
        print(f"reanuda en el paso {paso} ({gastado / 3600:.2f} h gastadas)", flush=True)
    presupuesto = o.hours * 3600

    def guardar() -> None:
        _save(
            {
                "model": model.state_dict(),
                "ema": ema.module.state_dict(),
                "ema_updates": ema.updates,
                "opt": opt.state_dict(),
                "step": paso,
                "elapsed": gastado + time.monotonic() - t0,
                "best_ap": mejor,
            },
            ultimo,
        )

    def evaluar() -> None:
        nonlocal mejor
        t = time.monotonic()
        m = evaluate(ema.module, gt, o.data / "val2017", ids_val, device)
        linea = {"step": paso, "hours": (gastado + time.monotonic() - t0) / 3600, **m}
        with (o.out / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(linea) + "\n")
        if m["ap"] > mejor:
            mejor = m["ap"]
            _save(
                {k: v.detach().cpu() for k, v in ema.module.state_dict().items()}, o.out / "best.pt"
            )
        print(
            f"eval paso {paso}: AP person {m['ap']:.4f} AP50 {m['ap50']:.4f} "
            f"(mejor {mejor:.4f}, {time.monotonic() - t:.0f} s)",
            flush=True,
        )
        model.train()

    t0 = time.monotonic()
    ultimo_ckpt = t0
    terminado = False
    while not terminado:
        disponibles = [e for e in indice if (o.data / "train2017" / e["file"]).exists()]
        print(f"época: {len(disponibles)} imágenes de {len(indice)} en disco", flush=True)
        cargador = DataLoader(
            CocoPersonMosaic(disponibles, o.data / "train2017", MosaicConfig(alto, ancho)),
            batch_size=o.batch,
            shuffle=True,
            num_workers=o.workers,
            drop_last=True,
            worker_init_fn=_worker_init,
            prefetch_factor=4 if o.workers else None,
        )
        t_log, imgs_log = time.monotonic(), 0
        for imagenes, objetivos in cargador:
            progreso = (gastado + time.monotonic() - t0) / presupuesto
            if progreso >= 1 or (o.max_steps and paso >= o.max_steps):
                terminado = True
                break
            lr = lr_at(paso, progreso, o.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = imagenes.to(device).float() * PIXEL_SCALE
            objetivos_dev = {k: v.to(device) for k, v in objetivos.items()}
            perdida, partes = centernet_loss(model(x), objetivos_dev)
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ema.update(model)
            paso += 1
            imgs_log += len(imagenes)
            if paso % LOG_EVERY == 0 or paso == 1:
                ahora = time.monotonic()
                print(
                    f"paso {paso}: pérdida {float(perdida.detach()):.4f} "
                    f"(heatmap {float(partes['heatmap']):.4f}, size {float(partes['size']):.3f}, "
                    f"offset {float(partes['offset']):.3f}) lr {lr:.2e} "
                    f"{imgs_log / max(ahora - t_log, 1e-6):.1f} img/s "
                    f"{(gastado + ahora - t0) / 3600:.2f} h",
                    flush=True,
                )
                if not math.isfinite(float(perdida.detach())):
                    print("pérdida no finita: se para sin guardar", flush=True)
                    return 1
                t_log, imgs_log = ahora, 0
            if paso % o.eval_every == 0:
                evaluar()
            if time.monotonic() - ultimo_ckpt >= o.ckpt_min * 60:
                guardar()
                ultimo_ckpt = time.monotonic()
    guardar()
    evaluar()
    print(f"fin: {paso} pasos, mejor AP person {mejor:.4f}, en {o.out / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
