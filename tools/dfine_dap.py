"""ΔAP de D-FINE-N entre torch fp32 y Core ML fp16, sobre COCO val2017 (ML-16).

    uv run python tools/dfine_dap.py \\
        --images <dir>/val2017 \\
        --annotations <dir>/instances_val2017.json \\
        --checkpoint models/pretrained/dfine_n_coco.pth \\
        --package dist/dfine-n-band.mlpackage \\
        --n 500 --out dist/dap.json

Las imágenes van con relleno y sin deformar (letterbox a la esquina, gris 114,
la de la receta DEIM) y el decode es el estándar DETR: top 300 del plano
query x clase. Informa el AP de las 80 clases y el de `person`, que es la que
nos llevamos (person → player, ADR 0020). Solo macOS: la pata de Core ML es la
que se mide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.players.dfine import BAND_SIZE, build

LETTERBOX_FILL = 114
"""El gris del relleno, el mismo de la receta DEIM/YOLO."""

TOP_DETECTIONS = 300
"""Detecciones por imagen que entran al eval: el estándar DETR en COCO."""

PERSON_CATEGORY = 1
"""La categoría COCO de `person`, la única que cruza al producto (ADR 0020)."""


def letterbox(imagen: Any, alto: int, ancho: int) -> tuple[np.ndarray, float]:
    """La imagen escalada sin deformar y pegada en (0,0) sobre gris. Y su escala."""
    from PIL import Image  # noqa: PLC0415 — llega con torchvision

    escala = min(ancho / imagen.width, alto / imagen.height)
    nuevo = (max(1, round(imagen.width * escala)), max(1, round(imagen.height * escala)))
    lienzo = Image.new("RGB", (ancho, alto), (LETTERBOX_FILL,) * 3)
    lienzo.paste(imagen.convert("RGB").resize(nuevo, Image.BILINEAR), (0, 0))
    return np.asarray(lienzo, dtype=np.uint8), escala


def decode(  # noqa: PLR0913 — la geometría entera del letterbox, todo con nombre
    logits: np.ndarray,
    boxes: np.ndarray,
    *,
    escala: float,
    alto: int,
    ancho: int,
    orig_w: int,
    orig_h: int,
    label2category: dict[int, int],
) -> list[dict[str, Any]]:
    """Top 300 del plano query x clase, devueltos en píxeles ORIGINALES (COCO xywh)."""
    puntuaciones = 1.0 / (1.0 + np.exp(-logits.reshape(-1).astype(np.float64)))
    clases = logits.shape[-1]
    mejores = np.argsort(-puntuaciones)[:TOP_DETECTIONS]
    detecciones = []
    for indice in mejores:
        consulta, clase = divmod(int(indice), clases)
        cx, cy, w, h = boxes[consulta].astype(np.float64)
        x1 = (cx - w / 2) * ancho / escala
        y1 = (cy - h / 2) * alto / escala
        ancho_px = w * ancho / escala
        alto_px = h * alto / escala
        x1 = float(np.clip(x1, 0, orig_w))
        y1 = float(np.clip(y1, 0, orig_h))
        detecciones.append(
            {
                "category_id": label2category[clase],
                "bbox": [
                    x1,
                    y1,
                    float(min(ancho_px, orig_w - x1)),
                    float(min(alto_px, orig_h - y1)),
                ],
                "score": float(puntuaciones[indice]),
            }
        )
    return detecciones


def _ap(gt: Any, resultados: list[dict[str, Any]], ids: list[int], cats: list[int] | None) -> float:
    from faster_coco_eval import COCOeval_faster  # noqa: PLC0415 — grupo train

    ev = COCOeval_faster(gt, gt.loadRes(resultados), "bbox")
    ev.params.imgIds = ids
    if cats is not None:
        ev.params.catIds = cats
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="el directorio val2017")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True, help="el .mlpackage fp16")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--size", default=f"{BAND_SIZE[0]}x{BAND_SIZE[1]}", help="alto x ancho")
    parser.add_argument("--out", type=Path, required=True)
    opciones = parser.parse_args(argv)

    import coremltools as ct  # noqa: PLC0415 — grupo apple
    import torch  # noqa: PLC0415 — grupo train
    from faster_coco_eval import COCO  # noqa: PLC0415 — grupo train
    from PIL import Image  # noqa: PLC0415

    alto, ancho = (int(v) for v in opciones.size.split("x"))
    red = build(opciones.checkpoint, size=(alto, ancho))
    from engine.data.dataset import mscoco_label2category  # noqa: PLC0415 — en third_party

    paquete = ct.models.MLModel(str(opciones.package), compute_units=ct.ComputeUnit.CPU_AND_NE)

    gt = COCO(str(opciones.annotations))
    ids = sorted(gt.getImgIds())[: opciones.n]
    resultados: dict[str, list[dict[str, Any]]] = {"torch_fp32": [], "coreml_fp16": []}
    for cuantos, imagen_id in enumerate(ids, start=1):
        info = gt.loadImgs([imagen_id])[0]
        ruta = opciones.images / info["file_name"]
        imagen = Image.open(ruta)
        lienzo, escala = letterbox(imagen, alto, ancho)

        tensor = torch.from_numpy(lienzo.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            logits, boxes = red(tensor)
        por_backend = {
            "torch_fp32": (logits[0].numpy(), boxes[0].numpy()),
        }
        salida = paquete.predict({"image": Image.fromarray(lienzo, "RGB")})
        por_backend["coreml_fp16"] = (
            np.asarray(salida["logits"])[0],
            np.asarray(salida["boxes"])[0],
        )
        for backend, (lg, bx) in por_backend.items():
            for det in decode(
                lg,
                bx,
                escala=escala,
                alto=alto,
                ancho=ancho,
                orig_w=info["width"],
                orig_h=info["height"],
                label2category=mscoco_label2category,
            ):
                resultados[backend].append({"image_id": imagen_id, **det})
        if cuantos % 50 == 0:
            print(f"{cuantos}/{len(ids)} imágenes", file=sys.stderr)

    informe: dict[str, Any] = {"samples": len(ids), "size": [alto, ancho]}
    for backend, res in resultados.items():
        informe[f"ap_{backend}"] = _ap(gt, res, ids, None)
        informe[f"ap_person_{backend}"] = _ap(gt, res, ids, [PERSON_CATEGORY])
    informe["dap"] = informe["ap_torch_fp32"] - informe["ap_coreml_fp16"]
    informe["dap_person"] = informe["ap_person_torch_fp32"] - informe["ap_person_coreml_fp16"]

    opciones.out.parent.mkdir(parents=True, exist_ok=True)
    opciones.out.write_text(json.dumps(informe, indent=2), encoding="utf-8")
    print(json.dumps(informe, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
