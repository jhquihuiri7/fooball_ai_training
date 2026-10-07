"""Baja la parte `person` de COCO 2017 para preentrenar CenterNet-MNv4 (ADR 0020, REF-33).

    uv run python tools/fetch_coco_person.py                # índice + val + todo train
    uv run python tools/fetch_coco_person.py --train 20000  # un subconjunto con semilla

Deja en `datasets/coco2017/` (en .gitignore):

- `annotations/instances_{train,val}2017.json`, del zip oficial;
- `person_train2017.json`: el índice compacto que lee el entreno. Una entrada por imagen
  con al menos una persona que no sea `iscrowd`, con sus cajas xywh en píxeles, las de
  multitud aparte y el id de la licencia Flickr de la imagen;
- `val2017/`: las `--val` primeras imágenes por id, el protocolo de `dfine_dap.py` (ML-16),
  para que el AP de persona se compare con el de D-FINE-N a 1920x576;
- `train2017/`: las imágenes del índice, en un orden barajado con semilla, de modo que lo
  bajado a medias ya es una muestra representativa;
- `PROCEDENCIA.txt`: de dónde sale y bajo qué términos.

Por qué COCO y en qué términos: las anotaciones son CC BY 4.0 y cada imagen lleva la
licencia Flickr que eligió su autor; el 69 % de las imágenes con personas son NC. El ADR
0002 acepta el preentreno COCO como riesgo residual (es el de `dfine_n_coco.pth`). El
índice guarda la licencia por imagen para poder entrenar solo con las comerciales
(`train_centernet.py --licencias 4,5,6,7,8`) si el propietario lo pide.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "coco2017"
BASE = "http://images.cocodataset.org"
ANNOTATIONS_ZIP = f"{BASE}/annotations/annotations_trainval2017.zip"
PERSON_CATEGORY = 1
"""La categoría COCO de `person`."""
SEED = 2026
"""La semilla del orden de bajada (y del subconjunto con --train)."""
TIMEOUT_S = 60
RETRIES = 4
WORKERS = 32
"""Conexiones a la vez: la bajada la limita el ancho de banda, no el servidor."""
PROVENANCE = """COCO 2017 (cocodataset.org), solo la categoría person.
Anotaciones: CC BY 4.0 (COCO Consortium). Imágenes: la licencia Flickr de cada una
(campo `license` del índice; 1-3 son NC, 4-8 admiten uso comercial).
Uso: preentreno de CenterNet-MNv4 (plan B de jugadores, ADR 0020). Riesgo residual
aceptado en el ADR 0002, el mismo de dfine_n_coco.pth. No es un dato propio.
Imágenes con persona en train2017: {n_train} ({n_comercial} con licencia comercial).
"""


def download(url: str, dest: Path) -> int:
    """Baja `url` a `dest` si no está ya, con reintentos. Devuelve los bytes."""
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for intento in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, tmp.open("wb") as f:  # noqa: S310 - URL fija de COCO
                while trozo := r.read(1 << 20):
                    f.write(trozo)
            tmp.rename(dest)
            return dest.stat().st_size
        except OSError:
            if intento == RETRIES - 1:
                raise
            time.sleep(2**intento)
    return 0


def person_index(instances: dict[str, Any]) -> list[dict[str, Any]]:
    """Las imágenes con alguna persona que no sea multitud, con sus cajas, por id."""
    imagenes = {i["id"]: i for i in instances["images"]}
    cajas: dict[int, list[list[float]]] = {}
    multitud: dict[int, list[list[float]]] = {}
    for a in instances["annotations"]:
        if a["category_id"] != PERSON_CATEGORY:
            continue
        destino = multitud if a["iscrowd"] else cajas
        destino.setdefault(a["image_id"], []).append([float(v) for v in a["bbox"]])
    return [
        {
            "id": i,
            "file": imagenes[i]["file_name"],
            "width": imagenes[i]["width"],
            "height": imagenes[i]["height"],
            "license": imagenes[i]["license"],
            "boxes": cajas[i],
            "crowd": multitud.get(i, []),
        }
        for i in sorted(cajas)
    ]


def _bajar_o_fallar(par: tuple[str, Path]) -> int:
    try:
        return download(*par)
    except OSError as e:
        print(f"falla {par[0]}: {e}", file=sys.stderr, flush=True)
        return -1


def _bajar_todas(urls: list[tuple[str, Path]], etiqueta: str) -> int:
    """Baja en paralelo; una imagen que falla no para las demás. Devuelve las fallidas."""
    hechas, fallidas, total_bytes, t0 = 0, 0, 0, time.monotonic()
    with ThreadPoolExecutor(WORKERS) as ex:
        for n in ex.map(_bajar_o_fallar, urls):
            hechas += 1
            fallidas += n < 0
            total_bytes += max(n, 0)
            if hechas % 1000 == 0 or hechas == len(urls):
                mb_s = total_bytes / 1e6 / max(time.monotonic() - t0, 1e-6)
                print(f"{etiqueta}: {hechas}/{len(urls)} ({mb_s:.1f} MB/s)", flush=True)
    return fallidas


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=0, help="cuántas de train (0: todas)")
    p.add_argument("--val", type=int, default=500, help="primeras N de val2017 por id")
    o = p.parse_args(argv)

    anotaciones = ROOT / "annotations"
    if not (anotaciones / "instances_train2017.json").exists():
        zip_path = ROOT / "annotations_trainval2017.zip"
        download(ANNOTATIONS_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for nombre in ("instances_train2017.json", "instances_val2017.json"):
                z.extract(f"annotations/{nombre}", ROOT)

    val = json.loads((anotaciones / "instances_val2017.json").read_text())
    ids_val = sorted(i["id"] for i in val["images"])[: o.val]
    nombres_val = {i["id"]: i["file_name"] for i in val["images"]}
    fallidas = _bajar_todas(
        [(f"{BASE}/val2017/{nombres_val[i]}", ROOT / "val2017" / nombres_val[i]) for i in ids_val],
        "val2017",
    )

    indice = person_index(json.loads((anotaciones / "instances_train2017.json").read_text()))
    (ROOT / "person_train2017.json").write_text(json.dumps(indice))
    comerciales = sum(1 for e in indice if e["license"] >= 4)  # noqa: PLR2004 - ids Flickr 4-8
    (ROOT / "PROCEDENCIA.txt").write_text(
        PROVENANCE.format(n_train=len(indice), n_comercial=comerciales)
    )
    orden = list(indice)
    random.Random(SEED).shuffle(orden)  # noqa: S311 - orden de bajada, no cripto
    if o.train:
        orden = orden[: o.train]
    fallidas += _bajar_todas(
        [(f"{BASE}/train2017/{e['file']}", ROOT / "train2017" / e["file"]) for e in orden],
        "train2017",
    )
    print(f"fallidas: {fallidas} (vuelve a lanzarlo: lo bajado no se repite)")
    return 1 if fallidas else 0


if __name__ == "__main__":
    sys.exit(main())
