"""Verifica grabaciones del soporte y deja un informe JSON (ML-07).

    uv sync --group data --group ref
    uv run python tools/check_recording.py left-123.mov [right-123.mov] [-o informe.json]

Con un fichero comprueba códec, formato, cadencia, bitrate y el rigMs (legible y
monótono, leído con la referencia). Con dos, además el solape de rigMs entre
cámaras. Sale con 0 si todo está OK y con 1 si algo falló: el motivo exacto queda
en el JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Lanzado como script, `sys.path[0]` es tools/ y no la raiz: como en export_onnx.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ftrain.recording import check_pair_overlap, check_recording


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="la grabación (o la cámara izquierda)")
    parser.add_argument("right", type=Path, nargs="?", help="la cámara derecha, si hay pareja")
    parser.add_argument("-o", "--out", type=Path, help="dónde escribir el informe JSON")
    opciones = parser.parse_args(argv)

    izquierda = check_recording(opciones.left)
    documento: dict[str, object] = {"left": izquierda.to_json()}
    ok = izquierda.ok
    if opciones.right is not None:
        derecha = check_recording(opciones.right)
        pareja = check_pair_overlap(izquierda, derecha)
        documento["right"] = derecha.to_json()
        documento["pair"] = pareja
        ok = ok and derecha.ok and bool(pareja["ok"])
    documento["ok"] = ok

    texto = json.dumps(documento, ensure_ascii=False, indent=2) + "\n"
    if opciones.out is not None:
        opciones.out.write_text(texto, encoding="utf-8")
    print(texto, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
