"""Lo que `tools/runpod.sh` le pide a RunPod y lo que lee de vuelta (ML-04).

    python tools/runpod_pod.py cuerpo                        # el JSON del pod, desde el entorno
    python tools/runpod_pod.py campo portMappings.22 < respuesta.json
    python tools/runpod_pod.py referencia [pyproject.toml]   # «<url> <rev>» de football-ai

Está aparte del shell para poder probarlo sin red, y para que, si RunPod cambia el esquema,
se vea de un vistazo qué se le pide: `runpod.sh` imprime este cuerpo junto al error de la API.
Solo usa la biblioteca estándar, así que `runpod.sh` lo lanza con el `python` que haya en la
máquina y no con `uv run`. Es el mismo reparto que `deploy/_crear.py` y `deploy/_json.py`
del repo de detección.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

IMAGEN: Final = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
"""La del pod del directo en el repo de detección: arranca su propio `sshd` con la clave de
la cuenta, que es por donde entra `runpod.sh`. Su torch no se usa: `uv sync` pone el del lock."""

GPU: Final = "NVIDIA GeForce RTX 4090"
"""La de entrenar del ADR 0002 §3 (RTX 4090 o L40S). Para una prueba vale una más barata."""

NUBES: Final = ("SECURE", "COMMUNITY")
"""Los dos tipos de nube de RunPod. Por defecto la segura, que trae IP pública para el ssh."""

DISCO_GB: Final = 40
"""Disco del contenedor, en GB: la caché de uv con torch y las bibliotecas de CUDA pasa de 10."""

VOLUMEN_GB: Final = 60
"""Volumen en `/workspace`, en GB: el repo con su `.venv` y la caché de los datasets."""

MONTAJE: Final = "/workspace"
"""Donde se monta el volumen. `runpod_bootstrap.sh` trabaja ahí."""

PUERTO_SSH: Final = "22/tcp"
"""El único puerto: por él se suben los repos y se corre el arranque."""

PAQUETE_REFERENCIA: Final = "football-ai"
"""El nombre de la referencia en `[tool.uv.sources]` de pyproject.toml (ML-03)."""

ARGUMENTOS_CAMPO: Final = 2
"""`campo` y la ruta del campo."""


def tipos_de_gpu(texto: str) -> list[str]:
    """Las GPU que valen, separadas por comas, en orden de preferencia."""
    tipos = [tipo.strip() for tipo in texto.split(",") if tipo.strip()]
    if not tipos:
        msg = f"GPU_TYPE vacío: hace falta al menos un tipo, p. ej. «{GPU}»"
        raise ValueError(msg)
    return tipos


def _gigas(entorno: Mapping[str, str], variable: str, defecto: int) -> int:
    texto = entorno.get(variable, str(defecto))
    if not texto.isdigit() or int(texto) == 0:
        msg = f"{variable} tiene que ser un número de GB mayor que cero, no «{texto}»"
        raise ValueError(msg)
    return int(texto)


def cuerpo(entorno: Mapping[str, str]) -> dict[str, object]:
    """Lo que se le manda a `POST /v1/pods`."""
    nube = entorno.get("CLOUD_TYPE", NUBES[0])
    if nube not in NUBES:
        msg = f"CLOUD_TYPE es {' o '.join(NUBES)}, no «{nube}»"
        raise ValueError(msg)
    return {
        "name": entorno.get("POD_NAME", "football-ai-training"),
        "imageName": entorno.get("POD_IMAGE", IMAGEN),
        "gpuTypeIds": tipos_de_gpu(entorno.get("GPU_TYPE", GPU)),
        # En el orden de GPU_TYPE: si no, RunPod elige y puede dar la más cara de la lista.
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "computeType": "GPU",
        "cloudType": nube,
        "supportPublicIp": True,
        "containerDiskInGb": _gigas(entorno, "DISK_GB", DISCO_GB),
        "volumeInGb": _gigas(entorno, "VOLUME_GB", VOLUMEN_GB),
        "volumeMountPath": MONTAJE,
        "ports": [PUERTO_SSH],
    }


def campo(texto: str, ruta: str) -> str:
    """Un campo anidado (`a.b.c`) de una respuesta JSON, o "" si no está.

    La cadena vacía es «todavía no»: `runpod.sh` sigue esperando, y si nunca llega enseña la
    respuesta entera, que dice más que un error de aquí. Lo mismo si no es JSON.
    """
    try:
        dato: object = json.loads(texto)
    except json.JSONDecodeError:
        return ""
    for parte in ruta.split("."):
        dato = dato.get(parte) if isinstance(dato, dict) else None
        if dato is None:
            return ""
    return str(dato)


def referencia(pyproject: str) -> tuple[str, str]:
    """La URL y el commit de football-ai que fija `[tool.uv.sources]`.

    `runpod.sh` sube esa copia desde la máquina local: el repo es privado y el pod no la
    puede bajar de GitHub.
    """
    fuentes = tomllib.loads(pyproject).get("tool", {}).get("uv", {}).get("sources", {})
    fuente = fuentes.get(PAQUETE_REFERENCIA)
    if not isinstance(fuente, dict) or not fuente.get("git"):
        msg = f"pyproject.toml no fija {PAQUETE_REFERENCIA} por git en [tool.uv.sources]"
        raise ValueError(msg)
    if not fuente.get("rev"):
        msg = f"{PAQUETE_REFERENCIA} en [tool.uv.sources] no lleva rev: hace falta un commit"
        raise ValueError(msg)
    return str(fuente["git"]), str(fuente["rev"])


def main(argv: list[str] | None = None) -> int:
    argumentos = sys.argv[1:] if argv is None else argv
    orden = argumentos[0] if argumentos else ""
    try:
        if orden == "cuerpo":
            print(json.dumps(cuerpo(os.environ)))
        elif orden == "campo" and len(argumentos) == ARGUMENTOS_CAMPO:
            print(campo(sys.stdin.read(), argumentos[1]))
        elif orden == "referencia":
            ruta = Path(argumentos[1] if len(argumentos) > 1 else "pyproject.toml")
            url, rev = referencia(ruta.read_text(encoding="utf-8"))
            print(url, rev)
        else:
            sys.stderr.write("uso: runpod_pod.py cuerpo | campo <a.b.c> | referencia [pyproject]\n")
            return 2
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
