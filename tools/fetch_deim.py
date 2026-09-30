"""Trae DEIM v1 al commit fijado y los pesos COCO de D-FINE-N (ML-15).

DEIM (Apache-2.0) es la receta con la que se entrena D-FINE-N para los jugadores
(ADR 0002 §2). Lo controla Intellindust, el mismo titular que relicenció DEIMv2 como no
comercial el 2026-08-24, así que aquí no se da nada por hecho:

- el repo se clona **a un commit fijado**, no a la cabeza de `main`;
- su LICENSE se compara con la copia versionada en `licenses/`, y si cambió una sola línea
  la herramienta se para: un relicenciado se decide con una persona delante, no se hereda;
- los pesos de partida son `dfine_n_coco.pth`, de las releases de D-FINE, comprobados por
  su sha256. Nunca `*_obj365` ni `*_obj2coco` (ADR 0020 del repo de detección).

Todo acaba en carpetas que git ignora: `third_party/DEIM` y `models/pretrained/`.

    uv run python tools/fetch_deim.py
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEIM_URL: Final = "https://github.com/ShihuaHuang95/DEIM"
DEIM_COMMIT: Final = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
"""El commit de DEIM v1 que se usa. Moverlo es una decisión: exige leer su LICENSE, copiarla
a `licenses/` con el nombre nuevo y anotarlo en `docs/DEPENDENCIES.md`."""

DEIM_DIR: Final = Path("third_party/DEIM")
LICENSE_PIN: Final = Path("licenses/DEIM-LICENSE-09d35d5.txt")
"""La LICENSE de DEIM tal como estaba en `DEIM_COMMIT`, versionada aquí."""

WEIGHTS_DIR: Final = Path("models/pretrained")
FORBIDDEN_WEIGHTS: Final = ("obj365", "obj2coco")
"""Pesos de D-FINE preentrenados con Objects365, cuyos términos no admiten uso comercial."""

_CHUNK: Final = 1 << 20
"""Bytes por lectura al descargar y al calcular el sha256: 1 MiB."""


@dataclass(frozen=True, slots=True)
class Weights:
    """Un fichero de pesos de partida, con el sha256 con el que se fijó."""

    name: str
    url: str
    sha256: str


DFINE_N_COCO: Final = Weights(
    name="dfine_n_coco.pth",
    url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth",
    sha256="41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c",
)
"""D-FINE-N entrenado en COCO (42,8 AP, 4 M de parámetros). 15.489.558 bytes."""


class DeimError(RuntimeError):
    """No se pudo traer DEIM o sus pesos tal como están fijados. El mensaje dice qué falta."""


def check_weights_name(name: str) -> None:
    """Rechaza los pesos vetados, por su nombre, antes de bajar nada."""
    vetado = next((f for f in FORBIDDEN_WEIGHTS if f in name), None)
    if vetado is not None:
        msg = f"{name}: los pesos *_{vetado} están vetados (Objects365 no es comercial)"
        raise DeimError(msg)


def _git(*args: str) -> str:
    try:
        salida = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detalle = getattr(exc, "stderr", "") or exc
        msg = f"git {' '.join(args)}: {detalle}"
        raise DeimError(msg) from exc
    return salida.stdout.strip()


def clone(dest: Path = DEIM_DIR, *, url: str = DEIM_URL, commit: str = DEIM_COMMIT) -> str:
    """Deja `dest` en `commit`, limpio, y devuelve el hash de su árbol.

    Si ya está clonado, se mueve al commit fijado. Si alguien tocó el árbol a mano, se para:
    un DEIM con cambios locales ya no es el de la licencia que se comprobó.
    """
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--no-checkout", url, str(dest))
        _git("-C", str(dest), "checkout", "--quiet", "--detach", commit)
    elif _git("-C", str(dest), "rev-parse", "HEAD") != commit:
        _git("-C", str(dest), "fetch", "--quiet", "origin")
        _git("-C", str(dest), "checkout", "--quiet", "--detach", commit)
    if _git("-C", str(dest), "status", "--porcelain"):
        msg = f"{dest} tiene cambios locales: bórralo y vuelve a correr la herramienta"
        raise DeimError(msg)
    return _git("-C", str(dest), "rev-parse", "HEAD^{tree}")


def _normalized(text: str) -> list[str]:
    # Con `core.autocrlf` el clon puede llegar con CRLF y la copia con LF, o al revés.
    return text.replace("\r\n", "\n").splitlines()


def check_license(clone_dir: Path = DEIM_DIR, pin: Path = LICENSE_PIN) -> None:
    """Para si la LICENSE del clon no es, línea a línea, la versionada en `pin`."""
    if not pin.is_file():
        msg = f"falta {pin}: la LICENSE de DEIM se versiona antes de usarlo"
        raise DeimError(msg)
    licencia = clone_dir / "LICENSE"
    if not licencia.is_file():
        msg = f"falta {licencia}: ¿está DEIM clonado?"
        raise DeimError(msg)
    esperada = _normalized(pin.read_text(encoding="utf-8"))
    actual = _normalized(licencia.read_text(encoding="utf-8"))
    if actual != esperada:
        diferencia = "\n".join(
            list(difflib.unified_diff(esperada, actual, str(pin), str(licencia), lineterm=""))[:12]
        )
        msg = f"la LICENSE de DEIM cambió; no se usa hasta revisarla:\n{diferencia}"
        raise DeimError(msg)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fichero:
        while bloque := fichero.read(_CHUNK):
            digest.update(bloque)
    return digest.hexdigest()


def download(
    weights: Weights = DFINE_N_COCO,
    dest_dir: Path = WEIGHTS_DIR,
    *,
    opener: Callable[[str], BinaryIO] = urllib.request.urlopen,
) -> Path:
    """Baja `weights` a `dest_dir` y comprueba su sha256. Si ya están y cuadran, no baja nada.

    Se escribe primero en un `.part`: si el sha no cuadra, se borra y no queda en disco un
    fichero con el nombre bueno y el contenido malo.
    """
    check_weights_name(weights.name)
    if not weights.url.startswith("https://"):
        msg = f"{weights.url}: solo se baja por https"
        raise DeimError(msg)
    destino = dest_dir / weights.name
    if destino.is_file() and sha256_file(destino) == weights.sha256:
        return destino
    dest_dir.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".part")
    digest = hashlib.sha256()
    with opener(weights.url) as respuesta, parcial.open("wb") as fichero:
        while bloque := respuesta.read(_CHUNK):
            digest.update(bloque)
            fichero.write(bloque)
    if digest.hexdigest() != weights.sha256:
        parcial.unlink()
        msg = (
            f"{weights.name}: sha256 {digest.hexdigest()}, se esperaba {weights.sha256}. "
            "Los pesos publicados cambiaron: no se usan hasta revisarlos"
        )
        raise DeimError(msg)
    parcial.replace(destino)
    return destino


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEIM_DIR, help="dónde clonar DEIM")
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--skip-weights", action="store_true", help="solo el código")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        arbol = clone(args.dir)
        check_license(args.dir)
        print(f"DEIM      : {args.dir} @ {DEIM_COMMIT[:7]}, árbol {arbol[:12]}")
        print(f"licencia  : igual a {LICENSE_PIN} (Apache-2.0)")
        if not args.skip_weights:
            pesos = download(dest_dir=args.weights_dir)
            print(f"pesos     : {pesos}, sha256 {DFINE_N_COCO.sha256[:12]}… comprobado")
    except DeimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
