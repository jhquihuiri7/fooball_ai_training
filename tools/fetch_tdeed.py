"""Trae T-DEED y lee su firma: qué forma tiene el clip que espera (TASK T1).

T-DEED es **GPL-3.0**. Se clona en `third_party/`, que está en `.gitignore`: se ejecuta,
no se vendoriza y no se redistribuye. Ver `docs/DEPENDENCIES.md`, donde está también lo
que queda por resolver antes de que algo suyo entre en un producto que se vende.

Lo que hace esta herramienta es lo que pide la T1: dejar el repo clonado y **leer de su
código** —no de lo que uno recuerde de un paper— el largo del clip, el ritmo de muestreo,
el tamaño del frame y el orden de las clases. Esos cuatro números son la mitad de la ficha
que el repo de detección va a exigir (su ADR 0013 §3), y son justo los que no se pueden
deducir mirando un `.onnx`.

    uv run python tools/fetch_tdeed.py

Los **pesos** no se pueden bajar desde aquí: están en un Google Drive y hay que cogerlos a
mano. La herramienta dice dónde y dónde ponerlos.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

TDEED_URL: Final = "https://github.com/arturxe2/T-DEED"
TDEED_DIR: Final = Path("third_party/T-DEED")
CHECKPOINTS_URL: Final = "https://drive.google.com/drive/folders/1sxZalU_hCwL8ITZCU9VqSWE8dB94lJty"
"""Los pesos publicados. Google Drive, a mano: no hay URL directa estable."""

DEFAULT_CONFIG: Final = "SoccerNetBall_challenge1"
"""La configuración de SoccerNet Ball Action Spotting, que es la que nos toca: sus clases
incluyen GOAL, SHOT y FREE KICK. **No incluye córner**; el córner está en la otra tarea,
la de 17 clases de SoccerNet-v2. Es una decisión que hay que tomar con esto delante."""

EXTRACTOR: Final = "extract_frames_snb.py"
"""De aquí sale el tamaño al que se extraen los frames para esta tarea."""

EVAL_SOURCE: Final = "util/eval.py"
"""De aquí sale el orden de las clases. Se **lee**, no se copia: si T-DEED lo cambia, esto
falla en vez de quedarse con una copia vieja, que es el error que el registro del repo de
detección existe para evitar."""

_CLASS_PATTERN: Final = re.compile(r'"([A-Z][A-Z ]*)"\s*:\s*(\d+)')
_SIZE_PATTERN: Final = re.compile(r"^TARGET_(HEIGHT|WIDTH)\s*=\s*(\d+)", re.MULTILINE)
_STRIDE_PATTERN: Final = re.compile(r"^STRIDE_SNB\s*=\s*(\d+)", re.MULTILINE)

EXTRACTED_FPS: Final = 25
"""Fps a los que su propio script documenta extraer los frames de esta tarea (el ejemplo
de su docstring es `--sample_fps 25`; el valor por defecto del argumento, 2, es el de la
otra tarea). Es el único de los cuatro números que no está en una constante del código,
así que **hay que confirmarlo** al correr la inferencia de verdad."""


class TdeedError(RuntimeError):
    """No se pudo traer o leer T-DEED. El mensaje dice qué falta."""


@dataclass(frozen=True, slots=True)
class Signature:
    """Lo que hay que saber para darle un clip a este modelo."""

    config: str
    clip_len: int
    stride: int
    height: int
    width: int
    classes: tuple[str, ...]
    feature_arch: str
    temporal_arch: str

    @property
    def sample_fps(self) -> float:
        """Ritmo real del clip: los frames extraídos, uno de cada `stride`."""
        return EXTRACTED_FPS / self.stride

    @property
    def window_s(self) -> float:
        """Cuántos segundos de partido ve el modelo de una vez."""
        return self.clip_len / self.sample_fps

    def as_lines(self) -> list[str]:
        clases = ", ".join(f"{n}={c}" for n, c in enumerate(self.classes))
        return [
            f"config    : {self.config}",
            (
                f"clip      : {self.clip_len} frames a {self.sample_fps:g} fps "
                f"= {self.window_s:g} s de ventana"
            ),
            f"frame     : {self.width}x{self.height}",
            f"backbone  : {self.feature_arch}  ·  temporal: {self.temporal_arch}",
            f"clases    : {len(self.classes)} — {clases}",
        ]


def clone(dest: Path = TDEED_DIR, *, url: str = TDEED_URL) -> str:
    """Clona T-DEED si no está. Devuelve el commit que quedó en disco.

    Superficial a propósito: interesa el código de hoy, no su historia, y así son unos
    megabytes en vez de decenas.
    """
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            msg = f"no se pudo clonar {url} en {dest}: {exc}"
            raise TdeedError(msg) from exc
    salida = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return salida.stdout.strip()


def read_classes(source: str) -> tuple[str, ...]:
    """Las clases de SoccerNet Ball, en el orden del modelo, leídas de su `eval.py`.

    El orden **es** el índice de salida: cambiarlo sin reexportar intercambia las
    etiquetas sin dar ningún error.
    """
    encontradas = {int(indice): nombre for nombre, indice in _CLASS_PATTERN.findall(source)}
    if not encontradas or sorted(encontradas) != list(range(len(encontradas))):
        msg = (
            f"no se pudieron leer las clases de {EVAL_SOURCE}: T-DEED habrá cambiado su "
            "`EVENT_DICTIONARY`. Hay que mirarlo antes de seguir, no adivinarlo"
        )
        raise TdeedError(msg)
    return tuple(encontradas[indice] for indice in sorted(encontradas))


def read_frame_size(source: str) -> tuple[int, int]:
    """`(alto, ancho)` al que su extractor deja los frames de esta tarea."""
    valores = dict(_SIZE_PATTERN.findall(source))
    if "HEIGHT" not in valores or "WIDTH" not in valores:
        msg = f"no se encontraron TARGET_HEIGHT/TARGET_WIDTH en {EXTRACTOR}"
        raise TdeedError(msg)
    return int(valores["HEIGHT"]), int(valores["WIDTH"])


def read_stride(source: str) -> int:
    """Uno de cada cuántos frames extraídos entra en el clip."""
    encontrado = _STRIDE_PATTERN.search(source)
    if encontrado is None:
        msg = "no se encontró STRIDE_SNB en inference.py"
        raise TdeedError(msg)
    return int(encontrado.group(1))


def _text(root: Path, relative: str) -> str:
    fichero = root / relative
    if not fichero.is_file():
        msg = f"falta {fichero}: ¿está T-DEED clonado? Corre esta herramienta sin argumentos"
        raise TdeedError(msg)
    return fichero.read_text(encoding="utf-8", errors="replace")


def signature(root: Path = TDEED_DIR, config: str = DEFAULT_CONFIG) -> Signature:
    """Lee del repo clonado todo lo que define la forma del clip."""
    ruta = root / "config" / config.split("_", 1)[0] / f"{config}.json"
    if not ruta.is_file():
        msg = f"no existe la configuración {ruta}"
        raise TdeedError(msg)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    alto, ancho = read_frame_size(_text(root, EXTRACTOR))
    return Signature(
        config=config,
        clip_len=int(datos["clip_len"]),
        stride=read_stride(_text(root, "inference.py")),
        height=alto,
        width=ancho,
        classes=read_classes(_text(root, EVAL_SOURCE)),
        feature_arch=str(datos["feature_arch"]),
        temporal_arch=str(datos["temporal_arch"]),
    )


def checkpoint_path(config: str = DEFAULT_CONFIG) -> Path:
    """Dónde espera `inference.py` el fichero de pesos, exactamente.

    Lo arma él como `checkpoints/<Dataset>/<config>/checkpoint_best.pt`, donde `<Dataset>`
    es el prefijo del nombre de la configuración. Ponerlo un nivel más arriba —que es lo
    que uno haría mirando la carpeta— falla con un `FileNotFoundError` a mitad de carga.
    """
    return Path("checkpoints") / config.split("_", 1)[0] / config / "checkpoint_best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_tdeed",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, default=TDEED_DIR, help="donde clonar T-DEED")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="que configuracion leer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        commit = clone(args.dir)
        firma = signature(args.dir, args.config)
    except TdeedError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"T-DEED    : {args.dir} @ {commit}  (GPL-3.0: se ejecuta, no se distribuye)")
    for linea in firma.as_lines():
        print(linea)
    print()
    print(f"pesos     : {CHECKPOINTS_URL}")
    print(f"            a mano, y como {args.dir / checkpoint_path(args.config)}")
    print(f"aviso     : los {EXTRACTED_FPS} fps de extraccion son lo que documenta su")
    print("            script, no una constante: confirmalos al correr la inferencia")
    return 0


if __name__ == "__main__":
    sys.exit(main())
