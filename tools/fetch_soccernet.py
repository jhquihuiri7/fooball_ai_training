"""Trae los datos de SoccerNet, y dice antes lo que va a costar (TASK T2).

Hay **dos tareas distintas** y elegir mal cuesta semanas:

| Tarea | Clases | ¿Córner? | De dónde |
|---|---|---|---|
| `spotting-ball-2024` | 12, centradas en el balón | **No** | Hugging Face, ~19 GB |
| `spotting-2023` | 17, acciones de partido | **Sí** | SoccerNet, con contraseña del NDA |

Las cuatro clases del MVP son gol, tiro, córner y tiro libre. Las tres primeras menos el
córner están en `spotting-ball-2024`, que es además la tarea para la que T-DEED publica
sus pesos. El córner solo está en `spotting-2023`. **No hay una sola descarga que dé las
cuatro**, y eso hay que decidirlo con esto delante, no a mitad de camino.

    uv run python tools/fetch_soccernet.py --check            # el plan, sin bajar nada
    uv run python tools/fetch_soccernet.py --task spotting-ball-2024 --split valid

La contraseña del NDA, cuando hace falta, va en `SOCCERNET_PASSWORD`. Nunca en un
argumento: los argumentos los ve cualquiera que liste los procesos de la máquina.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_DIR: Final = Path("datasets/soccernet")
"""Dónde caen los datos. Está en `.gitignore`: son decenas de gigabytes con términos de
uso propios y no tienen nada que hacer en un repositorio."""

PASSWORD_ENV: Final = "SOCCERNET_PASSWORD"
"""Variable con la contraseña que SoccerNet manda por correo tras firmar su NDA."""

NDA_URL: Final = "https://www.soccer-net.org/data"
"""De donde sale el formulario. Es una acción humana: hay que firmarlo y esperar."""


@dataclass(frozen=True, slots=True)
class Task:
    """Una tarea de SoccerNet y lo que cuesta traerla."""

    name: str
    classes: int
    has_corner: bool
    source: str
    needs_password: bool
    size_gb: float
    notes: str

    def plan(self, splits: list[str]) -> list[str]:
        """Qué va a pasar si se descarga, en líneas para una persona."""
        corner = "sí" if self.has_corner else "no"
        lineas = [
            f"tarea     : {self.name} ({self.classes} clases, córner: {corner})",
            f"origen    : {self.source}",
            f"splits    : {', '.join(splits)}",
            f"tamaño    : ~{self.size_gb:g} GB el conjunto completo",
            f"nota      : {self.notes}",
        ]
        if self.needs_password:
            lineas.append(f"acceso    : firma el NDA en {NDA_URL} y exporta {PASSWORD_ENV}")
        else:
            lineas.append("acceso    : cuenta de Hugging Face; acepta los términos del dataset")
        return lineas


TASKS: Final = {
    "spotting-ball-2024": Task(
        name="spotting-ball-2024",
        classes=12,
        has_corner=False,
        source="Hugging Face · SoccerNet/SN-BAS-2024",
        needs_password=False,
        size_gb=19.3,
        notes=(
            "la tarea para la que T-DEED publica pesos. El dataset se declara GPL-3.0, "
            "que para unos datos es raro y hay que mirarlo antes de vender nada"
        ),
    ),
    "spotting-2023": Task(
        name="spotting-2023",
        classes=17,
        has_corner=True,
        source="servidor de SoccerNet",
        needs_password=True,
        size_gb=50.0,
        notes="la única con córner; sus vídeos son de transmisión de TV profesional",
    ),
}
"""Solo las dos que importan aquí. El paquete `SoccerNet` conoce muchas más."""

DEFAULT_TASK: Final = "spotting-ball-2024"
DEFAULT_SPLITS: Final = ("train", "valid", "test")


class SoccerNetError(RuntimeError):
    """No se puede descargar. El mensaje dice qué falta."""


def resolve(task: str) -> Task:
    """La tarea por su nombre, o un error que dice cuáles hay."""
    elegida = TASKS.get(task)
    if elegida is None:
        msg = f"tarea desconocida: {task} (aquí se usan {', '.join(TASKS)})"
        raise SoccerNetError(msg)
    return elegida


def password_for(task: Task, env: dict[str, str] | None = None) -> str | None:
    """La contraseña del NDA si esa tarea la necesita. Falla pronto si falta."""
    if not task.needs_password:
        return None
    entorno = os.environ if env is None else env
    clave = entorno.get(PASSWORD_ENV, "").strip()
    if not clave:
        msg = (
            f"{task.name} necesita la contraseña del NDA. Fírmalo en {NDA_URL} y expórtala "
            f"en {PASSWORD_ENV}; no la pases por argumento, que la ve quien liste procesos"
        )
        raise SoccerNetError(msg)
    return clave


def download(task: Task, splits: list[str], directory: Path) -> None:
    """Descarga de verdad. Se importa aquí dentro para que `--check` no necesite nada."""
    from SoccerNet.Downloader import SoccerNetDownloader  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    descargador = SoccerNetDownloader(LocalDirectory=str(directory))
    clave = password_for(task)
    if clave is not None:
        descargador.password = clave
    descargador.downloadDataTask(task=task.name, split=splits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_soccernet",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="que tarea traer")
    parser.add_argument("--split", action="append", default=None, metavar="SPLIT", help="repetible")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="donde dejarlo")
    parser.add_argument(
        "--check", action="store_true", help="ensenar el plan y salir sin descargar nada"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    splits = args.split or list(DEFAULT_SPLITS)
    try:
        tarea = resolve(args.task)
        for linea in tarea.plan(splits):
            print(linea)
        print(f"destino   : {args.dir}")
        if args.check:
            return 0
        password_for(tarea)
        download(tarea, splits, args.dir)
    except SoccerNetError as exc:
        print(f"ERROR: {exc}")
        return 1
    print("descargado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
