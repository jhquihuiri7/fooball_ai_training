"""Guardia de licencias (ML-02, ADR 0002).

Lo que este repo entrega va dentro de la app, y tiene que poder venderse. La guardia
recorre `ftrain/` y `tools/` con `ast` y falla si algo trae código que no se puede
distribuir:

- `ultralytics`, que es AGPL-3.0;
- T-DEED, que es GPL-3.0: sus paquetes `model` y `util`, o su carpeta `third_party/T-DEED`.

Las únicas excepciones son las dos herramientas de la línea base, congeladas y sin
mantenimiento. Los cuadernos quedan fuera del recorrido: son esa misma línea base y no se
entregan.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

CARPETAS = ("ftrain", "tools")
"""Lo que se recorre, relativo a la raíz del repo."""

EXCEPCIONES = frozenset({"tools/fetch_tdeed.py", "tools/export_onnx.py"})
"""Las herramientas de la línea base de T-DEED (ADR 0002 §4). Ninguna otra."""

PROHIBIDOS = {
    "ultralytics": "ultralytics (AGPL-3.0)",
    "model": "T-DEED (GPL-3.0)",
    "util": "T-DEED (GPL-3.0)",
}
"""Paquete de primer nivel y por qué no puede entrar. `model` y `util` son los de T-DEED,
que se importan así cuando su carpeta está en `sys.path`."""

RUTA_TDEED = re.compile(r"(^|[/\\])(third_party|T-DEED)([/\\]|$)", re.IGNORECASE)
"""Una cadena que nombra la carpeta de T-DEED como parte de una ruta. «T-DEED» dentro de
una frase no cuenta."""


def _prohibido(modulo: str) -> str | None:
    return PROHIBIDOS.get(modulo.partition(".")[0])


def _es_import_dinamico(funcion: ast.expr) -> bool:
    if isinstance(funcion, ast.Name):
        return funcion.id in {"__import__", "import_module"}
    return isinstance(funcion, ast.Attribute) and funcion.attr == "import_module"


def _revisar_nodo(nodo: ast.AST, prosa: set[int]) -> list[tuple[int, str]]:
    if isinstance(nodo, ast.Import):
        return [
            (nodo.lineno, f"import {alias.name}: {motivo}")
            for alias in nodo.names
            if (motivo := _prohibido(alias.name))
        ]
    # `level > 0` es un import relativo: `from .model import X` es de este paquete.
    if isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
        if motivo := _prohibido(nodo.module):
            return [(nodo.lineno, f"from {nodo.module}: {motivo}")]
        return []
    if (
        isinstance(nodo, ast.Call)
        and _es_import_dinamico(nodo.func)
        and nodo.args
        and isinstance(nodo.args[0], ast.Constant)
        and isinstance(nodo.args[0].value, str)
        and (motivo := _prohibido(nodo.args[0].value))
    ):
        return [(nodo.lineno, f"import dinámico de {nodo.args[0].value!r}: {motivo}")]
    if (
        isinstance(nodo, ast.Constant)
        and isinstance(nodo.value, str)
        and id(nodo) not in prosa
        and RUTA_TDEED.search(nodo.value)
    ):
        return [(nodo.lineno, f"ruta a T-DEED {nodo.value!r}: T-DEED (GPL-3.0)")]
    return []


def _revisar(fichero: Path) -> list[tuple[int, str]]:
    arbol = ast.parse(fichero.read_text(encoding="utf-8"), filename=str(fichero))
    # Una cadena suelta como sentencia (un docstring) no hace nada: citar la ruta ahí es
    # documentarla, no usarla.
    prosa = {
        id(nodo.value)
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Expr) and isinstance(nodo.value, ast.Constant)
    }
    return sorted(h for nodo in ast.walk(arbol) for h in _revisar_nodo(nodo, prosa))


def infracciones(raiz: Path) -> list[str]:
    """Todo lo que la guardia rechaza bajo `raiz`, como `ruta:línea: motivo`."""
    encontradas: list[str] = []
    for carpeta in CARPETAS:
        for fichero in sorted((raiz / carpeta).rglob("*.py")):
            relativa = fichero.relative_to(raiz).as_posix()
            if relativa in EXCEPCIONES:
                continue
            encontradas.extend(
                f"{relativa}:{linea}: {motivo}" for linea, motivo in _revisar(fichero)
            )
    return encontradas


def _escribir(raiz: Path, relativa: str, codigo: str) -> None:
    fichero = raiz / relativa
    fichero.parent.mkdir(parents=True, exist_ok=True)
    fichero.write_text(codigo, encoding="utf-8")


# --------------------------------------------------------------------------- #
# El repo tal como está
# --------------------------------------------------------------------------- #


def test_el_repo_pasa_la_guardia() -> None:
    assert infracciones(RAIZ) == []


def test_las_excepciones_existen() -> None:
    # Si una se renombra o se borra, la excepción se quedaría abierta para otro fichero.
    assert all((RAIZ / relativa).is_file() for relativa in EXCEPCIONES)


# --------------------------------------------------------------------------- #
# Lo que rechaza
# --------------------------------------------------------------------------- #

RECHAZADOS = [
    "import ultralytics",
    "import ultralytics.engine.model",
    "from ultralytics import YOLO",
    "from model.model import TDEEDModel",
    "import model.model",
    "from util.io import load_json",
    "from util import dataset",
    "def cargar():\n    from model.model import TDEEDModel\n    return TDEEDModel\n",
    "import importlib; importlib.import_module('ultralytics')",
    "from importlib import import_module; import_module('util.io')",
    "__import__('model.model')",
    "from pathlib import Path; RAIZ = Path('third_party/T-DEED')",
    "from pathlib import Path; RAIZ = Path('third_party') / 'T-DEED'",
    "import sys; sys.path.insert(0, 'C:\\\\trabajo\\\\T-DEED')",
]


@pytest.mark.parametrize("carpeta", CARPETAS)
@pytest.mark.parametrize("codigo", RECHAZADOS)
def test_rechaza(tmp_path: Path, carpeta: str, codigo: str) -> None:
    _escribir(tmp_path, f"{carpeta}/nuevo.py", codigo)

    encontradas = infracciones(tmp_path)

    assert encontradas
    assert all(e.startswith(f"{carpeta}/nuevo.py:") for e in encontradas)


def test_dice_el_fichero_la_linea_y_el_motivo(tmp_path: Path) -> None:
    _escribir(tmp_path, "ftrain/export.py", "import numpy\n\nfrom ultralytics import YOLO\n")

    esperado = "ftrain/export.py:3: from ultralytics: ultralytics (AGPL-3.0)"

    assert infracciones(tmp_path) == [esperado]


def test_busca_en_subcarpetas(tmp_path: Path) -> None:
    _escribir(tmp_path, "ftrain/ball/loss.py", "from model.model import TDEEDModel")

    assert infracciones(tmp_path) == ["ftrain/ball/loss.py:1: from model.model: T-DEED (GPL-3.0)"]


# --------------------------------------------------------------------------- #
# Lo que deja pasar
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("relativa", sorted(EXCEPCIONES))
def test_las_dos_herramientas_de_la_linea_base_quedan_fuera(tmp_path: Path, relativa: str) -> None:
    _escribir(tmp_path, relativa, "from model.model import TDEEDModel\nimport ultralytics\n")

    assert infracciones(tmp_path) == []


@pytest.mark.parametrize(
    "codigo",
    [
        "from .model import Detector",
        "from . import util",
        "import models",
        "import utils",
        "import timm",
        '"""Nada de aquí toca third_party/T-DEED."""',
        'RUTA = 1\n"""La carpeta third_party/T-DEED no se toca."""\n',
        'MENSAJE = "la tarea para la que T-DEED publica pesos"',
    ],
)
def test_deja_pasar(tmp_path: Path, codigo: str) -> None:
    _escribir(tmp_path, "ftrain/nuevo.py", codigo)

    assert infracciones(tmp_path) == []


@pytest.mark.parametrize(
    "relativa",
    ["notebooks/linea_base.py", "tests/test_algo.py", "third_party/T-DEED/model/model.py"],
)
def test_solo_mira_ftrain_y_tools(tmp_path: Path, relativa: str) -> None:
    _escribir(tmp_path, relativa, "import ultralytics\nfrom model.model import TDEEDModel\n")

    assert infracciones(tmp_path) == []
