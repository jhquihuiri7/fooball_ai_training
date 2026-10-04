"""La referencia Python de football-ai, fijada por commit (ML-03).

Lo que este repo necesita del de detección —leer el código de tiempo pintado, la homografía
del campo, el postproceso de los detectores, el registro de modelos y el lienzo de la
franja— se importa de allí y nunca se copia. Así Python sigue siendo una sola referencia, y
la que prepara los datos es la misma que fija los dorados de la app. La versión la fija el
sha del grupo `ref` en `pyproject.toml`, y es el único modo de moverla:

    uv sync --group ref

Los nombres se resuelven al pedirlos (PEP 562). Sin el grupo, importar este módulo no
falla: falla pedir un nombre, con `RefMissingError` y el comando que lo arregla.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

ORIGINS: Final = {
    "read_timecode_ms": "libs.vision.source.timecode",
    "PitchModel": "libs.vision.pitch",
    "decode_boxes_to_corners": "libs.vision.postprocess",
    "sigmoid": "libs.vision.postprocess",
    "heatmap_peaks": "libs.vision.postprocess",
    "load_registry": "libs.vision.registry",
    "compose_band_input": "libs.vision.band",
    # ML-19: el paso nativo↔lienzo usa la MISMA geometría que la app, no una copia.
    "BandGeometry": "libs.vision.band",
    "recover_camera": "libs.vision.band",
    "strip_height": "libs.vision.source.timecode",
    "TIMECODE_BITS": "libs.vision.source.timecode",
}
"""Cada nombre que expone la fachada, con el módulo de `libs.vision` del que sale."""


class RefMissingError(ImportError):
    """Falta la referencia de football-ai: el grupo `ref` no está instalado."""


def __getattr__(nombre: str) -> Any:
    modulo = ORIGINS.get(nombre)
    if modulo is None:
        msg = f"ftrain.ref no expone {nombre!r}"
        raise AttributeError(msg)
    try:
        return getattr(import_module(modulo), nombre)
    except ModuleNotFoundError as exc:
        msg = f"falta {exc.name} de la referencia de football-ai: `uv sync --group ref`"
        raise RefMissingError(msg) from exc
