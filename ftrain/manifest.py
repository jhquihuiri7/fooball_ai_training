"""El manifiesto de un partido grabado (ML-06).

Cada partido de la campaña lleva un `matches/<match_id>.yaml` que dice CÓMO se grabó:
sin eso, un dataset es un montón de vídeos de procedencia dudosa y la matriz de la
campaña (canchas, luces, retranqueos) no se puede auditar. El manifiesto se valida al
leerlo y cada error nombra el campo: lo rellena una persona en la banda, de noche.

La división por partidos completos (CLAUDE.md §1) se decide aquí con `split_hint`:
una cancha `sagrada` jamás entra en train, pase lo que pase río abajo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "LIGHTS",
    "ROTATIONS_DEG",
    "SIDES",
    "SPLIT_HINTS",
    "ManifestError",
    "MatchManifest",
    "SideRecording",
]

LIGHTS: Final = ("day", "floodlight", "dusk")
"""Las tres luces de la matriz de la campaña. `dusk` es la hora mala a propósito."""

SPLIT_HINTS: Final = ("train", "val", "sagrada")
"""A qué lado de la división va el partido ENTERO. `sagrada` = jamás en train."""

ROTATIONS_DEG: Final = (0, 180)
"""Cómo va montado cada móvil en el soporte: derecho, o cabeza abajo (ADR 0012)."""

SIDES: Final = ("left", "right")


class ManifestError(ValueError):
    """El manifiesto no vale. El mensaje nombra el campo y el match_id si lo hay."""


def _text(data: Mapping[str, object], key: str, where: str) -> str:
    valor = data.get(key)
    if not isinstance(valor, str) or not valor.strip():
        msg = f"{where}: `{key}` tiene que ser un texto no vacío y es {valor!r}"
        raise ManifestError(msg)
    return valor.strip()


def _number(data: Mapping[str, object], key: str, where: str, *, minimo: float) -> float:
    valor = data.get(key)
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        msg = f"{where}: `{key}` tiene que ser un número y es {valor!r}"
        raise ManifestError(msg)
    if valor < minimo:
        msg = f"{where}: `{key}` no puede bajar de {minimo} y es {valor}"
        raise ManifestError(msg)
    return float(valor)


def _gs_uri(data: Mapping[str, object], key: str, where: str) -> str:
    valor = _text(data, key, where)
    if not valor.startswith("gs://"):
        msg = f"{where}: `{key}` tiene que ser una URI gs:// y es «{valor}»"
        raise ManifestError(msg)
    return valor


@dataclass(frozen=True, slots=True)
class SideRecording:
    """Lo grabado por un móvil del soporte."""

    phone_model: str
    ios_version: str
    tilt_deg: float
    rotation_deg: int
    recording_uri: str
    nv12_uri: str | None

    @classmethod
    def from_dict(cls, data: object, where: str) -> SideRecording:
        if not isinstance(data, dict):
            msg = f"{where}: cada lado tiene que ser un objeto y es {type(data).__name__}"
            raise ManifestError(msg)
        rotacion = data.get("rotation_deg")
        if isinstance(rotacion, bool) or rotacion not in ROTATIONS_DEG:
            msg = (
                f"{where}: `rotation_deg` tiene que ser 0 o 180 (cómo va montado el "
                f"móvil) y es {rotacion!r}"
            )
            raise ManifestError(msg)
        nv12 = data.get("nv12_uri")
        if nv12 is not None:
            nv12 = _gs_uri(data, "nv12_uri", where)
        return cls(
            phone_model=_text(data, "phone_model", where),
            ios_version=_text(data, "ios_version", where),
            tilt_deg=_number(data, "tilt_deg", where, minimo=-90.0),
            rotation_deg=int(rotacion),
            recording_uri=_gs_uri(data, "recording_uri", where),
            nv12_uri=nv12,
        )


@dataclass(frozen=True, slots=True)
class MatchManifest:
    """Un partido de la campaña, auditable: dónde, cómo y con qué se grabó."""

    match_id: str
    venue_id: str
    date: str
    light: str
    setback_m: float
    height_m: float
    sides: dict[str, SideRecording]
    rig_uri: str
    band_uri: str
    split_hint: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: object) -> MatchManifest:
        if not isinstance(data, dict):
            msg = f"el manifiesto tiene que ser un objeto YAML y es {type(data).__name__}"
            raise ManifestError(msg)
        match_id = _text(data, "match_id", "manifiesto")
        where = f"manifiesto `{match_id}`"

        luz = data.get("light")
        if luz not in LIGHTS:
            msg = f"{where}: `light` tiene que ser {', '.join(LIGHTS)} y es {luz!r}"
            raise ManifestError(msg)
        pista = data.get("split_hint")
        if pista not in SPLIT_HINTS:
            msg = f"{where}: `split_hint` tiene que ser {', '.join(SPLIT_HINTS)} y es {pista!r}"
            raise ManifestError(msg)

        lados_crudos = data.get("sides")
        if not isinstance(lados_crudos, dict) or set(lados_crudos) != set(SIDES):
            msg = f"{where}: `sides` tiene que llevar exactamente left y right"
            raise ManifestError(msg)
        lados = {
            lado: SideRecording.from_dict(lados_crudos[lado], f"{where}, lado {lado}")
            for lado in SIDES
        }

        calibracion = data.get("calibration")
        if not isinstance(calibracion, dict):
            msg = f"{where}: falta el bloque `calibration` (rig_uri y band_uri)"
            raise ManifestError(msg)

        return cls(
            match_id=match_id,
            venue_id=_text(data, "venue_id", where),
            date=_text(data, "date", where),
            light=str(luz),
            setback_m=_number(data, "setback_m", where, minimo=0.0),
            height_m=_number(data, "height_m", where, minimo=0.0),
            sides=lados,
            rig_uri=_gs_uri(calibracion, "rig_uri", f"{where}, calibration"),
            band_uri=_gs_uri(calibracion, "band_uri", f"{where}, calibration"),
            split_hint=str(pista),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def load(cls, path: Path) -> MatchManifest:
        try:
            crudo = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            msg = f"no se pudo leer el manifiesto {path}: {exc}"
            raise ManifestError(msg) from exc
        except yaml.YAMLError as exc:
            msg = f"el manifiesto {path} no es YAML válido: {exc}"
            raise ManifestError(msg) from exc
        return cls.from_dict(crudo)
