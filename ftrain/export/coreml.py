"""Export a Core ML: mlprogram fp16, iOS 18, formas fijas y sha estable (ML-09).

Tres cosas que este módulo garantiza y que un `ct.convert` a pelo no:

1. **La entrada declarada es la del entrenamiento**, leída de un YAML — ImageType con
   su orden de color y la escala 1/255, o TensorType fp16 — nunca de la memoria de
   nadie (CLAUDE.md §1).
2. **El paquete lleva sus metadatos**: clases, color, escala, región, frames, versión
   del dataset y commit, en `user_defined_metadata`. Un `.mlpackage` que aparece en
   un iPhone dentro de un año tiene que poder contarse a sí mismo.
3. **El sha256 es estable**: dos exports del mismo modelo dan el mismo fichero. Core
   ML mete UUIDs aleatorios en el manifiesto del paquete y el zip hereda mtimes del
   disco; aquí los UUIDs se reescriben de forma determinista (uuid5 de la ruta) y el
   zip fija orden, fecha y atributos. Es lo que permite que la ficha v2 del registro
   ancle el artefacto por hash.

La conversión en sí solo corre donde hay coremltools (Linux/macOS); el zip y los
metadatos son puros y se prueban en cualquier sitio.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "ExportError",
    "ExportSpec",
    "InputSpec",
    "build_metadata",
    "convert",
    "deterministic_zip",
    "load_spec",
    "normalize_manifest",
    "sha256_of",
]

MINIMUM_IOS: Final = 18
"""Target de despliegue del mlprogram (ADR 0020): por debajo no está el runtime."""

IMAGE_SCALE: Final = 1.0 / 255.0
"""La escala de un ImageType: el píxel a [0,1]. D-FINE no normaliza más (ADR 0020)."""

_ZIP_DATE: Final = (1980, 1, 1, 0, 0, 0)
"""La fecha fija de cada entrada del zip: el mtime del disco no toca el sha."""

_UUID_NAMESPACE: Final = uuid.UUID("f00b0a11-2026-4000-8000-000000000000")
"""Espacio de nombres propio para los uuid5 del manifiesto normalizado."""


class ExportError(ValueError):
    """El spec o el export no valen. El mensaje nombra el campo."""


# --------------------------------------------------------------------------- #
# El spec del export, en YAML
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class InputSpec:
    name: str
    kind: str  # image | tensor
    shape: tuple[int, ...]
    color: str = "RGB"  # el orden del ENTRENAMIENTO, no una preferencia

    @classmethod
    def from_dict(cls, data: object, where: str) -> InputSpec:
        if not isinstance(data, dict):
            msg = f"{where}: cada entrada tiene que ser un objeto"
            raise ExportError(msg)
        kind = data.get("kind")
        if kind not in ("image", "tensor"):
            msg = f"{where}: `kind` tiene que ser image o tensor y es {kind!r}"
            raise ExportError(msg)
        forma = data.get("shape")
        if (
            not isinstance(forma, list)
            or not forma
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in forma)
        ):
            msg = f"{where}: `shape` tiene que ser una lista de enteros positivos"
            raise ExportError(msg)
        color = str(data.get("color", "RGB"))
        if kind == "image" and color not in ("RGB", "BGR"):
            msg = f"{where}: `color` tiene que ser RGB o BGR y es {color!r}"
            raise ExportError(msg)
        nombre = data.get("name")
        if not isinstance(nombre, str) or not nombre:
            msg = f"{where}: falta `name`"
            raise ExportError(msg)
        return cls(name=nombre, kind=str(kind), shape=tuple(forma), color=color)


@dataclass(frozen=True, slots=True)
class ExportSpec:
    """Lo que el YAML del export declara. Los nombres de salida son EL CONTRATO."""

    model_name: str
    inputs: tuple[InputSpec, ...]
    output_names: tuple[str, ...]
    metadata: dict[str, str] = field(default_factory=dict)
    states: tuple[str, ...] = ()
    functions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> ExportSpec:
        if not isinstance(data, dict):
            msg = "el spec tiene que ser un objeto YAML"
            raise ExportError(msg)
        nombre = data.get("model_name")
        if not isinstance(nombre, str) or not nombre:
            msg = "spec: falta `model_name`"
            raise ExportError(msg)
        where = f"spec `{nombre}`"
        crudas = data.get("inputs")
        if not isinstance(crudas, list) or not crudas:
            msg = f"{where}: `inputs` tiene que ser una lista no vacía"
            raise ExportError(msg)
        salidas = data.get("outputs")
        if (
            not isinstance(salidas, list)
            or not salidas
            or any(not isinstance(s, str) or not s for s in salidas)
        ):
            msg = f"{where}: `outputs` tiene que ser la lista de nombres del contrato"
            raise ExportError(msg)
        metadatos = data.get("metadata", {})
        if not isinstance(metadatos, dict):
            msg = f"{where}: `metadata` tiene que ser un mapa"
            raise ExportError(msg)
        estados = data.get("states", [])
        if not isinstance(estados, list):
            msg = f"{where}: `states` tiene que ser una lista"
            raise ExportError(msg)
        funciones = data.get("functions", {})
        if not isinstance(funciones, dict):
            msg = f"{where}: `functions` tiene que ser un mapa nombre→builder"
            raise ExportError(msg)
        return cls(
            model_name=nombre,
            inputs=tuple(
                InputSpec.from_dict(cruda, f"{where}, entrada {i}")
                for i, cruda in enumerate(crudas)
            ),
            output_names=tuple(salidas),
            metadata={str(k): str(v) for k, v in metadatos.items()},
            states=tuple(str(e) for e in estados),
            functions={str(k): str(v) for k, v in funciones.items()},
        )


def load_spec(path: Path) -> ExportSpec:
    try:
        crudo = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"no se pudo leer el spec {path}: {exc}"
        raise ExportError(msg) from exc
    return ExportSpec.from_dict(crudo)


# --------------------------------------------------------------------------- #
# Metadatos y empaquetado determinista (puros: corren en Windows)
# --------------------------------------------------------------------------- #
def build_metadata(  # noqa: PLR0913 — la ficha completa, todo con nombre y obligatorio
    spec: ExportSpec,
    *,
    classes: Sequence[str],
    dataset_version: str,
    commit: str,
    region: str,
    frames: int,
) -> dict[str, str]:
    """`user_defined_metadata`: todo texto, porque Core ML solo guarda cadenas."""
    primera = spec.inputs[0]
    return {
        "classes": ",".join(classes),
        "color": primera.color if primera.kind == "image" else "tensor",
        "scale": repr(IMAGE_SCALE) if primera.kind == "image" else "1.0",
        "region": region,
        "frames": str(frames),
        "dataset_version": dataset_version,
        "commit": commit,
        **spec.metadata,
    }


def normalize_manifest(package_dir: Path) -> None:
    """Reescribe los UUIDs aleatorios del `Manifest.json` como uuid5 de su ruta.

    Es lo que hace estable el sha256 entre dos exports idénticos. El paquete sigue
    siendo válido: los identificadores solo tienen que ser únicos dentro del
    manifiesto, y el de la raíz apunta a una entrada existente."""
    manifiesto = package_dir / "Manifest.json"
    if not manifiesto.is_file():
        msg = f"no parece un .mlpackage: falta {manifiesto}"
        raise ExportError(msg)
    datos = json.loads(manifiesto.read_text(encoding="utf-8"))
    entradas = datos.get("itemInfoEntries", {})
    raiz_vieja = datos.get("rootModelIdentifier")
    nuevo_por_viejo: dict[str, str] = {}
    nuevas: dict[str, object] = {}
    for viejo, item in sorted(entradas.items(), key=lambda par: str(par[1].get("path", ""))):
        ruta = str(item.get("path", viejo))
        nuevo = str(uuid.uuid5(_UUID_NAMESPACE, ruta))
        nuevo_por_viejo[viejo] = nuevo
        nuevas[nuevo] = item
    datos["itemInfoEntries"] = nuevas
    if raiz_vieja in nuevo_por_viejo:
        datos["rootModelIdentifier"] = nuevo_por_viejo[raiz_vieja]
    manifiesto.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def deterministic_zip(package_dir: Path, out_zip: Path) -> str:
    """Zipea el `.mlpackage` con orden, fecha y atributos fijos. Devuelve su sha256."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    rutas = sorted(ruta for ruta in package_dir.rglob("*") if ruta.is_file())
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for ruta in rutas:
            relativa = f"{package_dir.name}/{ruta.relative_to(package_dir).as_posix()}"
            info = zipfile.ZipInfo(relativa, date_time=_ZIP_DATE)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, ruta.read_bytes())
    return sha256_of(out_zip)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fichero:
        while trozo := fichero.read(1 << 20):
            digest.update(trozo)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# La conversión (solo donde hay coremltools)
# --------------------------------------------------------------------------- #
def convert(
    module: Any,  # nn.Module; Any para no arrastrar torch a los módulos puros
    spec: ExportSpec,
    metadata: Mapping[str, str],
) -> Any:
    """`nn.Module` → mlprogram fp16 iOS 18, con las entradas y salidas del spec.

    Primero `torch.export` (el camino nuevo); si el grafo no lo traga, `jit.trace`
    de respaldo, que es lo que ML-05 validó en las tres plataformas."""
    import coremltools as ct  # noqa: PLC0415 — perezoso adrede (grupo apple)
    import torch  # noqa: PLC0415 — perezoso adrede (grupo train)

    if spec.states or spec.functions:
        # El spec ya los declara para que el YAML no cambie, pero implementarlos
        # llega con ML-51/SPK-53 (StateType) y SPK-52 (multifunción).
        msg = f"spec `{spec.model_name}`: states/functions aún sin implementar (ML-51, SPK-52/53)"
        raise ExportError(msg)

    module = module.eval()
    ejemplos = tuple(torch.zeros(entrada.shape) for entrada in spec.inputs)

    entradas_ct = []
    for entrada in spec.inputs:
        if entrada.kind == "image":
            entradas_ct.append(
                ct.ImageType(
                    name=entrada.name,
                    shape=entrada.shape,
                    scale=IMAGE_SCALE,
                    color_layout=ct.colorlayout.RGB
                    if entrada.color == "RGB"
                    else ct.colorlayout.BGR,
                )
            )
        else:
            import numpy as np  # noqa: PLC0415 — junto a sus consumidores

            entradas_ct.append(
                ct.TensorType(name=entrada.name, shape=entrada.shape, dtype=np.float16)
            )

    def _convertir(trazado: Any) -> Any:
        return ct.convert(
            trazado,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=getattr(ct.target, f"iOS{MINIMUM_IOS}"),
            inputs=entradas_ct,
        )

    try:
        # El camino nuevo entero: torch.export y su frontend. run_decompositions({})
        # baja el grafo al dialecto ATEN, que es el que coremltools acepta.
        modelo = _convertir(torch.export.export(module, ejemplos).run_decompositions({}))
    except Exception:  # noqa: BLE001 — el respaldo es parte del contrato
        # El respaldo cubre TAMBIÉN los fallos de ct.convert sobre el ExportedProgram
        # (p. ej. el frontend nuevo no traga el linear 3D del decoder de D-FINE):
        # el frontend de TorchScript es el maduro y el que ML-05 validó.
        with torch.no_grad():
            trazado = torch.jit.trace(module, ejemplos)
        modelo = _convertir(trazado)

    # Los nombres de salida son el contrato: se renombra lo que haya salido.
    # get_spec() devuelve una COPIA: se pide una vez y se trabaja sobre ella.
    especificacion = modelo.get_spec()
    actuales = [salida.name for salida in especificacion.description.output]
    if len(actuales) != len(spec.output_names):
        msg = (
            f"el modelo tiene {len(actuales)} salidas y el spec declara "
            f"{len(spec.output_names)}: {', '.join(spec.output_names)}"
        )
        raise ExportError(msg)
    for vieja, nueva in zip(actuales, spec.output_names, strict=True):
        if vieja != nueva:
            ct.utils.rename_feature(especificacion, vieja, nueva)
    modelo = ct.models.MLModel(especificacion, weights_dir=modelo.weights_dir)

    for clave, valor in metadata.items():
        modelo.user_defined_metadata[clave] = valor
    return modelo
