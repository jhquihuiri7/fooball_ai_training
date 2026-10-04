"""Vectores dorados legibles desde Swift: el bundle golden/<modelo>-<versión>/ (ML-12).

El bundle es lo que XCTest compara contra Core ML en el iPhone:

- `manifest.json` — formas, dtype, layout y, por salida y por ruta, su tolerancia
  OBLIGATORIA (ORT fp32 y Core ML fp16 no comparten atol y olvidarlo es un bug).
- `entradas/*.bin` y `salidas/<ruta>/*.bin` — arrays little-endian en orden C,
  con las entradas tal como las entrega la app (BGRA uint8 los jugadores, fp16
  planar en gris el balón).
- `detecciones.json` — lo que decide la referencia (ftrain/ref) sobre esas
  salidas, para que Swift pruebe también su postproceso.
- `README.md` — con el lector Swift de ~30 líneas.

El zip es el mismo determinista de ML-09: el sha256 ancla el bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .coreml import deterministic_zip

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = [
    "MANIFEST_VERSION",
    "ROUTES",
    "GoldenArray",
    "GoldenError",
    "pack_bundle",
    "read_bundle",
    "to_bgra8",
    "write_bundle",
]

MANIFEST_VERSION: Final = 1
"""La versión del formato del bundle; Swift la comprueba antes de leer nada."""

ROUTES: Final = ("torch_fp32", "ort_fp32", "coreml_fp16")
"""Las rutas posibles. torch fp32 es la referencia; las demás llevan tolerancia."""

TOLERANCE_ROUTES: Final = ("ort_fp32", "coreml_fp16")

_DTYPES: Final = {
    np.dtype(np.float32): "<f4",
    np.dtype(np.float16): "<f2",
    np.dtype(np.uint8): "|u1",
    np.dtype(np.int32): "<i4",
}
"""Los dtypes del formato, explícitos y little-endian. i64 no existe aquí, igual
que en los dorados v1 de la referencia: Swift lee Int32."""

_HWC_RANK: Final = 3
_RGB_CHANNELS: Final = 3

_BGRA_ALPHA: Final = 255
"""El alfa que la app entrega en los frames BGRA: opaco siempre."""


class GoldenError(ValueError):
    """El bundle no vale. El mensaje nombra el array o la tolerancia que falta."""


@dataclass(frozen=True, slots=True)
class GoldenArray:
    """Un array del bundle con el layout que Swift necesita saber."""

    name: str
    array: np.ndarray
    layout: str  # p. ej. "BGRA8", "NCHW", "gray_fp16_planar"


def to_bgra8(rgb: np.ndarray) -> np.ndarray:
    """HWC RGB uint8 → HWC BGRA uint8, tal como la app entrega los frames."""
    if rgb.ndim != _HWC_RANK or rgb.shape[-1] != _RGB_CHANNELS or rgb.dtype != np.uint8:
        msg = f"to_bgra8 espera HWC RGB uint8 y recibió {rgb.dtype} {rgb.shape}"
        raise GoldenError(msg)
    alto, ancho, _ = rgb.shape
    bgra = np.empty((alto, ancho, 4), dtype=np.uint8)
    bgra[..., 0] = rgb[..., 2]
    bgra[..., 1] = rgb[..., 1]
    bgra[..., 2] = rgb[..., 0]
    bgra[..., 3] = _BGRA_ALPHA
    return bgra


# --------------------------------------------------------------------------- #
# Escritura
# --------------------------------------------------------------------------- #
def _dtype_str(array: np.ndarray, donde: str) -> str:
    etiqueta = _DTYPES.get(array.dtype)
    if etiqueta is None:
        soportados = ", ".join(sorted(v for v in _DTYPES.values()))
        msg = f"{donde}: dtype {array.dtype} no está en el formato ({soportados})"
        raise GoldenError(msg)
    return etiqueta


def _write_array(ruta: Path, array: np.ndarray, etiqueta: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(np.ascontiguousarray(array).astype(np.dtype(etiqueta)).tobytes(order="C"))


def _check_tolerances(
    outputs: Mapping[str, Sequence[GoldenArray]],
    tolerances: Mapping[str, Mapping[str, float]],
) -> None:
    nombres = [salida.name for salida in outputs["torch_fp32"]]
    for ruta in outputs:
        if ruta not in ROUTES:
            msg = f"ruta desconocida {ruta!r}: las del formato son {', '.join(ROUTES)}"
            raise GoldenError(msg)
        if [s.name for s in outputs[ruta]] != nombres:
            msg = f"la ruta {ruta} no tiene las mismas salidas que torch_fp32: {nombres}"
            raise GoldenError(msg)
    for nombre in nombres:
        for ruta in TOLERANCE_ROUTES:
            if ruta in outputs and tolerances.get(nombre, {}).get(ruta) is None:
                msg = f"falta la tolerancia de {nombre!r} por la ruta {ruta}"
                raise GoldenError(msg)


def write_bundle(  # noqa: PLR0913 — el contrato del bundle entero, todo con nombre
    out_dir: Path,
    *,
    model: str,
    version: str,
    inputs: Sequence[GoldenArray],
    outputs: Mapping[str, Sequence[GoldenArray]],
    tolerances: Mapping[str, Mapping[str, float]],
    detections: Any = None,
) -> Path:
    """Escribe golden/<modelo>-<versión>/ bajo `out_dir` y devuelve su ruta.

    `outputs` va por ruta y tiene que incluir `torch_fp32`, la referencia.
    `tolerances` va por salida y por ruta, y es obligatoria para cada ruta
    presente que no sea torch."""
    if "torch_fp32" not in outputs:
        msg = "falta la ruta torch_fp32: sin referencia no hay dorado"
        raise GoldenError(msg)
    _check_tolerances(outputs, tolerances)

    bundle = out_dir / f"{model}-{version}"
    manifiesto: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "model": model,
        "model_version": version,
        "inputs": [],
        "outputs": {},
        "tolerances": {nombre: dict(rutas) for nombre, rutas in sorted(tolerances.items())},
    }
    for entrada in inputs:
        etiqueta = _dtype_str(entrada.array, f"entrada {entrada.name}")
        fichero = f"entradas/{entrada.name}.bin"
        _write_array(bundle / fichero, entrada.array, etiqueta)
        manifiesto["inputs"].append(
            {
                "name": entrada.name,
                "file": fichero,
                "dtype": etiqueta,
                "shape": list(entrada.array.shape),
                "layout": entrada.layout,
            }
        )
    for ruta, salidas in sorted(outputs.items()):
        manifiesto["outputs"][ruta] = []
        for salida in salidas:
            etiqueta = _dtype_str(salida.array, f"salida {salida.name} ({ruta})")
            fichero = f"salidas/{ruta}/{salida.name}.bin"
            _write_array(bundle / fichero, salida.array, etiqueta)
            manifiesto["outputs"][ruta].append(
                {
                    "name": salida.name,
                    "file": fichero,
                    "dtype": etiqueta,
                    "shape": list(salida.array.shape),
                    "layout": salida.layout,
                }
            )
    if detections is not None:
        (bundle / "detecciones.json").write_text(
            json.dumps(detections, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (bundle / "manifest.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (bundle / "README.md").write_text(_readme(model, version), encoding="utf-8")
    return bundle


# --------------------------------------------------------------------------- #
# Lectura y empaquetado
# --------------------------------------------------------------------------- #
def read_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """El manifiesto y los arrays: {"inputs": {n: arr}, "outputs": {ruta: {n: arr}}}."""
    manifiesto = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifiesto.get("version") != MANIFEST_VERSION:
        msg = f"versión de manifiesto {manifiesto.get('version')!r}: esta lee {MANIFEST_VERSION}"
        raise GoldenError(msg)

    def _leer(item: Mapping[str, Any]) -> np.ndarray:
        crudo = (bundle / str(item["file"])).read_bytes()
        return np.frombuffer(crudo, dtype=np.dtype(str(item["dtype"]))).reshape(item["shape"])

    arrays: dict[str, Any] = {
        "inputs": {str(item["name"]): _leer(item) for item in manifiesto["inputs"]},
        "outputs": {
            ruta: {str(item["name"]): _leer(item) for item in items}
            for ruta, items in manifiesto["outputs"].items()
        },
    }
    return manifiesto, arrays


def pack_bundle(bundle: Path, out_zip: Path) -> str:
    """El zip determinista del bundle (el de ML-09). Devuelve su sha256."""
    return deterministic_zip(bundle, out_zip)


def _readme(model: str, version: str) -> str:
    return f"""# Dorado `{model}-{version}`

Arrays little-endian en orden C. `manifest.json` lleva forma, dtype, layout y la
tolerancia por salida y por ruta (`ort_fp32`, `coreml_fp16`). Lector Swift:

```swift
import Foundation

struct GoldenItem: Decodable {{
    let name: String, file: String, dtype: String, shape: [Int], layout: String
}}
struct Manifest: Decodable {{
    let version: Int, model: String, inputs: [GoldenItem]
    let outputs: [String: [GoldenItem]]
    let tolerances: [String: [String: Double]]
}}

struct Golden {{
    let dir: URL
    let manifest: Manifest

    init(dir: URL) throws {{
        self.dir = dir
        let datos = try Data(contentsOf: dir.appending(path: "manifest.json"))
        self.manifest = try JSONDecoder().decode(Manifest.self, from: datos)
        precondition(manifest.version == {MANIFEST_VERSION})
    }}

    func floats(_ item: GoldenItem) throws -> [Float] {{
        let datos = try Data(contentsOf: dir.appending(path: item.file))
        switch item.dtype {{
        case "<f4": return datos.withUnsafeBytes {{ Array($0.bindMemory(to: Float32.self)) }}
        case "<f2": return datos.withUnsafeBytes {{
            $0.bindMemory(to: Float16.self).map(Float.init) }}
        case "|u1": return datos.map(Float.init)
        case "<i4": return datos.withUnsafeBytes {{
            $0.bindMemory(to: Int32.self).map(Float.init) }}
        default: fatalError("dtype \\(item.dtype)")
        }}
    }}
}}
```
"""
