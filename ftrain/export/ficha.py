"""La ficha v2 de registry.yaml, generada desde los artefactos (ML-13).

Generaliza el `registry_block` de tools/export_onnx.py: lo que importa es lo que
hay en los ficheros —formas, nombres y opset se leen del `.onnx`; target y
precisión, del spec del `.mlpackage`— y no la configuración con la que se
exportaron. La ficha que sale de aquí la carga `load_registry` del commit fijado
de football-ai (vía ftrain/ref): sus reglas de coherencia se comprueban AQUÍ, al
construir, para que el error salte donde se puede arreglar.

Nota: la tarjeta menciona la región `analysis_zone`, pero el lector de la
referencia solo admite full_frame, playable_band y roi; manda la referencia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import yaml

from .coreml import sha256_of

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "CoremlArtifact",
    "Ficha",
    "FichaError",
    "InputBlock",
    "OnnxArtifact",
    "OutputBlock",
    "ParityBlock",
    "coreml_facts",
    "onnx_facts",
    "render_registry",
]

REGISTRY_VERSION: Final = 2
"""La versión del formato que esta ficha escribe (REF-20)."""

POSTPROCESSES: Final = ("detr", "nms", "heatmap")
REGIONS: Final = ("full_frame", "playable_band", "roi")
GRAY_TEMPORAL: Final = "gray_temporal"
STACKED_FRAMES: Final = (1, 3)
COMPUTE_UNITS: Final = "cpu_and_ne"
"""El único `compute_units` del contrato: CPU y ANE, sin GPU (ADR 0020)."""

_SPEC_VERSION_TO_IOS: Final = {7: 16, 8: 17, 9: 18, 10: 19}
"""`specificationVersion` del proto de Core ML → versión mínima de iOS."""


class FichaError(ValueError):
    """La ficha no cuadra. El mensaje nombra el campo, como hace el lector."""


# --------------------------------------------------------------------------- #
# Los bloques de la ficha
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class InputBlock:
    name: str
    shape: tuple[int, ...]
    color: str = "RGB"
    layout: str = "NCHW"
    scale: float = 255
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    region: str = "full_frame"
    frames: int = 1
    sample_fps: float | None = None


@dataclass(frozen=True, slots=True)
class OutputBlock:
    name: str
    shape: tuple[int, ...]
    meaning: str


@dataclass(frozen=True, slots=True)
class OnnxArtifact:
    path: str  # relativo a models/, p. ej. onnx/<nombre>.onnx
    sha256: str
    opset: int


@dataclass(frozen=True, slots=True)
class CoremlArtifact:
    path: str  # p. ej. coreml/<nombre>.mlpackage.zip
    sha256: str
    min_ios: int
    ane_cost_pct: float  # obligatorio: el lector fijado lo exige (sale de SPK-50/ML-43)
    compute_units: str = COMPUTE_UNITS
    ops_off_ane: int = 0


@dataclass(frozen=True, slots=True)
class ParityBlock:
    golden: str  # p. ej. golden/<nombre>-<versión>.golden.zip
    atol: float
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Ficha:
    """Una entrada de `models:` lista para que la cargue la referencia."""

    name: str
    version: str
    postprocess: str
    box_format: str
    input: InputBlock
    outputs: tuple[OutputBlock, ...]
    classes: tuple[str, ...]
    license: str
    onnx: OnnxArtifact
    task: str = "detection"
    precision: str = "fp16"
    coreml: CoremlArtifact | None = None
    parity: ParityBlock | None = None
    heatmap_stride: int | None = None
    nms_iou: float | None = None
    dataset: str | None = None
    commit: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    exported: str | None = None  # ISO; sin él, hoy

    def _check(self) -> None:
        sitio = f"ficha `{self.name}`"
        if self.postprocess not in POSTPROCESSES:
            msg = f"{sitio}: `postprocess` es «{self.postprocess}» y vale {POSTPROCESSES}"
            raise FichaError(msg)
        es_heatmap = self.postprocess == "heatmap"
        if es_heatmap != (self.box_format == "heatmap_stride"):
            msg = (
                f"{sitio}: `box_format` «{self.box_format}» no cuadra con `postprocess` "
                f"«{self.postprocess}»: `heatmap_stride` va con `heatmap` y solo con él"
            )
            raise FichaError(msg)
        if es_heatmap != (self.heatmap_stride is not None):
            msg = f"{sitio}: `heatmap_stride` es obligatorio con `heatmap` y solo con él"
            raise FichaError(msg)
        if (self.postprocess == "nms") != (self.nms_iou is not None):
            msg = f"{sitio}: `nms_iou` es obligatorio con `nms` y solo con él"
            raise FichaError(msg)
        if self.input.region not in REGIONS:
            msg = f"{sitio}: `region` es «{self.input.region}» y vale {REGIONS}"
            raise FichaError(msg)
        if self.input.frames not in STACKED_FRAMES:
            msg = f"{sitio}: `frames` tiene que ser {STACKED_FRAMES} y es {self.input.frames}"
            raise FichaError(msg)
        if self.input.frames == 3 and self.input.color != GRAY_TEMPORAL:  # noqa: PLR2004
            msg = f"{sitio}: con `frames: 3` el `color` tiene que ser `{GRAY_TEMPORAL}`"
            raise FichaError(msg)
        if self.coreml is not None and self.coreml.compute_units != COMPUTE_UNITS:
            msg = f"{sitio}: `compute_units` solo vale `{COMPUTE_UNITS}` (ADR 0020)"
            raise FichaError(msg)

    def to_entry(self) -> dict[str, Any]:
        """El diccionario de la entrada, validado con las reglas del lector."""
        self._check()
        entrada: dict[str, Any] = {
            "version": self.version,
            "task": self.task,
            "postprocess": self.postprocess,
            "box_format": self.box_format,
            "precision": self.precision,
            "artifacts": {
                "onnx": {
                    "path": self.onnx.path,
                    "sha256": self.onnx.sha256,
                    "opset": self.onnx.opset,
                }
            },
            "license": self.license,
            "exported": self.exported or datetime.now(UTC).date().isoformat(),
            "input": self._input_entry(),
            "outputs": [
                {"name": s.name, "shape": list(s.shape), "meaning": s.meaning} for s in self.outputs
            ],
            "classes": list(self.classes),
        }
        if self.heatmap_stride is not None:
            entrada["heatmap_stride"] = self.heatmap_stride
        if self.nms_iou is not None:
            entrada["nms_iou"] = self.nms_iou
        if self.coreml is not None:
            entrada["artifacts"]["coreml"] = {
                "path": self.coreml.path,
                "sha256": self.coreml.sha256,
                "min_ios": self.coreml.min_ios,
                "compute_units": self.coreml.compute_units,
                "ops_off_ane": self.coreml.ops_off_ane,
                "ane_cost_pct": self.coreml.ane_cost_pct,
            }
        if self.parity is not None:
            bloque = {"golden": self.parity.golden, "atol": self.parity.atol}
            if self.parity.sha256 is not None:
                bloque["sha256"] = self.parity.sha256
            entrada["parity"] = bloque
        if self.dataset is not None:
            entrada["dataset"] = self.dataset
        if self.commit is not None:
            entrada["commit"] = self.commit
        if self.metrics:
            entrada["metrics"] = dict(self.metrics)
        return entrada

    def _input_entry(self) -> dict[str, Any]:
        bloque: dict[str, Any] = {
            "name": self.input.name,
            "shape": list(self.input.shape),
            "layout": self.input.layout,
            "color": self.input.color,
            "scale": self.input.scale,
            "mean": list(self.input.mean),
            "std": list(self.input.std),
            "region": self.input.region,
            "frames": self.input.frames,
        }
        if self.input.sample_fps is not None:
            bloque["sample_fps"] = self.input.sample_fps
        return bloque


def render_registry(fichas: dict[str, Ficha]) -> str:
    """El documento `registry.yaml` entero, listo para `load_registry`."""
    documento = {
        "version": REGISTRY_VERSION,
        "models": {nombre: ficha.to_entry() for nombre, ficha in fichas.items()},
    }
    return yaml.safe_dump(documento, sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# Lo que se lee de los artefactos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OnnxFacts:
    input_name: str
    input_shape: tuple[int, ...]
    outputs: tuple[tuple[str, tuple[int, ...]], ...]
    opset: int
    sha256: str


def onnx_facts(path: Path) -> OnnxFacts:
    """Formas, nombres, opset y sha, leídos del `.onnx` y no de ninguna config."""
    import onnx  # noqa: PLC0415 — junto a su consumidor

    modelo = onnx.load(str(path), load_external_data=False)
    entrada = modelo.graph.input[0]
    forma = tuple(d.dim_value for d in entrada.type.tensor_type.shape.dim)
    salidas = tuple(
        (s.name, tuple(d.dim_value for d in s.type.tensor_type.shape.dim))
        for s in modelo.graph.output
    )
    opsets = [imp.version for imp in modelo.opset_import if imp.domain in ("", "ai.onnx")]
    if not opsets:
        msg = f"{path}: el .onnx no declara opset del dominio estándar"
        raise FichaError(msg)
    return OnnxFacts(
        input_name=entrada.name,
        input_shape=forma,
        outputs=salidas,
        opset=int(opsets[0]),
        sha256=sha256_of(path),
    )


@dataclass(frozen=True, slots=True)
class CoremlFacts:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    min_ios: int
    precision: str


def coreml_facts(package: Path) -> CoremlFacts:
    """Entradas, salidas, target y precisión, leídos del spec del `.mlpackage`."""
    import coremltools as ct  # noqa: PLC0415 — perezoso adrede (grupo apple)

    from .ane_rules import ops_from_spec  # noqa: PLC0415 — evita el ciclo en los puros

    spec = ct.utils.load_spec(str(package))
    min_ios = _SPEC_VERSION_TO_IOS.get(int(spec.specificationVersion))
    if min_ios is None:
        msg = (
            f"{package}: specificationVersion {spec.specificationVersion} no está en la "
            f"tabla {_SPEC_VERSION_TO_IOS}"
        )
        raise FichaError(msg)
    ops, _cabezas, _entradas = ops_from_spec(spec)
    precision = "fp16" if any("fp16" in op.dtypes for op in ops) else "fp32"
    return CoremlFacts(
        inputs=tuple(e.name for e in spec.description.input),
        outputs=tuple(s.name for s in spec.description.output),
        min_ios=min_ios,
        precision=precision,
    )
