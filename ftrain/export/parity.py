"""Paridad torch ↔ ONNX ↔ Core ML: el mismo lote por cada backend (ML-11).

Los comparadores y los decodificadores de tarea son puros (numpy) y se prueban en
cualquier SO. Los runners importan su backend en perezoso: torch (grupo train),
onnxruntime (core) y coremltools (apple, y predecir solo en macOS).

La referencia es siempre torch fp32. La regla del repo se conserva: torch↔ORT
< 1e-3 (T6). El umbral de Core ML fp16 sale de la ficha v2 (`parity.atol`) o del
CLI, porque depende del modelo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from .coreml import ExportSpec

__all__ = [
    "Delta",
    "ParityError",
    "compare_lotes",
    "compare_outputs",
    "detector_metrics",
    "evaluate",
    "heatmap_metrics",
    "run_coreml",
    "run_ort",
    "run_torch",
    "seeded_batch",
    "spotter_metrics",
]

TORCH_ORT_ATOL: Final = 1e-3
"""La regla del repo (T6): torch fp32 y onnxruntime fp32 difieren menos que esto."""

REL_EPS: Final = 1e-6
"""Suelo del denominador del delta relativo: evita dividir por un cero numérico."""

DETECTOR_SCORE: Final = 0.3
"""Puntuación (sigmoide del mejor logit) desde la que una detección cuenta."""

DETECTOR_MATCH_TOL: Final = 0.01
"""Distancia máxima entre centros, normalizada al lado, para casar dos detecciones."""


class ParityError(ValueError):
    """Los lotes o las salidas no casan. El mensaje nombra qué."""


# --------------------------------------------------------------------------- #
# Comparadores, puros
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Delta:
    max_abs: float
    max_rel: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _delta(a: np.ndarray, b: np.ndarray) -> Delta:
    if a.shape != b.shape:
        msg = f"formas distintas: {a.shape} frente a {b.shape}"
        raise ParityError(msg)
    diferencia = np.abs(a.astype(np.float64) - b.astype(np.float64))
    relativo = diferencia / np.maximum(np.abs(a.astype(np.float64)), REL_EPS)
    return Delta(max_abs=float(diferencia.max()), max_rel=float(relativo.max()))


def compare_outputs(
    ref: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray]
) -> dict[str, Delta]:
    """max|Δ| y Δ relativo por salida. Las claves tienen que ser las mismas."""
    if set(ref) != set(other):
        msg = f"salidas distintas: {sorted(ref)} frente a {sorted(other)}"
        raise ParityError(msg)
    return {nombre: _delta(ref[nombre], other[nombre]) for nombre in sorted(ref)}


def compare_lotes(
    refs: Sequence[Mapping[str, np.ndarray]], others: Sequence[Mapping[str, np.ndarray]]
) -> dict[str, Delta]:
    """El peor delta de cada salida a lo largo del lote entero."""
    if len(refs) != len(others):
        msg = f"lotes de tamaños distintos: {len(refs)} frente a {len(others)}"
        raise ParityError(msg)
    peores: dict[str, Delta] = {}
    for ref, other in zip(refs, others, strict=True):
        for nombre, delta in compare_outputs(ref, other).items():
            previo = peores.get(nombre)
            if previo is None:
                peores[nombre] = delta
            else:
                peores[nombre] = Delta(
                    max_abs=max(previo.max_abs, delta.max_abs),
                    max_rel=max(previo.max_rel, delta.max_rel),
                )
    return peores


# --------------------------------------------------------------------------- #
# Decodificadores de tarea (puros, enchufables)
# --------------------------------------------------------------------------- #
def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _detecciones(salidas: Mapping[str, np.ndarray], score: float) -> tuple[np.ndarray, np.ndarray]:
    """Centros (cx, cy) y puntuaciones de las detecciones que superan el umbral.

    Espera el contrato del detector: logits [1,N,C] y boxes [1,N,4] cxcywh
    normalizado, con esos dos nombres (ML-13)."""
    for clave in ("logits", "boxes"):
        if clave not in salidas:
            msg = f"el decodificador detector espera la salida `{clave}`"
            raise ParityError(msg)
    puntuaciones = _sigmoide(salidas["logits"][0]).max(axis=-1)
    centros = salidas["boxes"][0, :, :2].astype(np.float64)
    quedan = puntuaciones >= score
    return centros[quedan], puntuaciones[quedan]


def detector_metrics(
    ref: Mapping[str, np.ndarray],
    other: Mapping[str, np.ndarray],
    *,
    band_of: Callable[[float, float], str],
    score: float = DETECTOR_SCORE,
    match_tol: float = DETECTOR_MATCH_TOL,
) -> dict[str, Any]:
    """Δrecall por banda y desplazamiento medio del centro, con torch de verdad.

    `band_of(cx, cy)` da la banda del centro normalizado; hasta que ML-13 enchufe
    la tabla fila→metros real, el CLI usa franjas horizontales de la imagen."""
    centros_ref, _ = _detecciones(ref, score)
    centros_other, _ = _detecciones(other, score)

    total: dict[str, int] = {}
    casadas: dict[str, int] = {}
    desplazamientos: list[float] = []
    for centro in centros_ref:
        banda = band_of(float(centro[0]), float(centro[1]))
        total[banda] = total.get(banda, 0) + 1
        if centros_other.size == 0:
            continue
        distancias = np.linalg.norm(centros_other - centro, axis=1)
        cercana = int(distancias.argmin())
        if distancias[cercana] <= match_tol:
            casadas[banda] = casadas.get(banda, 0) + 1
            desplazamientos.append(float(distancias[cercana]))

    recall_delta = {
        banda: 1.0 - casadas.get(banda, 0) / cuantas for banda, cuantas in sorted(total.items())
    }
    return {
        "recall_delta_por_banda": recall_delta,
        "desplazamiento_centro": float(np.mean(desplazamientos)) if desplazamientos else 0.0,
    }


def heatmap_metrics(
    ref: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Δ del pico: cuánto se mueve y cuánto cambia de valor, por salida."""
    resultado: dict[str, Any] = {}
    for nombre in sorted(ref):
        plano_ref = ref[nombre].reshape(ref[nombre].shape[-2:])
        plano_other = other[nombre].reshape(other[nombre].shape[-2:])
        pico_ref = np.unravel_index(int(plano_ref.argmax()), plano_ref.shape)
        pico_other = np.unravel_index(int(plano_other.argmax()), plano_other.shape)
        resultado[nombre] = {
            "peak_shift_px": float(np.linalg.norm(np.subtract(pico_ref, pico_other))),
            "peak_value_delta": float(
                abs(float(plano_ref[pico_ref]) - float(plano_other[pico_other]))
            ),
        }
    return resultado


def spotter_metrics(
    ref: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Δlogits: el peor delta absoluto por salida."""
    return {nombre: delta.max_abs for nombre, delta in compare_outputs(ref, other).items()}


DECODERS: Final[dict[str, str]] = {
    "detector": "Δrecall por banda y desplazamiento del centro",
    "heatmap": "Δ del pico",
    "spotter": "Δlogits",
}

DEFAULT_COREML_ATOL: Final = 5e-2
"""Umbral de Core ML fp16 si ni la ficha ni el CLI dicen otro: holgado adrede,
porque el bueno es por modelo y vive en `parity.atol` de la ficha v2."""

FALLBACK_BANDS: Final = 4
"""Franjas horizontales de la imagen para el Δrecall mientras ML-13 no enchufe
la tabla fila→metros real del band.json."""


def fallback_band_of(cx: float, cy: float) -> str:  # noqa: ARG001 — la firma es el contrato
    """La banda provisional de un centro normalizado: su franja horizontal."""
    indice = min(int(cy * FALLBACK_BANDS), FALLBACK_BANDS - 1)
    return f"franja-{indice}"


def aggregate_metrics(muestras: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """La media, hoja a hoja, de una lista de métricas anidadas (dicts y números)."""
    if not muestras:
        return {}
    primera = muestras[0]
    resultado: dict[str, Any] = {}
    for clave, valor in primera.items():
        valores = [muestra[clave] for muestra in muestras]
        if isinstance(valor, Mapping):
            resultado[clave] = aggregate_metrics(valores)
        else:
            resultado[clave] = float(np.mean([float(v) for v in valores]))
    return resultado


# --------------------------------------------------------------------------- #
# El lote y los runners
# --------------------------------------------------------------------------- #
def seeded_batch(spec: ExportSpec, n: int, seed: int = 0) -> list[dict[str, np.ndarray]]:
    """Ruido con semilla, en crudo: u8 HWC para una imagen, fp32 para un tensor."""
    rng = np.random.default_rng(seed)
    lote: list[dict[str, np.ndarray]] = []
    for _ in range(n):
        muestra: dict[str, np.ndarray] = {}
        for entrada in spec.inputs:
            if entrada.kind == "image":
                _, canales, alto, ancho = entrada.shape
                muestra[entrada.name] = rng.integers(
                    0, 256, size=(alto, ancho, canales), dtype=np.uint8
                )
            else:
                muestra[entrada.name] = rng.standard_normal(entrada.shape).astype(np.float32)
        lote.append(muestra)
    return lote


def _as_float(spec: ExportSpec, muestra: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """La muestra como la ven torch y ORT: imágenes a [0,1] NCHW fp32."""
    feeds: dict[str, np.ndarray] = {}
    for entrada in spec.inputs:
        cruda = muestra[entrada.name]
        if entrada.kind == "image":
            feeds[entrada.name] = (cruda.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        else:
            feeds[entrada.name] = cruda.astype(np.float32)
    return feeds


def run_torch(
    module: Any, spec: ExportSpec, lote: Sequence[Mapping[str, np.ndarray]]
) -> list[dict[str, np.ndarray]]:
    import torch  # noqa: PLC0415 — perezoso adrede (grupo train)

    module = module.eval()
    resultados: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for muestra in lote:
            feeds = _as_float(spec, muestra)
            salidas = module(*(torch.from_numpy(feeds[e.name]) for e in spec.inputs))
            if not isinstance(salidas, (tuple, list)):
                salidas = (salidas,)
            if len(salidas) != len(spec.output_names):
                msg = (
                    f"el modelo devuelve {len(salidas)} salidas y el spec declara "
                    f"{len(spec.output_names)}"
                )
                raise ParityError(msg)
            resultados.append(
                {
                    nombre: salida.detach().cpu().numpy().astype(np.float32)
                    for nombre, salida in zip(spec.output_names, salidas, strict=True)
                }
            )
    return resultados


def run_ort(
    onnx_path: Path, spec: ExportSpec, lote: Sequence[Mapping[str, np.ndarray]]
) -> list[dict[str, np.ndarray]]:
    import onnxruntime as ort  # noqa: PLC0415 — junto a su consumidor

    sesion = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    nombres = [salida.name for salida in sesion.get_outputs()]
    resultados: list[dict[str, np.ndarray]] = []
    for muestra in lote:
        salidas = sesion.run(None, _as_float(spec, muestra))
        resultados.append(
            {
                nombre: np.asarray(salida, dtype=np.float32)
                for nombre, salida in zip(nombres, salidas, strict=True)
            }
        )
    return resultados


def run_coreml(
    package: Path,
    spec: ExportSpec,
    lote: Sequence[Mapping[str, np.ndarray]],
    compute_units: str,
) -> list[dict[str, np.ndarray]]:
    """Predice con Core ML; `compute_units` es "cpu_and_ne" o "cpu_only" (solo macOS)."""
    import coremltools as ct  # noqa: PLC0415 — perezoso adrede (grupo apple)
    from PIL import Image  # noqa: PLC0415 — llega con coremltools

    unidades = {
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if compute_units not in unidades:
        msg = f"compute_units tiene que ser uno de {sorted(unidades)} y es {compute_units!r}"
        raise ParityError(msg)
    for entrada in spec.inputs:
        if entrada.kind == "image" and entrada.color != "RGB":
            msg = "el runner de paridad solo alimenta imágenes RGB de momento"
            raise ParityError(msg)

    modelo = ct.models.MLModel(str(package), compute_units=unidades[compute_units])
    resultados: list[dict[str, np.ndarray]] = []
    for muestra in lote:
        feeds: dict[str, Any] = {}
        for entrada in spec.inputs:
            cruda = muestra[entrada.name]
            feeds[entrada.name] = (
                Image.fromarray(cruda, "RGB") if entrada.kind == "image" else cruda
            )
        salidas = modelo.predict(feeds)
        resultados.append(
            {nombre: np.asarray(valor, dtype=np.float32) for nombre, valor in salidas.items()}
        )
    return resultados


# --------------------------------------------------------------------------- #
# El informe y los umbrales
# --------------------------------------------------------------------------- #
def evaluate(
    refs: Sequence[Mapping[str, np.ndarray]],
    others: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    atol: Mapping[str, float],
    metrics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """El informe de paridad y si todos los pares respetan su umbral.

    `others` y `atol` van por par ("torch_ort", "torch_coreml_ne"…); `metrics`
    son los resultados del decodificador de tarea, que viajan tal cual."""
    pares: dict[str, Any] = {}
    violaciones: list[str] = []
    for par, lote in others.items():
        if par not in atol:
            msg = f"el par {par!r} no tiene umbral"
            raise ParityError(msg)
        deltas = compare_lotes(refs, lote)
        pares[par] = {
            "atol": atol[par],
            "outputs": {nombre: delta.to_dict() for nombre, delta in deltas.items()},
        }
        for nombre, delta in deltas.items():
            if delta.max_abs > atol[par]:
                violaciones.append(f"{par}/{nombre}: max_abs {delta.max_abs:.3e} > {atol[par]:.3e}")
    informe = {
        "samples": len(refs),
        "pairs": pares,
        "metrics": dict(metrics or {}),
        "violations": violaciones,
    }
    return informe, not violaciones
