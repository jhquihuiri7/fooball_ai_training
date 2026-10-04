"""Lint estático del programa MIL: lo que el ANE no traga, antes del iPhone (ML-10).

El motor de reglas trabaja sobre una lista NEUTRA de ops (tipo, dtype, formas):
así cada regla se prueba sin coremltools. El walker que convierte el spec de un
`.mlpackage` en esa lista vive al final y solo importa el proto cuando se le llama.

La «cola» es el postproceso declarado: el sufijo contiguo del programa cuyos tipos
de op están en la lista blanca del modelo (más `cast`, que es pegamento, no
cómputo). Dentro de la cola se toleran topk/argsort/NMS y fp32; fuera, son bug.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "LintConfig",
    "LintError",
    "MilOp",
    "Violation",
    "lint",
    "load_config",
    "ops_from_package",
    "ops_from_spec",
]

CHANNEL_MULTIPLE: Final = 16
"""El ANE teselea las activaciones NCHW por canales de 16: otro ancho desperdicia."""

TAIL_GLUE: Final = frozenset({"cast", "const"})
"""Tipos que no cortan la cola aunque nadie los declare: literales y movimientos
de bits, no cómputo. Un const fp32 FUERA de la cola sigue cayendo por su regla."""

_RECURRENT: Final = frozenset({"gru", "lstm", "rnn"})
_CONTROL: Final = frozenset({"while_loop", "cond"})
_ORDERING: Final = frozenset({"topk", "argsort", "non_maximum_suppression"})
_NOT_ACTIVATION: Final = frozenset({"const", "cast"})
"""Pesos y pegamento: sus salidas no son activaciones y no pagan el múltiplo de 16."""

_RANK_5: Final = 5
_RANK_4: Final = 4
_CHANNEL_AXIS: Final = 1  # MIL es NCHW


class LintError(ValueError):
    """La configuración o el paquete no valen. El mensaje nombra el campo."""


@dataclass(frozen=True, slots=True)
class MilOp:
    """Una op del programa, ya neutra: lo único que las reglas necesitan ver."""

    type: str  # p. ej. "conv", "topk", "const"
    name: str  # el nombre de su primera salida, que es como MIL la nombra
    dtypes: tuple[str, ...] = ()  # por salida: "fp16", "fp32", "int32"…
    shapes: tuple[tuple[int | str, ...], ...] = ()  # por salida; str = simbólica


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    op: str
    type: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LintConfig:
    """La lista blanca de un modelo (configs/ane_lint/<modelo>.yaml)."""

    cola: frozenset[str] = field(default_factory=frozenset)  # tipos de op de la cola
    exentos: frozenset[str] = field(default_factory=frozenset)  # salidas sin múltiplo de 16


def load_config(path: Path) -> LintConfig:
    try:
        crudo = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        msg = f"no se pudo leer la lista blanca {path}: {exc}"
        raise LintError(msg) from exc
    if not isinstance(crudo, dict):
        msg = f"{path}: la lista blanca tiene que ser un mapa"
        raise LintError(msg)
    for clave in ("cola", "exentos"):
        valores = crudo.get(clave, [])
        if not isinstance(valores, list) or any(not isinstance(v, str) for v in valores):
            msg = f"{path}: `{clave}` tiene que ser una lista de cadenas"
            raise LintError(msg)
    return LintConfig(
        cola=frozenset(crudo.get("cola", [])),
        exentos=frozenset(crudo.get("exentos", [])),
    )


# --------------------------------------------------------------------------- #
# El motor de reglas (puro)
# --------------------------------------------------------------------------- #
def _tail_start(ops: Sequence[MilOp], config: LintConfig) -> int:
    """El índice donde empieza la cola: el sufijo contiguo de tipos declarados."""
    permitidos = config.cola | TAIL_GLUE
    inicio = len(ops)
    while inicio > 0 and ops[inicio - 1].type in permitidos:
        inicio -= 1
    return inicio


def _type_violations(op: MilOp) -> list[Violation]:
    """Las reglas por tipo de op, independientes de la posición."""
    violaciones: list[Violation] = []
    if op.type.startswith("conv") and any(len(forma) == _RANK_5 for forma in op.shapes):
        violaciones.append(Violation("conv3d", op.name, op.type, "conv sobre tensores de rango 5"))
    if op.type in _RECURRENT:
        violaciones.append(
            Violation("recurrente", op.name, op.type, "gru/lstm/rnn no corren en el ANE")
        )
    if op.type.startswith("scatter"):
        violaciones.append(Violation("scatter", op.name, op.type, "scatter* cae a CPU"))
    if op.type in _CONTROL:
        violaciones.append(
            Violation("control", op.name, op.type, "while_loop/cond parten el grafo")
        )
    return violaciones


def _shape_violations(op: MilOp, exentas: frozenset[str]) -> list[Violation]:
    """Las reglas por forma: rango 5, dimensiones simbólicas y canales."""
    violaciones: list[Violation] = []
    for forma in op.shapes:
        if len(forma) == _RANK_5:
            violaciones.append(
                Violation("rango-5", op.name, op.type, f"tensor de rango 5: {forma}")
            )
        if any(isinstance(dimension, str) for dimension in forma):
            violaciones.append(
                Violation("simbolica", op.name, op.type, f"dimensión simbólica: {forma}")
            )
    if op.type in _NOT_ACTIVATION or op.name in exentas:
        return violaciones
    for forma in op.shapes:
        if len(forma) != _RANK_4:
            continue
        canales = forma[_CHANNEL_AXIS]
        if isinstance(canales, int) and canales % CHANNEL_MULTIPLE != 0:
            violaciones.append(
                Violation(
                    "canales",
                    op.name,
                    op.type,
                    f"{canales} canales no son múltiplo de {CHANNEL_MULTIPLE}",
                )
            )
    return violaciones


def _is_input_preprocess(op: MilOp, inputs: frozenset[str]) -> bool:
    """El preproceso que coremltools inyecta sobre una entrada: la escala de un
    ImageType (`image__scaled__` y su const, en fp32) y el cast `image_to_fp16`.
    Sus vars se llaman como la entrada más un sufijo; es «la imagen de entrada»
    de la tarjeta y no paga ni canales ni fp32."""
    return any(op.name == entrada or op.name.startswith(f"{entrada}_") for entrada in inputs)


def lint(
    ops: Sequence[MilOp],
    config: LintConfig,
    heads: frozenset[str] = frozenset(),
    inputs: frozenset[str] = frozenset(),
) -> list[Violation]:
    """Todas las violaciones del programa.

    `heads` son las salidas del modelo e `inputs` sus entradas: las cabezas están
    exentas del múltiplo de 16 y el preproceso de la entrada, de canales y fp32."""
    cola = _tail_start(ops, config)
    exentas = config.exentos | heads
    violaciones: list[Violation] = []

    for indice, op in enumerate(ops):
        en_cola = indice >= cola
        preproceso = _is_input_preprocess(op, inputs)
        violaciones += _type_violations(op)
        violaciones += _shape_violations(op, exentas if not preproceso else exentas | {op.name})
        if op.type in _ORDERING and not en_cola:
            violaciones.append(
                Violation(
                    "orden-fuera-de-cola",
                    op.name,
                    op.type,
                    "topk/argsort/NMS solo en la cola declarada",
                )
            )
        if not en_cola and not preproceso and any(dtype == "fp32" for dtype in op.dtypes):
            violaciones.append(
                Violation("fp32-fuera-de-cola", op.name, op.type, "op en fp32 fuera de la cola")
            )

    return violaciones


# --------------------------------------------------------------------------- #
# El walker del spec (solo donde hay coremltools)
# --------------------------------------------------------------------------- #
_DTYPE_NAMES: Final = {"FLOAT16": "fp16", "FLOAT32": "fp32", "FLOAT64": "fp64"}


def _dtype(nombre: str) -> str:
    return _DTYPE_NAMES.get(nombre, nombre.lower())


def _tensor(tipo: Any, mil_pb2: Any) -> tuple[str | None, tuple[int | str, ...] | None]:
    if tipo.WhichOneof("type") != "tensorType":
        return None, None
    tensor = tipo.tensorType
    dims: list[int | str] = []
    for dimension in tensor.dimensions:
        if dimension.WhichOneof("dimension") == "constant":
            dims.append(int(dimension.constant.size))
        else:
            dims.append("?")
    return _dtype(mil_pb2.DataType.Name(tensor.dataType)), tuple(dims)


def _walk(bloque: Any, mil_pb2: Any, acumulador: list[MilOp]) -> None:
    for operacion in bloque.operations:
        dtypes: list[str] = []
        formas: list[tuple[int | str, ...]] = []
        for salida in operacion.outputs:
            dtype, forma = _tensor(salida.type, mil_pb2)
            if dtype is not None and forma is not None:
                dtypes.append(dtype)
                formas.append(forma)
        nombre = operacion.outputs[0].name if operacion.outputs else operacion.type
        acumulador.append(
            MilOp(
                type=operacion.type,
                name=nombre,
                dtypes=tuple(dtypes),
                shapes=tuple(formas),
            )
        )
        for anidado in operacion.blocks:
            _walk(anidado, mil_pb2, acumulador)


def ops_from_spec(spec: Any) -> tuple[tuple[MilOp, ...], frozenset[str], frozenset[str]]:
    """El programa MIL del spec, aplanado, con sus cabezas y sus entradas."""
    from coremltools.proto import MIL_pb2  # noqa: PLC0415 — perezoso adrede (grupo apple)

    if spec.WhichOneof("Type") != "mlProgram":
        msg = f"el spec no es un mlprogram: es {spec.WhichOneof('Type')!r}"
        raise LintError(msg)
    ops: list[MilOp] = []
    cabezas: set[str] = set()
    for funcion in spec.mlProgram.functions.values():
        bloque = funcion.block_specializations[funcion.opset]
        cabezas.update(bloque.outputs)
        _walk(bloque, MIL_pb2, ops)
    entradas = frozenset(entrada.name for entrada in spec.description.input)
    return tuple(ops), frozenset(cabezas), entradas


def ops_from_package(path: Path) -> tuple[tuple[MilOp, ...], frozenset[str], frozenset[str]]:
    import coremltools as ct  # noqa: PLC0415 — perezoso adrede (grupo apple)

    return ops_from_spec(ct.utils.load_spec(str(path)))
