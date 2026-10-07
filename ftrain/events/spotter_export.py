"""El modo paso de N4 para Core ML, con el estado de dos maneras (SPK-53).

- `N4StepStateful`: el estado vive DENTRO del modelo, en buffers que coremltools convierte
  en StateType (MLState, iOS 18). La app solo pasa el fotograma; Core ML lee y escribe el
  estado en cada predicción.
- `N4StepExplicit`: el estado entra y sale como tensores fp16 (`tsm0` → `tsm0_out`, ...,
  `h` → `h_out`). La app los recicla en doble búfer con `outputBackings`.

Las dos envuelven el MISMO `step` de ML-51 con los mismos pesos, y guardan el estado
empaquetado en filas de `N4_STATE_ROW` elementos (ver la constante): así la única
diferencia entre las dos es el mecanismo, que es lo que SPK-53 mide.

`golden_sequence` es la referencia: torch fp32 fotograma a fotograma desde el estado cero,
con los fotogramas redondeados a fp16 como los entrega la app. `TinyStepper` es un gemelo
diminuto con la misma interfaz, para los fixtures de los tests de Swift.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import torch
from torch import nn

from ftrain.constants import N4_INPUT_H, N4_INPUT_W, N4_STATE_ROW
from ftrain.events.spotter import N4Spotter, SpotterState, _ConvGru

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "FRAME_SCALE",
    "N4StepExplicit",
    "N4StepStateful",
    "TinyStepper",
    "build_spotter",
    "build_step_explicit",
    "build_step_stateful",
    "build_tiny",
    "build_tiny_explicit",
    "build_tiny_stateful",
    "frames_to_input",
    "golden_sequence",
    "packed_shape",
    "run_coreml_sequence",
    "seeded_frames",
    "state_names",
]

FRAME_SCALE: Final = 1.0 / 255.0
"""Del byte del fotograma al [0, 1] que ve N4. La app lo aplica al pasar el recorte a fp16
planar (como el balón): el modelo recibe un TensorType fp16, no un ImageType, porque el
cast y la escala de un ImageType se quedan en la CPU (medido en SPK-53: 18 % del coste)."""

TINY_INPUT_H: Final = 16
"""Píxeles. Alto del fotograma del gemelo diminuto (fixtures de Swift)."""

TINY_INPUT_W: Final = 128
"""Píxeles. Ancho del fotograma del gemelo diminuto: el mapa de su TSM, a paso 2, deja
filas de 64 elementos, múltiplo de N4_STATE_ROW."""

TINY_CHANNELS: Final = 16
"""Canales del gemelo diminuto (ANE_CHANNEL_QUANTUM)."""

TINY_SHIFT: Final = 2
"""Canales que el TSM del gemelo trae del fotograma anterior."""

TINY_HIDDEN: Final = 32
"""Estado de la GRU del gemelo: una fila de N4_STATE_ROW."""

TINY_OUTPUTS: Final = 4
"""Salidas del gemelo, las mismas que N4: fondo, dos clases y desplazamiento."""


class Stepper(Protocol):
    """Lo que necesita un modo paso: el estado cero y un paso."""

    def initial_state(self, batch: int, height: int, width: int) -> SpotterState: ...

    def step(
        self, frame: torch.Tensor, state: SpotterState
    ) -> tuple[torch.Tensor, SpotterState]: ...


def state_names(tsm_count: int) -> list[str]:
    """Los nombres del estado en Core ML, en orden: tsm0..tsm{n-1} y h (la GRU)."""
    return [*(f"tsm{i}" for i in range(tsm_count)), "h"]


def packed_shape(native: tuple[int, ...], row: int = N4_STATE_ROW) -> tuple[int, int, int, int]:
    """(1, C, n/row, row) si cada canal llena filas enteras; si no, (1, 1, N/row, row).

    El orden C no cambia: empaquetar es un reshape, y desempaquetar, el inverso."""
    total = math.prod(native[1:])
    canales = native[1]
    por_canal = total // canales
    if por_canal % row == 0:
        return (1, canales, por_canal // row, row)
    if total % row == 0:
        return (1, 1, total // row, row)
    msg = f"el estado {native} no llena filas de {row} elementos"
    raise ValueError(msg)


class _StepWrapper(nn.Module):
    """Lo común: las formas nativas y empaquetadas del estado, y el paso de unas a otras."""

    def __init__(self, red: nn.Module, height: int, width: int) -> None:
        super().__init__()
        self.red = red
        tsm, h = red.initial_state(1, height, width)  # type: ignore[operator]
        self.native = [tuple(t.shape) for t in (*tsm, h)]
        self.packed = [packed_shape(s) for s in self.native]
        self.names = state_names(len(tsm))

    def _unpack(self, packed: list[torch.Tensor]) -> SpotterState:
        valores = [p.reshape(s) for p, s in zip(packed, self.native, strict=True)]
        return valores[:-1], valores[-1]

    def _pack(self, state: SpotterState) -> list[torch.Tensor]:
        tsm, h = state
        return [v.reshape(s) for v, s in zip((*tsm, h), self.packed, strict=True)]


class N4StepStateful(_StepWrapper):
    """El fotograma entra; el estado, en buffers que Core ML convierte en MLState."""

    def __init__(self, red: nn.Module, height: int, width: int) -> None:
        super().__init__(red, height, width)
        for nombre, forma in zip(self.names, self.packed, strict=True):
            self.register_buffer(nombre, torch.zeros(forma))

    def reset(self) -> None:
        """El estado cero, el de antes del primer fotograma."""
        for nombre in self.names:
            self.get_buffer(nombre).zero_()

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        previos = [self.get_buffer(nombre) for nombre in self.names]
        y, estado = self.red.step(frame, self._unpack(previos))  # type: ignore[operator]
        for buffer, nuevo in zip(previos, self._pack(estado), strict=True):
            # slice + copy_: el patrón que coremltools convierte en write_state.
            buffer[:] = nuevo
        return y


class N4StepExplicit(_StepWrapper):
    """El fotograma y el estado entran; la salida y el estado nuevo salen."""

    def forward(self, frame: torch.Tensor, *state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y, nuevo = self.red.step(frame, self._unpack(list(state)))  # type: ignore[operator]
        return (y, *self._pack(nuevo))


class TinyStepper(nn.Module):
    """Un modo paso diminuto con la interfaz de N4 (un TSM y una GRU): los fixtures de
    Swift prueban el arnés con él sin cargar 1,5 M de parámetros."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, TINY_CHANNELS, 3, stride=2, padding=1)
        self.gru = _ConvGru(TINY_CHANNELS, TINY_HIDDEN)
        self.head = nn.Conv2d(TINY_HIDDEN, TINY_OUTPUTS, 1)

    def initial_state(self, batch: int, height: int, width: int) -> SpotterState:
        dev = next(self.parameters()).device
        tsm = torch.zeros(batch, TINY_SHIFT, height // 2, width // 2, device=dev)
        return [tsm], torch.zeros(batch, TINY_HIDDEN, 1, 1, device=dev)

    def step(self, frame: torch.Tensor, state: SpotterState) -> tuple[torch.Tensor, SpotterState]:
        anteriores, h = state
        x = torch.relu(self.stem(frame))
        nuevo = x[:, :TINY_SHIFT]
        x = torch.cat([anteriores[0], x[:, TINY_SHIFT:]], dim=1)
        h = self.gru(x.mean(dim=(2, 3), keepdim=True), h)
        return self.head(h).flatten(1), ([nuevo], h)


# --------------------------------------------------------------------------- #
# Builders para los CLI: pesos sembrados si no hay checkpoint
# --------------------------------------------------------------------------- #
def _load(m: nn.Module, checkpoint: Path | str | None) -> nn.Module:
    if checkpoint is not None:
        m.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return m.eval()


def build_spotter(checkpoint: Path | str | None) -> nn.Module:
    """N4 con la semilla 0: la referencia de la secuencia dorada."""
    torch.manual_seed(0)
    return _load(N4Spotter(), checkpoint)


def build_step_stateful(checkpoint: Path | str | None) -> nn.Module:
    return N4StepStateful(build_spotter(checkpoint), N4_INPUT_H, N4_INPUT_W).eval()


def build_step_explicit(checkpoint: Path | str | None) -> nn.Module:
    return N4StepExplicit(build_spotter(checkpoint), N4_INPUT_H, N4_INPUT_W).eval()


def build_tiny(checkpoint: Path | str | None) -> nn.Module:
    torch.manual_seed(0)
    return _load(TinyStepper(), checkpoint)


def build_tiny_stateful(checkpoint: Path | str | None) -> nn.Module:
    return N4StepStateful(build_tiny(checkpoint), TINY_INPUT_H, TINY_INPUT_W).eval()


def build_tiny_explicit(checkpoint: Path | str | None) -> nn.Module:
    return N4StepExplicit(build_tiny(checkpoint), TINY_INPUT_H, TINY_INPUT_W).eval()


# --------------------------------------------------------------------------- #
# La secuencia dorada
# --------------------------------------------------------------------------- #
def seeded_frames(n: int, height: int, width: int, seed: int = 0) -> np.ndarray:
    """n fotogramas de ruido con semilla, uint8 planar (n, 3, alto, ancho)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 3, height, width), dtype=np.uint8)


def frames_to_input(frames: np.ndarray) -> torch.Tensor:
    """uint8 (n, 3, alto, ancho) → lo que ve el modelo: byte·FRAME_SCALE redondeado a fp16
    (lo que entrega la app), de vuelta en fp32 para torch."""
    escalado = (frames.astype(np.float32) * np.float32(FRAME_SCALE)).astype(np.float16)
    return torch.from_numpy(escalado.astype(np.float32))


def golden_sequence(red: Stepper, frames: np.ndarray) -> dict[str, np.ndarray]:
    """La referencia torch fp32 desde el estado cero: por paso, la salida (`logits`) y el
    estado de la GRU (`h`, aplanado)."""
    entrada = frames_to_input(frames)
    _, _, alto, ancho = entrada.shape
    estado = red.initial_state(1, alto, ancho)
    logits, hs = [], []
    with torch.no_grad():
        for k in range(entrada.shape[0]):
            y, estado = red.step(entrada[k : k + 1], estado)
            logits.append(y.numpy()[0])
            hs.append(estado[1].numpy().reshape(-1))
    return {"logits": np.stack(logits).astype(np.float32), "h": np.stack(hs).astype(np.float32)}


def run_coreml_sequence(
    package: Path, frames: np.ndarray, compute_units: str = "cpu_and_ne"
) -> dict[str, np.ndarray]:
    """La misma secuencia por Core ML (solo macOS), con MLState si el modelo tiene estado
    o reciclando las salidas `<nombre>_out` si lo lleva explícito."""
    import coremltools as ct  # noqa: PLC0415 — perezoso adrede (grupo apple)

    unidades = {"cpu_and_ne": ct.ComputeUnit.CPU_AND_NE, "cpu_only": ct.ComputeUnit.CPU_ONLY}
    modelo = ct.models.MLModel(str(package), compute_units=unidades[compute_units])
    descripcion = modelo.get_spec().description
    estados = [e.name for e in descripcion.state]
    entradas = {
        e.name: tuple(e.type.multiArrayType.shape) for e in descripcion.input if e.name != "frame"
    }
    entrada = frames_to_input(frames).numpy().astype(np.float16)
    logits, hs = [], []
    if estados:
        estado = modelo.make_state()
        for k in range(entrada.shape[0]):
            salida = modelo.predict({"frame": entrada[k : k + 1]}, state=estado)
            logits.append(np.asarray(salida["logits"], dtype=np.float32).reshape(-1))
            hs.append(np.asarray(estado.read_state("h"), dtype=np.float32).reshape(-1))
    else:
        actual: dict[str, Any] = {n: np.zeros(f, dtype=np.float16) for n, f in entradas.items()}
        for k in range(entrada.shape[0]):
            salida = modelo.predict({"frame": entrada[k : k + 1], **actual})
            actual = {n: np.asarray(salida[f"{n}_out"], dtype=np.float16) for n in entradas}
            logits.append(np.asarray(salida["logits"], dtype=np.float32).reshape(-1))
            hs.append(np.asarray(salida["h_out"], dtype=np.float32).reshape(-1))
    return {"logits": np.stack(logits), "h": np.stack(hs)}
