"""Exporta T-DEED a ONNX y escribe su ficha (TASK T5 y T6).

Es lo único que cruza al repo de detección: un `.onnx` y el bloque de `registry.yaml` que
dice cómo ejecutarlo. Corre en Colab, donde vive torch; allí está prohibido.

    !python tools/export_onnx.py --tdeed third_party/T-DEED --out modelo/

**Sin ejecutar todavía**: se escribió leyendo el código de T-DEED, no corriéndolo. Lo que
falle va a `docs/PROGRESS.md`.

## Tres cosas del modelo que deciden cómo se exporta

**1. El modelo normaliza por dentro.** Su `forward` hace `x / 255` y luego la
estandarización de ImageNet. O sea que el `.onnx` espera píxeles **en crudo, 0-255**, y la
ficha tiene que declarar `scale: 1`, `mean: [0,0,0]`, `std: [1,1,1]`. Poner ahí los valores
de ImageNet —que es lo que uno copiaría del paper— normalizaría dos veces, y eso no da
error: solo hunde el score.

**2. Tiene dos cabezas.** El modelo se preentrena en SoccerNet (17 clases) y se afina en
SoccerNetBall (12), y la capa final emite las dos concatenadas. La nuestra es la primera,
`1 + 12` columnas contando el fondo. El recorte se hace **dentro del grafo**, que es un
`Slice` estático y exporta perfecto; así el repo de detección no tiene que saber nada de
esto.

**3. El desplazamiento no se puede meter en el grafo.** El modelo emite, además de las
puntuaciones, un desplazamiento temporal por frame, y su post-proceso lo aplica con un
doble bucle de Python con índices que dependen de los datos. Trazar eso a ONNX
**hornearía los desplazamientos del clip de ejemplo** dentro del grafo: no fallaría, y
daría mal todos los demás clips.

Así que el `.onnx` saca **dos salidas** —`logits` y `displacement`— y el desplazamiento lo
aplica el spotter, en numpy, donde se puede probar. La ficha lo declara con
`meaning: displacement` y quien no lo tenga no se entera.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKGROUND_CLASS: Final = "normal_play"
"""Nombre que se le da a la columna 0, que el modelo usa como fondo.

Se llama así y no «background» a propósito: es el nombre que el spotter del repo de
detección conoce como la clase que nunca produce candidatos (`SPOTTER_BACKGROUND_CLASS`).
Que coincida no es casualidad, es el contrato."""

OPSET: Final = 17
"""El opset que ya usa el `.onnx` del detector, y el que `onnxruntime` de allí sabe leer."""

DEFAULT_CONFIG: Final = "SoccerNetBall_challenge1"
INPUT_NAME: Final = "clip"
LOGITS_NAME: Final = "logits"
DISPLACEMENT_NAME: Final = "displacement"

LAYOUT: Final = "NTCHW"
"""Los frames van en el eje 1 y los canales en el 2: así lo desempaqueta su `forward`.
El repo de detección lo lee de aquí y no lo supone, desde el arreglo del 2026-09-20."""

VERIFY_TOLERANCE: Final = 1e-3
"""Diferencia máxima admitida entre torch y onnxruntime sobre el mismo clip (T6).

Es el umbral que fija §42.1 del blueprint. No es una formalidad: el backbone lleva
gate-shift, y si el `roll` temporal se degrada al exportar, es aquí donde se ve."""


class ExportError(RuntimeError):
    """No se pudo exportar o verificar. El mensaje dice qué falta."""


@dataclass(frozen=True, slots=True)
class ExportSpec:
    """Lo que hay que saber para exportar, leído de la configuración de T-DEED."""

    config: str
    clip_len: int
    height: int
    width: int
    classes: tuple[str, ...]
    sample_fps: float

    @property
    def input_shape(self) -> tuple[int, int, int, int, int]:
        """`(1, T, C, alto, ancho)`. Estático: el clip siempre mide lo mismo."""
        return (1, self.clip_len, 3, self.height, self.width)

    @property
    def head_width(self) -> int:
        """Columnas de nuestra cabeza: el fondo más las clases del dataset."""
        return len(self.classes) + 1

    @property
    def registry_classes(self) -> tuple[str, ...]:
        """Los nombres en el orden de salida: el fondo primero, luego las del dataset."""
        return (BACKGROUND_CLASS, *self.classes)


def build_wrapper(model: Any, spec: ExportSpec) -> Any:
    """Envuelve el modelo para que su `forward` acepte un clip y devuelva dos tensores.

    El de T-DEED acepta cuatro argumentos y devuelve un diccionario dentro de una tupla,
    que `torch.onnx.export` no sabe manejar. Y emite las dos cabezas concatenadas: aquí se
    recorta a la nuestra con un `Slice` estático, que exporta sin problemas.
    """
    import torch  # noqa: PLC0415

    class Exportable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model
            self.head_width = spec.head_width

        def forward(self, clip: Any) -> tuple[Any, Any]:
            salida, _ = self.inner(clip, inference=True)
            if not isinstance(salida, dict):
                msg = (
                    "el modelo no devolvió las dos cabezas; ¿tiene la config "
                    "`radi_displacement > 0`? Sin desplazamiento este export no vale"
                )
                raise ExportError(msg)
            # Sin softmax: la ficha dirá `activation: softmax` y lo aplica quien lee, que
            # es donde está probado. Cuanto menos haya en el grafo, menos hay que verificar.
            return salida["im_feat"][:, :, : self.head_width], salida["displ_feat"]

    envuelto = Exportable()
    envuelto.eval()
    return envuelto


def export(model: Any, spec: ExportSpec, destination: Path) -> Path:
    """Escribe el `.onnx` con formas estáticas. Devuelve su ruta."""
    import torch  # noqa: PLC0415

    destination.parent.mkdir(parents=True, exist_ok=True)
    ejemplo = torch.zeros(spec.input_shape, dtype=torch.float32)
    if next(model.parameters()).is_cuda:
        ejemplo = ejemplo.cuda()
    torch.onnx.export(
        model,
        (ejemplo,),
        str(destination),
        input_names=[INPUT_NAME],
        output_names=[LOGITS_NAME, DISPLACEMENT_NAME],
        opset_version=OPSET,
        # Nada dinámico a propósito (ADR 0013 §3 y BLUEPRINT §42.1): el clip siempre mide
        # lo mismo, y con ejes fijos la aritmética de formas del gate-shift se pliega a
        # constantes en vez de quedarse en el grafo.
        dynamic_axes=None,
        do_constant_folding=True,
    )
    return destination


def verify(model: Any, onnx_path: Path, spec: ExportSpec) -> float:
    """T6: el mismo clip por torch y por onnxruntime. Devuelve la diferencia máxima.

    Con ruido y no con ceros: un tensor de ceros pasa por cualquier grafo roto, porque
    casi todo multiplicado por cero sigue siendo cero.
    """
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    generador = np.random.default_rng(seed=0)
    clip = generador.integers(0, 256, size=spec.input_shape).astype(np.float32)

    tensor = torch.from_numpy(clip)
    if next(model.parameters()).is_cuda:
        tensor = tensor.cuda()
    with torch.no_grad():
        torch_logits, torch_displ = model(tensor)

    sesion = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_displ = sesion.run(None, {INPUT_NAME: clip})

    peor = max(
        float(np.abs(torch_logits.cpu().numpy() - onnx_logits).max()),
        float(np.abs(torch_displ.cpu().numpy() - onnx_displ).max()),
    )
    if peor > VERIFY_TOLERANCE:
        msg = (
            f"el export no coincide con torch: diferencia máxima {peor:.2e}, tolerancia "
            f"{VERIFY_TOLERANCE:.0e}. Mira el gate-shift: si su `roll` temporal se degradó, "
            "es aquí donde se nota"
        )
        raise ExportError(msg)
    return peor


def sha256_of(path: Path) -> str:
    """SHA-256 del fichero, a trozos. Igual que lo calcula el repo de detección."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    with path.open("rb") as fichero:
        while trozo := fichero.read(1 << 20):
            digest.update(trozo)
    return digest.hexdigest()


def registry_block(onnx_path: Path, spec: ExportSpec, *, name: str = "tdeed-snb") -> str:
    """El bloque de `models/registry.yaml` listo para pegar en el repo de detección.

    **Las formas y los nombres se leen del `.onnx` exportado**, no de la configuración con
    la que se exportó. Es el mismo principio de siempre: lo que importa es lo que hay en
    el fichero, porque es lo que se va a ejecutar.
    """
    import onnx  # noqa: PLC0415

    modelo = onnx.load(str(onnx_path), load_external_data=False)
    entrada = modelo.graph.input[0]
    forma = [d.dim_value for d in entrada.type.tensor_type.shape.dim]
    salidas = [
        (s.name, [d.dim_value for d in s.type.tensor_type.shape.dim]) for s in modelo.graph.output
    ]
    significados = {LOGITS_NAME: "logits", DISPLACEMENT_NAME: "displacement"}

    lineas = [
        f"  {name}:",
        '    version: "0.1.0"',
        "    task: spotting",
        f"    path: onnx/{onnx_path.name}",
        f"    sha256: {sha256_of(onnx_path)}",
        "    license: >",
        "      pesos de T-DEED (GPL-3.0) afinados sobre SoccerNet Ball Action Spotting.",
        "      SOLO PARA MEDIR: ni el código ni los datos son de uso comercial.",
        f'    exported: "{_today()}"',
        f"    opset: {OPSET}",
        "    input:",
        f"      name: {entrada.name}",
        f"      shape: {forma}",
        f"      layout: {LAYOUT}",
        "      color: RGB",
        "      # El modelo normaliza por dentro (x/255 y estadísticas de ImageNet), así",
        "      # que aquí no se toca el píxel: hacerlo lo normalizaría dos veces.",
        "      scale: 1",
        "      mean: [0.0, 0.0, 0.0]",
        "      std: [1.0, 1.0, 1.0]",
        f"      sample_fps: {spec.sample_fps:g}",
        "    activation: softmax",
        "    outputs:",
    ]
    for nombre, forma_salida in salidas:
        significado = significados.get(nombre, "aux")
        lineas.append(f"      - {{name: {nombre}, shape: {forma_salida}, meaning: {significado}}}")
    clases = ", ".join(spec.registry_classes)
    lineas += [
        f"    classes: [{clases}]",
        "    notes: >",
        "      Dos cabezas en el modelo original (SoccerNet 17 + SoccerNetBall 12); el",
        "      grafo ya viene recortado a la segunda. La salida `displacement` lleva el",
        "      desplazamiento temporal por frame y hay que aplicarla antes de buscar",
        "      picos, o los eventos salen movidos.",
    ]
    return "\n".join(lineas) + "\n"


def _today() -> str:
    from datetime import UTC, datetime  # noqa: PLC0415

    return datetime.now(UTC).date().isoformat()


def load_spec(tdeed: Path, config: str = DEFAULT_CONFIG) -> ExportSpec:
    """Lee de T-DEED todo lo que define el export. Reutiliza lo que ya hace T1."""
    from tools.fetch_tdeed import EXTRACTED_FPS, signature  # noqa: PLC0415

    firma = signature(tdeed, config)
    return ExportSpec(
        config=config,
        clip_len=firma.clip_len,
        height=firma.height,
        width=firma.width,
        classes=firma.classes,
        sample_fps=EXTRACTED_FPS / firma.stride,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_onnx",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tdeed", type=Path, default=Path("third_party/T-DEED"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("modelo"))
    parser.add_argument("--name", default="tdeed-snb", help="nombre en el registro")
    parser.add_argument(
        "--skip-verify", action="store_true", help="exportar sin comparar contra torch (T6)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(args.tdeed.resolve()))
    try:
        spec = load_spec(args.tdeed, args.config)
        print(f"clip      : {spec.input_shape} ({LAYOUT})")
        print(f"clases    : {len(spec.registry_classes)} con el fondo")

        modelo = _load_tdeed(args.tdeed, args.config, spec)
        destino = export(build_wrapper(modelo, spec), spec, args.out / f"{args.name}.onnx")
        print(f"exportado : {destino} ({destino.stat().st_size / 1e6:.0f} MB)")

        if not args.skip_verify:
            peor = verify(build_wrapper(modelo, spec), destino, spec)
            print(f"verificado: diferencia máxima {peor:.2e} (tolerancia {VERIFY_TOLERANCE:.0e})")

        ficha = args.out / "registry-block.yaml"
        ficha.write_text(registry_block(destino, spec, name=args.name), encoding="utf-8")
        print(f"ficha     : {ficha}  <- pegar en models/registry.yaml del repo de detección")
    except (ExportError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


def _load_tdeed(tdeed: Path, config: str, spec: ExportSpec) -> Any:
    """Monta el `TDEEDModel` y le carga los pesos, como hace su `inference.py`."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from model.model import TDEEDModel  # noqa: PLC0415
    from util.dataset import load_classes  # noqa: PLC0415
    from util.io import load_json  # noqa: PLC0415

    datos = load_json(str(tdeed / "config" / config.split("_", 1)[0] / f"{config}.json"))
    argumentos = argparse.Namespace(**datos)
    argumentos.crop_dim = None if datos["crop_dim"] <= 0 else datos["crop_dim"]
    argumentos.pretrain = datos.get("pretrain")

    modelo = TDEEDModel(args=argumentos)
    if argumentos.pretrain is not None:
        previas = load_classes(str(tdeed / "data" / argumentos.pretrain["dataset"] / "class.txt"))
        cabezas = [spec.head_width, len(previas) + 1]
        modelo._model.update_pred_head(cabezas)  # noqa: SLF001
        modelo._num_classes = int(np.array(cabezas).sum())  # noqa: SLF001

    pesos = tdeed / "checkpoints" / config.split("_", 1)[0] / config / "checkpoint_best.pt"
    if not pesos.is_file():
        msg = f"no están los pesos en {pesos}"
        raise ExportError(msg)
    modelo.load(torch.load(str(pesos)))
    return modelo._model  # noqa: SLF001


if __name__ == "__main__":
    sys.exit(main())
