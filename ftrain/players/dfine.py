"""D-FINE-N desde la config de DEIM, en deploy y sin PostProcessor (ML-16).

El modelo se construye con la MISMA receta de DEIM (YAMLConfig sobre la config
fijada por ML-15) y `eval_spatial_size` a nuestra banda, que es lo que fija los
anchors y el embedding posicional. El wrapper devuelve crudo lo que el contrato
declara — `logits` [1,300,C] y `boxes` [1,300,4] cxcywh normalizado — sin el
PostProcessor: su topk de detecciones finales no entra al grafo. El topk que SÍ
queda es el del selector de queries del decoder (300 de entre los anchors), que
es parte de la arquitectura, no del postproceso.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

DEIM_DIR: Final = Path(__file__).resolve().parent.parent.parent / "third_party" / "DEIM"
"""Donde deja el repo tools/fetch_deim.py (ML-15), al commit fijado."""

DFINE_N_CONFIG: Final = "configs/deim_dfine/dfine_hgnetv2_n_coco.yml"

BAND_SIZE: Final = (576, 1920)
"""(alto, ancho) de la banda jugable (ADR 0020). El orden es el de DEIM: h, w."""

SETBACK_SIZE: Final = (512, 1536)
"""(alto, ancho) del retranqueo de M3, la otra forma que mide SPK-51."""

_SIZE_BUFFERS: Final = ("decoder.anchors", "decoder.valid_mask")
"""Búferes que son función de `eval_spatial_size`: el checkpoint los trae de
640x640 y aquí se regeneran a nuestra forma, así que no se cargan."""


class DfineError(RuntimeError):
    """No se pudo montar D-FINE. El mensaje dice qué falta."""


def _patch_integral() -> None:
    """Reescribe `Integral.forward` para que exporte a Core ML.

    El original hace `F.linear(x, project)` con `project` como VECTOR (la función
    de pesos W(n) del DFL), y el op `linear` de MIL exige pesos de rango 2. Esto
    es la misma cuenta escrita como matvec —`x @ project`—, no una aproximación:
    la paridad torch↔ORT del export se mide con el modelo ya parcheado y el
    parche no cambia ni un bit la salida de torch (mismo kernel de matmul)."""
    import torch  # noqa: PLC0415 — perezoso adrede (grupo train)
    import torch.nn.functional as F  # noqa: N812, PLC0415 — el alias canónico
    from engine.deim import dfine_decoder  # noqa: PLC0415 — vive en third_party/DEIM

    def forward(self: Any, x: Any, project: Any) -> Any:
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        # Columna [R, 1] y no vector: el matmul de MIL exige rango 2 en los dos lados.
        columna = project.to(x.device, x.dtype).reshape(-1, 1)
        x = torch.matmul(x, columna).reshape(-1, 4)
        return x.reshape([*list(shape[:-1]), -1])

    dfine_decoder.Integral.forward = forward


def _load_state(checkpoint: Path) -> dict[str, Any]:
    """El estado del checkpoint de DEIM: el EMA si lo hay, si no el modelo.

    `weights_only=False` solo aquí: el fichero viene de la release fijada por
    sha256 en tools/fetch_deim.py, no de un origen desconocido."""
    import torch  # noqa: PLC0415 — perezoso adrede (grupo train)

    crudo = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if "ema" in crudo:
        return crudo["ema"]["module"]
    if "model" in crudo:
        return crudo["model"]
    return crudo


def build(
    checkpoint: Path | str | None,
    *,
    size: tuple[int, int] = BAND_SIZE,
    deim_dir: Path = DEIM_DIR,
) -> Any:
    """El nn.Module listo para exportar: deploy, eval y las dos salidas crudas."""
    import torch  # noqa: PLC0415 — perezoso adrede (grupo train)

    if not deim_dir.is_dir():
        msg = f"no está DEIM en {deim_dir}: corre `uv run python tools/fetch_deim.py`"
        raise DfineError(msg)
    if str(deim_dir) not in sys.path:
        sys.path.insert(0, str(deim_dir))
    from engine.core import YAMLConfig  # noqa: PLC0415 — vive en third_party/DEIM

    _patch_integral()
    cfg = YAMLConfig(str(deim_dir / DFINE_N_CONFIG))
    alto, ancho = size
    cfg.yaml_cfg["eval_spatial_size"] = [alto, ancho]
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    modelo = cfg.model
    if checkpoint is not None:
        estado = {
            clave: valor
            for clave, valor in _load_state(Path(checkpoint)).items()
            if clave not in _SIZE_BUFFERS
        }
        faltan, sobran = modelo.load_state_dict(estado, strict=False)
        raras = [clave for clave in faltan if clave not in _SIZE_BUFFERS]
        if raras or sobran:
            msg = f"el checkpoint no cuadra con el modelo: faltan {raras}, sobran {sobran}"
            raise DfineError(msg)
    modelo = modelo.deploy().eval()

    class _SinPostproceso(torch.nn.Module):
        """logits [1,300,C] y boxes [1,300,4] cxcywh normalizado, tal cual."""

        def __init__(self) -> None:
            super().__init__()
            self.inner = modelo

        def forward(self, images: Any) -> tuple[Any, Any]:
            salida = self.inner(images)
            return salida["pred_logits"], salida["pred_boxes"]

    return _SinPostproceso().eval()


def build_band(checkpoint: Path | str | None) -> Any:
    """El builder de la banda (576x1920), con la firma que piden los CLI."""
    return build(checkpoint, size=BAND_SIZE)


def build_setback(checkpoint: Path | str | None) -> Any:
    """El builder del retranqueo (512x1536), la otra forma de SPK-51."""
    return build(checkpoint, size=SETBACK_SIZE)


ONNX_OPSET: Final = 17
"""El opset del contrato (ADR 0020): el «0,5 AP» de terceros era TensorRT y se
arregló con el 17; el exportador dynamo lo ignora y clava 18, así que aquí se
usa el clásico."""


def export_onnx(module: Any, size: tuple[int, int], destination: Path) -> Path:
    """El `.onnx` fp32, opset 17 y TODO estático. Devuelve su ruta.

    El exportador clásico deja los ejes de salida sin inferir (0): aquí se
    infieren con `onnx.shape_inference` y el lote —que es 1 por construcción,
    el export no declara ejes dinámicos— se fija a mano. La ficha los lee del
    fichero, así que tienen que estar."""
    import onnx  # noqa: PLC0415 — junto a su consumidor
    import torch  # noqa: PLC0415 — perezoso adrede (grupo train)

    alto, ancho = size
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module.eval(),
        (torch.zeros(1, 3, alto, ancho),),
        str(destination),
        input_names=["image"],
        output_names=["logits", "boxes"],
        opset_version=ONNX_OPSET,
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )
    modelo = onnx.shape_inference.infer_shapes(onnx.load(str(destination)), data_prop=True)
    for salida in modelo.graph.output:
        lote = salida.type.tensor_type.shape.dim[0]
        if lote.dim_value != 1:
            lote.Clear()  # borra el dim_param simbólico que deja la inferencia
            lote.dim_value = 1
    onnx.save(modelo, str(destination))
    return destination
