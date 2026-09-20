"""Tests de las dos herramientas de T1 y T2.

Lo que se puede probar sin GPU, sin red y sin 19 GB: que la firma de T-DEED se **lee** de
su código en vez de estar copiada aquí, que el orden de clases se conserva, y que la
descarga dice lo que va a costar antes de costarlo.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tools.export_onnx import (
    BACKGROUND_CLASS,
    ExportSpec,
    registry_block,
    sha256_of,
)
from tools.fetch_soccernet import (
    DEFAULT_TASK,
    PASSWORD_ENV,
    SoccerNetError,
    password_for,
    resolve,
)
from tools.fetch_soccernet import (
    main as soccernet_main,
)
from tools.fetch_tdeed import (
    Signature,
    TdeedError,
    read_classes,
    read_frame_size,
    read_stride,
)

_EVAL = """
    EVENT_DICTIONARY = {"PASS":0, "DRIVE":1, "HEADER":2, "HIGH PASS":3, "OUT":4, "CROSS":5,
                        "THROW IN":6, "SHOT":7, "BALL PLAYER BLOCK":8,
                        "PLAYER SUCCESSFUL TACKLE":9, "FREE KICK":10, "GOAL":11}
"""

_EXTRACTOR = """
RECALC_FPS_ONLY = False
TARGET_HEIGHT = 448
TARGET_WIDTH = 796
"""

_INFERENCE = """
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
"""


# --------------------------------------------------------------------------- #
# T1: leer la firma de T-DEED
# --------------------------------------------------------------------------- #


def test_las_clases_salen_en_el_orden_del_modelo():
    # El orden ES el indice de salida: cambiarlo sin reexportar intercambia las etiquetas
    # sin dar ningun error.
    clases = read_classes(_EVAL)

    assert clases[0] == "PASS"
    assert clases[11] == "GOAL"
    assert len(clases) == 12


def test_un_diccionario_con_huecos_no_se_adivina():
    # Si T-DEED cambia su EVENT_DICTIONARY, esto falla en vez de quedarse con una copia
    # vieja, que es justo el error que el registro del otro repo existe para evitar.
    with pytest.raises(TdeedError, match="EVENT_DICTIONARY"):
        read_classes('{"PASS":0, "GOAL":5}')


def test_sin_clases_falla_en_vez_de_devolver_nada():
    with pytest.raises(TdeedError, match="EVENT_DICTIONARY"):
        read_classes("aqui no hay ningun diccionario")


def test_el_tamano_del_frame_se_lee_del_extractor():
    assert read_frame_size(_EXTRACTOR) == (448, 796)


def test_un_extractor_sin_tamanos_lo_dice():
    with pytest.raises(TdeedError, match="TARGET_HEIGHT"):
        read_frame_size("sin constantes")


def test_el_stride_es_el_de_ball_action_spotting():
    # STRIDE_SNB, no STRIDE ni STRIDE_SN: cada tarea muestrea a un ritmo distinto.
    assert read_stride(_INFERENCE) == 2


def test_la_ventana_sale_del_clip_y_del_stride():
    # 100 frames, uno de cada 2 de los 25 extraidos: 12,5 fps y 8 s de ventana. Es el
    # numero que decide cuanta memoria ocupa la ventana en el panel.
    firma = Signature(
        config="SoccerNetBall_challenge1",
        clip_len=100,
        stride=2,
        height=448,
        width=796,
        classes=read_classes(_EVAL),
        feature_arch="rny002_gsf",
        temporal_arch="ed_sgp_mixer",
    )

    assert firma.sample_fps == 12.5
    assert firma.window_s == 8.0
    assert "12 —" in " ".join(firma.as_lines())


# --------------------------------------------------------------------------- #
# T2: saber lo que cuesta antes de que cueste
# --------------------------------------------------------------------------- #


def test_la_tarea_del_balon_no_trae_corner():
    # Las cuatro clases del MVP son gol, tiro, corner y tiro libre, y no hay una sola
    # descarga que las de todas. Mejor saberlo antes que a mitad de camino.
    tarea = resolve("spotting-ball-2024")

    assert not tarea.has_corner
    assert tarea.classes == 12


def test_la_tarea_de_17_clases_si_lo_trae_pero_pide_el_nda():
    tarea = resolve("spotting-2023")

    assert tarea.has_corner
    assert tarea.needs_password


def test_una_tarea_desconocida_dice_cuales_hay():
    with pytest.raises(SoccerNetError, match="spotting-ball-2024"):
        resolve("la-que-yo-quiera")


def test_la_tarea_de_hugging_face_no_pide_contrasena():
    assert password_for(resolve("spotting-ball-2024"), {}) is None


def test_sin_contrasena_no_se_intenta_siquiera():
    # Fallar aqui es mucho mejor que fallar a los veinte minutos de descarga.
    with pytest.raises(SoccerNetError, match=PASSWORD_ENV):
        password_for(resolve("spotting-2023"), {})


def test_una_contrasena_en_blanco_es_un_olvido_no_una_contrasena():
    with pytest.raises(SoccerNetError, match=PASSWORD_ENV):
        password_for(resolve("spotting-2023"), {PASSWORD_ENV: "   "})


def test_la_contrasena_se_coge_del_entorno():
    assert password_for(resolve("spotting-2023"), {PASSWORD_ENV: "secreta"}) == "secreta"


def test_check_ensena_el_plan_y_no_descarga(capsys):
    assert soccernet_main(["--check"]) == 0

    salida = capsys.readouterr().out
    assert DEFAULT_TASK in salida
    assert "GB" in salida


def test_check_avisa_de_que_esa_tarea_no_tiene_corner(capsys):
    soccernet_main(["--check", "--task", "spotting-ball-2024"])

    assert "córner: no" in capsys.readouterr().out


def test_una_tarea_inventada_falla_sin_tocar_la_red(capsys):
    assert soccernet_main(["--task", "inventada"]) == 1

    assert "ERROR" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# T5: la ficha que cruza al repo de detección
# --------------------------------------------------------------------------- #


def _export_spec(**cambios):
    base = {
        "config": "SoccerNetBall_challenge1",
        "clip_len": 100,
        "height": 448,
        "width": 796,
        "classes": read_classes(_EVAL),
        "sample_fps": 12.5,
    }
    return ExportSpec(**{**base, **cambios})


def test_el_clip_que_se_exporta_es_estatico_y_ntchw():
    # Ejes fijos a proposito: con ellos la aritmetica de formas del gate-shift se pliega
    # a constantes en vez de quedarse en el grafo.
    assert _export_spec().input_shape == (1, 100, 3, 448, 796)


def test_la_cabeza_lleva_el_fondo_ademas_de_las_clases():
    # El modelo emite 1 + 12 columnas para nuestra cabeza. Recortar a 12 se comeria el
    # fondo y desplazaria todas las etiquetas una posicion.
    spec = _export_spec()

    assert spec.head_width == 13
    assert len(spec.registry_classes) == 13


def test_la_columna_cero_se_llama_como_el_fondo_del_spotter():
    # No es casualidad: `SPOTTER_BACKGROUND_CLASS` del repo de deteccion es ese nombre, y
    # asi la clase de fondo nunca produce candidatos sin que nadie configure nada.
    spec = _export_spec()

    assert spec.registry_classes[0] == BACKGROUND_CLASS
    assert spec.registry_classes[1] == "PASS"
    assert spec.registry_classes[-1] == "GOAL"


def _onnx_de_prueba(path, forma=(1, 100, 3, 448, 796), clases=13):
    """Un `.onnx` con la firma que tendra el exportado: dos salidas y ejes fijos.

    `clases` es el ancho de la cabeza ya recortada, o sea el fondo mas las doce del
    dataset. Tiene que cuadrar con la lista de `classes` de la ficha: el repo de deteccion
    lo comprueba al cargar, y con razon.
    """
    ceros_logits = np.zeros((1, forma[1], clases), dtype=np.float32)
    ceros_displ = np.zeros((1, forma[1]), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["logits"],
                value=helper.make_tensor(
                    "l", TensorProto.FLOAT, ceros_logits.shape, ceros_logits.flatten()
                ),
            ),
            helper.make_node(
                "Constant",
                [],
                ["displacement"],
                value=helper.make_tensor(
                    "d", TensorProto.FLOAT, ceros_displ.shape, ceros_displ.flatten()
                ),
            ),
        ],
        "stub",
        [helper.make_tensor_value_info("clip", TensorProto.FLOAT, list(forma))],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, forma[1], clases]),
            helper.make_tensor_value_info("displacement", TensorProto.FLOAT, [1, forma[1]]),
        ],
    )
    modelo = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    modelo.ir_version = 10
    onnx.save(modelo, str(path))
    return path


def test_la_ficha_sale_del_onnx_y_no_de_la_configuracion(tmp_path):
    # Lo que importa es lo que hay en el fichero, porque es lo que se va a ejecutar.
    ruta = _onnx_de_prueba(tmp_path / "tdeed-snb.onnx", forma=(1, 50, 3, 224, 398))

    bloque = registry_block(ruta, _export_spec())

    assert "shape: [1, 50, 3, 224, 398]" in bloque
    assert "layout: NTCHW" in bloque


def test_la_ficha_no_vuelve_a_normalizar_el_pixel(tmp_path):
    # El modelo hace x/255 y las estadisticas de ImageNet por dentro. Copiar aqui los
    # valores del paper normalizaria dos veces: no da error, solo hunde el score.
    bloque = registry_block(_onnx_de_prueba(tmp_path / "m.onnx"), _export_spec())

    assert "scale: 1" in bloque
    assert "mean: [0.0, 0.0, 0.0]" in bloque
    assert "std: [1.0, 1.0, 1.0]" in bloque


def test_la_ficha_marca_la_salida_del_desplazamiento(tmp_path):
    # El spotter tiene que aplicarlo antes de buscar picos, o los eventos salen movidos.
    bloque = registry_block(_onnx_de_prueba(tmp_path / "m.onnx"), _export_spec())

    assert "meaning: displacement" in bloque
    assert "meaning: logits" in bloque


def test_la_ficha_lleva_el_sha_del_fichero_exportado(tmp_path):
    ruta = _onnx_de_prueba(tmp_path / "m.onnx")

    assert f"sha256: {sha256_of(ruta)}" in registry_block(ruta, _export_spec())


def test_la_ficha_avisa_de_que_los_pesos_no_son_para_vender(tmp_path):
    bloque = registry_block(_onnx_de_prueba(tmp_path / "m.onnx"), _export_spec())

    assert "GPL-3.0" in bloque
    assert "SOLO PARA MEDIR" in bloque
