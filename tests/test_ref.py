"""La referencia de football-ai, fijada por commit (`ftrain/ref.py`, ML-03).

Con el grupo `ref`, cada nombre llega de `libs.vision`, y el código de tiempo que pinta la
referencia se lee a través de la fachada. Sin él, esos tests se saltan con su motivo, y
pedir un nombre da `RefMissingError` con el comando que lo arregla.
"""

from __future__ import annotations

import pytest

from ftrain import ref

MOTIVO = "falta el grupo ref: uv sync --group ref"
FRAME_4K_BGR = (2160, 3840, 3)
TIEMPO_MS = 5_400_123


def test_lee_el_codigo_de_tiempo_que_pinta_la_referencia():
    np = pytest.importorskip("numpy")
    timecode = pytest.importorskip("libs.vision.source.timecode", reason=MOTIVO)
    frame = np.zeros(FRAME_4K_BGR, dtype=np.uint8)

    timecode.write_timecode(frame, TIEMPO_MS)

    assert ref.read_timecode_ms(frame) == TIEMPO_MS


def test_cada_nombre_sale_del_modulo_de_la_referencia_que_dice():
    pytest.importorskip("libs.vision", reason=MOTIVO)

    for nombre, modulo in ref.ORIGINS.items():
        objeto = getattr(ref, nombre)
        # Casi todo son funciones o clases; TIMECODE_BITS es la constante del
        # ancho del código, que ML-19 necesita tal cual (un int no tiene módulo).
        if isinstance(objeto, int):
            continue
        assert callable(objeto), nombre
        assert objeto.__module__ == modulo, nombre


def test_sin_el_grupo_ref_lo_dice_con_el_comando(monkeypatch):
    def sin_referencia(nombre):
        raise ModuleNotFoundError(name=nombre)

    monkeypatch.setattr(ref, "import_module", sin_referencia)

    with pytest.raises(ref.RefMissingError, match="uv sync --group ref"):
        ref.read_timecode_ms  # noqa: B018 - pedir el nombre es lo que se prueba


def test_un_nombre_que_no_expone_es_un_attribute_error():
    with pytest.raises(AttributeError, match="no expone"):
        ref.write_timecode  # noqa: B018 - pedir el nombre es lo que se prueba
