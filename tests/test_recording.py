"""Verificación de grabaciones (ML-07), con vídeos sintéticos de PyAV.

Los vídeos de prueba son mpeg4 pequeños con el código de tiempo pintado POR LA
REFERENCIA (grupo `ref`): los umbrales son inyectables y aquí se verifican contra lo
que el sintético hace a propósito (sin código, a 25 fps, retrocediendo…).
"""

from __future__ import annotations

import numpy as np
import pytest

av = pytest.importorskip("av", reason="el grupo `data` no está instalado")
pytest.importorskip("libs.vision", reason="el grupo `ref` no está instalado")

from libs.vision.source.timecode import write_timecode  # noqa: E402

from ftrain.recording import RecordingLimits, check_pair_overlap, check_recording  # noqa: E402

W, H, FPS = 640, 360, 30


def _video(
    path,
    *,
    frames: int = 90,
    fps: int = FPS,
    rig_ms0: int | None = 1000,
    step_ms: int | None = None,
):
    """Un clip sintético con el código pintado (o no, con rig_ms0=None)."""
    paso = step_ms if step_ms is not None else 1000 // fps
    with av.open(str(path), "w") as contenedor:
        stream = contenedor.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 4_000_000
        for indice in range(frames):
            gris = np.full((H, W), 90, np.uint8)
            gris[H // 2 :, :] = (40 + 3 * (indice % 50)) % 256  # algo de movimiento
            if rig_ms0 is not None:
                write_timecode(gris, rig_ms0 + indice * paso)
            frame = av.VideoFrame.from_ndarray(np.dstack([gris] * 3), format="rgb24")
            for paquete in stream.encode(frame):
                contenedor.mux(paquete)
        for paquete in stream.encode():
            contenedor.mux(paquete)
    return path


def _check(path, **kwargs):
    base = {
        "codec": "mpeg4",
        "width": W,
        "height": H,
        "fps": float(FPS),
        "min_bitrate_bps": 100_000.0,
        "sample_interval_s": 0.2,
    }
    base.update(kwargs)
    return check_recording(path, limits=RecordingLimits(**base))


def test_un_clip_bueno_da_ok(tmp_path):
    informe = _check(_video(tmp_path / "ok.mov"))

    assert informe.ok, informe.checks
    assert informe.rig_ms_first == 1000
    assert informe.rig_ms_last is not None
    assert informe.rig_ms_last > 1000


def test_sin_timecode_falla_con_el_motivo_exacto(tmp_path):
    informe = _check(_video(tmp_path / "mudo.mov", rig_ms0=None))

    assert not informe.ok
    fallo = next(c for c in informe.checks if c["name"] == "timecode_readable")
    assert fallo["ok"] is False
    assert "legibles" in str(fallo["detail"])


def test_a_veinticinco_fps_falla_nombrando_la_cadencia(tmp_path):
    informe = _check(_video(tmp_path / "lento.mov", fps=25))

    assert not informe.ok
    fallo = next(c for c in informe.checks if c["name"] == "fps")
    assert fallo["ok"] is False
    assert "25" in str(fallo["detail"])


def test_un_rigms_que_retrocede_se_caza(tmp_path):
    informe = _check(_video(tmp_path / "atras.mov", step_ms=-20, rig_ms0=900_000))

    assert not informe.ok
    fallo = next(c for c in informe.checks if c["name"] == "timecode_monotonic")
    assert fallo["ok"] is False


def test_un_bitrate_hundido_falla(tmp_path):
    informe = _check(_video(tmp_path / "flaco.mov"), min_bitrate_bps=1e9)

    assert not informe.ok
    assert any(c["name"] == "bitrate" and not c["ok"] for c in informe.checks)


def test_el_codec_equivocado_falla(tmp_path):
    informe = _check(_video(tmp_path / "otro.mov"), codec="hevc")

    assert not informe.ok
    assert any(c["name"] == "codec" and not c["ok"] for c in informe.checks)


def test_el_solape_entre_camaras_se_mide_sobre_la_union(tmp_path):
    izquierda = _check(_video(tmp_path / "l.mov", rig_ms0=1000, frames=90))
    derecha = _check(_video(tmp_path / "r.mov", rig_ms0=1033, frames=90))

    pareja = check_pair_overlap(izquierda, derecha)
    assert pareja["ok"] is True

    # La derecha arranca cuando la izquierda va por la mitad: el solape se hunde.
    tarde = _check(_video(tmp_path / "tarde.mov", rig_ms0=2500, frames=90))
    pareja = check_pair_overlap(izquierda, tarde, min_overlap=0.95)
    assert pareja["ok"] is False
    assert "%" in str(pareja["detail"])
