"""El paquete multifunción (SPK-52): fusión con pesos deduplicados."""

from __future__ import annotations

import sys

import pytest
import yaml

torch = pytest.importorskip("torch", reason="el grupo `train` no está instalado")
ct = pytest.importorskip("coremltools", reason="sin ruedas de coremltools aquí")

from tools.export_multifunction import export_multifunction  # noqa: E402


def _spec(tmp_path, nombre: str, forma: list[int]):
    ruta = tmp_path / f"{nombre}.yaml"
    ruta.write_text(
        yaml.safe_dump(
            {
                "model_name": "roilite-multi-test",
                "inputs": [{"name": "frames", "kind": "tensor", "shape": forma}],
                "outputs": ["heatmap", "offset"],
            }
        ),
        encoding="utf-8",
    )
    return ruta


def test_fusiona_dos_funciones_y_los_pesos_se_comparten(tmp_path):
    paquete, sha = export_multifunction(
        "ftrain.ball.model:build",
        {
            "global": _spec(tmp_path, "global", [1, 3, 64, 64]),
            "roi": _spec(tmp_path, "roi", [2, 3, 32, 32]),
        },
        tmp_path / "dist",
        classes=["ball"],
    )

    assert paquete.is_dir()
    assert len(sha) == 64

    # La deduplicación es la gracia: el multifunción pesa ≤1,1x un export suelto.
    def _peso(ruta):
        return sum(f.stat().st_size for f in ruta.rglob("*") if f.is_file())

    suelto, _ = export_multifunction(
        "ftrain.ball.model:build",
        {"global": _spec(tmp_path, "solo", [1, 3, 64, 64])},
        tmp_path / "solo",
        classes=["ball"],
    )
    assert _peso(paquete) <= 1.1 * _peso(suelto)

    if sys.platform == "darwin":  # predecir pide macOS
        import numpy as np  # noqa: PLC0415

        for funcion, forma in (("global", (1, 3, 64, 64)), ("roi", (2, 3, 32, 32))):
            modelo = ct.models.MLModel(str(paquete), function_name=funcion)
            salida = modelo.predict({"frames": np.zeros(forma, dtype=np.float16)})
            assert set(salida) == {"heatmap", "offset"}
