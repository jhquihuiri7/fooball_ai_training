"""Tests de `tools/runpod_pod.py` y de los dos scripts del pod (ML-04).

Sin red ni RunPod: el cuerpo que se pide, el campo que se lee de una respuesta y la
referencia de football-ai que se saca de pyproject.toml. De los scripts de shell, que bash
los lea y que la clave de GCS solo salga por la tubería de ssh.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.runpod_pod import (
    GPU,
    IMAGEN,
    MONTAJE,
    PUERTO_SSH,
    campo,
    cuerpo,
    main,
    referencia,
    tipos_de_gpu,
)

RAIZ = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def test_el_cuerpo_por_defecto_pide_una_4090_con_ssh_y_volumen():
    pedido = cuerpo({})

    assert pedido["gpuTypeIds"] == [GPU]
    assert pedido["imageName"] == IMAGEN
    assert pedido["ports"] == [PUERTO_SSH]
    assert pedido["volumeMountPath"] == MONTAJE
    assert pedido["gpuCount"] == 1
    assert pedido["cloudType"] == "SECURE"


def test_varias_gpu_en_orden_de_preferencia():
    assert tipos_de_gpu("NVIDIA RTX A4000, NVIDIA RTX A5000 ,") == [
        "NVIDIA RTX A4000",
        "NVIDIA RTX A5000",
    ]
    assert cuerpo({"GPU_TYPE": "NVIDIA RTX A4000"})["gpuTypeIds"] == ["NVIDIA RTX A4000"]


def test_sin_gpu_falla_nombrando_la_variable():
    with pytest.raises(ValueError, match="GPU_TYPE"):
        cuerpo({"GPU_TYPE": " , "})


def test_los_discos_salen_del_entorno_y_se_validan():
    pedido = cuerpo({"DISK_GB": "80", "VOLUME_GB": "100"})

    assert pedido["containerDiskInGb"] == 80
    assert pedido["volumeInGb"] == 100
    with pytest.raises(ValueError, match="DISK_GB"):
        cuerpo({"DISK_GB": "mucho"})
    with pytest.raises(ValueError, match="VOLUME_GB"):
        cuerpo({"VOLUME_GB": "0"})


def test_la_nube_es_secure_o_community():
    assert cuerpo({"CLOUD_TYPE": "COMMUNITY"})["cloudType"] == "COMMUNITY"
    with pytest.raises(ValueError, match="CLOUD_TYPE"):
        cuerpo({"CLOUD_TYPE": "barata"})


def test_campo_anidado_y_lo_que_falta_sale_vacio():
    respuesta = '{"id": "abc", "portMappings": {"22": 40022}, "machine": null}'

    assert campo(respuesta, "id") == "abc"
    assert campo(respuesta, "portMappings.22") == "40022"
    assert campo(respuesta, "publicIp") == ""
    assert campo(respuesta, "machine.dataCenterId") == ""
    assert campo("<html>error</html>", "id") == ""


def test_la_referencia_sale_del_pyproject_del_repo():
    url, rev = referencia((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))

    assert url.startswith("https://github.com/")
    assert len(rev) == 40
    int(rev, 16)  # un sha entero, no una rama


def test_sin_referencia_fijada_falla_diciendo_que_falta():
    with pytest.raises(ValueError, match="football-ai"):
        referencia("[project]\nname = 'x'\n")
    with pytest.raises(ValueError, match="rev"):
        referencia('[tool.uv.sources]\nfootball-ai = { git = "https://x" }\n')


def test_main_imprime_la_referencia_y_rechaza_lo_desconocido(capsys):
    assert main(["referencia", str(RAIZ / "pyproject.toml")]) == 0
    url, rev = capsys.readouterr().out.split()
    assert (url, rev) == referencia((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))

    assert main(["otra"]) == 2
    assert main(["campo"]) == 2


def test_la_clave_de_gcs_solo_viaja_por_la_tuberia():
    """runpod.sh lee la clave una vez, para el prólogo que va por ssh, y nunca la pone en
    una orden ni en el cuerpo del pod."""
    texto = (RAIZ / "tools" / "runpod.sh").read_text(encoding="utf-8")
    lectura = '"$GOOGLE_APPLICATION_CREDENTIALS")'
    lecturas = [linea for linea in texto.splitlines() if lectura in linea]

    assert len(lecturas) == 1
    assert lecturas[0].strip().startswith("printf 'export GOOGLE_APPLICATION_CREDENTIALS_JSON=%q")


@pytest.mark.skipif(BASH is None, reason="sin bash en el PATH")
@pytest.mark.parametrize("script", ["runpod.sh", "runpod_bootstrap.sh"])
def test_bash_lee_los_scripts(script):
    # En bytes: en modo texto, Windows convertiría los saltos de línea en CRLF por el camino.
    texto = (RAIZ / "tools" / script).read_bytes()
    resultado = subprocess.run([BASH, "-n"], input=texto, capture_output=True, check=False)

    assert resultado.returncode == 0, resultado.stderr.decode("utf-8", "replace")
