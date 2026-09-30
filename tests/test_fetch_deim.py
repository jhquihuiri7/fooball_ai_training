"""Tests de `tools/fetch_deim.py` (ML-15), sin red.

Lo que importa probar es que la herramienta se **para**: si la LICENSE de DEIM cambia, si
los pesos no son los fijados o si alguien pide unos de Objects365. El clon se prueba contra
un repo de git local, en `tmp_path`.
"""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
from pathlib import Path

import pytest

from tools.fetch_deim import (
    DEIM_COMMIT,
    DFINE_N_COCO,
    LICENSE_PIN,
    DeimError,
    Weights,
    check_license,
    check_weights_name,
    clone,
    download,
    sha256_file,
)

RAIZ = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# El pin
# --------------------------------------------------------------------------- #


def test_el_pin_de_la_licencia_es_el_del_commit_fijado():
    assert re.fullmatch(r"[0-9a-f]{40}", DEIM_COMMIT)
    assert LICENSE_PIN.name == f"DEIM-LICENSE-{DEIM_COMMIT[:7]}.txt"
    assert (RAIZ / LICENSE_PIN).is_file()


def test_la_licencia_versionada_es_apache_2():
    texto = (RAIZ / LICENSE_PIN).read_text(encoding="utf-8")

    assert "DEIM is licensed under the Apache License" in texto
    assert "Version 2.0, January 2004" in texto


def test_los_pesos_fijados_son_los_coco_de_d_fine_n():
    assert DFINE_N_COCO.name == "dfine_n_coco.pth"
    assert DFINE_N_COCO.url.startswith("https://")
    assert re.fullmatch(r"[0-9a-f]{64}", DFINE_N_COCO.sha256)
    check_weights_name(DFINE_N_COCO.name)


# --------------------------------------------------------------------------- #
# La licencia
# --------------------------------------------------------------------------- #


def _clon_con_licencia(tmp_path: Path, texto: str) -> Path:
    clon = tmp_path / "DEIM"
    clon.mkdir()
    (clon / "LICENSE").write_bytes(texto.encode("utf-8"))
    return clon


def test_la_misma_licencia_pasa_aunque_cambien_los_finales_de_linea(tmp_path: Path):
    pin = RAIZ / LICENSE_PIN
    texto = pin.read_text(encoding="utf-8").replace("\r\n", "\n")

    check_license(_clon_con_licencia(tmp_path, texto.replace("\n", "\r\n")), pin)


def test_cambiar_una_linea_de_la_licencia_para_el_fetch(tmp_path: Path):
    pin = RAIZ / LICENSE_PIN
    texto = pin.read_text(encoding="utf-8").replace(
        "DEIM is licensed under the Apache License.",
        "DEIM is licensed for non-commercial use only.",
    )

    with pytest.raises(DeimError, match="non-commercial"):
        check_license(_clon_con_licencia(tmp_path, texto), pin)


def test_sin_la_copia_versionada_no_se_usa(tmp_path: Path):
    with pytest.raises(DeimError, match="se versiona antes"):
        check_license(_clon_con_licencia(tmp_path, "Apache"), tmp_path / "no-existe.txt")


def test_sin_clon_lo_dice(tmp_path: Path):
    with pytest.raises(DeimError, match="clonado"):
        check_license(tmp_path / "vacio", RAIZ / LICENSE_PIN)


# --------------------------------------------------------------------------- #
# Los pesos
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nombre", ["dfine_s_obj365.pth", "dfine_m_obj2coco.pth"])
def test_los_pesos_de_objects365_estan_vetados(nombre: str):
    with pytest.raises(DeimError, match="vetados"):
        check_weights_name(nombre)


def _pesos(contenido: bytes, nombre: str = "dfine_n_coco.pth") -> Weights:
    return Weights(nombre, "https://ejemplo/pesos", hashlib.sha256(contenido).hexdigest())


def test_baja_y_comprueba_el_sha(tmp_path: Path):
    contenido = b"pesos de prueba"

    destino = download(_pesos(contenido), tmp_path, opener=lambda _url: io.BytesIO(contenido))

    assert destino.read_bytes() == contenido
    assert sha256_file(destino) == hashlib.sha256(contenido).hexdigest()
    assert not list(tmp_path.glob("*.part"))


def test_un_sha_distinto_no_deja_nada_en_disco(tmp_path: Path):
    pesos = _pesos(b"los que se fijaron")

    with pytest.raises(DeimError, match="cambiaron"):
        download(pesos, tmp_path, opener=lambda _url: io.BytesIO(b"otros"))

    assert list(tmp_path.iterdir()) == []


def test_si_ya_estan_y_cuadran_no_baja_nada(tmp_path: Path):
    contenido = b"ya bajados"
    (tmp_path / "dfine_n_coco.pth").write_bytes(contenido)

    def sin_red(url: str) -> io.BytesIO:
        pytest.fail(f"no tenía que bajar {url}")

    download(_pesos(contenido), tmp_path, opener=sin_red)


def test_unos_pesos_vetados_no_se_llegan_a_pedir(tmp_path: Path):
    def sin_red(url: str) -> io.BytesIO:
        pytest.fail(f"no tenía que bajar {url}")

    with pytest.raises(DeimError, match="vetados"):
        download(_pesos(b"x", "dfine_l_obj365.pth"), tmp_path, opener=sin_red)


# --------------------------------------------------------------------------- #
# El clon, contra un repo local
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    salida = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return salida.stdout.strip()


@pytest.fixture
def origen(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "origen"
    repo.mkdir()
    _git("init", "--quiet", str(repo))
    (repo / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    _git("-C", str(repo), "add", "LICENSE")
    _git("-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a")
    return repo, _git("-C", str(repo), "rev-parse", "HEAD")


def test_clona_al_commit_fijado_y_devuelve_su_arbol(tmp_path: Path, origen: tuple[Path, str]):
    repo, commit = origen
    destino = tmp_path / "third_party" / "DEIM"

    arbol = clone(destino, url=str(repo), commit=commit)

    assert arbol == _git("-C", str(repo), "rev-parse", "HEAD^{tree}")
    assert _git("-C", str(destino), "rev-parse", "HEAD") == commit
    # Una segunda vez no clona ni cambia nada.
    assert clone(destino, url=str(repo), commit=commit) == arbol


def test_un_clon_tocado_a_mano_se_para(tmp_path: Path, origen: tuple[Path, str]):
    repo, commit = origen
    destino = tmp_path / "DEIM"
    clone(destino, url=str(repo), commit=commit)
    (destino / "LICENSE").write_text("otra cosa\n", encoding="utf-8")

    with pytest.raises(DeimError, match="cambios locales"):
        clone(destino, url=str(repo), commit=commit)
