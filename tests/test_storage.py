"""La envoltura de GCS (ML-04): órdenes exactas, idempotencia por sha y secretos."""

from __future__ import annotations

import pytest

from ftrain.storage import BUCKET_ENV, CREDENTIALS_ENV, Storage, StorageError, sha256_of

ENTORNO = {BUCKET_ENV: "fai-campania", CREDENTIALS_ENV: "/ruta/secreta/clave.json"}


class RunnerFalso:
    """Un gcloud de mentira: apunta las órdenes y contesta lo que se le diga."""

    def __init__(self) -> None:
        self.ordenes: list[list[str]] = []
        self.respuestas: dict[str, str] = {}
        self.fallan: set[str] = set()

    def __call__(self, command):
        self.ordenes.append(list(command))
        clave = " ".join(command)
        if clave in self.fallan:
            msg = "no esta"
            raise StorageError(msg)
        for patron, salida in self.respuestas.items():
            if patron in clave:
                return salida
        return ""


def test_sin_bucket_falla_con_mensaje_claro():
    with pytest.raises(StorageError, match=BUCKET_ENV):
        Storage(environ={CREDENTIALS_ENV: "x"})


def test_sin_credenciales_falla_nombrando_la_variable():
    with pytest.raises(StorageError, match=CREDENTIALS_ENV):
        Storage(environ={BUCKET_ENV: "fai-campania"})


def test_el_dry_run_planifica_las_ordenes_exactas(tmp_path):
    fichero = tmp_path / "left-123.mov"
    fichero.write_bytes(b"cuatro bytes")
    almacen = Storage(environ=ENTORNO, dry_run=True)

    assert almacen.upload(fichero, "raw/m_2026/left/left-123.mov") is True

    assert almacen.commands == [
        ["gcloud", "storage", "cat", "gs://fai-campania/raw/m_2026/left/left-123.mov.sha256"],
        ["gcloud", "storage", "cp", str(fichero), "gs://fai-campania/raw/m_2026/left/left-123.mov"],
        [
            "gcloud",
            "storage",
            "cp",
            str(fichero) + ".sha256",
            "gs://fai-campania/raw/m_2026/left/left-123.mov.sha256",
        ],
    ]
    # El sidecar temporal no se queda tirado ni en dry-run.
    assert not (tmp_path / "left-123.mov.sha256").exists()


def test_subir_dos_veces_el_mismo_fichero_no_resube(tmp_path):
    fichero = tmp_path / "datos.bin"
    fichero.write_bytes(b"inmutable")
    runner = RunnerFalso()
    almacen = Storage(environ=ENTORNO, runner=runner)

    assert almacen.upload(fichero, "datasets/jugadores/v1/datos.bin") is True
    # El bucket ya tiene el sidecar con ese sha: la segunda vez, ni un cp.
    runner.respuestas["cat gs://fai-campania/datasets/jugadores/v1/datos.bin.sha256"] = (
        sha256_of(fichero) + "\n"
    )
    antes = len(runner.ordenes)

    assert almacen.upload(fichero, "datasets/jugadores/v1/datos.bin") is False
    despues = [" ".join(orden) for orden in runner.ordenes[antes:]]
    assert all("cp" not in orden.split() for orden in despues)


def test_un_contenido_distinto_si_se_resube(tmp_path):
    fichero = tmp_path / "datos.bin"
    fichero.write_bytes(b"version nueva")
    runner = RunnerFalso()
    runner.respuestas["cat"] = "unshaviejo\n"
    almacen = Storage(environ=ENTORNO, runner=runner)

    assert almacen.upload(fichero, "datasets/jugadores/v2/datos.bin") is True
    assert any("cp" in orden for orden in runner.ordenes[-2])


def test_la_ruta_de_la_clave_no_aparece_en_ninguna_orden(tmp_path):
    fichero = tmp_path / "x.bin"
    fichero.write_bytes(b"x")
    almacen = Storage(environ=ENTORNO, dry_run=True)
    almacen.upload(fichero, "raw/x.bin")
    almacen.ls("raw/")
    almacen.exists("raw/x.bin")
    almacen.download("raw/x.bin", tmp_path / "bajada.bin")

    todas = " ".join(" ".join(orden) for orden in almacen.commands)
    assert "/ruta/secreta" not in todas
    assert "clave.json" not in todas


def test_exists_y_ls_pasan_por_gcloud():
    runner = RunnerFalso()
    runner.respuestas["ls gs://fai-campania/raw/"] = (
        "gs://fai-campania/raw/a.mov\ngs://fai-campania/raw/b.mov\n"
    )
    runner.fallan.add("gcloud storage ls gs://fai-campania/raw/no-esta.mov")
    almacen = Storage(environ=ENTORNO, runner=runner)

    assert almacen.ls("raw/") == [
        "gs://fai-campania/raw/a.mov",
        "gs://fai-campania/raw/b.mov",
    ]
    assert almacen.exists("raw/no-esta.mov") is False
