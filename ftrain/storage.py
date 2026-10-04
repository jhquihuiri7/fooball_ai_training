"""Los datos en GCS: la envoltura de `gcloud storage` (ML-04, ADR 0002 §3).

Todo lo que entra al bucket es **inmutable y versionado**: una versión nueva es una
ruta nueva, nunca una sobreescritura, y cada fichero viaja con su `sha256` en un
sidecar (`<fichero>.sha256`). La idempotencia sale de ahí: subir dos veces lo mismo
compara el sha y no resube. El árbol del bucket está en `docs/DATOS.md`.

El runner es inyectable y hay `dry_run` a propósito: los tests comprueban las órdenes
exactas sin tocar la red, y una campaña se puede ensayar antes de gastar ancho de
banda. El bucket sale de `FAI_GCS_BUCKET` y las credenciales de
`GOOGLE_APPLICATION_CREDENTIALS`; de esa variable solo se comprueba que exista:
su valor —la ruta de la clave— no aparece en órdenes, errores ni logs.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

__all__ = ["BUCKET_ENV", "CREDENTIALS_ENV", "Storage", "StorageError", "sha256_of"]

BUCKET_ENV: Final = "FAI_GCS_BUCKET"
"""Variable de entorno con el nombre del bucket, sin el prefijo `gs://`."""

CREDENTIALS_ENV: Final = "GOOGLE_APPLICATION_CREDENTIALS"
"""Variable con la RUTA de la clave de servicio. Solo se comprueba su presencia."""

_HASH_CHUNK_BYTES: Final = 1 << 20
"""Trozo con el que se lee un fichero para su sha256: 1 MiB, como en la referencia."""

Runner = "Callable[[Sequence[str]], str]"
if TYPE_CHECKING:
    Runner = Callable[[Sequence[str]], str]  # type: ignore[misc]


class StorageError(RuntimeError):
    """El bucket no se puede usar. El mensaje dice qué falta, nunca un secreto."""


def sha256_of(path: Path) -> str:
    """SHA-256 del fichero, leído a trozos."""
    digest = hashlib.sha256()
    with path.open("rb") as fichero:
        while trozo := fichero.read(_HASH_CHUNK_BYTES):
            digest.update(trozo)
    return digest.hexdigest()


def _default_runner(command: Sequence[str]) -> str:
    resultado = subprocess.run(list(command), capture_output=True, text=True, check=False)
    if resultado.returncode != 0:
        msg = f"fallo `{command[0]} {command[1]} {command[2]}…` (código {resultado.returncode})"
        raise StorageError(msg)
    return resultado.stdout


class Storage:
    """El bucket de la campaña, con órdenes de `gcloud storage` y nada más."""

    def __init__(
        self,
        bucket: str | None = None,
        *,
        runner: Runner | None = None,
        dry_run: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        entorno = os.environ if environ is None else environ
        nombre = bucket if bucket is not None else entorno.get(BUCKET_ENV, "")
        if not nombre:
            msg = (
                f"falta {BUCKET_ENV}: el bucket de la campaña no se adivina. "
                f"Exporta {BUCKET_ENV}=<bucket> (sin gs://)"
            )
            raise StorageError(msg)
        if CREDENTIALS_ENV not in entorno:
            msg = (
                f"falta {CREDENTIALS_ENV}: gcloud necesita la clave de servicio. "
                "Solo se comprueba que la variable exista; su valor no se toca"
            )
            raise StorageError(msg)
        self._bucket = nombre.removeprefix("gs://").strip("/")
        self._runner: Callable[[Sequence[str]], str] = runner or _default_runner
        self._dry_run = dry_run
        self.commands: list[list[str]] = []
        """Las órdenes exactas, ejecutadas o planificadas (dry-run): es lo que los
        tests congelan y lo que se enseña antes de gastar ancho de banda."""

    # ------------------------------------------------------------------ #
    def _url(self, remote: str) -> str:
        return f"gs://{self._bucket}/{remote.lstrip('/')}"

    def _run(self, command: list[str]) -> str:
        self.commands.append(command)
        if self._dry_run:
            return ""
        return self._runner(command)

    # ------------------------------------------------------------------ #
    def exists(self, remote: str) -> bool:
        """Si el objeto está en el bucket. En dry-run, `False`: aún no hay nada."""
        comando = ["gcloud", "storage", "ls", self._url(remote)]
        self.commands.append(comando)
        if self._dry_run:
            return False
        try:
            self._runner(comando)
        except StorageError:
            return False
        return True

    def ls(self, prefix: str) -> list[str]:
        """Los objetos bajo un prefijo, uno por línea, como los lista gcloud."""
        salida = self._run(["gcloud", "storage", "ls", self._url(prefix)])
        return [linea for linea in salida.splitlines() if linea]

    def remote_sha(self, remote: str) -> str | None:
        """El sha256 del sidecar del objeto, o `None` si no está subido."""
        comando = ["gcloud", "storage", "cat", self._url(remote) + ".sha256"]
        self.commands.append(comando)
        if self._dry_run:
            return None
        try:
            return self._runner(comando).strip() or None
        except StorageError:
            return None

    def upload(self, local: Path, remote: str) -> bool:
        """Sube el fichero y su sidecar sha256. Idempotente: con el mismo sha no
        resube. Devuelve si hubo subida."""
        if not local.is_file():
            msg = f"no existe el fichero local: {local}"
            raise StorageError(msg)
        sha = sha256_of(local)
        if self.remote_sha(remote) == sha:
            return False
        sidecar = local.with_name(local.name + ".sha256")
        sidecar.write_text(sha + "\n", encoding="ascii")
        try:
            self._run(["gcloud", "storage", "cp", str(local), self._url(remote)])
            self._run(["gcloud", "storage", "cp", str(sidecar), self._url(remote) + ".sha256"])
        finally:
            sidecar.unlink(missing_ok=True)
        return True

    def download(self, remote: str, local: Path) -> None:
        """Baja un objeto tal cual. La caché por versión la gobierna quien llama:
        las versiones son inmutables, así que un fichero ya bajado no caduca."""
        local.parent.mkdir(parents=True, exist_ok=True)
        self._run(["gcloud", "storage", "cp", self._url(remote), str(local)])
