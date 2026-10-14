#!/usr/bin/env bash
# Arranca un pod de RunPod para entrenar (ML-04): repo al commit, entorno y dataset.
#
# Lo normal es no lanzarlo a mano: `tools/runpod.sh`, desde tu máquina, crea el pod, sube
# los dos repos y lo corre dentro con todo puesto. A mano, en un pod que ya tengas:
#
#   export GOOGLE_APPLICATION_CREDENTIALS_JSON='<contenido del json de la cuenta>'
#   export FAI_GCS_BUCKET=<bucket>            # sin gs://
#   bash tools/runpod_bootstrap.sh <commit> [datasets/<nombre>/<version>]
#
# De dónde sale el código (lo pone runpod.sh; a mano, por defecto, de GitHub):
#   FAI_TRAINING_SRC   este repo: una URL o un `git bundle`.
#   FAI_REF_SRC        football-ai en un `git bundle`, con FAI_REF_URL y FAI_REF_REV: la URL
#                      y el commit que fija pyproject.toml. Ese repo es privado y el pod no lo
#                      puede bajar de GitHub; git lo sirve desde el bundle cuando uv lo pide.
#
# Modelado sobre tools/export_on_vm.sh. La clave llega POR VARIABLE DE ENTORNO y se
# escribe a un fichero 600 dentro del pod: nunca por argv, nunca en la imagen y nunca
# por la salida. El dataset se baja con caché en el volumen de red (/workspace/cache):
# las versiones son inmutables (docs/DATOS.md), así que lo cacheado no caduca.

set -euo pipefail

COMMIT="${1:?falta el commit de fooball_ai_training a clonar}"
DATASET="${2:-}"

: "${GOOGLE_APPLICATION_CREDENTIALS_JSON:?falta GOOGLE_APPLICATION_CREDENTIALS_JSON (el contenido del json, por entorno)}"
: "${FAI_GCS_BUCKET:?falta FAI_GCS_BUCKET (el bucket, sin gs://)}"

TRABAJO=/workspace
REPO="${TRABAJO}/fooball_ai_training"
CACHE="${TRABAJO}/cache"
FUENTE="${FAI_TRAINING_SRC:-https://github.com/jhquihuiri7/fooball_ai_training}"
REFERENCIA="${TRABAJO}/referencia"   # football-ai servido desde el bundle, sin `.git` al final
GCLOUD_DIR=/opt                       # donde se instala el SDK de Google si la imagen no lo trae

echo "== 1/4 · repo al commit ======================================================"
if [ ! -d "${REPO}/.git" ]; then
    git clone -q "${FUENTE}" "${REPO}"
fi
# Todas las ramas y la HEAD de la fuente: el commit puede no estar en la rama por defecto,
# y un bundle hecho desde una HEAD suelta solo trae esa referencia.
git -C "${REPO}" fetch -q "${FUENTE}" '+refs/heads/*:refs/remotes/fuente/*' HEAD
git -C "${REPO}" checkout -q "${COMMIT}"
echo "training : $(git -C "${REPO}" rev-parse --short HEAD)"

if [ -n "${FAI_REF_SRC:-}" ]; then
    : "${FAI_REF_URL:?con FAI_REF_SRC hace falta FAI_REF_URL, la URL de football-ai en pyproject.toml}"
    : "${FAI_REF_REV:?con FAI_REF_SRC hace falta FAI_REF_REV, el commit que fija pyproject.toml}"
    # Un repo desnudo con todo lo del bundle y una etiqueta en el commit fijado: la punta de
    # una referencia se puede pedir siempre por su sha, que es como la pide uv.
    rm -rf "${REFERENCIA}.git"
    git init -q --bare "${REFERENCIA}.git"
    git -C "${REFERENCIA}.git" fetch -q "${FAI_REF_SRC}" '+refs/*:refs/*'
    git -C "${REFERENCIA}.git" tag -f fijada "${FAI_REF_REV}" >/dev/null
    # uv baja las dependencias de git con el `git` del sistema, así que esta regla lo desvía
    # al bundle sin tocar pyproject.toml ni uv.lock. Sin `.git` en la regla: git prueba ese
    # sufijo solo, y así vale tanto si uv pide la URL con `.git` como sin él.
    git config --global "url.${REFERENCIA}.insteadOf" "${FAI_REF_URL}"
    echo "football-ai : ${FAI_REF_REV:0:7}, desde el bundle subido"
fi

echo "== 2/4 · entorno (uv sync --group train --group ref) ========================="
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="${HOME}/.local/bin:${PATH}"
fi
# -q y no >/dev/null: un fallo de la instalación tiene que verse, no cortar el script en blanco.
(cd "${REPO}" && uv sync -q --group train --group ref && uv run -q python -c \
    "import torch; print('torch', torch.__version__, '· cuda', torch.cuda.is_available())")

echo "== 3/4 · credenciales de GCS ================================================="
if ! command -v gcloud >/dev/null; then
    # Las imágenes de runpod/pytorch no lo traen. El instalador oficial, sin preguntas.
    curl -sSL https://sdk.cloud.google.com |
        bash -s -- --disable-prompts --install-dir="${GCLOUD_DIR}" >/dev/null
    export PATH="${GCLOUD_DIR}/google-cloud-sdk/bin:${PATH}"
fi
CLAVE="${HOME}/.config/gcloud/fai-key.json"
mkdir -p "$(dirname "${CLAVE}")"
umask 077
printf '%s' "${GOOGLE_APPLICATION_CREDENTIALS_JSON}" >"${CLAVE}"
export GOOGLE_APPLICATION_CREDENTIALS="${CLAVE}"
# La salida normal no, los errores sí: gcloud no escribe la clave en ninguno de los dos.
gcloud auth activate-service-account --key-file="${CLAVE}" --quiet >/dev/null
gcloud storage ls "gs://${FAI_GCS_BUCKET}" >/dev/null
echo "gcloud   : autenticado y con acceso al bucket (la clave vive en un fichero 600 del pod)"

echo "== 4/4 · dataset ============================================================="
if [ -n "${DATASET}" ]; then
    DESTINO_CACHE="${CACHE}/${DATASET}"
    DESTINO="${TRABAJO}/${DATASET}"
    mkdir -p "${DESTINO_CACHE}" "$(dirname "${DESTINO}")"
    # rsync sobre la caché del volumen: una versión inmutable solo baja una vez.
    gcloud storage rsync --recursive "gs://${FAI_GCS_BUCKET}/${DATASET}" "${DESTINO_CACHE}"
    ln -sfn "${DESTINO_CACHE}" "${DESTINO}"
    echo "dataset  : ${DATASET} → ${DESTINO} ($(find -L "${DESTINO}" -type f | wc -l) ficheros)"
else
    echo "dataset  : ninguno pedido (pasa datasets/<nombre>/<version> como 2º argumento)"
fi

echo "listo: cd ${REPO} && uv run …"
