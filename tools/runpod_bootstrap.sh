#!/usr/bin/env bash
# Arranca un pod de RunPod para entrenar (ML-04): repo al commit, entorno y dataset.
#
#   export GOOGLE_APPLICATION_CREDENTIALS_JSON='<contenido del json de la cuenta>'
#   export FAI_GCS_BUCKET=<bucket>            # sin gs://
#   bash tools/runpod_bootstrap.sh <commit> [datasets/<nombre>/<version>]
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

echo "== 1/4 · repo al commit ======================================================"
if [ ! -d "${REPO}/.git" ]; then
    git clone -q https://github.com/jhquihuiri7/fooball_ai_training "${REPO}"
fi
git -C "${REPO}" fetch -q origin
git -C "${REPO}" checkout -q "${COMMIT}"
echo "training : $(git -C "${REPO}" rev-parse --short HEAD)"

echo "== 2/4 · entorno (uv sync --group train --group ref) ========================="
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="${HOME}/.local/bin:${PATH}"
fi
(cd "${REPO}" && uv sync --group train --group ref >/dev/null && uv run python -c \
    "import torch; print('torch', torch.__version__, '· cuda', torch.cuda.is_available())")

echo "== 3/4 · credenciales de GCS ================================================="
CLAVE="${HOME}/.config/gcloud/fai-key.json"
mkdir -p "$(dirname "${CLAVE}")"
umask 077
printf '%s' "${GOOGLE_APPLICATION_CREDENTIALS_JSON}" >"${CLAVE}"
export GOOGLE_APPLICATION_CREDENTIALS="${CLAVE}"
gcloud auth activate-service-account --key-file="${CLAVE}" --quiet >/dev/null 2>&1
echo "gcloud   : autenticado (la clave vive en un fichero 600 del pod)"

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
