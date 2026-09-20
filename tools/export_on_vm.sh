#!/usr/bin/env bash
# Exporta T-DEED a ONNX en una máquina de Vertex AI (o cualquier VM con Python y red).
#
# Existe porque el export no cabe en Colab: el trazador mantiene vivas las activaciones de
# los 100 frames de 448x796 y eso son ~15 GB. Lo que hace falta NO es GPU —trazar es una
# pasada hacia delante y en CPU sale el mismo grafo— sino memoria. Por eso esto está
# pensado para una máquina de alta memoria y torch de CPU, que además evita el trámite de
# la cuota de GPU y cuesta una fracción.
#
#   bash export_on_vm.sh gs://TU-BUCKET/tdeed/checkpoint_best.pt gs://TU-BUCKET/modelo
#
# Al terminar deja en el destino `tdeed-snb.onnx` y `registry-block.yaml`.

set -euo pipefail

PESOS_URI="${1:?falta la URI de los pesos, p.ej. gs://bucket/tdeed/checkpoint_best.pt}"
DESTINO_URI="${2:?falta la URI de salida, p.ej. gs://bucket/modelo}"

TRABAJO="${HOME}/export-tdeed"
TDEED="${TRABAJO}/T-DEED"
TRAINING="${TRABAJO}/training"
SALIDA="${TRABAJO}/modelo"

echo "== 1/6 · memoria disponible =================================================="
free -g || true
echo "Si la columna 'total' baja de 32 GB, el export puede morir por falta de RAM."

echo "== 2/6 · repositorios ========================================================"
mkdir -p "${TRABAJO}"
rm -rf "${TDEED}" "${TRAINING}"
git clone -q --depth 1 https://github.com/arturxe2/T-DEED "${TDEED}"
git clone -q --depth 1 https://github.com/jhquihuiri7/fooball_ai_training "${TRAINING}"
echo "T-DEED   : $(git -C "${TDEED}" rev-parse --short HEAD)  (GPL-3.0: se ejecuta, no se distribuye)"
echo "training : $(git -C "${TRAINING}" rev-parse --short HEAD)"

echo "== 3/6 · dependencias ========================================================"
# torch de CPU a propósito: son ~200 MB en vez de los ~2,5 GB de la rueda con CUDA, y en
# esta máquina no hay GPU que usar. Si ya hay un torch instalado, se respeta.
if python3 -c "import torch" 2>/dev/null; then
    echo "torch    : ya instalado, $(python3 -c 'import torch; print(torch.__version__)')"
else
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q timm tabulate wandb onnx onnxruntime onnxscript onnxsim
python3 - <<'PY'
import onnx, onnxruntime, onnxscript, timm, torch
print(f"torch {torch.__version__} · onnx {onnx.__version__} · onnxruntime {onnxruntime.__version__}")
PY

echo "== 4/6 · pesos ==============================================================="
PESOS="${TRABAJO}/checkpoint_best.pt"
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "${PESOS_URI}" "${PESOS}"
else
    gsutil cp "${PESOS_URI}" "${PESOS}"
fi
ls -lh "${PESOS}"

echo "== 5/6 · export y verificación ==============================================="
# El export tarda bastante en CPU: son 100 frames por un CNN. La verificación (T6) lo
# vuelve a ejecutar dos veces más, una por torch y otra por onnxruntime.
cd "${TDEED}"
WANDB_MODE=disabled python3 "${TRAINING}/tools/export_onnx.py" \
    --tdeed "${TDEED}" \
    --weights "${PESOS}" \
    --out "${SALIDA}"

echo "== 6/6 · subida =============================================================="
if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "${SALIDA}/tdeed-snb.onnx" "${SALIDA}/registry-block.yaml" "${DESTINO_URI}/"
else
    gsutil cp "${SALIDA}/tdeed-snb.onnx" "${SALIDA}/registry-block.yaml" "${DESTINO_URI}/"
fi

echo ""
echo "Listo. En ${DESTINO_URI}/ quedan:"
echo "  tdeed-snb.onnx       -> a models/onnx/ del repo de detección"
echo "  registry-block.yaml  -> su contenido, dentro de models: en models/registry.yaml"
echo ""
echo "La ficha, para revisarla ahora mismo:"
cat "${SALIDA}/registry-block.yaml"
