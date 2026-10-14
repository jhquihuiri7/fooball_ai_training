#!/usr/bin/env bash
# Crea un pod de RunPod listo para entrenar, con un solo comando y desde esta máquina (ML-04).
#
#   export RUNPOD_API_KEY=...                       # runpod.io -> Settings -> API Keys
#   export FAI_GCS_BUCKET=<bucket>                  # sin gs://
#   export GOOGLE_APPLICATION_CREDENTIALS=<ruta>    # el json de la cuenta de servicio
#   bash tools/runpod.sh [datasets/<nombre>/<version>]
#
# Lo hace todo: pide el pod, espera al ssh, sube este repo y el de football-ai desde aquí y
# corre dentro `tools/runpod_bootstrap.sh` (entorno, gcloud, credenciales y dataset). Nada
# sale de GitHub ni hace falta token: football-ai es privado, así que viaja como `git bundle`
# desde la copia local, al commit que fija pyproject.toml.
#
# Sube el ÚLTIMO COMMIT de la rama en la que estés, no lo que haya sin commitear: un
# entrenamiento tiene que poder repetirse desde su commit.
#
# Opcionales:
#   PRUEBA=1         borra el pod al terminar, salga bien o mal: la prueba de ML-04
#   POD_ID=xxxx      usa un pod que ya existe en vez de crear otro
#   GPU_TYPE=...     una o varias, separadas por comas (por defecto una RTX 4090)
#   CLOUD_TYPE, DISK_GB, VOLUME_GB, POD_IMAGE, POD_NAME: ver tools/runpod_pod.py
#   FAI_REF_REPO     la copia local de football-ai (por defecto ../football-ai)
#   RUNPOD_SSH_KEY   la clave ssh dada de alta en RunPod (por defecto la de runpodctl)
#
# **La clave de GCS no viaja por argumento ni se guarda en RunPod**: va en el prólogo que se
# le mete al arranque por la tubería de ssh, como las contraseñas de `deploy/runpod.sh` en el
# repo de detección. No queda en el historial, ni en `ps` del pod, ni en su plantilla.
set -euo pipefail

API=https://rest.runpod.io/v1
SUBIDA=/workspace/subida

aviso() { printf '%s\n' "$*" >&2; }
morir() {
    aviso "ERROR: $*"
    exit 1
}

for orden in curl ssh git; do
    command -v "$orden" >/dev/null || morir "hace falta $orden"
done
PYTHON=$(command -v python || command -v python3) || morir "hace falta python 3.11 o más"
[ -f pyproject.toml ] && [ -f tools/runpod_bootstrap.sh ] ||
    morir "lanza esto desde la raíz de football-ai-training"
RAIZ=$(pwd)
DATASET=${1:-}

for variable in RUNPOD_API_KEY FAI_GCS_BUCKET GOOGLE_APPLICATION_CREDENTIALS; do
    [ -n "${!variable:-}" ] || morir "falta $variable. Ver la cabecera de este fichero."
done
# Sin enseñar la ruta: es la de la clave.
[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || morir "GOOGLE_APPLICATION_CREDENTIALS no es un fichero"
# El dataset va dentro de la orden remota de ssh: solo caracteres de ruta.
case $DATASET in
    *[!A-Za-z0-9._/-]*) morir "el dataset es una ruta del bucket, p. ej. datasets/jugadores/v1" ;;
esac

pod_py() { "$PYTHON" "$RAIZ/tools/runpod_pod.py" "$@"; }

api() {
    # $1 método, $2 ruta, $3 cuerpo (opcional). El JSON de la respuesta por stdout.
    local argumentos=(-sS -X "$1" "$API$2" -H "Authorization: Bearer $RUNPOD_API_KEY")
    if [ -n "${3:-}" ]; then
        argumentos+=(-H 'Content-Type: application/json' -d "$3")
    fi
    curl "${argumentos[@]}"
}

# --------------------------------------------------------------------------- #
# Lo que se sube: este repo al commit y football-ai al commit que fija pyproject
# --------------------------------------------------------------------------- #
COMMIT=$(git rev-parse HEAD)
aviso "== training: $(git rev-parse --abbrev-ref HEAD) $(git log -1 --format='%h %s' | cut -c1-60) =="
SIN_COMMIT=$(git status --porcelain | wc -l | tr -d ' ')
if [ "$SIN_COMMIT" -gt 0 ]; then
    aviso "   AVISO: $SIN_COMMIT ficheros sin commitear NO se suben: el pod va al último commit"
fi

REFERENCIA=$(pod_py referencia pyproject.toml) || morir "no se pudo leer la referencia de pyproject.toml"
URL_REF=${REFERENCIA% *}
REV_REF=${REFERENCIA#* }
FAI_REF_REPO=${FAI_REF_REPO:-$RAIZ/../football-ai}
git -C "$FAI_REF_REPO" cat-file -e "${REV_REF}^{commit}" 2>/dev/null ||
    morir "$FAI_REF_REPO no tiene el commit ${REV_REF:0:7} que fija pyproject.toml: haz git fetch ahí"
# Una rama que lo contenga: un bundle se hace de referencias, no de un sha suelto.
RAMA_REF=$(git -C "$FAI_REF_REPO" for-each-ref --count=1 --contains "$REV_REF" \
    --format='%(refname)' refs/heads refs/remotes)
[ -n "$RAMA_REF" ] || morir "ninguna rama de $FAI_REF_REPO contiene ${REV_REF:0:7}"

TEMPORAL=""
CREADO=""
al_salir() {
    local estado=$?
    if [ -n "$TEMPORAL" ]; then rm -rf "$TEMPORAL"; fi
    if [ -n "$CREADO" ] && [ "${PRUEBA:-}" = 1 ]; then
        if curl -fsS -X DELETE "$API/pods/$CREADO" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null; then
            aviso "== prueba: pod $CREADO borrado =="
        else
            aviso "AVISO: no se pudo borrar el pod $CREADO, y cobra: bórralo en la consola de RunPod"
        fi
    elif [ -n "${POD_ID:-}" ] && [ "$estado" -ne 0 ]; then
        aviso "el pod $POD_ID sigue creado y cobra. Para reintentar sobre él: POD_ID=$POD_ID bash tools/runpod.sh"
    fi
}
trap al_salir EXIT

TEMPORAL=$(mktemp -d)
git bundle create -q "$TEMPORAL/training.bundle" HEAD
git -C "$FAI_REF_REPO" bundle create -q "$TEMPORAL/football-ai.bundle" "$RAMA_REF"

# --------------------------------------------------------------------------- #
# El pod
# --------------------------------------------------------------------------- #
if [ -z "${POD_ID:-}" ]; then
    aviso "== creando el pod =="
    CUERPO=$(pod_py cuerpo) || morir "el pedido del pod no vale"
    RESPUESTA=$(api POST /pods "$CUERPO")
    POD_ID=$(printf '%s' "$RESPUESTA" | pod_py campo id)
    if [ -z "$POD_ID" ]; then
        aviso "RunPod no devolvió un pod. Lo que pedimos:"
        aviso "$CUERPO"
        aviso "Lo que contestó:"
        morir "${RESPUESTA:-nada: suele ser una RUNPOD_API_KEY que no vale}"
    fi
    CREADO=$POD_ID
    aviso "   pod $POD_ID"
else
    aviso "== usando el pod $POD_ID =="
fi

aviso "== esperando a que arranque =="
IP=""
PUERTO_SSH=""
for _ in $(seq 60); do
    ESTADO=$(api GET "/pods/$POD_ID")
    IP=$(printf '%s' "$ESTADO" | pod_py campo publicIp)
    PUERTO_SSH=$(printf '%s' "$ESTADO" | pod_py campo portMappings.22)
    [ -n "$IP" ] && [ -n "$PUERTO_SSH" ] && break
    sleep 5
done
[ -n "$IP" ] && [ -n "$PUERTO_SSH" ] || morir "el pod no llegó a tener IP y ssh. Último estado: $ESTADO"

# BatchMode: si la clave no vale, falla en el acto en vez de esperar una contraseña que en
# RunPod no existe. runpodctl guarda la suya fuera de ~/.ssh, así que ssh no la ofrecería sola.
SSH=(ssh -p "$PUERTO_SSH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR -o ConnectTimeout=10 -o BatchMode=yes)
CLAVE_SSH=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
if [ -f "$CLAVE_SSH" ]; then
    SSH+=(-i "$CLAVE_SSH" -o IdentitiesOnly=yes)
fi
SSH+=("root@$IP")

aviso "== esperando al ssh ($IP:$PUERTO_SSH) =="
for _ in $(seq 60); do
    "${SSH[@]}" true 2>/dev/null && break
    sleep 5
done
"${SSH[@]}" true || morir "no se pudo entrar por ssh con $CLAVE_SSH. Su pública tiene que estar en
RunPod -> Settings -> SSH Public Keys ANTES de crear el pod: se inyecta al arrancar."

# --------------------------------------------------------------------------- #
# Subir y arrancar
# --------------------------------------------------------------------------- #
aviso "== subiendo training (${COMMIT:0:7}) y football-ai (${REV_REF:0:7}) =="
"${SSH[@]}" "mkdir -p $SUBIDA && cat > $SUBIDA/training.bundle" <"$TEMPORAL/training.bundle"
"${SSH[@]}" "cat > $SUBIDA/football-ai.bundle" <"$TEMPORAL/football-ai.bundle"

aviso "== montando el pod (la primera vez tarda unos minutos: torch) =="
{
    # La única vez que la clave sale de esta máquina, y va por la tubería.
    printf 'export GOOGLE_APPLICATION_CREDENTIALS_JSON=%q\n' "$(cat "$GOOGLE_APPLICATION_CREDENTIALS")"
    printf 'export FAI_GCS_BUCKET=%q\n' "$FAI_GCS_BUCKET"
    printf 'export FAI_TRAINING_SRC=%q\n' "$SUBIDA/training.bundle"
    printf 'export FAI_REF_SRC=%q\n' "$SUBIDA/football-ai.bundle"
    printf 'export FAI_REF_URL=%q FAI_REF_REV=%q\n' "$URL_REF" "$REV_REF"
    # En Windows el fichero puede llevar CRLF, y el bash del pod se come el CR como parte
    # de la última palabra de cada línea.
    sed $'s/\r$//' tools/runpod_bootstrap.sh
} | "${SSH[@]}" "bash -s -- $COMMIT $DATASET"

if [ "${PRUEBA:-}" = 1 ]; then
    aviso "== prueba: el arranque pasó entero en un pod limpio =="
    exit 0
fi

cat <<RESUMEN

--------------------------------------------------------------------------
 POD      $POD_ID
 ENTRAR   ssh -p $PUERTO_SSH -i $CLAVE_SSH root@$IP
 REPO     /workspace/fooball_ai_training, al commit ${COMMIT:0:7}
 BORRAR   en la consola de RunPod, o
          curl -X DELETE $API/pods/$POD_ID -H "Authorization: Bearer \$RUNPOD_API_KEY"
--------------------------------------------------------------------------
 Parado sigue cobrando el volumen: bórralo cuando acabes.

RESUMEN
