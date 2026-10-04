# Los datos de la campaña: GCS (ML-04, ADR 0002 §3)

Todo lo que pesa vive en un bucket de GCS. El nombre llega por `FAI_GCS_BUCKET` y las
credenciales por `GOOGLE_APPLICATION_CREDENTIALS`; ninguno de los dos se escribe en un
log, un fichero del repo ni un commit. La envoltura es `ftrain/storage.py` (con
`dry_run` para ensayar sin gastar) y el arranque de un pod, `tools/runpod_bootstrap.sh`.

## El árbol del bucket

```
gs://$FAI_GCS_BUCKET/
  raw/<partido>/<lado>/        # las grabaciones 4K tal cual salen del iPhone
  nv12/<partido>/<lado>/       # los volcados crudos de IOS-15, para ML-18
  frames/<version>/            # fotogramas extraídos (ML-07), una versión por corte
  labels/<version>/            # anotaciones exportadas de CVAT (ML-08)
  datasets/<nombre>/<version>/ # datasets armados: lo que baja un pod para entrenar
  models/<nombre>/<version>/   # los cuatro artefactos de cada export (ADR 0002)
  golden/<nombre>/<version>/   # bundles dorados de paridad
  runs/<fecha>-<nombre>/       # salidas de entrenamientos: logs, curvas, checkpoints
```

`<partido>` es el `match_id` del manifiesto de grabación (ML-06): `m_20260703_2030_a1b2`.

## La política: versiones inmutables, sha256 por fichero

- **Una versión no se toca jamás.** Corregir es publicar `<version+1>`; quien entrene
  contra `datasets/jugadores/v3` tiene que poder reproducirlo dentro de un año.
- **Cada fichero sube con su sidecar `<fichero>.sha256`.** Es lo que hace idempotente
  la subida (mismo sha → no se resube) y lo que permite verificar una bajada sin
  volver a bajar: `ftrain.storage.Storage.upload/remote_sha` lo gestionan solos.
- **Nada de URLs firmadas en documentos**: el acceso es por cuenta de servicio, y la
  clave solo existe en la variable de entorno de quien la usa.

## El arranque de un pod (RunPod)

```bash
# En el pod, con la clave de servicio YA en el entorno (nunca en una imagen):
export GOOGLE_APPLICATION_CREDENTIALS_JSON='<contenido del json>'
export FAI_GCS_BUCKET=<bucket>
bash tools/runpod_bootstrap.sh <commit> datasets/jugadores/v1
```

El script clona este repo AL COMMIT pedido, hace `uv sync --group train --group ref`,
autentica `gcloud` escribiendo la clave a un fichero 600 dentro del pod (sin pasarla
jamás por argv ni enseñarla), y baja la versión del dataset a `/workspace/datasets/`
con caché en el volumen de red (`/workspace/cache/`): una versión es inmutable, así
que lo cacheado no caduca y el segundo pod del día arranca en segundos.
