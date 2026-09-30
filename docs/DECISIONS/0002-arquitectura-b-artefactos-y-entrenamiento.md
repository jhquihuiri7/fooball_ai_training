# ADR 0002 — Arquitectura B: qué entrega este repo, qué entrena y dónde

- **Fecha:** 2026-09-30
- **Estado:** ACEPTADO. No abre decisiones nuevas: pone por escrito, para este repo, las que el
  propietario ya tomó. Son la arquitectura B (ADR 0019 de `fooball_ai_streaming`, 2026-09-28) y
  las decisiones del plan de migración (2026-09-29).
- **Tarea:** ML-01 del plan de migración
  (`docs/plan-dos-moviles/entrenamiento.md` del repo de detección, rama `migracion/dos-moviles`)
- **Afecta a:** README (los tres repos, la ficha y el estado), CLAUDE.md §0-§4, las
  consecuencias del ADR 0001 y el tablero de PROGRESS. Todo se enmienda en este commit.

## Problema

Hasta ahora este repo producía una sola cosa: un `.onnx` con su ficha, que el pod del repo de
detección corría en una GPU. Con la arquitectura B el directo lo hacen los dos iPhone: cada uno
detecta en el Neural Engine sobre su propia cámara. Eso cambia tres cosas:

1. **El modelo tiene que llegar al móvil**, en Core ML y en fp16. El ANE no es onnxruntime: un
   export que pierde precisión en fp16 no da error, solo detecta peor, y en el ANE no hay capas
   que puedan quedarse en fp32.
2. **Los modelos de hoy no se pueden vender.** `rfdetr-small` se afinó con imágenes del Kaggle
   DFL, T-DEED es GPL-3.0 y SoccerNet va con NDA. Hay que reentrenar con datos propios.
3. **Una parte del trabajo necesita un Mac.** Core ML solo predice en macOS, y `coremltools` no
   tiene ruedas para Windows.

## Decisión

### 1. Por modelo se entregan cuatro artefactos

| Artefacto | Para qué | Tarea |
|---|---|---|
| `.onnx` | La referencia de Python y el diferido, en onnxruntime | T5 (existe) |
| `.mlpackage.zip` | El móvil: mlprogram en fp16, target iOS 18 y formas fijas. El zip es determinista para que su sha256 sea estable | ML-09 |
| Ficha v2 | El bloque de `models/registry.yaml`: la ficha de hoy más `artifacts.coreml` y `parity` | ML-13 |
| Bundle dorado | Entradas y salidas de referencia en `.bin` little-endian, para que XCTest compruebe en Swift el modelo y su postproceso | ML-12 |

- **La regla de la ficha no cambia:** los valores se leen de los ficheros exportados.
- **El `.mlpackage.zip` solo lo llevan los modelos que corren en el móvil:** jugadores, balón y
  N4. N3 corre en ONNX por CPU, en el VPS y en el diferido (ADR 0021 del repo de detección), así
  que entrega los otros tres.
- **Cómo cruzan:** todo se publica en GCS y pasa al repo de detección por su sha256. La entrada
  de `registry.yaml` la escribe allí la tarea que la integra (REF-34 los jugadores, REF-37 el
  balón), que es su dueña.

### 2. La pila de modelos

| Modelo | Qué | Entrada y clases | Dónde corre |
|---|---|---|---|
| Jugadores | D-FINE-N (Apache-2.0) con la receta DEIM v1 (commit y LICENSE fijados), desde pesos COCO | 1920×576, la franja jugable a ×0,5; `[goalkeeper, player, referee]` | ANE de cada móvil, a 7,5 Hz |
| Balón | ROI-lite: heatmap propio de 3 frames en gris, stride 2 y offset | Un mosaico de la franja, y un lote `[2,3,256,256]` de ROIs | ANE, después del MVP de jugadores |
| N3 | TCN causal sobre posiciones, de ≤100.000 parámetros | La serie del registro N0 a 7,5 Hz | VPS y diferido, ONNX por CPU |
| N4 | Spotter en streaming: MobileNetV4 + TSM + GRU causal, reimplementado desde E2E-Spot (BSD-3) | Un recorte de 448×256 a 7,5 Hz | ANE de cada móvil |

- **El detector lo decide el repo de detección, no este.** Su ADR 0020 está PROPUESTO hasta que
  SPK-51 y SPK-52 midan en el A19 (REF-33). Si la medida lo tumba, entra su plan B: primero
  D-FINE-N a 1536×512 y después YOLOX-Tiny. Entonces se enmienda esta tabla.
- **Pesos vetados:** los `*_obj365` y `*_obj2coco` de D-FINE, y los XL/2XL de RF-DETR (PML 1.0).
- **Los eventos N3 y N4 llevan su propio ADR (el 0004).** Este solo dice cuáles son.

### 3. Dónde se trabaja

| Dónde | Qué |
|---|---|
| GCS | Todo dato y todo artefacto. Cada versión es inmutable y lleva sha256 por fichero (ML-04) |
| RunPod (RTX 4090 o L40S) | Entrenar, el maestro y la autoanotación, con caché de GCS en el volumen de red |
| Vertex AI, CPU de alta memoria | Solo los exports pesados (`tools/export_on_vm.sh`) |
| Mac con Apple Silicon | Lo que pide Core ML: predecir en fp16, la paridad, los dorados y el banco en el iPhone. El de Alexander es el de referencia (Xcode 26.6), y se consigue un segundo para ML |
| Colab | Solo cuadernos y pruebas pequeñas |
| Windows | Código y tests. Los tests que necesitan `coremltools` se saltan con motivo |

### 4. T-DEED se congela y best.pt queda fuera

- **T-DEED queda como línea base de medida, fuera del producto** (ADR 0021 del repo de
  detección). N3 y N4 tienen que superar su cifra sobre los mismos partidos propios.
- **Se cancelan T3 y T4:** no se reduce ni se afina T-DEED. SoccerNet `spotting-ball-2024`
  (19 GB) no se descarga salvo que haga falta para medir.
- **Se conservan sin mantenimiento**, para poder reproducir la línea base:
  `notebooks/t1_baseline.ipynb`, `notebooks/t5_export.ipynb`, `tools/fetch_tdeed.py` y
  `tools/export_onnx.py`. Ningún código nuevo importa T-DEED; la guardia de licencias lo
  comprobará (ML-02).
- **`best.pt` (YOLOv8n de Ultralytics, AGPL-3.0) queda fuera:** no se usa, no se reentrena desde
  él y `ultralytics` no entra en este repo. Su procedencia la cerró ML-46 en el repo de
  detección. La única copia conocida es un fichero local, ignorado por git, en la máquina
  Windows del propietario, y se borra a mano.

## Alternativas descartadas

**Entregar solo el `.onnx` y convertir a Core ML en otro sitio.** Convertir necesita torch y
`coremltools`, y la puerta fp16 se mide contra la referencia de torch. Las dos cosas solo existen
aquí.

**Entrenar en Colab.** No cabe: el export de T5 murió allí por memoria, tanto en CPU como en la
T4. Además las sesiones caducan. Queda para cuadernos.

**Afinar T-DEED (T3 y T4).** Por mucho que mejorase, es GPL y sus datos van con NDA: no se
podría vender.

## Consecuencias

- **El repo deja de ser «un `.onnx` y nada más».** Tendrá varios modelos, cada uno con sus
  cuatro artefactos, y una cadena común de export (ML-09 a ML-13).
- **Un export se da por bueno cuando pasa dos pruebas:**
  - la verificación torch↔onnxruntime (< 1e-3), como hasta ahora;
  - en los modelos que van al móvil, la puerta fp16 de su tarea: Δrecall ≤2 pp por banda de
    distancia y centro de caja ≤1 px en mediana en los jugadores; Δ del pico ≤1 px y Δrecall
    por tamaño ≤2 pp en el balón (ADR 0020 §5).

  Si no pasa, el export falla.
- **El Mac entra en la ruta crítica.** Sin él no hay paridad fp16 ni dorados, y los jugadores y el
  balón (ML-E4 y ML-E5) se paran en el export. Por eso hace falta el segundo.
- **La frontera con el repo de detección se amplía:** ya no cruza solo un `.onnx` (su ADR 0013
  §2, ampliado por su ADR 0020 §4).
- **Los datos son propios,** así que su licencia deja de ser un problema. Los preentrenos (COCO
  en D-FINE-N, ImageNet en MobileNetV4) son un riesgo residual del sector que el propietario ya
  aceptó.
