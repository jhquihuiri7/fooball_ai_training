# PROGRESS

Estado de las tareas. Se actualiza al cerrar cada una: qué se hizo, qué quedó fuera,
siguiente paso. **Lo que se mide va aquí**: es lo único que luego no se recuerda.

Leyenda: ✅ hecha · 🚧 en curso · ⛔ bloqueada · ⬜ pendiente · ✖ cancelada

## Tablero de la arquitectura B

Las tarjetas están en `docs/plan-dos-moviles/entrenamiento.md` del repo de detección, en la
rama `migracion/dos-moviles`. Una tarea no empieza hasta que sus dependencias estén en ✅
aquí. Las que no llevan marca están ⬜.

| EPIC | Qué | Tareas |
|---|---|---|
| ML-E1 | Infraestructura y contrato del repo | ML-01 ✅ · ML-02 ✅ · ML-03 ✅ · ML-04 · ML-05 🚧 |
| ML-E2 | Campaña de datos: grabación, ingesta, extracción y anotación | ML-06 · ML-07 · ML-08 · ML-17 · ML-18 · ML-19 · ML-20 · ML-21 · ML-22 · ML-23 · ML-58 |
| ML-E3 | Autoanotación con maestro y players-v1 | ML-24 · ML-25 · ML-26 · ML-27 · ML-28 · ML-29 · ML-30 · ML-31 |
| ML-E4 | Jugadores: D-FINE-N a 1920×576 | ML-15 ✅ · ML-16 · ML-32 · ML-33 · ML-34 · ML-35 |
| ML-E5 | Balón: heatmap ROI-lite de 3 frames en gris | ML-14 ✅ · ML-36 · ML-37 · ML-38 · ML-39 · ML-40 · ML-41 · ML-42 |
| ML-E6 | Export a Core ML y validación | ML-09 · ML-10 · ML-11 · ML-12 · ML-13 · ML-43 · ML-45 |
| ML-E7 | Spikes de modelo en el iPhone 17 | SPK-50 · SPK-51 · SPK-52 · SPK-53 · SPK-54 · SPK-56 |
| ML-E8 | Eventos aprendidos N3 y N4 | ML-47 · ML-48 · ML-49 · ML-50 · ML-51 · ML-52 · ML-53 · ML-54 · ML-55 · ML-57 |
| ML-E9 | Retiradas | ML-46 ✅ (en el repo de detección) |

## Línea base de T-DEED, congelada (ADR 0002)

| TASK | Qué | Estado |
|---|---|---|
| **T0** | Repo, entorno y contrato con el repo de detección | ✅ |
| **T1** | T-DEED clonado, su firma leída y la línea base corrida y contrastada | ✅ |
| **T2** | Datos de SoccerNet | ✅ herramienta · ✅ tarea elegida ([ADR 0001](DECISIONS/0001-ball-action-spotting-sin-corner.md)) · ⬜ descarga |
| T3 | Reducir a las clases que interesan | ✖ cancelada ([ADR 0002](DECISIONS/0002-arquitectura-b-artefactos-y-entrenamiento.md)) |
| T4 | Fine-tuning, con división por partidos completos | ✖ cancelada ([ADR 0002](DECISIONS/0002-arquitectura-b-artefactos-y-entrenamiento.md)) |
| T5 | Export a ONNX con shapes estáticas **y su ficha** | ✅ |
| T6 | Verificación numérica torch vs onnxruntime (< 1e-3) | ✅ **9,54e-06** |

---

## 2026-10-03 · ML-03 — la referencia Python de football-ai, fijada por commit · ✅

**Hecho**
- `pyproject.toml`: grupo `ref` con `football-ai` por git, fijado en `[tool.uv.sources]` al
  commit `efed47b` de `migracion/dos-moviles` (NUBE-05c, que ya trae REF-23 y REF-25).
  Moverla es cambiar ese sha. Como aquel paquete no declara su runtime, el grupo lleva lo
  que importan los módulos que se usan: scipy (postproceso) y opencv (campo y franja).
  **No lleva structlog**, que pedía la tarjeta: ninguno de esos módulos lo importa ya.
- `ftrain/ref.py`: fachada de import perezoso (PEP 562) para `read_timecode_ms`,
  `PitchModel`, `decode_boxes_to_corners`, `sigmoid`, `heatmap_peaks`, `load_registry` y
  `compose_band_input`, el lienzo de la franja de REF-25. Sin el grupo, importar el módulo
  no falla. Lo que falla es pedir un nombre, con `RefMissingError` y el comando que lo
  arregla.
- `DEPENDENCIES.md`: la tabla del grupo `ref`.

**Tests**: `tests/test_ref.py` (4). El código de tiempo que pinta `write_timecode` de la
referencia se lee por la fachada en un frame 4K; cada nombre sale del módulo que dice; sin
la referencia, `RefMissingError`; y un nombre que no expone es un `AttributeError`. Los
dos primeros se saltan con motivo si falta el grupo.

**Ejecutado aquí (Windows)**
- `uv lock` resuelve el commit fijado, y `uv sync --group train --group ref` instala.
- ruff ✅ · ruff format ✅. pytest: los 4 de `test_ref.py` pasan con el grupo instalado.
- **El resto de la suite no se puede correr entero en esta máquina.** El Control de
  aplicaciones de Windows (error 4551) bloquea las extensiones nativas de torch
  (`torch_python.dll`) y de onnx (`onnx_cpp2py_export.pyd`, que nadie ha tocado desde el
  19 de septiembre). Caen al recogerlos `test_tools.py` y `test_ball_model.py`, y falla el
  test de la red de `test_mac_smoke.py`. Del resto, 71 pasan y 2 se saltan (Core ML).
  cv2, scipy, numpy y onnxruntime sí cargan.
- Ojo: `uv sync --group ref` a secas desinstala los grupos que no se nombran, torch
  incluido. Aquí se sincroniza con `--group train --group ref`.

**Siguiente**: con ML-03, ML-19 (bandas de distancia y paso de etiquetas nativo↔lienzo)
tiene sus dependencias, porque REF-25 y REF-29 ya están; es de H2. ML-12 y ML-13 usarán
esta referencia cuando llegue su turno.

## 2026-09-30 · ML-05 — grupo `apple` con coremltools y el Mac de referencia · ✅

**Hecho**
- `pyproject.toml`: el grupo `apple`, que instala `coremltools>=9` salvo en Windows, donde no
  hay ruedas. El lock resuelve `coremltools` 9.0.
- `DEPENDENCIES.md`: `coremltools`, BSD-3, que arrastra attrs y cattrs (MIT) y pyaml
  (WTFPL); ninguno viaja en el `.mlpackage`.
- `docs/MAC.md`: qué se hace en el Mac y qué tarea lo pide, qué Mac (el de Alexander como
  referencia, y un segundo para ML), los requisitos (Apple Silicon, macOS ≥15, Xcode 26,
  uv, el mismo commit) y cómo se deja listo.
- `tools/mac_smoke.py`: convierte una red de tres convs a mlprogram fp16 para iOS 18 con
  formas fijas y la guarda. En macOS, además, predice con `CPU_AND_NE` y `CPU_ONLY` y sale
  con error si alguna se aleja 1e-2 o más de torch en fp32. En Windows dice que no hay
  `coremltools` y sale con 1.
- `tests/test_mac_smoke.py`: cada parte se prueba donde se puede. La red de prueba, con
  torch; convertir, con `coremltools`; predecir, solo en macOS; y el aviso, donde falta.

**Ejecutado aquí (Windows)**
- La red de prueba y el aviso de que falta `coremltools` pasan.
- Los tests de convertir y de predecir **se saltan con su motivo**, que es lo que pide la
  tarjeta para Windows.
- `tools/mac_smoke.py` sale con 1 y con el mensaje.

**Ejecutado en el Mac de referencia (2026-10-03)**
- MacBook Air M4, macOS 26.3, Xcode 26.6; coremltools 9.0, torch 2.14.0 (aviso de
  coremltools: probado hasta 2.7; el humo pasa igual).
- `uv run python tools/mac_smoke.py`: convertido `runs/mac_smoke/smoke.mlpackage`
  (mlprogram, fp16, iOS 18) y **CPU_AND_NE: Δ = 4,68e-04 · CPU_ONLY: Δ = 4,68e-04**,
  las dos muy por debajo del 1e-2 de la aceptación.
- `pytest tests/test_mac_smoke.py`: 3 passed, 1 skipped (el aviso de Windows, aquí
  no aplica). Convertir y guardar también pasó en macOS.

**Ejecutado en Linux (2026-10-03, el VPS del soporte, Ubuntu 24.04.4)**
- Árbol subido por ssh (sin credenciales de git en el servidor), `uv sync --group
  train --group apple` y **pytest: 111 passed, 4 skipped** — el test de convertir en
  verde; los de predecir, saltados con su motivo (solo macOS). Con esto las tres
  patas (Windows/Mac/Linux) están, y la tarea cierra.
- Ojo al correr en un Linux pelado: con `LANG=C` la guardia de licencias lee en
  ASCII (PYTHONUTF8=1 lo arregla), y un tar hecho en macOS sin `COPYFILE_DISABLE=1`
  cuela ficheros `._*` AppleDouble que esa guardia no puede decodificar.

ruff en verde y pytest con 109 tests y 2 saltados (eran 107).

**Siguiente**: ML-09 (`export_coreml`) depende de esta. La parte pura de ML-09 (el zip
determinista y los metadatos) se puede escribir ya, pero la tarea no pasa a ✅ hasta que
ML-05 lo esté.

---

## 2026-09-30 · ML-14 — arquitectura ROI-lite (heatmap de 3 frames en gris) · ✅

**Hecho**
- `ftrain/ball/model.py`: `RoiLite` y `RoiLiteConfig`, diseño propio sin código ni pesos de
  FootAndBall ni de WASB.
  - **Tallo:** conv 3x3 de paso 2 a 16 canales.
  - **Cuerpo:** 4 etapas `[16, 32, 64, 128]` a strides 2, 4, 8 y 16, con bloques de 3x3 +
    1x1 densas, BatchNorm, ReLU y atajo residual. Los bloques son `(1, 2, 2, 2)`: la etapa
    de stride 2 lleva uno solo porque es la más cara por píxel.
  - **Decoder:** de arriba abajo hasta stride 2 (1x1, upsample nearest ×2, suma y 3x3).
  - **Cabezas:** heatmap de 1 canal y offset de 2.
  - **Ancho configurable:** se redondea a múltiplos de 16, con 16 de mínimo.
- `ftrain/flops.py`: `count_macs` por hooks sobre `Conv2d`, `ConvTranspose2d` y `Linear`.
  Solo necesita torch.
- Constantes nuevas: `PIXEL_SCALE` (1/255 dentro del grafo, ADR 0020 §3) y
  `ANE_CHANNEL_QUANTUM` (16).
- `tests/test_ball_model.py`, con importorskip de torch. Comprueba:
  - las formas del lote de ROIs y del mosaico;
  - los dos presupuestos de MAC;
  - que no hay Conv3d, GRU, LSTM ni RNN, y que todas las convs son densas;
  - que los canales van de 16 en 16, salvo la entrada y las cabezas;
  - el redondeo del ancho;
  - el contador contra cuentas hechas a mano.

**Decidido aquí, por ser lo mínimo**
- **El heatmap sale en logits.** La sigmoide la pone el export (ML-42): la pérdida focal
  de ML-39 la quiere fuera, y `heatmap_peaks` del repo de detección, con su umbral de 0,30,
  espera 0-1. El ADR 0020 §3 solo saca del grafo la NMS, el `topk`, el `argmax` y los picos.
- **El offset sale sin activar**, en fracciones de celda: canal 0 en x y 1 en y, como
  `refine_offset` de REF-23.

**Medido** (CPU, torch 2.14, ancho 1,0)

| Entrada | MAC | Objetivo |
|---|---|---|
| ROI `[1,3,256,256]` | **0,515 G** | ≤1,0 G |
| Mosaico `[1,3,896,1920]` | **13,5 G** | ≤17 G |
| Parámetros | 594 k | — |

Con ancho 1,5 sale 1,38 G en la ROI: se sale del presupuesto. El margen para crecer está
en los bloques, no en el ancho.

**Fuera**
- La latencia en el ANE es de SPK-52, que también decide la altura del mosaico.
- **El grupo `train` está instalado en esta máquina Windows** (torch 2.14 de CPU). Los
  tests de torch ya no se saltan aquí.

ruff en verde y pytest con 107 tests (eran 91).

**Siguiente**: ML-05 (el grupo `apple`), y detrás ML-09. La pérdida y el conjunto del
balón (ML-38 y ML-39) esperan a la campaña de datos.

---

## 2026-09-30 · ML-15 — DEIM v1 fijado y pesos COCO de D-FINE-N · ✅

**Hecho**
- `tools/fetch_deim.py` hace tres cosas:
  - clona DEIM en `third_party/DEIM` al commit fijado `09d35d5`, y se para si el árbol tiene
    cambios locales;
  - compara su LICENSE, línea a línea y sin mirar los finales de línea, con
    `licenses/DEIM-LICENSE-09d35d5.txt`, que va en git, y se para con el diff si cambió;
  - baja `dfine_n_coco.pth` a `models/pretrained/` con su sha256 fijado. Descarga a un
    `.part` y rechaza por el nombre los `*_obj365` y `*_obj2coco`.
- **Lo que se vio al fijarlo.** La LICENSE de DEIM cambió el 2025-07-21 (`bc11dfe`). Solo
  añadió arriba el copyright de Intellindust; sigue siendo Apache-2.0. Es la versión que
  queda fijada.
- **Dependencias de DEIM, una a una**, en el grupo `train` y en DEPENDENCIES:
  - `faster-coco-eval`, Apache-2.0: PyPI no declara la licencia; está leída en su repo;
  - `scipy`, BSD-3;
  - `calflops`, MIT, que arrastra `accelerate` y `huggingface-hub`, Apache-2.0 los dos;
  - `transformers` no entra: está en su `requirements.txt`, pero el código no lo importa.
- **La guardia de ML-02 era demasiado ancha.** Rechazaba cualquier ruta con
  `third_party/`, y ahí vive también DEIM. Ahora solo rechaza el componente `T-DEED`. Sigue
  cazando `Path("third_party") / "T-DEED"`, y hay un caso nuevo que deja pasar
  `third_party/DEIM`.

**Medido** (fetch real en Windows, 2026-09-30)

| | |
|---|---|
| DEIM | `09d35d53d39e`, árbol `20da1766e866` en las dos pasadas |
| `dfine_n_coco.pth` | 15.489.558 bytes, sha256 `41973938d278…`; la segunda pasada no baja nada |
| LICENSE con una línea cambiada | el fetch sale con 1 y enseña el diff |

ruff en verde y pytest con 91 tests (eran 75).

**Siguiente**: ML-14 (el modelo del balón) y ML-05 (el grupo `apple`). ML-16 espera a
ML-09, ML-10 y ML-11.

---

## 2026-09-30 · ML-02 — paquete `ftrain/`, constantes con unidades y guardia de licencias · ✅

**Hecho**
- `ftrain/`: la librería del repo, con `ftrain/constants.py`. Tiene las catorce constantes de
  la tarjeta, cada una con su comentario, sus unidades y de dónde sale: las bandas de
  distancia, el muestreo disperso, las clases de jugadores y del maestro, la entrada de
  1920×576 y la escala de la franja, lo del balón (ROIs, frames, stride y gaussiano) y las
  tres puertas del export.
- `tests/test_guardia_licencias.py`: recorre con `ast` `ftrain/` y `tools/`, y falla si
  algo trae `ultralytics` (AGPL) o T-DEED (GPL). Detecta tres cosas:
  - los imports de `ultralytics` o de los paquetes `model` y `util` de T-DEED, también
    dentro de una función o con `importlib.import_module` y `__import__`;
  - una ruta a `third_party/` o a `T-DEED`.

  Los imports relativos y los docstrings no cuentan. Las únicas excepciones son
  `tools/fetch_tdeed.py` y `tools/export_onnx.py`, y un test comprueba que siguen
  existiendo.
- `tests/test_constants.py`: las bandas crecen de 0 a ∞; las clases no se repiten y van en
  el orden declarado; la entrada es la franja 4K por su escala; las ROIs caben en la
  rejilla del heatmap.
- ruff: T201 (`print`) deja de estar ignorado en todo el repo. Solo lo está en `tools/` y
  en los cuadernos, así que `ftrain/` no puede imprimir, como decidió el plan. Va anotado
  en CLAUDE.md §2.
- La cabecera y la descripción de `pyproject.toml` ya no dicen que el repo produce solo un
  `.onnx`: se me pasó en ML-01.
- ruff en verde y pytest con 75 tests (eran 25).

**Fuera**
- `BALL_GAUSS_D_PX = 2.5` es la «d» del plan, que no dice si es la sigma o el diámetro del
  gaussiano. Lo fija ML-38 al construir el objetivo.
- `tools/export_onnx.py` sigue con su propia tolerancia de 1e-3 en vez de
  `ONNX_VERIFY_TOL`. Es de la línea base congelada; ML-13 lo envuelve.
- **En esta máquina Windows, pytest se lanza con el intérprete base de uv.** El Control de
  aplicaciones bloquea el `python.exe` del venv (os error 4551), así que
  `uv run pytest` no arranca. Los tests pasan igual cargando `.venv/Lib/site-packages`.

**Siguiente**: ML-03 (la referencia de football-ai fijada por commit), ML-04 (GCS y
RunPod) o ML-05 (el grupo `apple`). Las tres dependen solo de esta. ML-03 fija un sha de
la rama `migracion/dos-moviles` de football-ai, y ese sha tiene que estar subido a GitHub.

---

## 2026-09-30 · ML-01 — ADR 0002: qué produce el repo con la arquitectura B · ✅

**Hecho**
- [ADR 0002](DECISIONS/0002-arquitectura-b-artefactos-y-entrenamiento.md), ACEPTADO. Pone
  por escrito, para este repo, decisiones que el propietario ya tomó el 2026-09-28 y el
  2026-09-29:
  - los cuatro artefactos por modelo;
  - la pila (D-FINE-N, ROI-lite, N3 y N4);
  - dónde se trabaja (GCS, RunPod, Vertex, Mac y Colab);
  - T-DEED congelado y `best.pt` fuera.
- Enmiendas:
  - el README: los tres repos, la ficha v2 y el estado;
  - CLAUDE.md §0-§4: el entregable, lo que tiene que poder venderse, la puerta fp16, dónde va
    cada cosa y la rama `migracion/dos-moviles`;
  - las consecuencias del ADR 0001: el córner vuelve por N1 y N3;
  - este tablero, con T3 y T4 canceladas.
- La rama `migracion/dos-moviles` sale de `main` en este commit.

**Decidido aquí, por ser lo mínimo**
- El `.mlpackage.zip` solo lo llevan los modelos que corren en el móvil. N3 corre en ONNX
  por CPU (ADR 0021 del repo de detección), así que entrega los otros tres artefactos.
- CLAUDE.md §4 no lista comandos de herramientas que todavía no existen: cada uno se añade
  con la tarea que lo crea.

**Fuera**
- `best.pt` sigue en `c:/dev/football-ai/models/onnx/`, en la máquina Windows. Git lo
  ignora y nada lo referencia. Borrarlo es una acción del propietario (ML-46).

**Siguiente**: ML-02, el paquete `ftrain/` con sus constantes y la guardia de licencias.

---

## 2026-09-19 · T0 y T1 — el repo y la firma de T-DEED · ✅

**Por qué existe este repo.** El ADR 0004 del repo de detección sacó el entrenamiento de
allí en septiembre de 2026 y desde entonces «el repo de entrenamiento» aparecía en cinco
sitios de la documentación sin existir. Mientras tanto llegó un `rfdetr-small.onnx` de
algún sitio —hay un `best.pt` al lado— **sin código versionado de cómo se hizo**: hoy no se
puede reproducir ni reexportar a otra resolución, que es justo lo que hará falta para
medir el riesgo R-P7-1.

### La firma de T-DEED, leída de su código

`uv run python tools/fetch_tdeed.py` clona el repo y **lee** de sus ficheros lo que define
la forma del clip. No se copia a mano ninguno de estos números: si T-DEED los cambia, la
herramienta falla en vez de quedarse con una copia vieja.

Medido en el commit `d953b4d`, configuración `SoccerNetBall_challenge1`:

| | Valor | De dónde sale |
|---|---|---|
| Clip | **100 frames** | `clip_len` de la config |
| Ritmo | **12,5 fps** | `STRIDE_SNB = 2` sobre frames extraídos a 25 |
| Ventana | **8 s** | los dos anteriores |
| Frame | **796 × 448** | `TARGET_HEIGHT/WIDTH` de `extract_frames_snb.py` |
| Backbone | `rny002_gsf` | config |
| Temporal | `ed_sgp_mixer` | config |
| Clases | **12** | `EVENT_DICTIONARY` de `util/eval.py` |

Orden de clases (el orden **es** el índice de salida): `PASS, DRIVE, HEADER, HIGH PASS,
OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL`.

### Tres cosas que salieron de mirarlo, y cambian el plan

1. **No hay córner.** Las cuatro clases del MVP son gol, tiro, córner y tiro libre. Ball
   Action Spotting tiene las tres primeras y **no** el córner: en su lugar tiene THROW IN
   y OUT. El córner está en la otra tarea, la de 17 clases. Hay que elegir, y no es obvio:
   la tarea con córner no tiene pesos publicados de T-DEED para SoccerNetBall.
2. **La ventana es de 8 s, no de 2.** Eso son 100 × 448 × 796 × 3 = **107 MB** de ventana
   viva en el panel, no los 13 MB que estimé al escribir la E6 con números de ejemplo. El
   spotter la deriva de la ficha, así que no hay que tocar código; pero el número real
   conviene tenerlo escrito antes de medir en el pod.
3. **El backbone es `rny002_gsf`**, o sea gate-shift. Es exactamente la parte que avisé
   que puede degradarse al exportar a ONNX. La verificación numérica de T6 no es un
   trámite: es donde se sabrá si este camino sirve.

### La línea base: `notebooks/t1_baseline.ipynb` (2026-09-20)

**No se ha ejecutado**: se escribió sin GPU delante. Va a Colab, que es donde el
propietario entrena; RunPod queda solo para el directo.

**No necesita los 19 GB.** `inference.py` corre sobre **un vídeo**, así que la línea base
se hace con metraje propio, que además contesta mejor la pregunta: no «¿reproduce el
paper?» sino «¿ve algo en nuestra cámara?».

**Tres trampas del código de T-DEED**, encontradas leyéndolo y ya resueltas en el notebook:

1. **Los pesos van en `checkpoints/SoccerNetBall/SoccerNetBall_challenge1/checkpoint_best.pt`.**
   `inference.py` arma esa ruta con el prefijo del nombre del modelo. Ponerlos donde uno
   los pondría mirando la carpeta —un nivel más arriba— falla a mitad de la carga.
   `tools/fetch_tdeed.py` decía la ruta equivocada; corregido.
2. **El vídeo tiene que ir a 25 fps.** `ActionSpotInferenceDataset` lee con OpenCV y se
   queda con uno de cada dos frames **nativos**: no remuestrea nada. Con un vídeo a 30
   fps el clip le llega a 15 y el modelo ve la jugada acelerada respecto a todo lo que
   aprendió. No da error, solo acierta menos.
3. **No instalar su `requirements.txt`.** Pinea `torch==2.3.1` y `numpy==1.26.4` y pelea
   con Colab. Bastan `timm`, `tabulate`, `wandb` —que `inference.py` importa y no usa— y
   `SoccerNet`, que **no está en su `requirements.txt`** y hace falta igual (ver abajo).

**Primera ejecución, 2026-09-20**: muere en el import, `ModuleNotFoundError: No module
named 'SoccerNet'` desde `util/eval.py:13`. Es un fallo de su `requirements.txt`, que no
lista el paquete aunque `util/eval.py` lo importe arriba del todo y `inference.py` importe
ese módulo. Se instala con `--no-deps`: el paquete arrastra `boto3`, `scikit-video` y
`pycocoevalcap`, y de todo él solo se usan `average_mAP` y `LoadJsonFromZip`, que se
importan con numpy y tqdm y nada más —comprobado en el entorno local—. Ya estaba en
`docs/DEPENDENCIES.md` (MIT el paquete, otra cosa los datos), así que no cambia la tabla.

La salida cae en `inference_output/results_inference.json` con el frame nativo, la clase y
la confianza.

**Almacenamiento**: el propietario servirá los datos desde Google Cloud Storage de un
proyecto suyo, así que el tope de 15 GB de Drive deja de ser un problema para T4.

---

## 2026-09-20 · T5 y T6, cerradas · ✅

El export salió en la máquina de Vertex AI (`n1-highmem-8`, **CPU**, 50 GB):

| | |
|---|---|
| Tamaño | **50 MB** (venía de 1028) |
| Verificación numérica | **9,54e-06**, tolerancia 1e-03 |
| Exportador | clásico, por trazado |
| Capas gate-shift parcheadas | 11 |
| SHA-256 | `dde3c1d8…` |

**Los 50 MB son la prueba de que el arreglo era el correcto**: los pesos del modelo son
49 MB, así que en el fichero ya no hay nada más que el modelo. El gigabyte eran ceros.

**Y los 9,54e-06 son la respuesta a la pregunta que estaba abierta desde el principio.**
Esa cifra compara el `.onnx` contra el modelo de torch **sin parchear**, así que dice dos
cosas a la vez:

1. **El gate-shift sobrevive al export.** Era la duda que hacía dudar de todo el camino de
   T-DEED: su backbone `rny002_gsf` lleva desplazamiento temporal de canales, y no estaba
   claro que llegara entero al otro lado. Llega, con dos órdenes de magnitud de margen
   sobre la tolerancia.
2. **El `Concat` es equivalente al `zeros_like`.** El parche no es una aproximación.

**Lo que costó llegar**, por si vuelve a pasar con otro modelo:

| Parada | Qué era |
|---|---|
| Colab, CPU | SIGKILL a los ~12 GB de RAM |
| Colab, T4 | `torch.OutOfMemoryError` con 14,27 de 14,56 GB |
| Vertex AI, 1.ª | Salió, pero 1,03 GB y `onnxruntime` no podía abrirlo |
| `onnxsim` | Lo dejó en 2,4 GB, pasó del límite de protobuf y quedó ilegible |
| `torch.export` | No traza `if (std == 0).any()` de `torchvision.Normalize` |
| Vertex AI, 2.ª | **50 MB y 9,54e-06** |

De las seis, solo dos eran del modelo. Las otras cuatro fueron entorno, y todas están
resueltas dentro de `export_on_vm.sh` para que no haya que redescubrirlas.

**Siguiente paso**: el artefacto cruza al repo de detección. Aquí quedan T3 (reducir
clases) y T4 (fine-tuning con datos propios), que solo tienen sentido después de medir
este modelo sobre partidos reales.

## 2026-09-20 · El gigabyte de ceros: causa y arreglo · 🚧

Segunda vuelta del export, con datos de verdad. Tres cosas quedaron claras:

**1. El exportador de `torch.export` no puede con este modelo, y se sabe por qué.** Falla
en `standarize`, dentro de `torchvision.Normalize`:

    if (std == 0).any():

Es una rama que depende del **valor** de un tensor. `torch.export` la convierte en un
símbolo sin respaldo (`Sym(Eq(u0, 1))`) y se rinde. Pasa a ser `legacy` por defecto: ya no
es una preferencia, es que el otro no puede.

**2. `onnxsim` empeora el problema.** Sobre este grafo lo dejó en **2,4 GB**, pasó del
límite de 2 GB de protobuf, lo guardó como datos externos y el resultado ni se pudo
parsear. Pasa a ser opt-in con esa advertencia escrita; §42.1 lo prescribe en general, y
en general está bien, pero aquí no.

**3. El gigabyte de ceros se quita en el modelo, no en el exportador.** Su origen:

    y = torch.zeros_like(x)          # <- esto se graba entero al trazar
    y[:, :fold] = self.gs(x[:, :fold])
    y[:, fold:] = x[:, fold:]

Escrito con un `Concat` es la misma operación y no materializa nada:

    y = cat([gs(x[:, :fold]), x[:, fold:]], dim=1)

`patch_gated_shift` reescribe el `forward` de las once capas antes de exportar.

**Y que sea equivalente no se da por hecho.** La referencia de la verificación numérica se
calcula ahora **antes** de parchear, así que T6 comprueba dos cosas de una vez: que el
export es fiel y que el `Concat` hace lo mismo que el `zeros_like`. Si el parche estuviera
mal, saltaría ahí.

**Sin ejecutar todavía.**

## 2026-09-20 · El export salió, y salió inservible · ⚠️ arreglado con `onnxsim`

El export terminó en la máquina de Vertex AI (`n1-highmem-8`, CPU) y **la verificación
numérica pasó**: el `.onnx` es fiel a torch. Pero el fichero salió de **1,03 GB** para un
modelo de 12,3 M de parámetros, y al abrirlo en el repo de detección:

    onnxruntime ... RUNTIME_EXCEPTION : Exception during initialization: bad allocation

**Dónde estaba el gigabyte.** Los pesos de verdad son 49 MB en 534 constantes. Los otros
**978 MB están en 1641 constantes dentro de nodos**, con formas de activación:
`[100, 152, 28, 50]`, `[100, 56, 56, 100]`… todas dentro de las capas `conv1` de RegNet,
o sea las que llevan gate-shift.

Es el `torch.zeros_like(x)` de `GatedShift.forward`. Al trazar, esa llamada se graba como
**un tensor de ceros del tamaño completo de la activación**. Once capas gate-shift y clips
de 100 frames: un gigabyte de ceros escritos en el fichero.

**El arreglo estaba en el blueprint y me lo salté.** §42.1 pone `onnxsim` entre el export
y la verificación numérica. No es cosmético: es lo que dobla esos ceros y deja el grafo
utilizable. Ahora `export_onnx.py` lo hace en un paso aparte, después de escribir el
fichero y antes de verificar.

**Aparte y no dentro del export** (`do_constant_folding`) a propósito: así su pico de
memoria no se suma al del trazado. Ese pico junto es exactamente lo que no cabía en Colab.

**Queda por medir**: cuánto baja el fichero y cuánta memoria y tiempo pide una inferencia.
Eso último importa para el pod, que tiene que correr esto al lado del panel.

## 2026-09-20 · Dónde se puede exportar, y por qué no en Colab · ⚠️

El export de T5 **no cabe en Colab**, ni en CPU ni con una T4. Está medido:

| Dónde | Qué pasa |
|---|---|
| Colab CPU (~12 GB de RAM) | SIGKILL: el vigilante de memoria mata el proceso |
| Colab T4 (14,56 GB de VRAM) | `torch.OutOfMemoryError` con 14,27 GB ya ocupados |

**Por qué**: el modelo mete los 100 frames de 448×796 en el backbone como **un solo
lote**, y el trazador clásico mantiene vivas todas las activaciones intermedias mientras
construye el grafo. Eso son ~15 GB, y no se puede reducir el clip: el codificador
posicional del modelo está fijado a 100 frames.

**Lo que hace falta no es GPU, es memoria.** Trazar es una pasada hacia delante; en CPU
sale exactamente el mismo grafo. Eso tiene dos consecuencias prácticas buenas: no hace
falta pelearse con la cuota de GPU de GCP —que es un formulario y una espera— y una
máquina de alta memoria cuesta una fracción de lo que cuesta una con acelerador.

**La vía**: `tools/export_on_vm.sh` en una máquina de Vertex AI de alta memoria
(`n1-highmem-8`, 52 GB, unos 0,47 $/h para un trabajo de minutos). Instala torch de CPU
—200 MB en vez de 2,5 GB—, clona los dos repos, baja los pesos de GCS, exporta, verifica
y sube el `.onnx` con su ficha de vuelta al bucket.

**Queda por probar** si el exportador de `torch.export` —que traza con tensores falsos y
casi no materializa nada— entra donde el clásico no. El script ya lo intenta primero, así
que si en la máquina grande funciona, sabremos además que en Colab habría bastado con eso.

## 2026-09-20 · T5 y T6 — el export y su verificación · ✅ escritos, ⬜ sin correr

`tools/export_onnx.py`. Corre en Colab, donde vive torch. **No se ha ejecutado**: se
escribió leyendo el código de T-DEED, no corriéndolo.

### Tres cosas del modelo que decidieron el diseño, y las tres se descubrieron leyendo

**1. El modelo normaliza por dentro.** Su `forward` hace `x / 255` y después la
estandarización de ImageNet. O sea que el `.onnx` espera píxeles **en crudo, 0-255**, y la
ficha declara `scale: 1`, `mean: [0,0,0]`, `std: [1,1,1]`. Copiar ahí los valores de
ImageNet —que es lo que uno haría mirando el paper— normalizaría dos veces. No da error:
hunde el score y a otra cosa.

**2. Tiene dos cabezas.** Se preentrena en SoccerNet (17 clases) y se afina en
SoccerNetBall (12); la capa final emite las dos concatenadas. La nuestra es la primera,
`1 + 12` columnas contando el fondo. El recorte va **dentro del grafo**: es un `Slice`
estático, exporta limpio, y así el repo de detección no sabe nada de esto.

**3. El desplazamiento no cabe en el grafo, y es lo importante.** El modelo emite, además
de las puntuaciones, un desplazamiento temporal por frame; su post-proceso lo aplica con
un **doble bucle de Python con índices que dependen de los datos**. Trazar eso a ONNX
hornearía los desplazamientos del clip de ejemplo dentro del grafo: no fallaría, y daría
mal todos los demás clips.

Así que el `.onnx` saca **dos salidas** —`logits` y `displacement`— y el desplazamiento lo
aplicará el spotter, en numpy, donde se puede probar. La ficha lo declara con
`meaning: displacement`.

### La columna 0 se llama `normal_play` a propósito

El modelo usa la primera columna como fondo. En la ficha se llama `normal_play`, que es
exactamente el nombre que el spotter del repo de detección conoce como la clase que nunca
produce candidatos. No es casualidad: es el contrato, y hace que la clase de fondo se
excluya sola sin que nadie configure nada.

### Probado lo que se puede probar sin torch, y el contrato entre repos

La generación de la ficha se prueba contra un `.onnx` sintético: que las formas salgan del
fichero y no de la configuración, que no se vuelva a normalizar el píxel, que la salida
del desplazamiento quede marcada y que el aviso de licencia esté.

Y se comprobó **de punta a punta entre los dos repositorios**: la ficha que genera este
repo la carga el de detección, le verifica el SHA-256, abre `VideoOnnxBackend` y monta el
`Spotter` con su ventana de 8 s. De paso, su comprobación de clases cazó un stub que
declaraba 13 clases y emitía 3 columnas — que es justo para lo que se escribió.

### Lo que queda, y es una trampa conocida

**El spotter todavía no aplica el desplazamiento.** La ficha lo declara y nadie lo lee,
que es **exactamente** el fallo del `layout` que se arregló esta mañana. No se puede dejar
así: sin aplicarlo, los eventos salen movidos hasta cuatro frames (`radi_displacement: 4`),
o sea un tercio de segundo. Es lo siguiente en el repo de detección.

**Siguiente paso**: correr el export en Colab con los pesos que ya están bajados. Si T6
pasa, el `.onnx` cruza y se acabó el camino de ida.

## 2026-09-20 · T1 — la línea base, corrida · ✅

**Ejecutada por el propietario en Colab** con los pesos publicados de
`SoccerNetBall_challenge1` sobre `videoGP.MP4` (11 min 50 s, 1280×720, 29,97 fps
convertidos a 25). Umbral 0.2.

### Lo que salió

**292 eventos** en 11:50. Por clase: PASS 97, DRIVE 63, HIGH PASS 47, OUT 30, SHOT 22,
BALL PLAYER BLOCK 14, GOAL 10, THROW IN 4, CROSS 3, HEADER 2. **FREE KICK: ninguno**, y
PLAYER SUCCESSFUL TACKLE tampoco.

Tras aplicar el NMS temporal de 2 s que usa el spotter, las 10 detecciones de GOAL se
quedan en **6 momentos** y las 22 de SHOT en **19**:

| GOAL | confianza |
|---|---|
| 02:05 | 0.599 |
| 03:24 | 0.581 |
| 04:24 | **0.962** |
| 05:28 | **0.888** |
| 06:00 | 0.255 |
| 07:57 | **0.888** |

### El hallazgo, y es el bueno

**Los 10 GOAL, sin una sola excepción, van precedidos de un SHOT entre 0,24 y 2,16 s
antes.** Eso no lo produce un modelo disparando al azar: es la estructura de una jugada
real, tiro y después gol, y aparece las diez veces.

De ahí sale una regla de fusión que **no hay que inventarse, ya está medida**: el par
«SHOT seguido de GOAL en menos de ~2,5 s» es mucho más fuerte que un GOAL suelto. Es
gratis de implementar y es justo lo que el documento de origen pedía para reducir falsos
positivos.

### Lo que esto valida del repo de detección

Las dos constantes que se eligieron a ciegas resultan ser las correctas:

- `SPOTTER_NMS_S = 2.0` colapsa exactamente las crestas de este modelo: los pares de GOAL
  separados 0,88–1,12 s se funden en uno, y los momentos distintos (a más de 50 s) se
  conservan. Sin él saldrían 10 clips donde hay 6 jugadas.
- El error temporal está **por debajo de 2 s**, y el clip se corta con granularidad de
  2 s sobre un pre-roll de 20 y un post-roll de 10. O sea que la imprecisión del modelo es
  irrelevante frente al tamaño del clip: cae dentro del margen por diseño.

### Contrastado contra el vídeo: **los seis son goles**

El propietario los revisó el 2026-09-20 y confirma que los seis momentos corresponden a
goles reales. Con eso, sobre este vídeo y a umbral 0.2:

**Precisión de GOAL: 6 de 6.** Ni un falso positivo, incluido el de confianza 0.255 —o
sea que el umbral podría bajarse todavía más sin ensuciar—.

Lo que **sigue sin medirse** es el recall: no se ha contado si el vídeo tenía más goles
que el modelo no vio. Para las metas de E8 (recall ≥ 90 %, precisión ≥ 85 %) hace falta
saber el denominador, y eso pide un partido completo con los goles anotados a mano. La
precisión, que era la mitad más preocupante, está.

Tampoco se sabe nada de FREE KICK: cero detecciones puede ser que no hubo ninguno o que
la clase no dispara.

**Veredicto**: el camino sigue, y con margen. Precisión 6/6 en goles, error temporal por
debajo de la granularidad del clip, ruido concentrado en clases que no usamos (PASS,
DRIVE y HIGH PASS son 207 de los 292) y una regla de fusión medida de regalo. Se pasa a
T5: exportar a ONNX con su ficha.

## 2026-09-19 · T2 — datos de SoccerNet · ✅ la herramienta, ⬜ la descarga

`uv run python tools/fetch_soccernet.py --check` dice lo que va a costar antes de costarlo.

**El acceso ha cambiado desde lo que dice la documentación vieja.** `spotting-ball-2024`
ya **no** se baja con la contraseña del NDA: viene de Hugging Face
(`SoccerNet/SN-BAS-2024`, ~19,3 GB) y el acceso lo controlan los permisos del repositorio.
El paquete `SoccerNet` avisa por su cuenta de que la contraseña se ignora para esa tarea.
La tarea de 17 clases sí sigue pidiendo el NDA.

**No se ha descargado nada**: son 19 GB y hace falta una cuenta de Hugging Face que acepte
los términos del dataset. Es una acción con nombre y apellidos, no algo que deba hacer una
herramienta por su cuenta.

**Y hay un aviso de licencia**: ese dataset se declara **GPL-3.0**. Para unos datos es
raro, y hay que mirarlo con calma antes de que nada entrenado sobre ellos entre en un
producto que se vende. Está anotado en `docs/DEPENDENCIES.md` junto al otro problema, el
de que T-DEED es GPL-3.0 también.

**Decidido el 2026-09-20**: Ball Action Spotting, y el córner fuera del primer modelo.
Está razonado en el [ADR 0001](DECISIONS/0001-ball-action-spotting-sin-corner.md). Lo que
manda es que solo esa tarea tiene pesos publicados, y sin pesos no hay línea base: la
primera pregunta no es cuántas clases se detectan sino si esto funciona con nuestra
cámara.

**Siguiente paso**: la línea base. No necesita los 19 GB —`inference.py` corre sobre **un
vídeo**— así que se puede hacer con metraje propio en cuanto haya pesos y una GPU. La
descarga de SoccerNet solo hace falta para T4 y para medir contra su ground truth.
