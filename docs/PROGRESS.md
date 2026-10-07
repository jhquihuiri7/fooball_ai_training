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
| ML-E1 | Infraestructura y contrato del repo | ML-01 ✅ · ML-02 ✅ · ML-03 ✅ · ML-04 🚧 · ML-05 ✅ |
| ML-E2 | Campaña de datos: grabación, ingesta, extracción y anotación | ML-06 ✅ · ML-07 ✅ · ML-08 · ML-17 · ML-18 · ML-19 ✅ · ML-20 · ML-21 ✅ · ML-22 ✅ · ML-23 · ML-58 |
| ML-E3 | Autoanotación con maestro y players-v1 | ML-24 · ML-25 · ML-26 ✅ · ML-27 · ML-28 · ML-29 · ML-30 · ML-31 |
| ML-E4 | Jugadores: D-FINE-N a 1920×576 | ML-15 ✅ · ML-16 · ML-32 · ML-33 · ML-34 · ML-35 |
| ML-E5 | Balón: heatmap ROI-lite de 3 frames en gris | ML-14 ✅ · ML-36 ✅ · ML-37 · ML-38 · ML-39 · ML-40 · ML-41 · ML-42 |
| ML-E6 | Export a Core ML y validación | ML-09 · ML-10 · ML-11 · ML-12 · ML-13 · ML-43 · ML-45 |
| ML-E7 | Spikes de modelo en el iPhone 17 | SPK-50 · SPK-51 · SPK-52 · SPK-53 · SPK-54 · SPK-56 |
| ML-E8 | Eventos aprendidos N3 y N4 | ML-47 🚧 · ML-48 ✅ · ML-49 · ML-50 🚧 · ML-51 ✅ · ML-52 · ML-53 · ML-54 · ML-55 · ML-57 |
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

## 2026-10-07 · Plan B de jugadores: preentreno de CenterNet-MNv4 con COCO person (ADR 0020, para REF-33) · 🚧 en pausa en el paso 527, reanudable

Para medir mañana en el iPhone un CenterNet-MNv4 con pesos reales, no los sembrados. Sin
datos propios (ML-17) ni bucket (ML-04), el preentreno se hace en este Mac con datos
abiertos. Commit del código: `0837cdd`.

**Decidido** (pendiente de revisión del propietario)
- **Datos: COCO 2017, solo `person`.** Anotaciones CC BY 4.0; cada imagen trae su licencia
  Flickr, y el 69 % de las 64.115 imágenes con personas son NC. El ADR 0002 acepta el
  preentreno COCO como riesgo residual: es el mismo de `dfine_n_coco.pth`. El índice guarda
  la licencia de cada imagen, y `--licencias 4,5,6,7,8` entrena solo con las 19.746
  comerciales si el propietario lo prefiere.
- **Tronco:** MobileNetV4-Conv-S de ImageNet (`timm/mobilenetv4_conv_small.e2400_r224_in1k`),
  Apache-2.0 según su ficha, en el commit `331fb80` y con el sha256 fijado. Con la regla de
  ML-52, la licencia del checkpoint se ha mirado antes de usarlo.
- **Clases:** las personas van a `player`. `goalkeeper` y `referee` se entrenan como
  negativos puros, para que en el iPhone no den cajas al azar. Separarlas es del afinado
  con datos propios.
- **Contrato de las salidas** (lo tendrá que decodificar la app): heatmap con sigmoid a paso
  4; `size` = ancho y alto **en celdas** (×4 = px de entrada); `offset` = centro dentro de
  la celda; picos por máximo local de 3×3, sin NMS. Está en `ftrain/players/centernet_data.py`.
  Si REF-33 elige CenterNet, se fija en un ADR del repo de detección.
- **Resolución:** se entrena con mosaicos en fila de 384×1152 (las imágenes de COCO, una al
  lado de otra y a escala al azar, con un alto del 25 % al 150 % del lienzo). Así las
  personas salen a tamaños de jugador lejano y cercano. Se evalúa a 576×1920, la franja.
  La red es convolucional entera: lo que cuenta es el tamaño de las personas en píxeles,
  no el del lienzo. El throughput en Mpx/s es el mismo a 576×1920, pero con lotes de 16 en
  vez de 8.
- **Horas:** `--hours 12.5`, con el lr en coseno por **tiempo**: acaba a su hora aunque el
  Mac vaya más lento de lo medido. El presupuesto cuenta las horas de todas las tandas.
- **En pausa.** Arrancó el 2026-10-07 a las 16:09 y se paró a las 16:25, porque el
  propietario necesitaba el Mac. Se paró justo después del primer `last.pt`: paso 527,
  0,25 h gastadas y todavía sin evaluación ni `best.pt`. La bajada de COCO también se
  paró, con 39.885 de 64.115 imágenes en disco.

**Medido** (MacBook Air M4 sin ventilador, 16 GB, MPS, fp32)
- Throughput en frío: unos 10,5 Mpx/s a cualquier forma (b16 384×1152: 676 ms por paso;
  b8 576×1920: 764 ms). channels_last y autocast fp16 van 3-5 veces **más lentos** en MPS.
  En caliente, y compartiendo la máquina con compilaciones de Xcode, el Air baja a 6,6-10,5
  img/s por estrangulamiento térmico.
- La bajada va a ~3 MB/s (unas 160 KB por imagen, 10 GB en total). El entreno arrancó con
  21.453 imágenes en disco y cada época vuelve a mirar cuántas hay.
- **Primera pérdida:** 169,4 en el paso 1 (focal 166,9, size 13,9, offset 1,15); 3,68 en el
  paso 50, 2,36 en el 350 (focal 1,81) y 2,65 en el 500, con el lr todavía subiendo. Humo
  previo de 120 pasos: AP de persona 0,010 (AP50 0,042) en 100 imágenes.

**Reanudar** (el mismo comando del arranque: si encuentra `last.pt`, sigue desde él)
```bash
# 1. Terminar la bajada (lo que ya está en disco no se repite):
uv run python tools/fetch_coco_person.py
# 2. El entreno corre desde un worktree fijado en 0837cdd (runs/centernet-coco/code),
#    separado del árbol de trabajo: así nadie le cambia el código mientras corre.
cd runs/centernet-coco/code
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -ims \
  ../../../.venv/bin/python tools/train_centernet.py --data ../../../datasets/coco2017 \
  --out .. --hours 12.5 --workers 4 >> ../train.log 2>&1 < /dev/null &
```
Si el worktree se ha borrado: `git worktree add --detach runs/centernet-coco/code 0837cdd`.
Pararlo limpio es matar el proceso justo después de un `last.pt`, que se escribe de forma
atómica cada 15 min. Se pierde como mucho lo hecho desde el último.
- Log: `runs/centernet-coco/train.log`, con la pérdida cada 50 pasos y el AP cada 2000.
- AP por evaluación: `runs/centernet-coco/metrics.jsonl`.
- `runs/centernet-coco/last.pt` se guarda cada 15 min. Si se corta, el mismo comando
  reanuda y descuenta las horas ya gastadas.
- `runs/centernet-coco/best.pt` es el state_dict de la EMA con mejor AP de persona: lo carga
  `ftrain.players.plan_b:build_centernet` tal cual.
- El Air tiene que seguir enchufado y con la tapa abierta: `caffeinate` evita el reposo por
  inactividad, pero no el de cerrar la tapa.

**Al terminar: exportar el mejor checkpoint y medirlo**
```bash
uv sync --group train --group apple
uv run python tools/export_coreml.py \
  --spec configs/export/players-centernet-mnv4.yaml \
  --builder ftrain.players.plan_b:build_centernet \
  --checkpoint runs/centernet-coco/best.pt \
  --classes goalkeeper,player,referee \
  --dataset-version coco2017-person-0837cdd \
  --out runs/centernet-coco/export
cd runs/centernet-coco/export && unzip -o players-centernet-mnv4.mlpackage.zip -d pkg && cd -
uv run python tools/ane_lint.py runs/centernet-coco/export/pkg/players-centernet-mnv4.mlpackage
# el .mlpackage descomprimido es el que copia el banco del iPhone (SPK-50)
```
El grafo es el mismo que el sembrado, así que la latencia tiene que seguir en unos 8 ms. Lo
nuevo es que detecta, y la app necesita el decodificador del contrato de arriba.

**Fuera**
- La puerta fp16 (ML-11) y los dorados de este checkpoint.
- La paridad: es un preentreno para medir, no un artefacto que se entregue.
- Afinar con la franja propia, cuando existan ML-17 y ML-28.

**Siguiente paso**: reanudar cuando el Mac esté libre (una noche entera), leer
`metrics.jsonl`, exportar `best.pt` y medirlo en el iPhone 17.
Con la cifra, que el propietario decida REF-33.

## 2026-10-05 · Plan B de jugadores (ADR 0020, para REF-33): CenterNet-MNv4 y YOLOX-Tiny medidos en el iPhone 17

SPK-51 dejó a D-FINE-N fuera: 0 % de ANE, 150 ms en GPU. Así que se construyen los dos
candidatos del plan B con pesos sembrados y se miden en el mismo banco (SPK-50), como
pide el ADR 0020 («dos exports con pesos sembrados, una tarde»).

**Hecho**
- `ftrain/players/plan_b.py`:
  - **CenterNet-MNv4:** MobileNetV4-Conv-S de timm, cuello FPN ligero hasta paso 4 (64
    canales), y heatmap de 3 clases, tamaño y desplazamiento; 1,31 M parámetros;
  - **YOLOX-Tiny:** reimplementado (Focus, CSPDarknet, SPP, PAFPN y cabeza desacoplada),
    sin NMS en el grafo; 5,03 M parámetros;
  - builders deterministas para `export_coreml.py`.
- Specs en `configs/export/players-{centernet-mnv4,yolox-tiny}.yaml` (franja de
  1920x576, RGB).
- Tests (3): formas, canales de CenterNet múltiplos de 16 y determinismo.

**Medido**
- **ane_lint**: CenterNet tiene 474 ops y 7 avisos (la salida de 3 clases y 6 ops en
  fp32). YOLOX tiene 861 ops y 33 capas con canales que no son múltiplos de 16 (24, 12,
  4, 3, 1), más 4 en fp32.
- **iPhone 17** (iPhone18,3; 500 predicciones tras calentar; CPU_AND_NE; térmica fair):

  | Modelo | Coste en el ANE | p50 | p99 | Compilación | Carga |
  |---|---|---|---|---|---|
  | CenterNet-MNv4 | 83,1 % | **8 ms** | 8 ms | 66 ms | 0,48 s |
  | YOLOX-Tiny | 67,9 % | 33 ms | 50 ms | 77 ms | 0,85 s |
  | D-FINE-N (SPK-51) | 0 % | 150 ms (GPU) | — | 155 ms | 3,5 s |

  Informe en `bench/model-bench-1791234120.json` del repo de la app.

**Recomendación para REF-33**: CenterNet-MNv4. Cabe 16 veces en el ciclo de 133 ms y en el
ANE, sin quitarle GPU al render. YOLOX-Tiny también cabe, pero cuatro veces más lento y con
un tercio fuera del ANE. La decisión es del propietario (REF-33): con ella se replantean
ML-32 a ML-35 sobre CenterNet.

## 2026-10-05 · ML-51 — el spotter N4 en streaming (modo clip y modo paso) · ✅

**Hecho**
- `ftrain/events/spotter.py`:
  - **backbone:** MobileNetV4-Conv-S de timm (Apache-2.0), con etapas de 32, 64, 96, 128 y
    960 canales. Se crea sin pesos: los pesos llegan con la destilación de DINOv2-S
    (ML-52), y la licencia del checkpoint se mira entonces;
  - **TSM causal** con estado explícito en la entrada de las etapas 1-3: 1/8 de los canales
    viene del fotograma anterior, con slice y concat;
  - **GRU desenrollada** con conv 1x1, sigmoid y tanh (64 canales de estado);
  - **cabeza** por fotograma: fondo, shot, goal y el desplazamiento al instante del evento.
- Dos modos con los mismos pesos: `forward(clip)` para entrenar y `step(frame, estado)`
  para exportar; `initial_state` da el estado en ceros.
- Constantes `N4_*` con su motivo.
- Tests (3, con importorskip de torch y timm): clip y bucle de pasos coinciden con
  Δ < 1e-5 en fp32; es causal; no hay Conv3d ni GRU/LSTM/RNN nativas ni scatter; las
  conv internas tienen canales múltiplos de 16.

**Fuera**: el export a Core ML del modo paso y la decisión sobre el estado (MLState o E/S
explícita) son de SPK-53 y ML-55. El entrenamiento espera a ML-52 y ML-53, que necesitan
metraje propio.

## 2026-10-05 · ML-48 ✅ y ML-50 🚧 — SkillCorner a rejillas de N3 y el primer preentreno en el Mac

Se construye según el ADR 0004, que sigue PROPUESTO. El propietario pidió seguir sin
esperar su respuesta, y queda pendiente de su revisión.

**Hecho**
- `tools/fetch_skillcorner.py`: SkillCorner/opendata al commit `4340d27` (2026-09-14). Son
  20 partidos de la A-League 2024/25, con tracking de la retransmisión a 10 Hz y eventos
  dinámicos. El LICENSE del repositorio es MIT y queda comprobado; los datos no traen
  licencia aparte, como deja escrito `PROCEDENCIA.txt`. Ocupan 1,8 GB en `datasets/`, sin
  commit.
- `ftrain/events/skillcorner.py`:
  - posiciones a 105x68 con el convenio de `PitchModel`, sin equipos y con porteros;
  - remuestreo de 10 a 7,5 Hz;
  - etiquetas a partir de las reanudaciones de `dynamic_events`, fechadas cuando quien
    reanuda pone el balón en juego, más el gol y el saque inicial de cada parte.
- `ftrain/events/features.py`: rejilla de 16x10 con 7 canales (jugadores, vx, vy,
  porteros, árbitros, balón y silbato), espejos del campo como aumento y etiqueta por
  paso dilatada ±1.
- `ftrain/events/tcn.py`: la TCN causal, toda en conv2d. Tiene 76 215 parámetros (tope
  100 000) y 127 pasos de campo receptivo (17 s), con dropout en los bloques.
  `ftrain/events/spotting.py`: decodificación por picos con supresión de ±2 s y P/R con
  tolerancia. `tools/convert_skillcorner.py` y `tools/train_e3.py` (MPS del Mac).
- El ADR 0004 queda alineado con la rejilla de la tarjeta (16x10, 7 canales).

**Medido**
- Conversión de los 20 partidos en 49 s, unas 33 h a 7,5 Hz. Etiquetas: córner 138,
  saque de banda 835, saque de puerta 259, saque inicial 82, falta 396 y gol 63.
- Preentreno: 12 partidos para entrenar, 4 para elegir umbrales y 4 para dar la cifra;
  3000 pasos en unos 7 min en el MPS. F1 de validación por clase:

  | Corrida | córner | banda | puerta | inicial | falta | gol |
  |---|---|---|---|---|---|---|
  | v1, sin aumento, umbrales en train | 0,10 | 0,18 | 0,03 | 0,23 | 0,03 | 0,00 |
  | v2, espejos + dropout + umbrales aparte | 0,19 | 0,31 | 0,15 | 0,25 | 0,15 | 0,06 |
  | v3, = v2 fechando al poner el balón en juego | 0,13 | 0,22 | 0,11 | 0,24 | 0,15 | 0,10 |

  La v1 memorizaba: F1 0,93 en entrenamiento.

**Lectura**: con 4 partidos de validación, la diferencia entre v2 y v3 es ruido. El
techo probable es la señal: el tracking de retransmisión solo ve parte del campo y el
resto lo extrapola. El N0 propio verá el campo entero fundido, y ahí está el afinado
(ML-49) y la comparación con N1 (EV-08). Ninguna clase se automatiza con esto (ADR 0013).

**Validación cruzada por partidos** (5 pliegues; cada uno da la cifra con 4 partidos y
elige los umbrales con otros 4; `train_e3.py --fold k`; v3, 3000 pasos). Agregado sobre
los 20 partidos:

| Clase | Verdad | P | R | F1 | F1 por pliegue |
|---|---|---|---|---|---|
| córner | 138 | 0,34 | 0,43 | **0,38** | 0,22-0,48 |
| saque de banda | 835 | 0,38 | 0,42 | **0,39** | 0,21-0,50 |
| saque de puerta | 259 | 0,30 | 0,31 | **0,31** | 0,14-0,40 |
| saque inicial | 82 | 0,45 | 0,51 | **0,48** | 0,25-0,68 |
| falta | 396 | 0,25 | 0,30 | **0,27** | 0,16-0,40 |
| gol | 63 | 0,12 | 0,19 | **0,15** | 0,06-0,22 |

La división fija de antes (los 4 últimos partidos) era el pliegue más flojo: de ahí el
0,15-0,31. Es la línea base del preentreno con SkillCorner. El objetivo de ML-50 (mejor F1
que N1 en saque de banda y en falta) se mide con el N0 propio.

**Siguiente paso**: ML-49 cuando haya partidos propios; con ellos, el afinado y la
comparación con N1.

## 2026-10-05 · ML-26 — evaluación por bandas (M4, R-P7-1) y tamaño del balón (M5) · ✅ el código, ⬜ las cifras

**Hecho**
- `ftrain/eval/detection.py`:
  - **emparejado** de la más segura a la menos, por clase y sin clase: IoU ≥0,5, y por
    distancia de centros (≤4 px) para las cajas de verdad de menos de 8 px de ancho;
  - **métricas por grupo** (banda por defecto; también luz y partido): recall, precisión
    a 0,5 y AP50 con interpolación de todos los puntos;
  - **intervalos del recall** por bootstrap sobre PARTIDOS (1000 remuestreos, semilla
    fija).
- `tools/eval_players.py`: verdad en COCO canónico (ML-22) con `match` y `light` por
  imagen y `band` por caja; predicciones en .npz de torch o de Core ML. Escribe
  `<out>.json` y una tabla Markdown por banda con el IC.
- `tools/ball_size_table.py`: el diámetro del balón (la media del ancho y el alto de su
  caja) en p5/p50/p95 por banda.
- Constantes de evaluación con su motivo en `ftrain/constants.py`.
- Tests (9): recall y precisión conocidos por banda, luz y partido; cajas pequeñas por
  centro; sin clase; una caja de verdad una sola vez; AP de una lista conocida; bootstrap
  determinista que contiene el recall; las dos herramientas de punta a punta.

**Fuera**
- La entrada directa de un .onnx por onnxruntime en `eval_players`: necesita el
  preproceso de la franja en este repo.
- Las cifras de M4 y M5, y borrar sus filas de MEDICIONES: esperan a la campaña anotada
  (ML-17 y ML-28).

## 2026-10-05 · ML-22 — formato canónico de etiquetas y conversores · ✅

**Hecho**
- `ftrain/labels.py`, con `FrameRef`, `Box` y `LabelSet`:
  - **COCO extendido nativo**, de ida y vuelta. Clases `MASTER_CLASSES` con ids 0..4.
    `occluded`, `truncated` y `blurred` son obligatorios (sin ellos, `LabelsError`);
    cada caja lleva su fuente (human/master/auto) y su confianza, y cada imagen su
    `unusable` y su sha256.
  - **CVAT XML 1.1**, de imágenes y de pistas de vídeo, de ida y vuelta. Las pistas se
    interpolan entre fotogramas clave, y lo interpolado sale con fuente `auto`.
  - **DEIM:** solo jugadores, ids 0..2 en `PLAYER_CLASSES`, en el lienzo de 1920x576 a
    través del `map_box` de `SideBand` (ML-19). Se quedan fuera los frames inservibles,
    y las cajas que no caen en el lienzo se cuentan en `skipped`.
  - **RF-DETR por teselas nativas (ML-24):** las cajas se recortan a la tesela si se ve
    al menos `LABEL_TILE_MIN_VISIBLE` (50 %) de su área.
- El balón es una caja con `center` y `diameter`, no un punto. La tarjeta habla de un
  `BallPoint`, pero manda el ADR 0003 §3.
- Tests (8): COCO→CVAT→COCO conserva cajas (<0,01 px), atributos y orden de clases;
  rechazos; balón; pistas con interpolación; DEIM; RF-DETR por teselas.

**Siguiente paso**: ML-26 (evaluación por bandas).

## 2026-10-05 · ML-47 — ADR 0004: N3 y N4 distribuibles y su formato · 🚧 PROPUESTO

**Hecho**
- `docs/DECISIONS/0004-eventos-n3-n4-distribuibles.md`. Baja a algo construible lo que el
  ADR 0021 de football-ai ya decidió:
  - N3: rejillas de 12×8 celdas con 2 canales (jugadores por celda y balón) a 7,5 Hz, 16 s
    de contexto causal, TCN de ≤100.000 parámetros y 6 clases más «nada»;
  - N4: un recorte de 448×256 por cámara a 7,5 Hz, MNv4-Conv-S destilado de DINOv2-S, TSM y
    GRU causal, con shot y goal más «nada»; la fusión de las dos cámaras va en el maestro
    (IOS-78);
  - etiqueta dilatada ±1 paso, decodificación por máximo local con supresión, y los
    umbrales en la ficha del modelo;
  - la salida es un `mark` del N0 v1 sin campos nuevos (`source` positions o spotter, y
    `by` n3-positions@x o n4-spotter@x). Los dos ejemplos pasan `validate_record` de
    `libs/vision/match_log.py`;
  - se mide por partido completo con ±2 s (N3) y ±1 s (N4): P, R, AP y error de fecha por
    clase, contra N1 (EV-08) y contra T-DEED.
- La tarjeta habla de `whistle` y `side`; manda el ADR 0021 (un silbato no es una marca y
  el registro no lleva cámara). Queda anotado en el ADR.
- `DEPENDENCIES.md`: E2E-Spot, MobileNetV4, DINOv2-S y SkillCorner como referencias o
  datos, con su licencia.

**Falta**: que el propietario acepte el ADR. Con eso empiezan ML-48 y ML-51.

## 2026-10-05 · ML-36 — trayectorias del balón para etiquetar · ✅

**Hecho**
- `ftrain/ball/trajectory.py`, en dos fases:
  1. en línea: un Kalman de velocidad constante por eje en píxeles nativos a 30 fps, una
     puerta de 70 px alrededor de la predicción (70 px por fotograma transcurrido
     mientras la pista tiene una sola detección y aún no sabe su velocidad) y
     asociación voraz por distancia;
  2. un repaso offline de cada pista en bruto contra todos los candidatos (`refine`): en
     cada fotograma, el candidato a menos de 12 px de una cuadrática robusta de sus
     vecinas (la de los dos lados, la de antes o la de después; la robusta quita la
     peor mientras alguna se aleje más de 12 px), y la prolongación por los extremos.
     Las pistas se rehacen de la más larga a la más corta y no reutilizan candidatos:
     los trozos de la misma pelota se quedan vacíos.
- Huecos de ≤8 fotogramas rellenos con una cuadrática de 3+3 vecinas (`fill_gaps`);
  puntuación de pista (suma de confianzas) y banderas `long_gap`, `rival_nearby` (<120
  px en el mismo fotograma), `low_confidence` (media <0,35) y `outside_mask`.
- Constantes nuevas con su derivación en `ftrain/constants.py` (BALL_GATE_PX_PER_FRAME
  y siguientes) y dos invariantes en test_constants.
- Tests (13): la parábola con un 20 % de huecos y 2 fantasmas por fotograma en 10
  semillas, un fantasma a destiempo, hueco largo, corte en dos pistas, rivales,
  confianza y máscara, pistas cortas fuera, recta con dos puntos y un bote.

**Medido** (parábola de 120 fotogramas, 20 % de huecos, 200 semillas):

| Fantasmas/fotograma | Pistas de más | Fantasmas aceptados | Incompletas | Error interpolado máx |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 6e-12 px |
| 2 | 0 | 0 | 0 | 6e-12 px |
| 4 | 0 | 2 | 0 | 1,8 px |

Los 2 de 4/fotograma son fantasmas que caen por azar a menos de 12 px de la curva. Un bote
con pérdida conserva sus 70 detecciones sin banderas. La primera versión (puerta que
crecía con los huecos y sin repaso) daba 14 pistas en vez de 1 con 2 fantasmas por
fotograma.

**Fuera**: los umbrales (12 px, 120 px, 0,35) están puestos para los sintéticos; se
ajustan con las primeras detecciones reales de la campaña (ML-37).

**Siguiente paso**: ML-47 (el ADR del spotter), que no necesita datos.

## 2026-10-04 · SPK-52 (la pata de export) — paquete multifunción con pesos deduplicados · ✅

**Hecho**
- `tools/export_multifunction.py`: cada función (global/roi) se convierte suelta
  con el convert de ML-09 y las fusiona `ct.utils.save_multifunction`, que
  DEDUPLICA los pesos; manifiesto normalizado y zip determinista con sha.
- `ftrain.ball.model.build`: el builder sembrado para los CLI (sin checkpoint,
  pesos aleatorios con semilla: la latencia no depende de ellos y el sha sí).
- Medido con ROI-lite real (global 896x1920 / roi 2x256x256): **1,036x el peso
  de un export suelto** (aceptación: ≤1,1x) ✓, y en el Mac las dos funciones
  cargan por `function_name` y predicen con sus formas.
- La guardia de ML-09 ahora dice la verdad: `functions` no va dentro de
  `convert` — la fusión es de este tool; `states` sigue esperando a ML-51.

Las latencias por función y el coste de cambiar se miden en el iPhone
(football-ai-capture, SPK-52).

## 2026-10-03 · ML-24 — teselado nativo, TTA y fusión · ✅

**Hecho**
- `ftrain/tiling.py`, todo numpy: rejilla de teselas del lado del modelo con
  solape en constante (`TILE_OVERLAP_PX = 256`, mayor que el objeto más alto:
  es lo que garantiza que nada quede partido en todas las teselas a la vez;
  la última tesela de cada eje se clava al borde), `crop` como vista,
  `to_native`, fusión por clase (`fuse`: NMS numpy con `FUSE_IOU = 0.6`, se
  queda con la confianza MÁXIMA, nunca un promedio), y el TTA con inversa:
  volteo horizontal y ampliación de las filas lejanas (`zoom_far_rows` con cv2
  perezoso; su inversa divide).
- Exactitud del volteo medida de verdad: con cajas float32 (lo que emite un
  detector) la involución es bit a bit **si la resta se hace en float64** — con
  el array en float32, numpy restaba en float32 por la promoción débil del
  escalar y se perdía un ulp. Está en el docstring y en el test.
- Tests (9, puros): cobertura y borde de la rejilla, imagen menor que la
  tesela, solape inválido, vista+vuelta a nativo, **el objeto sobre la costura
  sale UNA vez y con la confianza máxima**, fusión por clases, involución
  exacta del volteo, inversa exacta de la ampliación, validaciones.

**Siguiente paso**: ML-17 (muestreo de frames) si sus deps lo permiten; si no,
quedan las tareas que esperan al propietario (ML-04/06/21) o a hubs/REF-33.

## 2026-10-03 · ML-16 — D-FINE-N COCO a 1920×576, exportado y con su fp16 medido · ✅ criterios (🚧 subida a GCS, espera el bucket de ML-04)

**Hecho**
- `ftrain/players/dfine.py`: `build()` desde la config de DEIM (YAMLConfig,
  `eval_spatial_size` nuestro, HGNetv2 sin pretrained), checkpoint EMA con los
  búferes `decoder.anchors/valid_mask` filtrados (son función del tamaño y se
  regeneran), deploy, wrapper → `logits [1,300,80]` y `boxes [1,300,4]` cxcywh
  en [0,1], sin PostProcessor; y `export_onnx()` (opset 17, TODO estático,
  formas de salida fijadas tras `shape_inference`). `_patch_integral`: el
  `F.linear(x, project)` del DFL usa un VECTOR de pesos y MIL exige rango 2 —
  se reescribe como matmul con columna [R,1], misma cuenta, cero aproximación.
- `transformers` al grupo train: calflops lo importa AL IMPORTARSE
  (`engine.misc` → profiler_utils → calflops → transformers); la nota vieja de
  pyproject («su código no lo importa») era falsa por esa vía. Anotado en
  DEPENDENCIES.md en el mismo commit.
- `tools/dfine_dap.py`: ΔAP torch fp32 ↔ Core ML fp16 sobre COCO val2017 con
  letterbox (gris 114, esquina, sin deformar) y decode DETR top-300.
- `configs/ane_lint/dfine-n.yaml`: `permitidos: [topk]` con la excusa escrita;
  los DOS TopK del grafo (selector de queries y stats del LQE) son arquitectura
  de D-FINE, no postproceso — «sin topk en el grafo» literal es imposible aquí.

**Medido (banda 576×1920; el retranqueo 512×1536 exporta y linta igual)**
- torch↔ORT: por índice max|Δ| ≈ 1.0, pero es REORDENACIÓN del topk de queries
  (filas equivalentes intercambiadas entre backends); sobre valores ORDENADOS
  max|Δ| = **3.4e-04 < 1e-3** ✓ (la regla del repo, bien planteada para DETR).
- ane_lint con su lista blanca: **orden-fuera-de-cola = 0** ✓. El yaml no calla
  el resto: canales 87, rango-5 33 (atención deformable), fp32 3 → exit 1 a
  propósito: es la foto de lo que NO es ANE-limpio, para REF-33/SPK-51.
- **ΔAP fp16 = −0.28 puntos AP** (500 img; all-classes 9.95→10.22, person
  4.18→4.28: fp16 ni resta) ✓ objetivo ≤ 1.0.
- El harness está validado: a 640×640 (forma nativa) da **49.2 AP** en 100 img
  (paper: 42.8 en las 5k). El AP bajo a 1920×576 es REAL: el COCO-preentrenado
  se degrada fuerte en la forma de banda sin afinar — la razón de ser de
  ML-32/ML-33, y dato para REF-33.

**Qué quedó fuera**: la subida de los paquetes a GCS para SPK-51 — espera el
bucket de ML-04 (propietario). Los artefactos se regeneran con
`ftrain.players.dfine` + `tools/export_coreml.py --builder
ftrain.players.dfine:build_band` en minutos.

**Siguiente paso**: ML-24 (teselado/TTA, H2).

## 2026-10-03 · ML-13 — ficha v2 para registry.yaml · ✅

**Hecho**
- `ftrain/export/ficha.py`: `Ficha` (dataclasses por bloque: `InputBlock`,
  `OutputBlock`, `OnnxArtifact`, `CoremlArtifact`, `ParityBlock`) → `to_entry()`
  valida AL CONSTRUIR las mismas coherencias que el lector de la referencia
  (postprocess↔box_format, heatmap_stride/nms_iou condicionales, frames 3 →
  gray_temporal, compute_units solo cpu_and_ne) y `render_registry()` escribe el
  documento v2 entero. `onnx_facts()` lee formas/nombres/opset/sha del `.onnx` y
  `coreml_facts()` lee target (specificationVersion→iOS) y precisión (fp16 si el
  programa MIL la usa, vía el walker de ML-10) del `.mlpackage`.
- `tools/export_onnx.py` pasa a leer el `.onnx` con `onnx_facts` y delega su
  `sha256_of` en el de ML-09, sin cambiar un carácter de la ficha de tdeed-snb.
- Dos cosas medidas contra el commit FIJADO de la referencia: `ane_cost_pct` es
  obligatorio en artifacts.coreml (el lector lo exige; la ficha lo refleja), y
  `analysis_zone` NO existe en sus REGIONS (full_frame, playable_band, roi) —
  la tarjeta la menciona, manda la referencia; anotado en el docstring.
- Tests (5): entrada completa y render, coherencias que saltan al construir,
  `onnx_facts` sobre un .onnx sintético, **contrato**: `ref.load_registry` carga
  un registro con la ficha detr (coreml+parity) y la heatmap (frames 3) sin
  error, y `coreml_facts` sobre un mlpackage real (min_ios 18, fp16).

**Aceptación**: load_registry de la referencia carga la ficha generada ✓; los
tests de T5 siguen en verde (30 passed en test_tools) ✓.

**Siguiente paso**: ML-16 si está libre; si no, ML-24 (H2).

## 2026-10-03 · ML-12 — vectores dorados legibles desde Swift · ✅

**Hecho**
- `ftrain/export/golden.py`: el bundle `golden/<modelo>-<versión>/` con
  `manifest.json` (formas, dtype, layout y tolerancia OBLIGATORIA por salida y
  por ruta — falta una y `write_bundle` falla nombrándola), arrays `.bin`
  little-endian en orden C (dtypes explícitos `<f4 <f2 |u1 <i4`; i64 no entra,
  igual que en los dorados v1), `detecciones.json` con lo que decide la
  referencia, y `README.md` con el lector Swift de ~30 líneas. El zip
  determinista es el MISMO de ML-09 (`deterministic_zip`), así que el sha256
  ancla el bundle.
- `tools/golden.py`: corre el lote con semilla por torch fp32, ORT fp32 y (en
  macOS con `--package`) Core ML fp16; guarda las entradas como las entrega la
  app (BGRA uint8 vía `to_bgra8`, fp16 planar los tensores) y decodifica con
  `ftrain.ref` (`--decode detr|heatmap`: sigmoid + decode_boxes_to_corners, o
  heatmap_peaks).
- Tests (8, puros): ida y vuelta idéntica, sha estable con mtimes distintos y
  distinto si cambia un array, tolerancias obligatorias, little-endian byte a
  byte, i64 rechazado, versión del manifiesto comprobada, cruce BGRA.
- Humo del CLI en el Mac: bundle con las TRES rutas, detecciones de la
  referencia, y el mismo sha en dos ejecuciones completas.

**Siguiente paso**: ML-24 (teselado/TTA, H2); ML-13 sigue esperando a ML-15/REF-33.

## 2026-10-03 · ML-11 — paridad torch ↔ ONNX ↔ Core ML · ✅

**Hecho**
- `ftrain/export/parity.py`: comparadores puros (max|Δ| y Δ relativo por salida,
  peor delta del lote), decodificadores de tarea enchufables (detector: Δrecall
  por banda y desplazamiento del centro, con `fallback_band_of` de franjas
  horizontales hasta que ML-13 enchufe la tabla fila→metros; heatmap: Δ del pico;
  spotter: Δlogits), lote con semilla (u8 HWC para imágenes, fp32 para tensores)
  y runners de torch, onnxruntime y Core ML (CPU_AND_NE / CPU_ONLY, solo macOS).
- `tools/parity.py`: el CLI corre el mismo lote por los tres backends, escribe el
  informe JSON y sale con 1 si algún par supera su umbral. torch↔ORT conserva la
  regla del repo (1e-3); el de Core ML sale de `parity.atol` de la ficha
  (`--ficha`) o de `--atol-coreml`.
- Tests (11): puros en cualquier SO; ORT contra numpy sobre un .onnx sintético
  (ojo: el onnx instalado escribe IR 14 y ORT lee hasta 13 — el test fija
  `ir_version`); torch↔ORT < 1e-3 con una red pequeña; y en el Mac aparece y
  PASA la columna de Core ML (CPU_ONLY, delta < 5e-2). Humo del CLI: violación
  forzada → código 1 e informe escrito.

**Qué quedó fuera**: el Δrecall usa franjas horizontales de la imagen como bandas
provisionales; la tabla real fila→metros del band.json entra con ML-13.

**Siguiente paso**: ML-24 (teselado/TTA, H2) o ML-12 cuando toque por orden.

## 2026-10-03 · ML-10 — ane_lint: lint estático del programa MIL · ✅

**Hecho**
- `ftrain/export/ane_rules.py`: motor de reglas sobre una lista NEUTRA de ops
  (`MilOp`: tipo, dtype, formas) — cada regla se prueba sin coremltools. Reglas:
  conv3d, gru/lstm/rnn, scatter*, topk/argsort/NMS fuera de la cola, while_loop/
  cond, rango 5, canales de rango 4 no múltiplos de 16, dimensiones simbólicas y
  fp32 fuera de la cola. La cola es el sufijo contiguo de tipos declarados en la
  lista blanca más `cast` y `const` (pegamento, no cómputo: los const de los
  parámetros del topk cortaban la cola si no).
- Exenciones medidas en un paquete real: las cabezas (salidas del programa) no
  pagan el múltiplo de 16, y el preproceso que coremltools inyecta sobre la
  entrada (`image__scaled__` y su const, en fp32 SIEMPRE con ImageType) no paga
  ni canales ni fp32 — sin eso, todo modelo con imagen fallaba de serie.
- Walker del proto MIL (`ops_from_spec`/`ops_from_package`, bloques anidados) y
  CLI `tools/ane_lint.py`: JSON con las violaciones y código 1 si hay alguna.
- Lista blanca por modelo en `configs/ane_lint/<modelo>.yaml` (plantilla
  comentada incluida): `cola` (tipos del postproceso) y `exentos` (nombres).
- Tests: `test_ane_rules.py` (9, puros) y `test_ane_lint_mil.py` (3, programas
  construidos con `mb` e importorskip). Aceptación cumplida: un topk en medio
  falla nombrando la op; humo del CLI sobre el demo de ML-09 → exit 1 con solo
  los positivos verdaderos (activaciones de 8 canales).

**Siguiente paso**: ML-11 (paridad torch/ORT/Core ML), también desbloqueada por ML-09.

## 2026-10-03 · ML-09 — export a Core ML: fp16, iOS 18, formas fijas y sha estable · ✅

**Hecho**
- `ftrain/export/coreml.py`: `ExportSpec` desde YAML (entradas ImageType RGB/BGR con
  escala 1/255 o TensorType fp16; los nombres de salida son el contrato),
  `convert()` por `torch.export` + `run_decompositions({})` (coremltools no traga el
  dialecto TRAINING de torch 2.14) con `jit.trace` de respaldo, `ct.convert` a
  mlprogram FLOAT16 iOS 18, renombrado de salidas sobre UNA copia del spec (cada
  `get_spec()` devuelve una nueva: renombrar sobre copias sueltas se perdía) y
  `user_defined_metadata` todo en texto.
- El sha256 estable: `normalize_manifest()` reescribe los UUIDs del `Manifest.json`
  como uuid5 de su ruta y `deterministic_zip()` fija orden, fecha (1980) y
  atributos. Dos exports del mismo modelo dan el mismo fichero.
- `tools/export_coreml.py`: CLI `--spec --builder modulo:funcion --checkpoint --out`
  que imprime `{package, sha256}` para la ficha v2 del registro.
- `tests/test_export_coreml.py`: 6 tests — spec, metadatos y zip puros (corren en
  Windows) + conversión real con importorskip. **En el Mac predice**: el paquete
  normalizado abre y `predict()` devuelve `heatmap (1,1,32,32)`.

**Qué quedó fuera**: StateType y multifunción. El spec ya declara `states` y
`functions` (el YAML no cambiará), pero `convert()` falla con un mensaje claro si
aparecen: implementarlos llega con ML-51/SPK-53 y SPK-52.

**Siguiente paso**: ML-10 (`ane_lint`), que ya está desbloqueada.

## 2026-10-03 · ML-21 — ADR 0003: CVAT, formato de etiquetas y guía de anotación · ✅ (aceptado el 2026-10-05)

**Hecho**
- `docs/DECISIONS/0003-anotacion-cvat-y-formato.md` (PROPUESTO): CVAT autohospedado
  frente a Label Studio — gana por la interpolación de pistas, que es el 80 % del
  coste de anotar el balón a 30 fps; formato canónico COCO extendido EN NATIVO (el
  lienzo caduca con cada band.json; lo nativo no, y el paso ya existe en ML-19);
  clases de MASTER_CLASSES con atributos booleanos siempre presentes; el balón es
  caja + atributos, nunca un punto (el heatmap entrena desde el centro y la caja
  conserva el tamaño por banda); eventos del N0 en JSONL aparte por rig_ms; doble
  anotación del 5 % repartida por bandas.
- `docs/GUIA_ANOTACION.md` (normativa, con un ejemplo por caso): jugadores con
  occluded/truncated y la máscara del campo como frontera; el balón según §40.4
  (parte visible + occluded, estela completa + blurred, distractor_ball, nada si no
  se ve, y unusable por debajo de 3 px); pistas que se cortan al desaparecer; qué
  entrega un lote.

**Pendiente para el ✅**: que el propietario acepte el ADR (es la aceptación de la
tarjeta). Sin código: pytest sigue en 152 passed, 1 skipped.

## 2026-10-03 · ML-19 — bandas de distancia y paso de etiquetas nativo↔lienzo · ✅

**Hecho**
- `ftrain/bands.py`: `load_band_spec` lee el band.json de REF-29 (los dos lados; la
  rotación por lado viene del manifiesto de ML-06, con el nominal «izquierdo
  invertido» por defecto); `band_of(d)` sobre DISTANCE_BANDS_M (20.0 ya es la banda
  [20,40)); la distancia por homografía (ref `PitchModel.image_to_pitch` + la
  posición del soporte de `recover_camera`) o por la tabla fila→metros del spec
  (interpolada, sujetando en los extremos); y el viaje nativo↔lienzo con la MISMA
  `BandGeometry` de la referencia vía la fachada — la rotación de 180° se aplica y
  deshace sola, una caja no se parte en la junta (la región la decide el centro,
  convención REF-21), y una etiqueta en la franja enmascarada del código o fuera de
  la banda queda MARCADA (`status`), nunca perdida en silencio. El código manda
  sobre todo: se pinta en el (0,0) de lo capturado en los dos montajes.
- La fachada `ftrain/ref.py` gana BandGeometry, recover_camera, strip_height y
  TIMECODE_BITS (test_ref acepta constantes); el band.json de REF-29 ganó en el
  repo de detección la tabla `row_to_m` como campo PASANTE (anclas geométricas:
  densas en el campo cercano, donde la perspectiva estira los metros por fila).
- `tests/test_bands.py` (7): ida y vuelta < 0,5 px, la rotación contada y
  deshecha, la caja entera en la junta del mosaico, el código y lo de fuera
  marcados, y la banda por homografía y por tabla coincidiendo en ±2 m sobre la
  nominal sintética.
- ruff ✅ · ruff format ✅ · pytest 152 passed, 1 skipped.

## 2026-10-03 · ML-07 — verificación de grabaciones · ✅

**Hecho**
- Grupo `data` nuevo (PyAV, BSD-3, anotado en DEPENDENCIES.md: la rueda lleva un
  FFmpeg con partes GPL que aquí solo DECODIFICA y no se distribuye).
- `ftrain/recording.py`: `check_recording` con umbrales inyectables
  (`RecordingLimits`; los de verdad en constants.py con unidades) — códec/forma,
  30±0,1 fps, bitrate medio ≥40 Mbit/s, rigMs legible con LA REFERENCIA en ≥99 % de
  una muestra a 1 fps y estrictamente monótono — y `check_pair_overlap`: el solape
  de rigMs entre cámaras sobre la UNIÓN de lo grabado (una cámara que arranca tarde
  cuenta en contra). Informe JSON apto tal cual.
- `tools/check_recording.py`: 1 o 2 ficheros, `-o informe.json`, sale 0/1.
- `tests/test_recording.py` (7): sintéticos mpeg4 con el código pintado por la
  referencia — el bueno OK; sin código, a 25 fps, retrocediendo, flaco de bitrate y
  de códec equivocado FALLAN nombrando el motivo; el solape se hunde si una cámara
  llega tarde.
- **Aceptación con grabación real** (left-1791038890-2.mov del soporte, 174,8 s):
  OK en todo — hevc 3840×2160, 30,015 fps, 45,1 Mbit/s, 174/174 códigos legibles y
  monótonos.
- ruff ✅ · ruff format ✅ · pytest 145 passed, 1 skipped.

## 2026-10-03 · ML-06 — protocolo de grabación y manifiesto de partido · ✅ (protocolo pendiente de revisión del propietario)

**Hecho**
- `docs/PROTOCOLO_GRABACION.md`: la lista previa (NTP, «Guardar vídeo» en los dos,
  audio, refrigeración, PD), la matriz de la campaña (6-10 partidos, ≥3 canchas, día
  y focos, dos retranqueos y dos inclinaciones, y la cancha SAGRADA que jamás entra
  en train), qué se captura además (pareja de calibración, marcas N0 y 3 ráfagas de
  10 s de NV12 por cámara y parte con el NV12_DUMP_S de IOS-15) y la hoja de campo.
- `matches/_plantilla.yaml` comentada campo a campo.
- `ftrain/manifest.py`: `MatchManifest`/`SideRecording` con validación que nombra el
  campo — luz ∈ {day, floodlight, dusk}, rotación 0/180 por lado, retranqueo y
  altura no negativos, URIs gs://, los dos lados obligatorios, el bloque de
  calibración (rig_uri y band_uri) y `split_hint` ∈ {train, val, sagrada}.
- `tests/test_manifest.py` (17): la plantilla valida; cada obligatorio, enumeración
  y URI se defiende nombrándose.
- ruff ✅ · ruff format ✅ · pytest 138 passed, 1 skipped.

**Pendiente**: la revisión del protocolo por el propietario antes del primer
partido, que es parte de la aceptación.

## 2026-10-03 · ML-04 — datos en GCS y arranque en RunPod · 🚧 falta el pod limpio

**Hecho**
- `docs/DATOS.md`: el árbol del bucket (raw/, nv12/, frames/, labels/, datasets/,
  models/, golden/, runs/) y la política — versiones INMUTABLES, sha256 por fichero
  en sidecar, credenciales solo por entorno.
- `ftrain/storage.py`: la envoltura de `gcloud storage` (cp, ls, cat del sidecar,
  exists, download) con runner inyectable y `dry_run` que deja las órdenes exactas
  en `commands`. El bucket sale de FAI_GCS_BUCKET y de las credenciales solo se
  comprueba que GOOGLE_APPLICATION_CREDENTIALS exista: su valor no aparece en
  órdenes, errores ni logs. La subida es idempotente por sha (mismo sha → ni un cp).
- `tools/runpod_bootstrap.sh` (modelado sobre export_on_vm.sh): clona AL COMMIT,
  `uv sync --group train --group ref`, autentica gcloud escribiendo la clave que
  llega por GOOGLE_APPLICATION_CREDENTIALS_JSON a un fichero 600 (jamás por argv ni
  por la salida) y baja la versión del dataset con caché en el volumen de red
  (/workspace/cache): inmutable, así que lo cacheado no caduca.
- `tests/test_storage.py` (7): sin bucket y sin credenciales fallan nombrando la
  variable; el dry-run congela las órdenes exactas; subir dos veces el mismo sha no
  resube; un contenido distinto sí; la ruta de la clave no aparece en ninguna orden.
- ruff ✅ · ruff format ✅ · pytest 121 passed, 1 skipped (en el Mac de referencia).

**Pendiente para el ✅**: la pasada a mano en un pod limpio (necesita la cuenta de
servicio de GCS y el bucket creados — propietario); se anota aquí al hacerla.

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
