"""Constantes del entrenamiento con la arquitectura B (ADR 0002).

Cada una lleva sus unidades y de dónde sale. Las que existen también en el repo de
detección tienen que valer lo mismo en los dos: una diferencia no da error, solo mueve las
detecciones.
"""

from __future__ import annotations

import math
from typing import Final

DISTANCE_BANDS_M: Final = (0.0, 20.0, 40.0, 60.0, 80.0, math.inf)
"""Bordes de las bandas de distancia a la cámara, en metros: 0-20, 20-40, 40-60, 60-80 y
más de 80. Todo se evalúa por banda (el recall, la puerta fp16, las cuotas del muestreo),
porque un jugador a 80 m ocupa pocos píxeles y la media del campo entero lo esconde."""

SPARSE_SAMPLE_INTERVAL_S: Final = 2.0
"""Segundos entre dos frames de la extracción dispersa. Dos frames más cercanos son casi la
misma imagen para el detector de jugadores; la extracción densa, a 30 fps, es para las
pistas del balón."""

PLAYER_CLASSES: Final = ("goalkeeper", "player", "referee")
"""Clases del detector de jugadores, en el orden del export: el índice es la columna de
`logits` (ADR 0020 §1 del repo de detección). Cambiar el orden no da error, cambia qué es
cada caja."""

MASTER_CLASSES: Final = ("ball", "distractor_ball", "goalkeeper", "player", "referee")
"""Clases de las semillas y del maestro de autoanotación (RF-DETR-M, ML-29), en su orden.
`distractor_ball` es un balón que no es el del partido (en la grada, en el calentamiento, uno
de repuesto): se anota aparte para que el modelo aprenda a distinguirlo (BLUEPRINT §40.4)."""

PLAYER_INPUT_W: Final = 1920
"""Ancho de la entrada del detector de jugadores, en píxeles: los 3840 de una cámara 4K por
`BAND_SCALE`."""

PLAYER_INPUT_H: Final = 576
"""Alto de la entrada del detector de jugadores, en píxeles. Una franja jugable de hasta
1152 filas nativas entra entera por `BAND_SCALE`; una más alta entra en mosaico por
distancia (ADR 0020 §1)."""

BAND_SCALE: Final = 0.5
"""Escala de la franja jugable al entrar al detector de jugadores, adimensional. Es la misma
en x y en y, para no deformar al jugador."""

BALL_ROI_SIDES_PX: Final = (256,)
"""Lados de las ROIs nativas del balón, en píxeles. La de 320 solo entra si la aprueba
SPK-52 (ADR 0020 §2)."""

BALL_TEMPORAL_FRAMES: Final = 3
"""Frames en gris apilados a la entrada del modelo del balón: t-2, t-1 y t, consecutivos a
30 fps. El movimiento es lo que separa un balón de 3-5 px de una cabeza o de una línea."""

BALL_HEATMAP_STRIDE: Final = 2
"""Píxeles de entrada por celda del heatmap del balón. Tiene que valer lo mismo que
`BALL_HEATMAP_STRIDE` en `libs/vision/constants.py` del repo de detección, que decodifica
los picos: si no, las posiciones salen escaladas."""

PIXEL_SCALE: Final = 1.0 / 255.0
"""Escala del píxel dentro del modelo: la entrada llega en 0-255, tal como la deja Metal, y
el propio grafo la lleva a 0-1 (ADR 0020 §3). Adimensional."""

ANE_CHANNEL_QUANTUM: Final = 16
"""Los canales internos de un modelo para el ANE son múltiplos de este número (ML-14):
así la capa no paga relleno. Se exceptúan la entrada y las cabezas."""

BALL_GAUSS_D_PX: Final = 2.5
"""Tamaño del gaussiano del objetivo del heatmap del balón, en píxeles de entrada. El plan
no dice si es la sigma o el diámetro: lo fija ML-38 al construir el objetivo."""

FP16_MAX_RECALL_DROP_PP: Final = 2.0
"""Caída máxima del recall de fp32 a fp16, en puntos porcentuales, en cada banda de
distancia. Es la puerta del export a Core ML (ADR 0002): en el ANE ninguna capa puede
quedarse en fp32, así que si no pasa, el export falla."""

CENTER_SHIFT_MAX_PX: Final = 1.0
"""Desplazamiento máximo de fp32 a fp16, en píxeles de entrada y en mediana: el del centro
de la caja en los jugadores y el del pico en el balón."""

ONNX_VERIFY_TOL: Final = 1e-3
"""Diferencia absoluta máxima entre las salidas de torch y de onnxruntime sobre la misma
entrada (CLAUDE.md §1). Con T-DEED, T6 midió 9,54e-06."""


# --------------------------------------------------------------------------- #
# Verificación de grabaciones (ML-07)
# --------------------------------------------------------------------------- #

RECORDING_WIDTH: Final = 3840
"""Píxeles. Ancho que graba el soporte: 4K, la verdad del partido (ADR 0012)."""

RECORDING_HEIGHT: Final = 2160
"""Píxeles. Alto de la grabación del soporte."""

RECORDING_FPS: Final = 30.0
"""Hz. Cadencia de la grabación."""

RECORDING_FPS_TOL_HZ: Final = 0.1
"""Hz de margen sobre la cadencia declarada: 30±0,1. Más desvío es un modo de
cámara mal puesto, no jitter."""

RECORDING_MIN_BITRATE_BPS: Final = 40_000_000
"""Bits por segundo mínimos. La app graba HEVC a 45 Mbit/s; por debajo de 40 algo
recomprimió el fichero por el camino y ya no sirve de dataset."""

RECORDING_SAMPLE_INTERVAL_S: Final = 1.0
"""Segundos entre fotogramas muestreados al leer el código de tiempo: uno por
segundo basta para legibilidad y monotonía sin decodificar el partido entero."""

RECORDING_MIN_TIMECODE_READABLE: Final = 0.99
"""Fracción mínima de la muestra con rigMs legible. Un código ilegible es un
fotograma que el emparejado no puede usar."""

RECORDING_MIN_RIG_OVERLAP: Final = 0.95
"""Fracción mínima del partido en la que los rigMs de las dos cámaras se solapan:
por debajo, una de las dos arrancó tarde o murió pronto."""

# --------------------------------------------------------------------------- #
# Trayectorias del balón para etiquetar (ML-36)
# --------------------------------------------------------------------------- #

BALL_GATE_PX_PER_FRAME: Final = 70.0
"""Píxeles nativos 4K por fotograma a 30 fps: la puerta de asociación. Lo más que se aleja
una detección de la predicción de su pista. Sale de los 58,5 px que recorre en un fotograma
un balón a 25 m/s visto a 20 m, con un 20 % de margen. No crece con los huecos: la
predicción de velocidad constante ya sigue al balón, y una puerta que crece deja entrar
fantasmas (con 2 por fotograma, 14 pistas en vez de 1)."""

BALL_OUTLIER_PX: Final = 12.0
"""Píxeles nativos. Una detección que queda más lejos que esto de lo que extrapolan
las de antes Y de lo que extrapolan las de después no es el balón: se quita y su
fotograma pasa a hueco. Un bote cae cerca de las dos extrapolaciones; un fantasma, no."""

BALL_TRACK_MAX_COAST_FRAMES: Final = 15
"""Fotogramas (0,5 s a 30 fps) que una pista sigue viva sin detecciones. Más es un corte:
lo que aparezca después empieza otra pista."""

BALL_INTERP_MAX_GAP_FRAMES: Final = 8
"""Fotogramas. Los huecos de hasta este largo se rellenan con la cuadrática; los más largos
quedan sin rellenar y marcan la pista con `long_gap` para que la mire una persona."""

BALL_INTERP_CONTEXT_POINTS: Final = 3
"""Detecciones a cada lado de un hueco con las que se ajusta la cuadrática. Tres y tres
sobran para una parábola y no arrastran el ajuste a un rebote lejano."""

BALL_RIVAL_RADIUS_PX: Final = 120.0
"""Píxeles nativos. Dos pistas que en el mismo fotograma pasan a menos de esto (dos
desplazamientos de un balón rápido) son ambiguas: cuál es el balón lo decide una persona."""

BALL_LOW_MEAN_SCORE: Final = 0.35
"""Confianza media de las detecciones de una pista por debajo de la cual se marca
`low_confidence`. Adimensional, 0-1."""

BALL_TRACK_MIN_DETECTIONS: Final = 5
"""Detecciones mínimas para que una pista salga. Las más cortas son fantasmas: un balón real
se ve al menos un sexto de segundo seguido."""

BALL_KF_ACCEL_STD_PX: Final = 8.0
"""Ruido de proceso del Kalman de velocidad constante: desviación de la aceleración, en
píxeles por fotograma al cuadrado (la gravedad vista de lejos, los efectos y los botes)."""

BALL_KF_MEAS_STD_PX: Final = 2.0
"""Ruido de medida del Kalman: desviación del centro detectado, en píxeles nativos."""

BALL_REFINE_CONTEXT_POINTS: Final = 12
"""Detecciones vecinas (las más cercanas en el tiempo) con las que el repaso offline
ajusta la cuadrática robusta de cada fotograma: unos seis fotogramas a cada lado."""

BALL_REFINE_PASSES: Final = 3
"""Vueltas máximas del repaso offline de una pista; para antes si una vuelta no cambia
nada."""

LABEL_TILE_MIN_VISIBLE: Final = 0.5
"""Fracción del área de una caja que tiene que verse en una tesela para entrar en el COCO
de RF-DETR (ML-22). Menos es una caja cortada que enseña al maestro a detectar medio
jugador; el solape de las teselas (ML-24) garantiza que la entera está en otra."""

# --------------------------------------------------------------------------- #
# Evaluación por bandas (ML-26)
# --------------------------------------------------------------------------- #

EVAL_IOU_MATCH: Final = 0.5
"""IoU mínimo para que una predicción empareje con una caja de verdad (AP50)."""

EVAL_SMALL_BOX_PX: Final = 8.0
"""Píxeles de ancho por debajo de los cuales una caja de verdad se empareja por
distancia de centros y no por IoU: a 8 px, un píxel de error ya hunde el IoU por
debajo de 0,5 aunque la detección sea buena (el balón lejano)."""

EVAL_SMALL_CENTER_PX: Final = 4.0
"""Distancia máxima entre centros, en píxeles, para emparejar una caja pequeña: medio
ancho de la mayor de ellas."""

EVAL_SCORE_THRESHOLD: Final = 0.5
"""Confianza a partir de la cual una predicción cuenta para el recall y la precisión.
El AP50 barre todas."""

EVAL_BOOTSTRAP_ROUNDS: Final = 1000
"""Remuestreos por partido para los intervalos de confianza."""

EVAL_CONFIDENCE: Final = 0.95
"""Nivel de los intervalos por bootstrap: percentiles 2,5 y 97,5."""

EVAL_BOOTSTRAP_SEED: Final = 2026
"""Semilla del bootstrap: el informe tiene que ser reproducible."""

# --------------------------------------------------------------------------- #
# Eventos aprendidos N3 (ML-48, ADR 0004)
# --------------------------------------------------------------------------- #

PITCH_LENGTH_M: Final = 105.0
"""Metros. El campo de referencia al que se normalizan las posiciones (el mismo valor por
defecto que `PITCH_LENGTH_M` de la referencia). Origen en el centro, x a lo largo e y a lo
ancho, como `PitchModel`."""

PITCH_WIDTH_M: Final = 68.0
"""Metros. El ancho del campo de referencia."""

EVENTS_HZ: Final = 7.5
"""Hz. La cadencia de la serie de N0 que ven N3 y N4 (ADR 0021 §1)."""

SKILLCORNER_HZ: Final = 10.0
"""Hz. La cadencia del tracking de SkillCorner."""

N3_GRID_W: Final = 16
"""Celdas de la rejilla de N3 a lo largo del campo (6,6 m cada una)."""

N3_GRID_H: Final = 10
"""Celdas de la rejilla de N3 a lo ancho del campo (6,8 m cada una)."""

N3_CHANNELS: Final = ("players", "vx", "vy", "goalkeeper", "referee", "ball", "whistle")
"""Canales de la rejilla de N3, en su orden: jugadores por celda, su velocidad media en
m/s (x e y), porteros, árbitros, el balón (1 en su celda) y el silbato de N2 (1 en toda la
rejilla mientras suena). En SkillCorner no hay árbitros ni audio: esos canales van a 0."""

N3_CLASSES: Final = ("corner", "throw_in", "goal_kick", "kickoff", "free_kick", "goal")
"""Clases de N3 en el orden de la salida; el índice 0 de la salida es «nada» y la clase k
va en k + 1 (ADR 0004 §1)."""

EVENT_LABEL_DILATION_STEPS: Final = 1
"""Pasos a cada lado del instante de un evento que llevan su etiqueta (±133 ms a 7,5 Hz),
como E2E-Spot (ADR 0004 §1)."""

EVENT_DEDUP_S: Final = 2.0
"""Segundos. Dos etiquetas de la misma clase más juntas que esto son el mismo evento (las
filas de SkillCorner repiten la reanudación en varias posesiones)."""

N3_WIDTH: Final = 48
"""Canales internos de la TCN de N3: múltiplo de 16 (ANE_CHANNEL_QUANTUM) y lo justo para
quedar por debajo de los 100.000 parámetros del ADR 0021 §1."""

N3_TEMPORAL_KERNEL: Final = 3
"""Ancho del kernel temporal de cada bloque de la TCN."""

N3_DILATIONS: Final = (1, 2, 4, 8, 16, 32)
"""Dilataciones de los bloques temporales: campo receptivo de 1 + 2·63 = 127 pasos, unos
17 s a 7,5 Hz (el contexto de 16 s del ADR 0004)."""

N3_MAX_PARAMS: Final = 100_000
"""Tope de parámetros de N3 (ADR 0021 §1): corre en CPU en el VPS junto a N1."""

N3_TOLERANCE_S: Final = 2.0
"""Segundos. Tolerancia de un acierto de N3 contra la etiqueta (ADR 0004 §4), y la ventana
de supresión de la decodificación por clase."""
