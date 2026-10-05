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
