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
