# ADR 0004 — N3 y N4: el spotter distribuible y su formato de eventos

- **Fecha:** 2026-10-05
- **Estado:** PROPUESTO (lo acepta el propietario; es la aceptación de ML-47)
- **Tarea:** ML-47 del plan de migración
  (`docs/plan-dos-moviles/entrenamiento.md` del repo de detección, rama `migracion/dos-moviles`)
- **Parte de:** el ADR 0021 del repo de detección (aceptado el 2026-09-30), que ya fija los
  cinco niveles, las licencias, el vocabulario de las marcas (§4) y el instante de un evento
  (§5). Este ADR no cambia nada de aquello: baja N3 y N4 a algo que se pueda construir.
- **Afecta a:** ML-48 a ML-55 y SPK-53 de este repo; IOS-78 (N4 en el ANE) y EV-08 (la
  evaluación de N1, con la que N3 y N4 se comparan) del plan.

## Problema

El ADR 0021 decide que los eventos aprendidos van en dos niveles distribuibles:

- **N3**, una TCN causal sobre las posiciones de N0;
- **N4**, un spotter de píxeles en streaming en el ANE de cada móvil.

Los dos se reimplementan desde E2E-Spot (BSD-3), sin T-DEED. Sin fijar la entrada, las
clases, la salida y cómo se mide, ML-48 a ML-55 no pueden empezar. Tres cosas no están
decididas en ninguna parte:

1. qué ve exactamente cada modelo y a qué cadencia;
2. cómo una predicción por fotograma se convierte en un registro `mark` del N0 v1;
3. cómo se mide, para que la cifra se pueda poner al lado de N1 (EV-08) y de T-DEED.

## Opciones

### La entrada de N3

- **Rejillas de ocupación del campo (elegida).** El campo de 105×68 m en celdas fijas. Un
  canal cuenta los jugadores de cada celda y otro lleva el balón. Las rejillas no dependen
  del número de jugadores ni de su orden, ni de saber de qué equipo es cada uno: N0 v1 no
  trae equipo. SkillCorner opendata se convierte a esto mismo (ML-48), así que el preentreno
  y los datos propios hablan el mismo idioma.
- **Conjuntos de jugadores con atención.** Expresan más, pero pasan de 100.000 parámetros y
  necesitan el equipo para sacarle partido. Descartada para la v1.

### La salida

- **Clasificación por paso, con la etiqueta dilatada en el tiempo, como E2E-Spot
  (elegida).** Una clase por paso más «nada». La etiqueta de un evento cubre ±1 paso
  alrededor de su instante. En inferencia, un máximo local por clase con supresión en una
  ventana. Es lo más simple que se entrena bien con pocos eventos.
- **Regresión del desfase al evento.** Fecha mejor, pero añade una cabeza y su pérdida. Se
  deja para cuando la tolerancia de ±1 s de N4 no baste.

### Dónde se funden las dos cámaras en N4

- **En el maestro, después de los modelos (elegida).** Cada móvil corre su N4 sobre su
  cámara (IOS-78). El maestro se queda, por clase, con la confianza máxima de los dos
  dentro de la tolerancia, y escribe una sola marca. El registro `mark` no tiene `side`
  (`additionalProperties: false`), y no lo necesita.
- **Un modelo que vea las dos mitades.** Dobla la entrada y obliga a mover píxeles entre
  móviles. Descartada.

## Decisión

### 1. N3: eventos sobre posiciones

| Qué | Valor |
|---|---|
| Entrada | La serie de N0 a 7,5 Hz: `fused` (jugadores en metros) y `ball` |
| Representación | Rejilla de 12×8 celdas (8,75 × 8,5 m), 2 canales: jugadores por celda y el balón (1 en su celda, 0 si no hay) |
| Ventana | Causal, 16 s (120 pasos) de contexto efectivo |
| Modelo | TCN causal de convoluciones dilatadas, ≤100.000 parámetros (ADR 0021 §1) |
| Clases | corner, throw_in, goal_kick, kickoff, free_kick, goal, más «nada» |
| Etiqueta | La clase en el paso del evento y en ±1 paso (±133 ms) |
| Preentreno | SkillCorner opendata en rejillas (ML-48), pendiente de su licencia de datos (ADR 0021, punto abierto) |
| Afinado | Datos propios desde N0 (ML-49) con las marcas del operador corregidas (ML-57) |
| Despliegue | ONNX en CPU, en el VPS y en el diferido, junto a N1 |

### 2. N4: spotter de píxeles por cámara

| Qué | Valor |
|---|---|
| Entrada | Un recorte de 448×256 del área de juego de cada cámara, a 7,5 Hz (ADR 0021 §1) |
| Backbone | MobileNetV4-Conv-S (timm, Apache-2.0), destilado de DINOv2-S sobre metraje propio (ML-52) |
| Tiempo | Desplazamiento temporal (TSM) en los bloques intermedios y una GRU causal a la salida |
| Estado | Se lleva de paso a paso: como MLState o como entrada y salida explícitas, lo decide SPK-53 |
| Clases | shot, goal, más «nada» |
| Etiqueta | Como N3: la clase en el paso del evento y en ±1 paso |
| Despliegue | Core ML en el ANE de cada móvil (IOS-78), con su export y sus dorados (ML-55) |

### 3. De la predicción a la marca

- **Decodificación**, por clase y en streaming:
  - se emite una marca en el máximo local de la probabilidad que pase el umbral de la clase;
  - después, la clase calla durante la tolerancia de su nivel (supresión).
  - Los umbrales salen de la medida (§4) y viajan en la ficha del modelo, no en el código.
- **El registro** es un `mark` del N0 v1 de EV-02, sin campos nuevos:

  ```json
  {"type": "mark", "rig_ms": 61050, "kind": "corner", "source": "positions",
   "by": "n3-positions@0.1.0", "confidence": 0.81}
  ```

  - `kind`: una de las clases del nivel, todas de `MARK_KINDS`.
  - `source`: `positions` para N3 y `spotter` para N4 (ADR 0021 §4). La tarjeta de ML-47
    habla aún de `whistle` y de `side`. Manda el ADR 0021: un silbato no es una marca, y
    la cámara no va en el registro.
  - `by`: `n3-positions@<versión>` o `n4-spotter@<versión>`, la versión de la ficha del
    modelo.
  - `confidence`: la probabilidad del máximo, de 0 a 1.
  - `rig_ms`: el del último paso de la ventana en N3 y el del fotograma en N4 (ADR 0021 §5).
- **N4 con dos cámaras:** el maestro funde las marcas de los dos móviles. Si las dos dan la
  misma clase dentro de la tolerancia, sale una sola marca, con el `rig_ms` y la confianza
  de la más segura.
- **Son sugerencias** (ADR 0021 §1, ADR 0013 §1). Una marca de N3 o N4 no saca nada al aire
  sin el operador.

### 4. Cómo se mide

- **Siempre por partido completo**, nunca por clips. La división de entrenamiento y
  validación es por partidos (regla dura del repo).
- **Un acierto** es una marca de la clase dentro de la tolerancia de una etiqueta humana,
  con emparejamiento uno a uno:
  - N3: ±2 s;
  - N4: ±1 s.
- **Por clase:** precisión, recall y AP, más la mediana del error de fecha de los aciertos.
- **Contra qué:**
  - N1 (EV-08) sobre los mismos partidos;
  - T-DEED, que es la línea base congelada de la tabla de PROGRESS. Solo se usa para medir:
    no entra en ningún artefacto.
- **Para automatizar una clase** sigue sin haber umbral: se decide con la medida delante
  (ADR 0021 §6).

### 5. Licencias

- **E2E-Spot (BSD-3):** referencia de diseño, como dice ADR 0021 §2. Se lee el paper y el
  repo `jhong93/spot`, y se reimplementa en `ftrain/events/`. No se vendoriza ni se copia
  código; si se copiara una función, iría con su aviso BSD-3. La guardia de licencias
  (ML-02) sigue prohibiendo cualquier línea de T-DEED.
- **MobileNetV4 de timm:** cada checkpoint lleva su licencia; antes de usar uno se
  comprueba que sea Apache-2.0 y se anota en `DEPENDENCIES.md`.
- **DINOv2-S:** Apache-2.0, solo como maestro de la destilación; no va en el artefacto.

## Consecuencias

- ML-48 (SkillCorner a rejillas) y ML-51 (el spotter en streaming) pueden empezar con este
  ADR aceptado.
- ML-49 y ML-53 necesitan partidos propios, así que esperan a la campaña (ML-17 y ML-58).
- El formato no toca el N0 v1: lo que escriben N3 y N4 lo valida ya
  `libs/vision/match_log.py` del repo de detección.
- La fusión de N4 entre cámaras es lógica del maestro: va en IOS-78, no en el modelo.

## Puntos abiertos

| Punto | Recomendación | Lo cierra |
|---|---|---|
| La licencia de los datos de SkillCorner | Como el ADR 0021: si no se aclara, N3 se entrena solo con datos propios y el preentreno queda para medir | ML-48, propietario |
| 12×8 celdas y 16 s de contexto | Son el punto de partida; se barren en ML-50 con los datos propios delante | ML-50 |
| Los umbrales por clase | Del punto de operación que elija el propietario sobre la curva P/R | ML-50, ML-54 |

## Aceptación

El propietario acepta el ADR. El formato de §3 se comprueba con el validador de N0 v1 de
EV-02 en el primer test de ML-48 o ML-51.
