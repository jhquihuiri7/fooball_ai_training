# ADR 0003 — Anotación: CVAT, formato canónico y guía

- **Fecha:** 2026-10-03
- **Estado:** PROPUESTO (lo acepta el propietario; es la aceptación de ML-21)
- **Tarea:** ML-21 del plan de migración
  (`docs/plan-dos-moviles/entrenamiento.md` del repo de detección, rama `migracion/dos-moviles`)
- **Afecta a:** `docs/GUIA_ANOTACION.md` (nueva), y aguas abajo ML-08 (ingesta), ML-20
  (extracción), ML-22/ML-23 (anotación y revisión) y ML-29 (autoanotación).

## Problema

La campaña (H2) va a producir 6-10 partidos de 4K que hay que anotar: jugadores en la
franja, el balón a resolución nativa y los eventos del N0. Sin decidir herramienta,
formato y reglas ANTES del primer lote, cada anotador inventa su criterio, los lotes no
se pueden mezclar y la doble anotación no mide nada. Y el balón es el caso duro: a 40 m
son 8 px, con estelas de desenfoque y balones de calentamiento en el fondo — justo lo
que una guía ambigua rompe.

## Opciones

### Herramienta

- **CVAT (MIT, autohospedado)** — interpolación de PISTAS nativa: se anota el balón
  cada N frames y la herramienta interpola la trayectoria, que el anotador solo
  corrige. Es exactamente el flujo de la revisión de trayectorias del balón, y lo que
  hace viable anotar a 30 fps sin morir. Formatos de export maduros (COCO incluido).
  Se hospeda en Docker en una máquina propia: los frames de la campaña no salen a un
  SaaS de terceros.
- **Label Studio (Apache-2.0)** — más cómodo para tareas mixtas (texto, audio), pero
  su vídeo no tiene la interpolación de pistas al nivel de CVAT, y ese es el 80 % del
  coste de anotar el balón.

### Formato canónico

- **COCO extendido, en coordenadas NATIVAS (3840×2160, frame tal cual se almacenó)** —
  COCO porque todo lo lee (las recetas DEIM/D-FINE entrenan de COCO directamente);
  extendido con `attributes` por anotación (lo que COCO puro no lleva); y NATIVO
  porque el lienzo 1920×576 y el mosaico CAMBIAN con la calibración de cada partido:
  una etiqueta anotada en el lienzo caduca con su band.json, una nativa no. El paso
  nativo→lienzo ya existe y está probado (`ftrain/bands.py`, ML-19), y marca —no
  pierde— lo que cae en la franja del código o fuera de la banda.
- YOLO txt (descartado: sin atributos, sin pistas), CVAT XML propio (descartado como
  canónico: útil como intermedio de la herramienta, pero el dataset se congela en COCO).

## Decisión

1. **CVAT autohospedado** para cajas y pistas. Versión fijada en el compose del
   despliegue (ML-22); MIT, sin datos fuera de casa.
2. **Formato canónico: COCO extendido en coordenadas nativas.**
   - Un `instances.json` por lote (`labels/<version>/`), con `images` apuntando a
     `frames/<version>/` por ruta relativa y `sha256` por imagen (la política de
     `docs/DATOS.md`).
   - Clases de detección: las de `ftrain/constants.py` — `MASTER_CLASSES =
     (ball, distractor_ball, goalkeeper, player, referee)`. Los modelos entregados
     derivan de ahí (los jugadores entrenan con `PLAYER_CLASSES`; el balón con
     `ball` + los negativos de `distractor_ball`).
   - `attributes` por anotación: `occluded`, `truncated`, `blurred` (balón) y
     `unusable` a nivel de imagen. Booleanos, siempre presentes (no opcionales), para
     que un export sin atributos cante en la validación.
3. **El balón es caja + atributos, nunca un punto.** El heatmap del ADR 0020 se
   entrena desde el CENTRO de la caja; guardar la caja conserva el tamaño real por
   banda de distancia, que es lo que piden las métricas (`DISTANCE_BANDS_M`).
4. **Los eventos del N0 van en JSONL aparte** (el match-log v1 de EV-02 cuando
   exista; hasta entonces, el JSONL de marcas del mando tal cual), emparejados por
   `rig_ms`. CVAT no los toca: no son cajas.
5. **Doble anotación del 5 %** de los frames de cada lote, repartida por bandas de
   distancia, para medir el acuerdo entre anotadores antes de creerse una métrica.

## Consecuencias

- ML-22 despliega CVAT con esta configuración y ML-23 usa la guía de abajo tal cual.
- El export de CVAT a COCO extendido pasa por un conversor nuestro (parte de ML-20)
  que valida clases, atributos y tamaños mínimos contra esta decisión.
- La guía de anotación (`docs/GUIA_ANOTACION.md`) es NORMATIVA: un lote anotado con
  otro criterio se reanota, no se mezcla.
- La revisión del propietario de este ADR cierra ML-21.
