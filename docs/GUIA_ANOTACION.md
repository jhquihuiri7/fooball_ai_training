# Guía de anotación (ML-21, ADR 0003)

Esta guía es NORMATIVA: un lote anotado con otro criterio se reanota, no se mezcla.
Se anota en CVAT, sobre el frame NATIVO tal cual se almacenó (aunque se vea girado:
la rotación la deshace `ftrain/bands.py`, no el anotador). El 5 % de cada lote se
anota por duplicado para medir el acuerdo.

## Reglas que valen para todo

- La caja se ajusta a LO VISIBLE, nunca a lo inferido.
- **Lo que queda fuera de la máscara del campo no se anota**: público, banquillo,
  fotógrafos y recogepelotas no existen para el dataset.
- La franja del código de tiempo (la tira de celdas de la esquina) no tapa nada que
  haya que anotar: si un objeto cae debajo, se anota igual — el pipeline lo marca y
  decide, no el anotador.
- En duda entre dos clases: se anota la más probable y se marca `occluded=true` si
  la duda viene de una oclusión. La duda sin oclusión va al canal de revisión.

## Jugadores (`goalkeeper`, `player`, `referee`)

- Caja de la persona entera visible, botas incluidas, sin el palo de la sombra.
- `goalkeeper`: el que está VESTIDO de portero, aunque ande fuera del área.
- `referee`: el trío arbitral completo (principal y asistentes).
- `occluded=true` cuando falta una parte del cuerpo tras otro jugador o el poste.
- `truncated=true` cuando el borde del frame corta la caja.
- **Ejemplo**: dos jugadores en un córner, uno tapa medio cuerpo del otro → dos
  cajas; la de atrás, `occluded=true` y ajustada a lo que se ve (no se «completa»).

## Balón (`ball`), según §40.4 del blueprint

- Caja ajustada al balón visible.
- **Parcialmente ocluido** (un pie encima, tras la red): caja de la parte visible,
  `occluded=true`.
  *Ejemplo: balón pisado — caja de la media luna visible, occluded=true.*
- **Con estela de desenfoque**: caja de la EXTENSIÓN COMPLETA de la estela,
  `blurred=true`.
  *Ejemplo: disparo cruzado a 30 fps — la caja cubre la estela entera, no un punto
  en su centro.*
- **Balón que no es EL balón** (gradas, calentamiento, otro campo): clase
  `distractor_ball`, nunca `ball`. Enseñar a distinguirlos vale más que ignorarlos.
  *Ejemplo: balón parado junto al banderín mientras se juega con otro →
  distractor_ball.*
- **Balón no visible**: no se anota nada; el frame vale como negativo.
- **Umbral de tamaño: 3 px.** Un balón de 3 px o más se anota; por debajo, el frame
  entero se marca `unusable=true` (no se adivina dónde está).
  *Ejemplo: balón altísimo contra el cielo, 2 px → frame unusable.*

## Pistas (tracks) en CVAT

- El balón se anota como PISTA: caja cada 5-10 frames y se corrige la
  interpolación, no frame a frame.
- Una pista se corta cuando el balón desaparece (fuera de cuadro, ocluido del todo)
  y se abre otra cuando reaparece: no se «puentea» a ciegas.

## Eventos del N0

- No se anotan en CVAT: vienen del mando (JSONL por `rig_ms`) y se revisan en su
  propia pasada (ML-23). Si al anotar se ve un gol sin marca, se apunta en la hoja
  del lote, no se inventa el evento.

## Qué entrega un lote

1. El export COCO extendido en nativo (`labels/<version>/instances.json`).
2. La lista de frames `unusable`.
3. La hoja del lote: anotador, fecha, dudas y el 5 % duplicado.
