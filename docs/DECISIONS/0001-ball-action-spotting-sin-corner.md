# ADR 0001 — Se entrena sobre Ball Action Spotting, y el córner se queda fuera

- **Fecha:** 2026-09-20
- **Estado:** ACEPTADO (decidido por el propietario, 2026-09-20)
- **Tarea:** T2, y condiciona T3–T6
- **Afecta a:** el ADR 0013 §10 del repo de detección, que lista cuatro clases para el MVP

## Problema

El ADR 0013 del repo de detección fija cuatro clases para el MVP: `goal`, `shot`,
`corner`, `free_kick`. Al leer los datos de SoccerNet (T2) resultó que **no hay una sola
tarea que dé las cuatro**:

| | Ball Action Spotting (`spotting-ball-2024`) | Action Spotting v2 (`spotting-2023`) |
|---|---|---|
| Clases | 12, centradas en el toque del balón | 17, acciones de partido |
| Gol, tiro, tiro libre | sí | sí |
| **Córner** | **no** (tiene THROW IN y OUT en su lugar) | sí |
| Pesos publicados de T-DEED | **sí** | no |
| Acceso | Hugging Face | NDA con espera |

## Decisión

**Ball Action Spotting, y el córner se queda fuera del primer modelo.**

Dos razones, en este orden:

1. **Sin pesos publicados no hay línea base.** Lo primero que hay que responder no es
   «¿cuántas clases detectamos?» sino «¿esto funciona con nuestra cámara?». Esa pregunta
   se contesta corriendo un modelo ya entrenado sobre metraje propio, y solo la tarea del
   balón tiene uno. Empezar por la otra significaría entrenar desde cero para descubrir si
   merecía la pena entrenar.
2. **El córner es el menos valioso de los cuatro.** Una repetición existe porque alguien
   quiere volver a ver algo. Nadie pide ver un córner otra vez; se pide ver el gol, el
   tiro que se estrelló en el palo y la falta que lo provocó.

El córner no se descarta para siempre: vuelve cuando haya datos propios etiquetados, que
es adonde lleva el camino de todas formas —ni T-DEED ni SoccerNet pueden entrar en el
producto, los dos son GPL-3.0 y sus datos son de investigación—.

## Alternativas descartadas

**Las 17 clases, para tener el córner desde el principio.** Cuesta entrenar desde cero
antes de saber si el problema es abordable, y añade la espera del NDA a la ruta crítica.

**Las dos tareas, una para cada cosa.** Dos datasets, dos entrenamientos y dos modelos en
el panel, para ganar una clase que casi no se pide. No.

## Consecuencias

- Las clases del primer modelo son las de Ball Action Spotting, y son **doce**, no cuatro.
  Al repo de detección le llegan las doce en la ficha; que el panel solo marque algunas es
  decisión suya, no del modelo.
- El ADR 0013 §10 del repo de detección queda desactualizado en su lista de cuatro clases.
  No se enmienda todavía: se hará cuando llegue el primer `.onnx` y se sepa qué clases
  sobreviven con recall aceptable.
- `spotting-2023` y su NDA salen de la ruta crítica. La contraseña sigue soportada en
  `tools/fetch_soccernet.py` por si vuelve.
