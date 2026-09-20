# football-ai-training

El repo donde se entrena. Produce **un artefacto y nada más**: un `.onnx` con su ficha,
que se copia al repo de detección.

## Los tres repos

| Repo | Qué hace | Qué sale de él |
|---|---|---|
| `fooball_ai_capturer` | La app de los dos iPhone (EPIC A del ADR 0012) | SRT con código de tiempo |
| `fooball_ai_streaming` | Detección, panel, emisión y repetición | El programa al aire |
| **`fooball_ai_training`** (este) | Datasets y entrenamiento | Un `.onnx` + su entrada de `registry.yaml` |

La frontera la fijan el [ADR 0004](https://github.com/jhquihuiri7/fooball_ai_streaming/blob/main/docs/DECISIONS/0004-recorte-a-repo-de-deteccion.md)
y el ADR 0013 §3 del repo de detección: **lo único que cruza es un `.onnx` acompañado de
su ficha**. `torch` vive aquí y está prohibido allí, donde lo verifica `.importlinter`.

## La ficha, que es el entregable de verdad

Un `.onnx` no dice cómo usarse. Su orden de clases, el color de entrada, la normalización
y qué significa cada salida no están en el fichero, y adivinarlos falla **en silencio**.
Ya pasó: `rfdetr-small.onnx` llegó en septiembre de 2026 con otra firma que la esperada y
con las clases en orden alfabético de Roboflow; el orden hubo que **medirlo** contando
áreas de caja sobre once frames. Con cinco clases de acción eso ya no se puede hacer: no
hay ninguna propiedad geométrica que distinga «gol» de «córner» en unos logits.

Por eso cada export produce, además del `.onnx`, el bloque YAML listo para pegar en
`models/registry.yaml` del repo de detección, con: `sha256`, forma de entrada, `T`, fps de
muestreo, normalización, activación, **orden de clases** y layout de salida.

## Estado

Ver [docs/PROGRESS.md](docs/PROGRESS.md). Hoy: T1 y T2 preparadas, ninguna ejecutada de
punta a punta —hacen falta una GPU y el NDA de SoccerNet—.

## Entorno

```bash
uv sync                  # herramientas, sin torch
uv sync --group train    # + torch y el resto: varios GB, y quiere GPU
```

`torch` está en un grupo aparte a propósito: el repo se puede clonar, leer y pasar el
linter sin bajarse varios gigabytes.

## Aviso de licencias, y es importante

Este repo usa herramientas que **no se pueden meter en el producto**. Está detallado en
[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md); el resumen es:

- **T-DEED es GPL-3.0.** Correr un GPL para entrenar es libre. Lo que no está claro es si
  un `.onnx` exportado desde su definición de modelo es obra derivada, y esa duda se
  resuelve **antes** de vender, no después.
- **SoccerNet exige un NDA** y sus términos son de investigación. Sus pesos y sus datos
  sirven para medir una línea base, **no** para el producto.

El camino limpio para vender es reentrenar con datos propios una arquitectura de licencia
permisiva. Lo demás es para medir.
