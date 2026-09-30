# football-ai-training

El repo donde se entrena. Produce **los modelos del sistema**, y por cada uno entrega cuatro
cosas: un `.onnx`, un `.mlpackage.zip` para el iPhone, su ficha y sus vectores dorados
([ADR 0002](docs/DECISIONS/0002-arquitectura-b-artefactos-y-entrenamiento.md)).

## Los tres repos

| Repo | Qué hace | Qué sale de él |
|---|---|---|
| `fooball_ai_capturer` | La app de los dos iPhone. Es el directo: captura, detecta en el Neural Engine con los modelos de aquí, compone el programa y lo emite | El programa por SRT al VPS |
| `fooball_ai_streaming` | La referencia en Python (`libs/vision`) con sus dorados, los servicios del VPS y el diferido en una GPU puntual | Clips, relé a las plataformas y el reprocesado en 4K |
| **`fooball_ai_training`** (este) | Datos, autoanotación, entrenamiento y export a ONNX y Core ML | Por modelo: `.onnx`, `.mlpackage.zip`, ficha y bundle dorado |

Es la arquitectura B: el directo lo hacen los dos móviles. La frontera la fijan el
[ADR 0019 §8](https://github.com/jhquihuiri7/fooball_ai_streaming/blob/migracion/dos-moviles/docs/DECISIONS/0019-arquitectura-b-dos-moviles.md),
que sustituye al alcance de su ADR 0004, y el §4 de su ADR 0020. **Lo que cruza son los
artefactos de cada modelo, cada uno por su sha256.** `torch` vive aquí y está prohibido allí,
donde lo verifica `.importlinter`.

## La ficha, que es el entregable de verdad

Un `.onnx` no dice cómo usarse. Su orden de clases, el color de entrada, la normalización
y qué significa cada salida no están en el fichero, y adivinarlos falla **en silencio**.
Ya pasó: `rfdetr-small.onnx` llegó en septiembre de 2026 con otra firma que la esperada y
con las clases en orden alfabético de Roboflow; el orden hubo que **medirlo** contando
áreas de caja sobre once frames. Con cinco clases de acción eso ya no se puede hacer: no
hay ninguna propiedad geométrica que distinga «gol» de «córner» en unos logits.

Por eso cada export produce, además del `.onnx`, el bloque YAML listo para pegar en
`models/registry.yaml` del repo de detección. Lleva `sha256`, forma de entrada, `T`, fps de
muestreo, normalización, activación, **orden de clases** y layout de salida.

La ficha v2 (ML-13) añade dos bloques:
- `artifacts.coreml`: la ruta, el sha256, la versión mínima de iOS y qué operaciones no caen en
  el ANE;
- `parity`: el bundle dorado y su tolerancia.

Con eso, la app comprueba en XCTest que el modelo y su postproceso en Swift dan lo mismo que
Python.

## Estado

Ver [docs/PROGRESS.md](docs/PROGRESS.md). T-DEED quedó exportado y verificado (T1, T5 y T6),
y se congela como línea base de medida: T3 y T4 se cancelan. Lo siguiente son las tareas ML y
SPK del plan de la arquitectura B, empezando por la infraestructura (ML-02 a ML-05).

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
permisiva. Es lo que fija el ADR 0002; lo demás es para medir.
