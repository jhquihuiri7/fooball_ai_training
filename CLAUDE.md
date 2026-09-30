# football-ai-training — instrucciones permanentes

Se carga en todas las sesiones de este repo. Es normativo, no informativo.

## 0. Qué es esto y qué no

El repo donde se entrena. Produce **los modelos del sistema**, y por cada uno entrega cuatro
artefactos: `.onnx`, `.mlpackage.zip` en fp16, ficha v2 y bundle dorado (ADR 0002). El
directo vive en la app de los dos iPhone (`fooball_ai_capturer`), que ejecuta los modelos en
el Neural Engine. La referencia de Python, el VPS y el diferido viven en `fooball_ai_streaming`.

La fuente de verdad arquitectónica sigue siendo `docs/BLUEPRINT.md` del repo de detección,
con sus enmiendas. La frontera entre repos la fijan su ADR 0019 §8, que sustituye al alcance
de su ADR 0004, y su ADR 0020 §4. El backlog es su `docs/plan-dos-moviles/entrenamiento.md`:
las tarjetas ML y SPK-50+ son de este repo. Cuando algo de aquí cambie lo que aquel repo
espera recibir, se anota **allí**, en un ADR, no aquí en silencio.

## 1. Reglas duras

**El entregable son los cuatro artefactos de cada modelo.** Un export sin su bloque de
`registry.yaml` está a medias: el repo de detección lo rechaza y hace bien. La ficha lleva
`sha256`, forma de entrada, `T`, fps de muestreo, normalización, activación, orden de clases
y layout de salida. La v2 añade `artifacts.coreml` y `parity`. Los valores se **leen de los
ficheros exportados**, nunca de lo que uno recuerde de la configuración del entrenamiento.

**Lo que se entrega tiene que poder venderse.** Los modelos de aquí van dentro de la app, así
que su código, sus pesos de partida y sus datos necesitan licencias que admitan uso
comercial. T-DEED (GPL), SoccerNet (NDA), `rfdetr-small` (datos DFL) y `ultralytics` (AGPL)
solo sirven para medir o para preanotar, y nunca terminan en un artefacto entregado. Ver
`docs/DEPENDENCIES.md`.

**Verificación obligatoria antes de dar un export por bueno.** Un export que se degrada no
da error, solo detecta peor. Tiene dos partes:
- el mismo clip por torch y por onnxruntime, con diferencia máxima < 1e-3;
- en los modelos que van al móvil, además, la puerta fp16 de su tarea en Core ML: Δrecall
  ≤2 pp por banda de distancia, y centro ≤1 px en mediana en los jugadores o pico ≤1 px en el
  balón (ADR 0002). En el ANE ninguna capa puede quedarse en fp32: si la puerta no pasa, el
  export falla.

**División por partidos completos, nunca por clips sueltos.** Dos clips del mismo partido
en train y en validación son casi la misma imagen, y la métrica sale inflada.

**Toda dependencia nueva se anota en `docs/DEPENDENCIES.md` en el mismo commit**, con su
licencia. Aquí importa más que en el otro repo, no menos.

**Secretos por variable de entorno**: la contraseña del NDA de SoccerNet nunca en un
fichero, un log ni un commit.

## 2. Protocolo

- Una TASK = un commit. Conventional Commits: `feat(export): ...`, con el ID de la tarjeta
  (`(ML-09)`).
- Todo se commitea en la rama `migracion/dos-moviles`. `main` no recibe nada, tampoco al
  cerrar un hito.
- Ninguna tarea está hecha sin tests de lo que sea testeable sin GPU.
- Al cerrar una tarea se actualiza `docs/PROGRESS.md`: qué se hizo, qué quedó fuera,
  siguiente paso. Y si algo se **midió**, la cifra va ahí: es lo único que no se recuerda.
- Lo que no se pueda ejecutar en la máquina de turno, se implementa igual y se dice que no
  se ejecutó. No se reporta verde lo que no se ha corrido.

## 3. Entorno

- Gestor: `uv`. Nunca `pip install` suelto ni conda.
- `uv sync` deja las herramientas; `uv sync --group train` añade torch y el resto.
- Dónde va cada cosa (ADR 0002 §3):
  - **datos y artefactos** en GCS, con versiones inmutables y sha256 por fichero;
  - **entrenar y autoanotar** en RunPod;
  - **exports pesados en CPU** en Vertex AI (`tools/export_on_vm.sh`);
  - **Core ML** (predecir en fp16, paridad, dorados y banco en el iPhone) en un Mac con Apple
    Silicon;
  - **Colab** solo para cuadernos.
- Windows sirve para el código y los tests. `coremltools` no tiene ruedas para Windows: los
  tests que lo necesitan se saltan con motivo, y lo que pida el Mac se dice que no se ejecutó.
- En Windows, `export UV_LINK_MODE=copy` antes de cualquier `uv`.

## 4. Comandos

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Solo para la línea base de T-DEED, congelada y sin mantenimiento (ADR 0002):

```bash
uv run python tools/fetch_tdeed.py        # clona T-DEED en third_party/ (GPL, no se vendoriza)
uv run python tools/fetch_soccernet.py    # etiquetas; los vídeos piden el NDA
```

Los comandos de la arquitectura B (ingesta, extracción de frames, autoanotación,
`export_coreml`, `ane_lint`, paridad y dorados) se añaden aquí con la tarea que crea cada
herramienta.
