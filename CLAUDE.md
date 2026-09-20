# football-ai-training — instrucciones permanentes

Se carga en todas las sesiones de este repo. Es normativo, no informativo.

## 0. Qué es esto y qué no

El repo donde se entrena. Produce **un `.onnx` con su ficha** y nada más. El producto vive
en `fooball_ai_streaming` y la app de las cámaras en `fooball_ai_capturer`.

La fuente de verdad arquitectónica sigue siendo `docs/BLUEPRINT.md` del repo de detección,
y la frontera entre repos la fijan su ADR 0004 y su ADR 0013 §3. Cuando algo de aquí
cambie lo que aquel repo espera recibir, se anota **allí**, en un ADR, no aquí en silencio.

## 1. Reglas duras

**El entregable es el par `.onnx` + ficha.** Un export sin su bloque de `registry.yaml`
está a medias: el repo de detección lo rechaza y hace bien. La ficha lleva `sha256`, forma
de entrada, `T`, fps de muestreo, normalización, activación, orden de clases y layout de
salida, y los valores se **leen del fichero exportado**, nunca de lo que uno recuerde de
la configuración del entrenamiento.

**Nada de aquí se distribuye.** Este repo usa GPL y datos con NDA. Es legítimo para
entrenar y para medir; meter cualquiera de las dos cosas en el producto es una decisión
que exige mirar la licencia primero. Ver `docs/DEPENDENCIES.md`.

**Verificación numérica obligatoria antes de dar un export por bueno**: el mismo clip por
torch y por onnxruntime, con diferencia máxima < 1e-3. Un export que se degrada no da
error, solo detecta peor.

**División por partidos completos, nunca por clips sueltos.** Dos clips del mismo partido
en train y en validación son casi la misma imagen, y la métrica sale inflada.

**Toda dependencia nueva se anota en `docs/DEPENDENCIES.md` en el mismo commit**, con su
licencia. Aquí importa más que en el otro repo, no menos.

**Secretos por variable de entorno**: la contraseña del NDA de SoccerNet nunca en un
fichero, un log ni un commit.

## 2. Protocolo

- Una TASK = un commit. Conventional Commits: `feat(export): ...`.
- Ninguna tarea está hecha sin tests de lo que sea testeable sin GPU.
- Al cerrar una tarea se actualiza `docs/PROGRESS.md`: qué se hizo, qué quedó fuera,
  siguiente paso. Y si algo se **midió**, la cifra va ahí: es lo único que no se recuerda.
- Lo que no se pueda ejecutar en la máquina de turno, se implementa igual y se dice que no
  se ejecutó. No se reporta verde lo que no se ha corrido.

## 3. Entorno

- Gestor: `uv`. Nunca `pip install` suelto ni conda.
- `uv sync` deja las herramientas; `uv sync --group train` añade torch y el resto.
- El entrenamiento y las medidas van en una GPU alquilada (RunPod), no en el portátil.
- En Windows, `export UV_LINK_MODE=copy` antes de cualquier `uv`.

## 4. Comandos

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run python tools/fetch_tdeed.py        # clona T-DEED en third_party/ (GPL, no se vendoriza)
uv run python tools/fetch_soccernet.py    # etiquetas; los vídeos piden el NDA
```
