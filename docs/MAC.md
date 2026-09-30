# El Mac del entrenamiento

Core ML solo predice en macOS. Todo lo que tiene que ver con cómo corre un modelo en el
iPhone pasa por un Mac con Apple Silicon (ADR 0002 §3). Sin él no se pueden cerrar la puerta
fp16 ni los dorados, y el export de los jugadores y del balón se queda parado.

## Qué se hace allí

| Qué | Tarea |
|---|---|
| Predecir en fp16 con el ANE (`CPU_AND_NE`) y en la CPU | ML-05 (humo), ML-09 |
| Paridad torch ↔ ONNX ↔ Core ML | ML-11 |
| Las salidas de Core ML de los vectores dorados | ML-12 |
| El banco de modelos en el iPhone (XCTest, desde `football-ai-capture`) | SPK-50 y siguientes |

Convertir no necesita el Mac: `coremltools` convierte también en Linux. Entrenar tampoco,
porque eso va en RunPod.

## Qué Mac

- **El de referencia** es el de Alexander, con Xcode 26.6. Es el que firma e instala la app
  en los iPhone.
- **Se consigue un segundo Mac con Apple Silicon para el trabajo de ML**, para no depender de
  una sola máquina (decisión del plan, 2026-09-29).

## Requisitos

- Apple Silicon (M1 o posterior): el ANE del Mac es de la misma familia que el del iPhone.
- macOS 15 o posterior.
- Xcode 26, para el banco en el iPhone.
- `uv` y `git`.
- **El mismo commit que el resto.** Un dorado o una paridad medidos en otro commit no valen:
  el modelo, el preproceso o la tolerancia pueden haber cambiado.

## Puesta a punto

```bash
git clone https://github.com/jhquihuiri7/fooball_ai_training
cd fooball_ai_training
git switch migracion/dos-moviles
uv sync --group train --group apple
uv run python tools/mac_smoke.py
```

`mac_smoke` convierte una red de tres convs a mlprogram fp16 para iOS 18, predice con el ANE
y con la CPU, y compara con torch en fp32. **El Mac está listo cuando las dos diferencias
salen por debajo de 1e-2 y la herramienta termina con 0.** Anota las dos cifras en
`docs/PROGRESS.md`, junto con el modelo del Mac y las versiones de macOS y de `coremltools`.
