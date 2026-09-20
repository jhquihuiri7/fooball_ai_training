# DEPENDENCIES

Qué usa este repo y con qué licencia. Aquí la tabla importa **más** que en el repo de
detección, no menos: este es el sitio donde entran las cosas que no se pueden vender.

**Política.** Igual que en el repo de detección: permisivas (MIT, BSD, Apache-2.0, ISC,
PSF, MPL-2.0) entran sin más; AGPL y GPL enlazado, no. La diferencia es que aquí hay una
tercera categoría: **herramientas que se ejecutan pero no se distribuyen**, que sí se
admiten y van con su aviso.

## `core` — herramientas, siempre instalado

| Paquete | Licencia | Notas |
|---|---|---|
| numpy | BSD-3 | |
| pyyaml | MIT | Escribir la ficha del modelo |
| typer | MIT | CLI de las herramientas |
| onnx | Apache-2.0 | Leer y comprobar el `.onnx` exportado |
| onnxruntime | MIT | La mitad onnx de la verificación numérica |

## `train` — opt-in, varios GB, quiere GPU

| Paquete | Licencia | Notas |
|---|---|---|
| torch, torchvision | BSD-3 | Solo aquí. En el repo de detección está prohibido y lo verifica `.importlinter` |
| onnxscript | MIT | Desde torch 2.6, `torch.onnx.export` lo importa siempre, use o no el exportador nuevo. Sin él, exportar muere con un `ModuleNotFoundError` |
| SoccerNet | MIT (el paquete) | El **paquete** es MIT; los **datos** que descarga, no: ver abajo |
| timm | Apache-2.0 | Backbones preentrenados. Cada checkpoint tiene su propia licencia, que hay que mirar una por una |
| opencv-python-headless | Apache-2.0 | |
| tensorboard | Apache-2.0 | |

## Herramientas externas — se ejecutan, no se distribuyen

| Herramienta | Licencia | Para qué |
|---|---|---|
| **T-DEED** | **GPL-3.0** | Action spotting. Se clona en `third_party/`, que está en `.gitignore`: **no se vendoriza ni se redistribuye** |
| FFmpeg (CLI) | GPL (con x264) | Extraer frames y recortar clips. Proceso aparte |

## Las dos cosas que hay que resolver antes de vender

### 1. T-DEED es GPL-3.0

Ejecutar un programa GPL es libre y no contagia nada: entrenar y medir con T-DEED no tiene
ningún problema, igual que el repo de detección usa OBS (GPLv2) y FFmpeg sin que eso afecte
a su código.

Lo que **no está resuelto** es si un `.onnx` exportado desde la definición de modelo de
T-DEED es obra derivada de ese código. Los pesos en sí difícilmente lo son —salen de los
datos—, pero el grafo exportado se genera a partir de las clases del modelo. Es territorio
discutido, y este proyecto se quiere vender.

**Cómo se trabaja mientras tanto**: T-DEED sirve para medir la línea base y para saber si
el problema es abordable. Si la respuesta es que sí, el camino limpio es una arquitectura
de licencia permisiva —X3D de PyTorchVideo (Apache-2.0), VideoMAE de Hugging Face
(Apache-2.0 o CC-BY-NC según el checkpoint: **hay que mirarlo**)— entrenada con datos
propios. No es un rodeo: es lo mismo que ya dice el roadmap del blueprint sobre usar datos
propios por la restricción de licencia.

### 2. SoccerNet exige un NDA y sus datos no son para vender

Los vídeos se descargan tras firmar un formulario NDA, que da una contraseña por correo.
Las anotaciones se bajan con el paquete `SoccerNet`. Los términos están en el propio NDA y
son de investigación: **nada de lo que salga de ahí entra en el producto**, ni los datos ni
un modelo entrenado sobre ellos.

Sirven para una cosa concreta y valiosa: saber cuánto recall se puede esperar antes de
gastar un céntimo en etiquetar partidos propios.

La contraseña va por variable de entorno (`SOCCERNET_PASSWORD`), nunca en un fichero.
