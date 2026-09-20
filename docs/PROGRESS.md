# PROGRESS

Estado de las tareas. Se actualiza al cerrar cada una: qué se hizo, qué quedó fuera,
siguiente paso. **Lo que se mide va aquí**: es lo único que luego no se recuerda.

Leyenda: ✅ hecha · 🚧 en curso · ⛔ bloqueada · ⬜ pendiente

## Tablero

| TASK | Qué | Estado |
|---|---|---|
| **T0** | Repo, entorno y contrato con el repo de detección | ✅ |
| **T1** | T-DEED clonado y su firma leída | ✅ código · ⛔ inferencia: faltan pesos y GPU |
| **T2** | Datos de SoccerNet | ✅ herramienta · ⬜ descarga: 19 GB y cuenta de Hugging Face |
| T3 | Reducir a las clases que interesan | ⬜ |
| T4 | Fine-tuning, con división por partidos completos | ⬜ |
| T5 | Export a ONNX con shapes estáticas **y su ficha** | ⬜ |
| T6 | Verificación numérica torch vs onnxruntime (< 1e-3) | ⬜ |

---

## 2026-09-19 · T0 y T1 — el repo y la firma de T-DEED · ✅

**Por qué existe este repo.** El ADR 0004 del repo de detección sacó el entrenamiento de
allí en septiembre de 2026 y desde entonces «el repo de entrenamiento» aparecía en cinco
sitios de la documentación sin existir. Mientras tanto llegó un `rfdetr-small.onnx` de
algún sitio —hay un `best.pt` al lado— **sin código versionado de cómo se hizo**: hoy no se
puede reproducir ni reexportar a otra resolución, que es justo lo que hará falta para
medir el riesgo R-P7-1.

### La firma de T-DEED, leída de su código

`uv run python tools/fetch_tdeed.py` clona el repo y **lee** de sus ficheros lo que define
la forma del clip. No se copia a mano ninguno de estos números: si T-DEED los cambia, la
herramienta falla en vez de quedarse con una copia vieja.

Medido en el commit `d953b4d`, configuración `SoccerNetBall_challenge1`:

| | Valor | De dónde sale |
|---|---|---|
| Clip | **100 frames** | `clip_len` de la config |
| Ritmo | **12,5 fps** | `STRIDE_SNB = 2` sobre frames extraídos a 25 |
| Ventana | **8 s** | los dos anteriores |
| Frame | **796 × 448** | `TARGET_HEIGHT/WIDTH` de `extract_frames_snb.py` |
| Backbone | `rny002_gsf` | config |
| Temporal | `ed_sgp_mixer` | config |
| Clases | **12** | `EVENT_DICTIONARY` de `util/eval.py` |

Orden de clases (el orden **es** el índice de salida): `PASS, DRIVE, HEADER, HIGH PASS,
OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL`.

### Tres cosas que salieron de mirarlo, y cambian el plan

1. **No hay córner.** Las cuatro clases del MVP son gol, tiro, córner y tiro libre. Ball
   Action Spotting tiene las tres primeras y **no** el córner: en su lugar tiene THROW IN
   y OUT. El córner está en la otra tarea, la de 17 clases. Hay que elegir, y no es obvio:
   la tarea con córner no tiene pesos publicados de T-DEED para SoccerNetBall.
2. **La ventana es de 8 s, no de 2.** Eso son 100 × 448 × 796 × 3 = **107 MB** de ventana
   viva en el panel, no los 13 MB que estimé al escribir la E6 con números de ejemplo. El
   spotter la deriva de la ficha, así que no hay que tocar código; pero el número real
   conviene tenerlo escrito antes de medir en el pod.
3. **El backbone es `rny002_gsf`**, o sea gate-shift. Es exactamente la parte que avisé
   que puede degradarse al exportar a ONNX. La verificación numérica de T6 no es un
   trámite: es donde se sabrá si este camino sirve.

### Lo que **no** se hizo, y por qué

**La inferencia de línea base no se ha ejecutado.** Hacen falta dos cosas que no están
aquí: los pesos, que están en un Google Drive y se bajan a mano, y una GPU. La herramienta
dice dónde están y dónde ponerlos. Correrla es lo primero que hay que hacer en el pod.

---

## 2026-09-19 · T2 — datos de SoccerNet · ✅ la herramienta, ⬜ la descarga

`uv run python tools/fetch_soccernet.py --check` dice lo que va a costar antes de costarlo.

**El acceso ha cambiado desde lo que dice la documentación vieja.** `spotting-ball-2024`
ya **no** se baja con la contraseña del NDA: viene de Hugging Face
(`SoccerNet/SN-BAS-2024`, ~19,3 GB) y el acceso lo controlan los permisos del repositorio.
El paquete `SoccerNet` avisa por su cuenta de que la contraseña se ignora para esa tarea.
La tarea de 17 clases sí sigue pidiendo el NDA.

**No se ha descargado nada**: son 19 GB y hace falta una cuenta de Hugging Face que acepte
los términos del dataset. Es una acción con nombre y apellidos, no algo que deba hacer una
herramienta por su cuenta.

**Y hay un aviso de licencia**: ese dataset se declara **GPL-3.0**. Para unos datos es
raro, y hay que mirarlo con calma antes de que nada entrenado sobre ellos entre en un
producto que se vende. Está anotado en `docs/DEPENDENCIES.md` junto al otro problema, el
de que T-DEED es GPL-3.0 también.

**Siguiente paso**: decidir entre las dos tareas —con córner y sin pesos, o con pesos y
sin córner— y, en paralelo, conseguir el acceso de Hugging Face. Después T3.
