# PROGRESS

Estado de las tareas. Se actualiza al cerrar cada una: qué se hizo, qué quedó fuera,
siguiente paso. **Lo que se mide va aquí**: es lo único que luego no se recuerda.

Leyenda: ✅ hecha · 🚧 en curso · ⛔ bloqueada · ⬜ pendiente

## Tablero

| TASK | Qué | Estado |
|---|---|---|
| **T0** | Repo, entorno y contrato con el repo de detección | ✅ |
| **T1** | T-DEED clonado, su firma leída y la línea base corrida | ✅ · ⬜ falta contrastar los instantes contra el vídeo |
| **T2** | Datos de SoccerNet | ✅ herramienta · ✅ tarea elegida ([ADR 0001](DECISIONS/0001-ball-action-spotting-sin-corner.md)) · ⬜ descarga |
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

### La línea base: `notebooks/t1_baseline.ipynb` (2026-09-20)

**No se ha ejecutado**: se escribió sin GPU delante. Va a Colab, que es donde el
propietario entrena; RunPod queda solo para el directo.

**No necesita los 19 GB.** `inference.py` corre sobre **un vídeo**, así que la línea base
se hace con metraje propio, que además contesta mejor la pregunta: no «¿reproduce el
paper?» sino «¿ve algo en nuestra cámara?».

**Tres trampas del código de T-DEED**, encontradas leyéndolo y ya resueltas en el notebook:

1. **Los pesos van en `checkpoints/SoccerNetBall/SoccerNetBall_challenge1/checkpoint_best.pt`.**
   `inference.py` arma esa ruta con el prefijo del nombre del modelo. Ponerlos donde uno
   los pondría mirando la carpeta —un nivel más arriba— falla a mitad de la carga.
   `tools/fetch_tdeed.py` decía la ruta equivocada; corregido.
2. **El vídeo tiene que ir a 25 fps.** `ActionSpotInferenceDataset` lee con OpenCV y se
   queda con uno de cada dos frames **nativos**: no remuestrea nada. Con un vídeo a 30
   fps el clip le llega a 15 y el modelo ve la jugada acelerada respecto a todo lo que
   aprendió. No da error, solo acierta menos.
3. **No instalar su `requirements.txt`.** Pinea `torch==2.3.1` y `numpy==1.26.4` y pelea
   con Colab. Bastan `timm`, `tabulate`, `wandb` —que `inference.py` importa y no usa— y
   `SoccerNet`, que **no está en su `requirements.txt`** y hace falta igual (ver abajo).

**Primera ejecución, 2026-09-20**: muere en el import, `ModuleNotFoundError: No module
named 'SoccerNet'` desde `util/eval.py:13`. Es un fallo de su `requirements.txt`, que no
lista el paquete aunque `util/eval.py` lo importe arriba del todo y `inference.py` importe
ese módulo. Se instala con `--no-deps`: el paquete arrastra `boto3`, `scikit-video` y
`pycocoevalcap`, y de todo él solo se usan `average_mAP` y `LoadJsonFromZip`, que se
importan con numpy y tqdm y nada más —comprobado en el entorno local—. Ya estaba en
`docs/DEPENDENCIES.md` (MIT el paquete, otra cosa los datos), así que no cambia la tabla.

La salida cae en `inference_output/results_inference.json` con el frame nativo, la clase y
la confianza.

**Almacenamiento**: el propietario servirá los datos desde Google Cloud Storage de un
proyecto suyo, así que el tope de 15 GB de Drive deja de ser un problema para T4.

---

## 2026-09-20 · T1 — la línea base, corrida · ✅

**Ejecutada por el propietario en Colab** con los pesos publicados de
`SoccerNetBall_challenge1` sobre `videoGP.MP4` (11 min 50 s, 1280×720, 29,97 fps
convertidos a 25). Umbral 0.2.

### Lo que salió

**292 eventos** en 11:50. Por clase: PASS 97, DRIVE 63, HIGH PASS 47, OUT 30, SHOT 22,
BALL PLAYER BLOCK 14, GOAL 10, THROW IN 4, CROSS 3, HEADER 2. **FREE KICK: ninguno**, y
PLAYER SUCCESSFUL TACKLE tampoco.

Tras aplicar el NMS temporal de 2 s que usa el spotter, las 10 detecciones de GOAL se
quedan en **6 momentos** y las 22 de SHOT en **19**:

| GOAL | confianza |
|---|---|
| 02:05 | 0.599 |
| 03:24 | 0.581 |
| 04:24 | **0.962** |
| 05:28 | **0.888** |
| 06:00 | 0.255 |
| 07:57 | **0.888** |

### El hallazgo, y es el bueno

**Los 10 GOAL, sin una sola excepción, van precedidos de un SHOT entre 0,24 y 2,16 s
antes.** Eso no lo produce un modelo disparando al azar: es la estructura de una jugada
real, tiro y después gol, y aparece las diez veces.

De ahí sale una regla de fusión que **no hay que inventarse, ya está medida**: el par
«SHOT seguido de GOAL en menos de ~2,5 s» es mucho más fuerte que un GOAL suelto. Es
gratis de implementar y es justo lo que el documento de origen pedía para reducir falsos
positivos.

### Lo que esto valida del repo de detección

Las dos constantes que se eligieron a ciegas resultan ser las correctas:

- `SPOTTER_NMS_S = 2.0` colapsa exactamente las crestas de este modelo: los pares de GOAL
  separados 0,88–1,12 s se funden en uno, y los momentos distintos (a más de 50 s) se
  conservan. Sin él saldrían 10 clips donde hay 6 jugadas.
- El error temporal está **por debajo de 2 s**, y el clip se corta con granularidad de
  2 s sobre un pre-roll de 20 y un post-roll de 10. O sea que la imprecisión del modelo es
  irrelevante frente al tamaño del clip: cae dentro del margen por diseño.

### Lo que **no** se sabe todavía

**Si esos 6 momentos son goles de verdad.** Seis goles en doce minutos es mucho para
fútbol corrido; si `videoGP.MP4` es un resumen, cuadra, y si es juego continuo, parte de
esos GOAL son ocasiones que el modelo llama gol. Sin contrastarlo no hay precisión
medida, solo una estructura coherente. **Es lo primero que hay que cerrar.**

Tampoco se sabe nada de FREE KICK: cero detecciones puede ser que no hubo ninguno o que
la clase no dispara.

**Veredicto**: suficiente para seguir. La estructura es coherente, la localización
temporal es buena y el ruido se concentra en clases que no usamos (PASS, DRIVE y HIGH
PASS son 207 de los 292). Se pasa a T5.

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

**Decidido el 2026-09-20**: Ball Action Spotting, y el córner fuera del primer modelo.
Está razonado en el [ADR 0001](DECISIONS/0001-ball-action-spotting-sin-corner.md). Lo que
manda es que solo esa tarea tiene pesos publicados, y sin pesos no hay línea base: la
primera pregunta no es cuántas clases se detectan sino si esto funciona con nuestra
cámara.

**Siguiente paso**: la línea base. No necesita los 19 GB —`inference.py` corre sobre **un
vídeo**— así que se puede hacer con metraje propio en cuanto haya pesos y una GPU. La
descarga de SoccerNet solo hace falta para T4 y para medir contra su ground truth.
