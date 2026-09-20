# PROGRESS

Estado de las tareas. Se actualiza al cerrar cada una: qué se hizo, qué quedó fuera,
siguiente paso. **Lo que se mide va aquí**: es lo único que luego no se recuerda.

Leyenda: ✅ hecha · 🚧 en curso · ⛔ bloqueada · ⬜ pendiente

## Tablero

| TASK | Qué | Estado |
|---|---|---|
| **T0** | Repo, entorno y contrato con el repo de detección | ✅ |
| **T1** | T-DEED clonado, su firma leída y la línea base corrida y contrastada | ✅ |
| **T2** | Datos de SoccerNet | ✅ herramienta · ✅ tarea elegida ([ADR 0001](DECISIONS/0001-ball-action-spotting-sin-corner.md)) · ⬜ descarga |
| T3 | Reducir a las clases que interesan | ⬜ |
| T4 | Fine-tuning, con división por partidos completos | ⬜ |
| T5 | Export a ONNX con shapes estáticas **y su ficha** | ✅ escrito · ⬜ sin correr (falta Colab) |
| T6 | Verificación numérica torch vs onnxruntime (< 1e-3) | ✅ escrita dentro de T5 · ⬜ sin correr |

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

## 2026-09-20 · El gigabyte de ceros: causa y arreglo · 🚧

Segunda vuelta del export, con datos de verdad. Tres cosas quedaron claras:

**1. El exportador de `torch.export` no puede con este modelo, y se sabe por qué.** Falla
en `standarize`, dentro de `torchvision.Normalize`:

    if (std == 0).any():

Es una rama que depende del **valor** de un tensor. `torch.export` la convierte en un
símbolo sin respaldo (`Sym(Eq(u0, 1))`) y se rinde. Pasa a ser `legacy` por defecto: ya no
es una preferencia, es que el otro no puede.

**2. `onnxsim` empeora el problema.** Sobre este grafo lo dejó en **2,4 GB**, pasó del
límite de 2 GB de protobuf, lo guardó como datos externos y el resultado ni se pudo
parsear. Pasa a ser opt-in con esa advertencia escrita; §42.1 lo prescribe en general, y
en general está bien, pero aquí no.

**3. El gigabyte de ceros se quita en el modelo, no en el exportador.** Su origen:

    y = torch.zeros_like(x)          # <- esto se graba entero al trazar
    y[:, :fold] = self.gs(x[:, :fold])
    y[:, fold:] = x[:, fold:]

Escrito con un `Concat` es la misma operación y no materializa nada:

    y = cat([gs(x[:, :fold]), x[:, fold:]], dim=1)

`patch_gated_shift` reescribe el `forward` de las once capas antes de exportar.

**Y que sea equivalente no se da por hecho.** La referencia de la verificación numérica se
calcula ahora **antes** de parchear, así que T6 comprueba dos cosas de una vez: que el
export es fiel y que el `Concat` hace lo mismo que el `zeros_like`. Si el parche estuviera
mal, saltaría ahí.

**Sin ejecutar todavía.**

## 2026-09-20 · El export salió, y salió inservible · ⚠️ arreglado con `onnxsim`

El export terminó en la máquina de Vertex AI (`n1-highmem-8`, CPU) y **la verificación
numérica pasó**: el `.onnx` es fiel a torch. Pero el fichero salió de **1,03 GB** para un
modelo de 12,3 M de parámetros, y al abrirlo en el repo de detección:

    onnxruntime ... RUNTIME_EXCEPTION : Exception during initialization: bad allocation

**Dónde estaba el gigabyte.** Los pesos de verdad son 49 MB en 534 constantes. Los otros
**978 MB están en 1641 constantes dentro de nodos**, con formas de activación:
`[100, 152, 28, 50]`, `[100, 56, 56, 100]`… todas dentro de las capas `conv1` de RegNet,
o sea las que llevan gate-shift.

Es el `torch.zeros_like(x)` de `GatedShift.forward`. Al trazar, esa llamada se graba como
**un tensor de ceros del tamaño completo de la activación**. Once capas gate-shift y clips
de 100 frames: un gigabyte de ceros escritos en el fichero.

**El arreglo estaba en el blueprint y me lo salté.** §42.1 pone `onnxsim` entre el export
y la verificación numérica. No es cosmético: es lo que dobla esos ceros y deja el grafo
utilizable. Ahora `export_onnx.py` lo hace en un paso aparte, después de escribir el
fichero y antes de verificar.

**Aparte y no dentro del export** (`do_constant_folding`) a propósito: así su pico de
memoria no se suma al del trazado. Ese pico junto es exactamente lo que no cabía en Colab.

**Queda por medir**: cuánto baja el fichero y cuánta memoria y tiempo pide una inferencia.
Eso último importa para el pod, que tiene que correr esto al lado del panel.

## 2026-09-20 · Dónde se puede exportar, y por qué no en Colab · ⚠️

El export de T5 **no cabe en Colab**, ni en CPU ni con una T4. Está medido:

| Dónde | Qué pasa |
|---|---|
| Colab CPU (~12 GB de RAM) | SIGKILL: el vigilante de memoria mata el proceso |
| Colab T4 (14,56 GB de VRAM) | `torch.OutOfMemoryError` con 14,27 GB ya ocupados |

**Por qué**: el modelo mete los 100 frames de 448×796 en el backbone como **un solo
lote**, y el trazador clásico mantiene vivas todas las activaciones intermedias mientras
construye el grafo. Eso son ~15 GB, y no se puede reducir el clip: el codificador
posicional del modelo está fijado a 100 frames.

**Lo que hace falta no es GPU, es memoria.** Trazar es una pasada hacia delante; en CPU
sale exactamente el mismo grafo. Eso tiene dos consecuencias prácticas buenas: no hace
falta pelearse con la cuota de GPU de GCP —que es un formulario y una espera— y una
máquina de alta memoria cuesta una fracción de lo que cuesta una con acelerador.

**La vía**: `tools/export_on_vm.sh` en una máquina de Vertex AI de alta memoria
(`n1-highmem-8`, 52 GB, unos 0,47 $/h para un trabajo de minutos). Instala torch de CPU
—200 MB en vez de 2,5 GB—, clona los dos repos, baja los pesos de GCS, exporta, verifica
y sube el `.onnx` con su ficha de vuelta al bucket.

**Queda por probar** si el exportador de `torch.export` —que traza con tensores falsos y
casi no materializa nada— entra donde el clásico no. El script ya lo intenta primero, así
que si en la máquina grande funciona, sabremos además que en Colab habría bastado con eso.

## 2026-09-20 · T5 y T6 — el export y su verificación · ✅ escritos, ⬜ sin correr

`tools/export_onnx.py`. Corre en Colab, donde vive torch. **No se ha ejecutado**: se
escribió leyendo el código de T-DEED, no corriéndolo.

### Tres cosas del modelo que decidieron el diseño, y las tres se descubrieron leyendo

**1. El modelo normaliza por dentro.** Su `forward` hace `x / 255` y después la
estandarización de ImageNet. O sea que el `.onnx` espera píxeles **en crudo, 0-255**, y la
ficha declara `scale: 1`, `mean: [0,0,0]`, `std: [1,1,1]`. Copiar ahí los valores de
ImageNet —que es lo que uno haría mirando el paper— normalizaría dos veces. No da error:
hunde el score y a otra cosa.

**2. Tiene dos cabezas.** Se preentrena en SoccerNet (17 clases) y se afina en
SoccerNetBall (12); la capa final emite las dos concatenadas. La nuestra es la primera,
`1 + 12` columnas contando el fondo. El recorte va **dentro del grafo**: es un `Slice`
estático, exporta limpio, y así el repo de detección no sabe nada de esto.

**3. El desplazamiento no cabe en el grafo, y es lo importante.** El modelo emite, además
de las puntuaciones, un desplazamiento temporal por frame; su post-proceso lo aplica con
un **doble bucle de Python con índices que dependen de los datos**. Trazar eso a ONNX
hornearía los desplazamientos del clip de ejemplo dentro del grafo: no fallaría, y daría
mal todos los demás clips.

Así que el `.onnx` saca **dos salidas** —`logits` y `displacement`— y el desplazamiento lo
aplicará el spotter, en numpy, donde se puede probar. La ficha lo declara con
`meaning: displacement`.

### La columna 0 se llama `normal_play` a propósito

El modelo usa la primera columna como fondo. En la ficha se llama `normal_play`, que es
exactamente el nombre que el spotter del repo de detección conoce como la clase que nunca
produce candidatos. No es casualidad: es el contrato, y hace que la clase de fondo se
excluya sola sin que nadie configure nada.

### Probado lo que se puede probar sin torch, y el contrato entre repos

La generación de la ficha se prueba contra un `.onnx` sintético: que las formas salgan del
fichero y no de la configuración, que no se vuelva a normalizar el píxel, que la salida
del desplazamiento quede marcada y que el aviso de licencia esté.

Y se comprobó **de punta a punta entre los dos repositorios**: la ficha que genera este
repo la carga el de detección, le verifica el SHA-256, abre `VideoOnnxBackend` y monta el
`Spotter` con su ventana de 8 s. De paso, su comprobación de clases cazó un stub que
declaraba 13 clases y emitía 3 columnas — que es justo para lo que se escribió.

### Lo que queda, y es una trampa conocida

**El spotter todavía no aplica el desplazamiento.** La ficha lo declara y nadie lo lee,
que es **exactamente** el fallo del `layout` que se arregló esta mañana. No se puede dejar
así: sin aplicarlo, los eventos salen movidos hasta cuatro frames (`radi_displacement: 4`),
o sea un tercio de segundo. Es lo siguiente en el repo de detección.

**Siguiente paso**: correr el export en Colab con los pesos que ya están bajados. Si T6
pasa, el `.onnx` cruza y se acabó el camino de ida.

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

### Contrastado contra el vídeo: **los seis son goles**

El propietario los revisó el 2026-09-20 y confirma que los seis momentos corresponden a
goles reales. Con eso, sobre este vídeo y a umbral 0.2:

**Precisión de GOAL: 6 de 6.** Ni un falso positivo, incluido el de confianza 0.255 —o
sea que el umbral podría bajarse todavía más sin ensuciar—.

Lo que **sigue sin medirse** es el recall: no se ha contado si el vídeo tenía más goles
que el modelo no vio. Para las metas de E8 (recall ≥ 90 %, precisión ≥ 85 %) hace falta
saber el denominador, y eso pide un partido completo con los goles anotados a mano. La
precisión, que era la mitad más preocupante, está.

Tampoco se sabe nada de FREE KICK: cero detecciones puede ser que no hubo ninguno o que
la clase no dispara.

**Veredicto**: el camino sigue, y con margen. Precisión 6/6 en goles, error temporal por
debajo de la granularidad del clip, ruido concentrado en clases que no usamos (PASS,
DRIVE y HIGH PASS son 207 de los 292) y una regla de fusión medida de regalo. Se pasa a
T5: exportar a ONNX con su ficha.

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
