# Evaluación Parcial N°1: Predicción de Popularidad de Canciones en Spotify (Caso C)

**Asignatura:** Machine Learning (MLY1101)

**Semestre:** 2026-2

## Integrantes del Equipo

* Lucas Concha
* Natalia Hormazábal
* Martín Villarroel

## Resumen del proyecto

Se prepara y analiza el *Spotify Tracks Dataset* (114.000 canciones) para predecir la popularidad (`popularity`, 0 a 100) de una canción a partir de sus atributos acústicos y metadatos. Este informe cubre las fases de **comprensión del negocio, comprensión de los datos y preparación de datos** (CRISP-DM).

**Hallazgos principales del EDA**

1. Los atributos de audio, por sí solos, se relacionan muy débilmente con `popularity` (|r| ≤ 0,10, con Pearson y con Spearman).
2. El género musical es la señal más fuerte: `track_genre` explica ~25% de la varianza de `popularity` (η² = 0,25).
3. El ~14% de las canciones tiene `popularity = 0` (concentradas en ciertos géneros) y solo el 4,3% supera 70 puntos ("hits").
4. El 35,3% de las filas corresponde a canciones repetidas en distintos géneros, lo que exige particionar por `track_id` al modelar.

## Herramientas Colaborativas

| Herramienta | Uso en el proyecto | Por qué |
| :--- | :--- | :--- |
| **Jupyter Notebook** | Desarrollo del EDA y la preparación de datos (`notebooks/Spotify_Notebook.ipynb`). | Reúne código, resultados y explicaciones en un solo archivo ejecutable, lo que permite reproducir cada etapa del informe. |
| **Git / GitHub** | Control de versiones y repositorio compartido del equipo. | Coordina el trabajo de los tres integrantes, mantiene trazabilidad de los cambios y comparte código, datos procesados y gráficos. |
| **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn) | Manipulación, visualización y escalado de datos. | Ecosistema estándar de la asignatura para ciencia de datos. |

## 1. Descripción del Problema de Negocio

### Contexto

Una plataforma de streaming musical busca comprender qué factores y atributos acústicos determinan que una canción logre un alto nivel de recepción por parte de la audiencia.

### Problema de Negocio

En una industria altamente saturada, la toma de decisiones sobre programación de playlists, inversión en campañas promocionales y firma de nuevos talentos se realiza muchas veces bajo criterios subjetivos, generando riesgo de inversión no optimizada en lanzamientos que no logran tracción comercial.

### Planteamiento del Problema Técnico

Se define como la predicción continua (regresión) del nivel de popularidad (`popularity`, escala 0 a 100) de una canción a partir de sus metadatos e indicadores acústicos extraídos de la API de Spotify.

## 2. Objetivos del Proyecto

### Objetivo General

Construir y validar un modelo de Machine Learning capaz de predecir la popularidad (`popularity`) de una canción en Spotify a partir de sus atributos de audio y características asociadas.

### Objetivos Específicos

1. Identificar e interpretar cuáles variables de audio (`danceability`, `energy`, `valence`, etc.) ejercen mayor influencia en la popularidad.

2. Detectar patrones y variaciones de popularidad según el género musical (`track_genre`).

3. Diagnosticar la calidad del dataset (nulos, duplicados y anomalías).

4. Preparar y transformar el conjunto de datos mediante codificación, escalado y selección de características para dejarlo listo para el modelamiento.

## 3. Definición de KPIs

| KPI | Tipo | Unidad de Medida | Descripción / Meta de Negocio | Estado |
| :--- | :--- | :--- | :--- | :--- |
| Error Global de Predicción (RMSE / MAE) | Técnico | Puntos de popularidad (0-100) | Minimizar la desviación promedio de las predicciones respecto al valor real de `popularity`. | Se mide en la etapa de modelado. |
| Tasa de Detección de Hits (% Hit Recall) | Negocio / Técnico | Porcentaje (%) | Porcentaje de canciones con popularidad alta (`popularity > 70`) correctamente identificadas para campañas de difusión. Solo el 4,3% del dataset son hits, por lo que es una clase escasa. | Se mide en la etapa de modelado. |
| Correlación Atributos vs. Popularidad | Exploratorio | Coeficiente de correlación (r) | Cuantificar el grado de asociación entre atributos clave de audio (`danceability`, `energy`, `valence`, `tempo`) y `popularity` con el fin de orientar la selección de canciones para playlists y campañas de difusión. | Medido en el EDA: \|r\| ≤ 0,10. |

## 4. Descripción de las Fuentes de Datos

* **Dataset:** *Spotify Tracks Dataset* (`Spotify_Tracks_Dataset.csv`), autor `maharshipandya`. Publicado en Kaggle. Los datos se obtuvieron mediante la API web de Spotify. Según la ficha del dataset en Hugging Face, su licencia es BSD; el uso de datos de Spotify está además sujeto a los términos de uso de su API.
* **Dimensiones iniciales:** 114.000 filas y 21 columnas. Cada fila es una canción asociada a un género (114 géneros).
* **Variable objetivo:** `popularity` (entero, 0 a 100). Es un valor calculado por Spotify que depende en gran parte de las reproducciones recientes de la canción.
* **Variables continuas de audio (9):** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
* **Variables discretas y categóricas:** `track_genre`, `key`, `mode`, `time_signature`, `explicit`. Además, `duration_ms` (duración en milisegundos).
* **Identificadores y texto:** `track_id`, `artists`, `album_name`, `track_name`, `Unnamed: 0` (índice redundante de exportación).

## 5. Preparación y Análisis Exploratorio de Datos (EDA)

Todo lo descrito en esta sección se reproduce en `notebooks/Spotify_Notebook.ipynb`.

### A. Limpieza de datos

| Paso | Qué se hizo | Justificación | Filas resultantes |
| :--- | :--- | :--- | :-: |
| Columna redundante | Se eliminó `Unnamed: 0` (21 → 20 columnas). | Índice residual de la exportación, sin información. | 114.000 |
| Valores nulos | 1 registro con nulos simultáneos en `artists`, `album_name` y `track_name`; se eliminó con `dropna`. | Una sola fila (0,001%) sin identificación utilizable: eliminar es más simple que imputar. | 113.999 |
| Duplicados exactos | 450 filas totalmente idénticas, eliminadas con `drop_duplicates()`. | Registros repetidos sin información nueva. | 113.549 |

### B. Duplicados por `track_id` (se conservan, con advertencia)

* **40.108 filas (35,3%)** tienen un `track_id` repetido, catalogado en distintos géneros (16.299 canciones aparecen en más de un género; hay 89.740 canciones únicas).
* Se **conservan** para el análisis por género.
* El **95,6%** de las canciones repetidas tiene exactamente la misma `popularity` en todos sus géneros. Por eso, si se particiona al azar, una misma canción puede caer en entrenamiento y prueba (fuga de información). **En el modelado se particionará agrupando por `track_id`** (por ejemplo, `GroupKFold`).

### C. Inspección de anomalías (outliers)

* **Tempo en cero:** 157 canciones con `tempo = 0`, asociadas a audio de ruido blanco, meditación o archivos corruptos.
* **Duración extrema:** mediante el rango intercuartílico (Q3 + 3×IQR) se estableció un umbral de 8,7 minutos (> 522.000 ms). 1.217 filas (1,07%) lo superan, correspondientes a 1.099 canciones distintas (el archivo de anomalías se deduplica por `track_id`). La duración máxima es de 87,3 minutos.
* **Compás (`time_signature`):** 163 filas con valor 0 (probablemente dato faltante) y 967 con valor 1; se dejan como observación.
* **Registro:** todas se documentan en `data_procesado/spotify_tracks_anomalias.csv` (1.256 filas = 157 de tempo cero + 1.099 de duración extrema, con la columna `anomalia`).

> **Decisión:** las anomalías se **documentan pero se conservan** en el dataset limpio. Representan ~1,1% de los datos, las duraciones largas pueden ser legítimas (álbumes en vivo, sets) y eliminarlas sin evidencia sería arbitrario. En el modelado se comparará el error con y sin ellas, y solo se excluirán si mejoran el desempeño.

### D. Hallazgos del análisis exploratorio

#### D.1 Variable objetivo (`popularity`)

![Distribución de popularity](images/distribucion_popularity.png)

* Distribución **no normal**: media 33,3 y mediana 35, con un pico marcado en 0 y otros dos modos cerca de 20 y 42.
* **15.843 canciones (14,0%) tienen `popularity = 0`.**
* Solo **4.843 canciones (4,3%) superan 70** (hits): clase escasa para el KPI de detección.

#### D.2 Duración de las canciones

![Boxplot de duración](images/boxplot_duration.png)

* La mayoría dura entre ~3 y ~4,5 minutos, pero hay una cola larga de hasta 87 minutos (asimetría 10,8), lo que justifica el umbral extremo usado en la sección C.

#### D.3 Correlaciones entre atributos de audio y popularidad

![Matriz de correlación](images/matriz_correlacion.png)

* **Con `popularity`:** ninguna variable de audio supera |r| = 0,09; la mayor es `instrumentalness` (−0,09). Ampliando a `duration_ms`, `explicit`, `key`, `mode` y `time_signature`, y usando Spearman, el máximo sigue siendo bajo (Pearson 0,095; Spearman 0,078): **los atributos de audio no bastan para explicar la popularidad**.
* Al excluir los `popularity = 0`, la relación de `instrumentalness` sube a −0,18: parte de la señal está diluida por los ceros.
* **Entre los atributos:** `energy`–`loudness` (0,76), `energy`–`acousticness` (−0,73) y `danceability`–`valence` (0,48).

#### D.4 Popularidad por género

![Popularidad por género](images/popularidad_por_genero.png)

* El género explica **~25% de la varianza** de `popularity` (η² = 0,25), y ~38% si se excluyen los ceros.
* Los géneros más populares son `pop-film` (59,3), `k-pop` (57,0) y `chill` (53,7); los menos, `iranian` (2,2), `romance` (3,5) y `latin` (8,4).
* Los ceros se concentran en ciertos géneros (`jazz` 68%, `iranian` 66%, `romance` 61%, `soul` 61%): parte de la diferencia entre géneros puede deberse a cómo se asigna la popularidad y no solo al gusto de la audiencia.
* Cada género tiene entre 904 y 1.000 canciones (equilibrado por construcción).

### E. Transformación y Encoding de Variables

1. `explicit` se convierte a entero (0 = False, 1 = True).
2. `track_genre` se codifica mediante One-Hot Encoding con `pd.get_dummies(drop_first=True)`, generando 113 variables dummy; la columna original se elimina del dataframe resultante.
3. `key`, `mode` y `time_signature` se conservan como enteros.
4. **Escalamiento:** se aplicó `MinMaxScaler` (Scikit-Learn) sobre las 9 variables continuas de audio, quedando en el rango [0,0, 1,0]. `duration_ms` no se escala en esta etapa (no es un atributo de audio y su tratamiento dependerá del modelo).
5. **Nota metodológica:** el escalador se ajusta aquí con todo el dataset por ser la etapa de preparación. En el modelado deberá ajustarse solo con el conjunto de entrenamiento para evitar fuga de información.

### F. Selección de características

* **Se excluyen como predictoras:** `track_id` (se usa solo para agrupar la partición) y `artists`, `album_name`, `track_name` (texto de alta cardinalidad). Permanecen en el CSV para trazabilidad.
* **Multicolinealidad moderada:** el VIF máximo entre los atributos de audio es 4,2 (`energy`), bajo el umbral habitual de 5, por lo que no se elimina ninguna variable de audio en esta etapa.
* **Selección final: 127 variables predictoras** (9 de audio, 5 discretas o de metadatos y 113 dummies de género). Ninguna cumple un criterio de exclusión (VIF > 5 o ser identificador/texto), por lo que se conservan todas; las dummies de género se mantienen por ser la señal más fuerte.

### G. Exportación de Datos Procesados

* Dataset limpio y transformado: `data_procesado/spotify_tracks_clean.csv` (113.549 filas, 132 columnas, sin nulos ni duplicados exactos; el notebook incluye validaciones automáticas).
* Registros atípicos identificados: `data_procesado/spotify_tracks_anomalias.csv` (1.256 filas, con columna `anomalia` que indica si corresponde a tempo cero o duración extrema).

## 6. Consideraciones Éticas, Sesgos y Privacidad

| Riesgo | Descripción | Mitigación |
| :--- | :--- | :--- |
| **Sesgo de exposición y temporal** | La popularidad refleja las reproducciones recientes y la visibilidad que recibe una canción (playlists, marketing), no solo su calidad musical o acústica. Es además una "foto" de un momento y puede cambiar con el tiempo. Un modelo entrenado sobre `popularity` puede aprender y reforzar ese sesgo. | Interpretar los resultados como *popularidad observada*, no como calidad; complementar con criterio de negocio. |
| **Sesgo de muestreo por género** | El dataset contiene entre 904 y 1.000 canciones por cada uno de los 114 géneros: está equilibrado *por diseño* y no refleja el peso real de cada género en el catálogo de Spotify. Los resultados no deben extrapolarse a toda la industria. | Reportar el error del modelo **por género**, no solo el global. |
| **Ruido en la variable objetivo** | El 14% de las canciones tiene `popularity = 0`, concentradas en ciertos géneros (`jazz`, `iranian`, `romance`, `soul`). Un modelo podría penalizar sistemáticamente esos géneros. | Evaluar el modelo con y sin los ceros y auditar sus errores por género. |
| **Duplicidad por `track_id`** | 16.299 canciones aparecen en varios géneros (35,3% de las filas): riesgo de sobreponderarlas y de fuga de información entre entrenamiento y prueba. | Particionar agrupando por `track_id` y evaluar con canciones únicas. |
| **Uso responsable de las predicciones** | Usar el modelo para decidir a qué artistas apoyar puede favorecer siempre a lo ya popular y perjudicar a artistas emergentes o a géneros con menor exposición. | Usar las predicciones como apoyo a la decisión, no como criterio único. |
| **Privacidad** | Los datos son públicos y provienen de la API de Spotify (metadatos e indicadores acústicos). No incluyen información personal de usuarios ni datos de comportamiento individual, pero sí nombres reales de artistas. | No publicar predicciones individuales asociadas a un artista específico y respetar los términos de uso de la API de Spotify. |

## 7. Metodología de Trabajo (CRISP-DM)

| Fase | Qué se hizo | Estado |
| :--- | :--- | :--- |
| **1. Comprensión del negocio** | Definición del problema, objetivos y KPIs (secciones 1 a 3). | Completada |
| **2. Comprensión de los datos** | Descripción de fuentes, inspección, calidad (nulos, duplicados, anomalías) y EDA: variable objetivo, correlaciones y género (secciones 4 y 5.A a 5.D). | Completada |
| **3. Preparación de datos** | Limpieza, encoding, escalado, selección de características y exportación (secciones 5.E a 5.G). | Completada (falta la partición train/test, que se define en el modelado) |
| **4. Modelado** | Partición agrupada por `track_id`, modelo base y modelos no lineales. | Pendiente |
| **5. Evaluación** | Medición de RMSE / MAE, Hit Recall y error por género. | Pendiente |
| **6. Despliegue** | — | Pendiente |

## 8. Estructura del Proyecto

```
data/                   dataset original (Spotify_Tracks_Dataset.csv)
data_procesado/         spotify_tracks_clean.csv, spotify_tracks_anomalias.csv
notebooks/              Spotify_Notebook.ipynb
images/                 gráficos generados en el EDA
requirements.txt        librerías necesarias
README.md
```