# Evaluacion Parcial N°1: Prediccion de Popularidad de Canciones en Spotify (Caso C)
**Asignatura:** Machine Learning (MLY1101)
**Semestre:** 2026-2

## Integrantes del Equipo
* Lucas Concha
* Natalia Hormazábal
* Martín Villarroel

## Herramientas Colaborativas
El desarrollo se realizo utilizando Jupyter Notebook como entorno de trabajo y Git/GitHub como sistema de control de versiones para coordinar el trabajo del equipo, mantener trazabilidad de los cambios y compartir el codigo y los datos procesados entre los integrantes.

## 1. Descripcion del Problema de Negocio

### Contexto
Una plataforma de streaming musical busca comprender que factores y atributos acusticos determinan que una cancion logre un alto nivel de recepcion por parte de la audiencia.

### Problema de Negocio
En una industria altamente saturada, la toma de decisiones sobre programacion de playlists, inversion en campañas promocionales y firma de nuevos talentos se realiza muchas veces bajo criterios subjetivos, generando riesgo de inversion no optimizada en lanzamientos que no logran traccion comercial.

### Planteamiento del Problema Tecnico
Se define como la prediccion continua del nivel de popularidad (`popularity`, escala 0 a 100) de una cancion a partir de sus metadatos e indicadores acusticos extraidos de la API de Spotify.

## 2. Objetivos del Proyecto

### Objetivo General
Construir y validar un modelo de Machine Learning capaz de predecir la popularidad (`popularity`) de una cancion en Spotify a partir de sus atributos de audio y caracteristicas asociadas.

### Objetivos Especificos
1. Identificar e interpretar cuales variables de audio (`danceability`, `energy`, `valence`, etc.) ejercen mayor influencia en la popularidad.
2. Detectar patrones y variaciones de popularidad segun el genero musical (`track_genre`).
3. Diagnosticar la calidad del dataset (nulos, duplicados y anomalias).
4. Preparar y transformar el conjunto de datos mediante codificacion, escalado y seleccion de caracteristicas para dejarlo listo para el modelamiento.

## 3. Definicion de KPIs

| KPI | Tipo | Unidad de Medida | Descripcion / Meta de Negocio |
| :--- | :--- | :--- | :--- |
| Error Global de Prediccion (RMSE / MAE) | Tecnico | Puntos de popularidad (0-100) | Minimizar la desviacion promedio de las predicciones respecto al valor real de `popularity`. |
| Tasa de Deteccion de Hits (% Hit Recall) | Negocio / Tecnico | Porcentaje (%) | Porcentaje de canciones con popularidad alta (`popularity > 70`) correctamente identificadas para campanas de difusion. |
| Correlacion Atributos vs. Popularidad | Exploratorio | Coeficiente de correlacion (r) | Medir la relacion entre variables clave (`danceability`, `energy`, `valence`, `tempo`) y `popularity` para orientar decisiones de curaduria. |

## 4. Descripcion de las Fuentes de Datos

* Dataset: Spotify Tracks Dataset (`Spotify_Tracks_Dataset.csv`)
* Dimensiones iniciales: 114.000 filas y 21 columnas.
* Variable objetivo: `popularity` (entero, 0 a 100).
* Variables continuas de audio (9): `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
* Variables discretas y categoricas: `track_genre`, `key`, `mode`, `time_signature`, `explicit`.
* Identificadores y texto: `track_id`, `artists`, `album_name`, `track_name`, `Unnamed: 0` (indice redundante de exportacion).

## 5. Preparacion y Analisis Exploratorio de Datos (EDA)

### A. Eliminacion de Columnas Redundantes
Se removio la columna indice residual `Unnamed: 0`. El dataframe pasa de 21 a 20 columnas.

### B. Tratamiento de Valores Nulos
Se identifico exactamente 1 registro con nulos simultaneos en `artists`, `album_name` y `track_name`. Se elimino mediante `dropna(subset=["artists", "album_name", "track_name"])`. El dataset queda en 113.999 registros.

### C. Auditoria y Eliminacion de Duplicados
* Duplicados exactos: se detectaron 450 filas totalmente identicas, eliminadas con `drop_duplicates()`. El dataset queda en 113.549 filas.
* Duplicados por `track_id`: se identificaron 40.108 registros (35.3% del total) cuyo `track_id` se repite catalogado en distintos generos. Se mantienen para el analisis exploratorio por genero, dejando constancia de esta particularidad para la agregacion previa al modelamiento.

### D. Inspeccion de Outliers
* Tempo en cero: 157 canciones con `tempo = 0`, asociadas a audio de ruido blanco, meditacion o archivos corruptos. Quedan documentadas en `spotify_tracks_anomalias.csv`.
* Duracion extrema: mediante el rango intercuartilico (Q3 + 3xIQR) se establecio un umbral en 8.7 minutos (> 522.000 ms). 1.217 canciones (1.07%) superan este umbral. Tambien quedan documentadas en `spotify_tracks_anomalias.csv`.

### E. Transformacion y Encoding de Variables Categoricas
1. `explicit` se convierte a entero (0 = False, 1 = True).
2. `track_genre` se codifica mediante One-Hot Encoding con `pd.get_dummies(drop_first=True)`, generando 113 variables dummy; la columna original `track_genre` se elimina del dataframe resultante.
3. `key`, `mode` y `time_signature` se conservan como enteros.

### F. Escalamiento de Caracteristicas de Audio
Se aplico `MinMaxScaler` de Scikit-Learn sobre las 9 variables continuas de audio, quedando acotadas en el rango [0.0, 1.0].

### G. Exportacion de Datos Procesados
* Dataset limpio y transformado: `spotify_tracks_clean.csv` (113.549 filas, 132 columnas, sin nulos ni duplicados exactos).
* Registros atipicos identificados: `spotify_tracks_anomalias.csv` (1.256 filas, con columna `anomalia` que indica si corresponde a tempo cero o duracion extrema).

## 6. Consideraciones Eticas, Sesgos y Privacidad

* Sesgo de exposicion: la popularidad en Spotify refleja en parte la promocion y visibilidad que recibe una cancion (posicionamiento en playlists, marketing), no solo su calidad musical o acustica. Un modelo entrenado sobre `popularity` puede aprender y reforzar ese sesgo en lugar de factores puramente musicales.
* Sesgo de representacion por genero: la cantidad de canciones por `track_genre` no es uniforme, lo que puede llevar a que el modelo prediga mejor para los generos sobrerrepresentados y peor para los minoritarios.
* Duplicidad por `track_id`: al mantener registros con el mismo `track_id` en distintos generos para el EDA, existe riesgo de sobreponderar esas canciones si no se agregan correctamente antes del modelamiento.
* Privacidad: el dataset contiene datos publicos obtenidos via la API de Spotify (metadatos e indicadores acusticos). No incluye informacion personal de usuarios ni datos de comportamiento individual, pero si nombres reales de artistas, lo que debe considerarse si se publican resultados o predicciones asociadas a un artista especifico.

## 7. Metodologia de Trabajo (CRISP-DM)

Comprension del Negocio -> Comprension de los Datos -> Preparacion de Datos -> (Modelado) -> (Evaluacion) -> (Despliegue)

## 8. Estructura del Proyecto

```
data/                 dataset original (Spotify_Tracks_Dataset.csv)
data_procesado/        spotify_tracks_clean.csv, spotify_tracks_anomalias.csv
notebooks/              Spotify_Notebook.ipynb
images/                 graficos generados en el EDA
README.md
```
