# Evaluación Parcial N°1: Predicción de Popularidad de Canciones en Spotify (Caso C)
**Asignatura:** Machine Learning (MLY1101)  
**Institución:** Duoc UC  
**Semestre:** 2026-1  

---

## 👥 Integrantes del Equipo
* **Lucas Concha**
* **Natalia Hormazabal**
* **Martin Villarroel**

---

## 📌 1. Descripción del Problema de Negocio

### Contexto
Una plataforma de *streaming* musical (o sello discográfico) busca comprender en profundidad qué factores y atributos acústicos determinan que una canción logre un alto nivel de recepción por parte de la audiencia.

### Problema de Negocio
En una industria altamente saturada, la toma de decisiones sobre programación de listas de reproducción (*playlists*), inversión en campañas promocionales y firma de nuevos talentos (A&R) se realiza muchas veces bajo criterios subjetivos. Esto genera un alto riesgo de inversión no optimizada en lanzamientos que no logran tracción comercial.

### Planteamiento del Problema Técnico
El problema se define como la predicción continua del nivel de popularidad (`popularity`, escala de 0 a 100) de una canción a partir de sus metadatos e indicadores acústicos extraídos de la API de Spotify.

---

## 🎯 2. Objetivos del Proyecto

### Objetivo General
Construir y validar un modelo de Machine Learning capaz de predecir la popularidad (`popularity`) de una canción en Spotify a partir de sus atributos de audio y características asociadas.

### Objetivos Específicos
1. Identificar e interpretar cuáles variables de audio (`danceability`, `energy`, `valence`, etc.) ejercen mayor influencia en la popularidad.
2. Detectar patrones y variaciones de popularidad según el género musical (`track_genre`).
3. Diagnosticar la calidad del dataset (identificación de nulos, duplicados y anomalías).
4. Preparar y transformar el conjunto de datos mediante codificación, escalado y selección de características para dejarlo listo para la fase de modelamiento.

---

## 📊 3. Definición de KPIs

Para evaluar la efectividad de la solución tanto a nivel técnico como de impacto en el negocio, se establecen los siguientes KPIs:

| KPI | Tipo | Unidad de Medida | Descripción / Meta de Negocio |
| :--- | :--- | :--- | :--- |
| **Error Global de Predicción (RMSE / MAE)** | Técnico (Modelo) | Puntos de popularidad (0-100) | Minimizar la desviación promedio de las predicciones respecto al valor real de `popularity`. |
| **Tasa de Detección de Hits (% Hit Recall)** | Negocio / Técnico | Porcentaje (%) | Porcentaje de canciones con popularidad alta (`popularity > 70`) correctamente clasificadas/identificadas para campañas de difusión. |
| **Correlación Atributos vs. Popularidad** | Exploratorio (Negocio) | Coeficiente de Correlación ($r$) | Medir la magnitud de relación entre variables clave (`danceability`, `energy`, `valence`, `tempo`) y `popularity` para orientar decisiones de curaduría. |

---

## 💾 4. Descripción de las Fuentes de Datos

### Origen y Dimensión Inicial
* **Dataset:** Spotify Tracks Dataset (`Spotify_Tracks_Dataset.csv`)
* **Dimensiones Iniciales:** 114,000 filas y 21 columnas.

### Estructura y Tipos de Variables
* **Variable Objetivo (Target):** `popularity` (entero, 0 a 100).
* **Variables Continuas de Audio (9):** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
* **Variables Discretas y Categóricas:** `track_genre`, `key`, `mode`, `time_signature`, `explicit` (Booleano).
* **Identificadores y Texto:** `track_id`, `artists`, `album_name`, `track_name`, `Unnamed: 0` (índice redundante de exportación).

---

## 🔍 5. Preparación y Análisis Exploratorio de Datos (EDA)

### 🧹 Inciso: Pipeline de Limpieza y Preprocesamiento de Datos

A partir de la inspección ejecutada en el notebook principal (`Spotify_Notebook.ipynb`), se aplicaron las siguientes etapas secuenciales de limpieza y transformación:

#### A. Eliminación de Columnas Redundantes
* **Operación:** Se removió la columna índice residual `Unnamed: 0`.
* **Resultado:** El dataframe pasa de 21 a 20 columnas (`shape: (114000, 20)`).

#### B. Tratamiento de Valores Nulos
* **Diagnóstico:** Se identificó exactamente 1 registro faltante de forma simultánea en las columnas `artists`, `album_name` y `track_name`.
* **Acción:** Se eliminó la fila mediante `dropna(subset=["artists", "album_name", "track_name"])`.
* **Resultado:** El dataset se ajusta a **113,999 registros**.

#### C. Auditoría y Eliminación de Duplicados
1. **Duplicados Exactos:**
   * **Diagnóstico:** Se detectaron 450 filas totalmente idénticas.
   * **Acción:** Eliminación mediante `drop_duplicates()`.
   * **Resultado:** Registros reducidos a **113,549 filas**.
2. **Duplicados por Identificador (`track_id`):**
   * **Diagnóstico:** Se identificaron **40,108 registros** (35.3% del total) cuyo `track_id` se repite catalogado en distintos géneros (`track_genre`). 
   * **Decisión Técnica:** Mantener los registros para la fase de análisis exploratorio por género, registrando esta particularidad para la agregación previa al modelamiento.

#### D. Inspección de Outliers y Valores Atípicos
* **Tempo en Cero (`tempo = 0`):** Se identificaron **157 canciones** con ritmo nulo, asociadas a audios de ruido blanco, meditación o archivos corruptos.
* **Duración Extrema (`duration_ms`):** Mediante el rango intercuartílico ($Q3 + 3 \times IQR$), se estableció un umbral atípico extremo en **8.7 minutos** ($> 522,000\text{ ms}$). Un total de **1,217 canciones (1.07%)** superan este umbral (temas progresivos, audiolibros o mezclas extensas).

#### E. Transformación y Encoding de Variables Categóricas
1. **Variable Booleana `explicit`:** Convertida a entero (`0` para False, `1` para True).
2. **Género Musical (`track_genre`):** Se generaron **113 variables dummy** (*One-Hot Encoding* con `drop_first=True`) mediante `pd.get_dummies()`.
3. **Variables Musicales Discretas:** `key`, `mode` y `time_signature` se conservaron como enteros.

#### F. Escalamiento de Características de Audio
* **Operación:** Se aplicó **`MinMaxScaler`** de Scikit-Learn sobre las 9 variables continuas de audio (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`).
* **Resultado:** Todas las métricas de audio quedaron acotadas estrictamente en el rango continuo $[0.0, 1.0]$.

#### G. Exportación de Datos Procesados
* El conjunto de datos limpio y transformado fue exportado a `../data_procesado/spotify_tracks_clean.csv`.

---

## 🔄 6. Metodología de Trabajo (CRISP-DM)

El proyecto se desarrolla bajo la estructura del estándar **CRISP-DM**:

```text
[Comprensión del Negocio] ➔ [Comprensión de los Datos] ➔ [Preparación de Datos] ➔ (Modelado)* ➔ (Evaluación)* ➔ (Despliegue)*