# Evaluación Parcial N°1: Predicción de Popularidad de Canciones en Spotify (Caso C)
**Asignatura:** Machine Learning (MLY1101)  
**Institución:** Duoc UC  
**Semestre:** 2026-1  

---

## 👥 Integrantes del Equipo
* **[Nombre Estudiante 1]** - Lucas Concha Perez
* **[Nombre Estudiante 2]** - Natalia Hormazabal Nuñez
* **[Nombre Estudiante 3]** - Martin Villarroel Morales

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

### Origen y Dimensión
* **Dataset:** Spotify Tracks Dataset
* **Volumen:** 114,000 registros y 21 columnas.

### Estructura y Tipos de Variables
* **Variable Objetivo (Target):**
  * `popularity`: Numérica entera (rango de 0 a 100).
* **Numéricas Continuas (Audio Features):**
  * `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`.
* **Categorical y Discretas:**
  * `track_genre`, `key`, `mode`, `time_signature`, `explicit` (Booleana).
* **Identificadores / Texto (No predictivas directas):**
  * `track_id`, `artists`, `album_name`, `track_name`, `Unnamed: 0` (índice residual).

### Diagnóstico Inicial de Calidad
* **Valores Nulos:** Presencia mínima de nulos (1 registro con faltantes en `artists`, `album_name` y `track_name`).
* **Duplicados:** Se debe auditar la presencia de duplicados en la columna `track_id` (canciones que aparecen en múltiples álbumes o géneros).

---

## 🔍 5. Preparación y Análisis Exploratorio de Datos (EDA)

### Estrategia de Análisis Exploratorio
1. **Estadística Descriptiva:** Inspección de medidas de tendencia central y dispersión para `popularity` y métricas de audio.
2. **Distribución por Género (`track_genre`):** Evaluación de la representatividad y balance de registros por cada género musical.
3. **Detección de Outliers y Anomalías:** Identificación de registros atípicos (ej. canciones con `duration_ms` excesivamente altos/bajos o registros con `tempo = 0`).
4. **Matriz de Correlación:** Análisis de relaciones lineales entre variables continuas y `popularity` mediante mapas de calor.

### Visualizaciones Clave a Generar
* Histogramas de distribución para `popularity` y *audio features*.
* Boxplots comparativos de popularidad según `track_genre`.
* Heatmap de correlaciones (Pearson / Spearman).

### Transformaciones y Preprocesamiento
* **Limpieza:** Imputación o eliminación del registro con valores nulos y remoción de registros duplicados en `track_id`.
* **Depuración de Columnas:** Eliminación de variables no predictivas o identificadores puros (`track_id`, `Unnamed: 0`).
* **Transformación Categórica:** Aplicación de técnicas de *Encoding* (ej. *One-Hot Encoding* o *Target Encoding*) para `track_genre` y variables de tono/modo.
* **Escalamiento:** Normalización/Estandarización (`StandardScaler` o `RobustScaler`) de variables numéricas continuas para homogenizar rangos.

---

## 🔄 6. Metodología de Trabajo (CRISP-DM)

El desarrollo del proyecto se estructura bajo las etapas del marco **CRISP-DM**:

```text
[Comprensión del Negocio] ➔ [Comprensión de los Datos] ➔ [Preparación de Datos] ➔ (Modelado)* ➔ (Evaluación)* ➔ (Despliegue)*