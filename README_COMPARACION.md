# Comparación de Algoritmos de ML No Supervisados para PdM

Este proyecto implementa y compara algoritmos de Machine Learning no supervisados para Mantenimiento Predictivo (PdM), organizados en dos categorías: **Clustering** y **Detección de Anomalías**.

## 📁 Estructura del Proyecto

```
Algoritmos-de-ML-no-supervisados-para-PdM/
│
├── 1. Clustering/
│   ├── DBSCAN/
│   │   ├── DBSCAN.py                    # Algoritmo DBSCAN
│   │   ├── data.csv                     # Dataset
│   │   ├── graficas_DBSCAN/             # Gráficas generadas
│   │   ├── metricas_DBSCAN/             # Métricas y outputs
│   │   └── modelos_entrenados_DBSCAN/   # Modelos guardados
│   │
│   ├── K-means/
│   │   ├── K-means.py                   # Algoritmo K-Means
│   │   ├── data.csv                     # Dataset
│   │   ├── graficas_KMeans/             # Gráficas generadas
│   │   ├── metricas_KMeans/             # Métricas y outputs
│   │   └── modelos_entrenados_KMeans/   # Modelos guardados
│   │
│   ├── comparar_clustering.py           # Script de comparación
│   └── comparacion_clustering/          # Resultados de comparación
│
├── 2. Detección de Anomalías/
│   ├── CBLOF (Cluster-Based Local Outlier Factor)/
│   │   ├── CBLOF.PY                     # Algoritmo CBLOF
│   │   ├── data.csv                     # Dataset
│   │   ├── graficas_CBLOF/              # Gráficas generadas
│   │   ├── metricas_CBLOF/              # Métricas y outputs
│   │   └── modelos_entrenados_CBLOF/    # Modelos guardados
│   │
│   ├── Isolation Forest/
│   │   ├── Isolation Forest.py          # Algoritmo Isolation Forest
│   │   ├── data.csv                     # Dataset
│   │   ├── graficas_IForest/            # Gráficas generadas
│   │   ├── metricas_IForest/            # Métricas y outputs
│   │   └── modelos_entrenados_IForest/  # Modelos guardados
│   │
│   ├── comparar_deteccion_anomalias.py  # Script de comparación
│   └── comparacion_deteccion_anomalias/ # Resultados de comparación
│
└── README_COMPARACION.md                 # Este archivo
```

## 🎯 Métricas Estandarizadas

### Clustering (DBSCAN y K-Means)

**Gráficas Generadas:**
- `clusters_2d_pca.png` - Visualización 2D con PCA
- `clusters_3d_pca.png` - Visualización 3D con PCA

**Métricas Comunes:**
- Número de clusters
- Silhouette Score (mayor es mejor)
- Calinski-Harabasz Score (mayor es mejor)
- Davies-Bouldin Index (menor es mejor)

**Archivos Generados:**
- `output.log` - Log de ejecución
- `metrics.txt` - Métricas en formato texto
- `metrics.csv` - Métricas en formato CSV para comparación
- `anomaly_scores.csv` - Scores de todos los puntos con fecha, XYZ, anomaly_score, is_outlier, cluster_id

### Detección de Anomalías (CBLOF e Isolation Forest)

**Gráficas Generadas:**
- `anomaly_scores.png` - Distribución de scores de anomalía
- `anomalies_3d.png` - Visualización 3D de anomalías

**Métricas Comunes:**
- Número de anomalías detectadas
- Porcentaje de anomalías (%)
- Separación de scores (P95-P50) - indica qué tan bien se distinguen anomalías de datos normales
- Score promedio

**Archivos Generados:**
- `output.log` - Log de ejecución
- `metrics.txt` - Métricas en formato texto
- `metrics.csv` - Métricas en formato CSV para comparación
- `anomaly_scores.csv` - Scores de todos los puntos con fecha, XYZ, is_outlier, anomaly_score
- `anomalies.csv` - Solo los puntos detectados como anomalías

## 📊 Formato Estandarizado de `metrics.csv`

Todos los algoritmos generan un archivo `metrics.csv` con las siguientes columnas:

```csv
algoritmo,params_json,n_clusters,silhouette_score,calinski_harabasz_score,davies_bouldin_score,pct_anomalias,p95_minus_p50,mean_score
```

- **algoritmo**: Nombre del algoritmo (DBSCAN, K-Means, CBLOF, Isolation_Forest)
- **params_json**: Parámetros óptimos en formato JSON
- **n_clusters**: Número de clusters (None para detección de anomalías)
- **silhouette_score**: Coeficiente Silhouette (None para detección de anomalías)
- **calinski_harabasz_score**: Índice Calinski-Harabasz (None para detección de anomalías)
- **davies_bouldin_score**: Índice Davies-Bouldin (None para detección de anomalías)
- **pct_anomalias**: Porcentaje de anomalías detectadas
- **p95_minus_p50**: Separación entre percentil 95 y 50 de scores
- **mean_score**: Score promedio de todos los puntos

## 🚀 Uso

### 1. Ejecutar Algoritmos Individuales

#### Clustering

```bash
# DBSCAN
cd "1. Clustering/DBSCAN"
python DBSCAN.py

# K-Means
cd "1. Clustering/K-means"
python K-means.py
```

#### Detección de Anomalías

```bash
# CBLOF
cd "2. Detección de Anomalías/CBLOF (Cluster-Based Local Outlier Factor)"
python CBLOF.PY

# Isolation Forest
cd "2. Detección de Anomalías/Isolation Forest"
python "Isolation Forest.py"
```

### 2. Ejecutar Comparaciones

#### Comparar Algoritmos de Clustering

```bash
cd "1. Clustering"
python comparar_clustering.py
```

**Outputs generados:**
- `comparacion_clustering/comparacion_metricas_clustering.png` - Gráfico de barras comparativo
- `comparacion_clustering/comparacion_radar_clustering.png` - Gráfico de radar multidimensional
- `comparacion_clustering/reporte_comparacion_clustering.txt` - Reporte detallado
- `comparacion_clustering/tabla_comparativa_clustering.csv` - Tabla comparativa en CSV

#### Comparar Algoritmos de Detección de Anomalías

```bash
cd "2. Detección de Anomalías"
python comparar_deteccion_anomalias.py
```

**Outputs generados:**
- `comparacion_deteccion_anomalias/comparacion_metricas_deteccion_anomalias.png` - Gráfico de barras comparativo
- `comparacion_deteccion_anomalias/comparacion_porcentaje_anomalias.png` - Comparación de porcentajes
- `comparacion_deteccion_anomalias/comparacion_separacion_scores.png` - Comparación de separación
- `comparacion_deteccion_anomalias/reporte_comparacion_deteccion_anomalias.txt` - Reporte detallado
- `comparacion_deteccion_anomalias/tabla_comparativa_deteccion_anomalias.csv` - Tabla comparativa en CSV

## 📦 Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn joblib h5py pyod
```

## 🔍 Interpretación de Métricas

### Clustering

- **Silhouette Score** (0 a 1): Mide qué tan bien están separados los clusters. Valores cercanos a 1 indican clusters bien definidos.
- **Calinski-Harabasz Score**: Ratio de dispersión entre clusters vs. dentro de clusters. Mayor es mejor.
- **Davies-Bouldin Index**: Promedio de similitud entre clusters. Menor es mejor.

### Detección de Anomalías

- **Separación de Scores (P95-P50)**: Mide qué tan bien se distinguen las anomalías de los datos normales. Mayor valor indica mejor separación.
- **Porcentaje de Anomalías**: Debe evaluarse en contexto del problema específico. No necesariamente "más es mejor".

## 🎓 Características de los Algoritmos

### DBSCAN
- ✅ Detecta clusters de forma arbitraria
- ✅ Identifica outliers automáticamente
- ✅ No requiere especificar número de clusters a priori
- ⚠️ Sensible a parámetros (eps, min_samples)

### K-Means
- ✅ Simple y rápido
- ✅ Escalable a grandes datasets
- ⚠️ Requiere especificar número de clusters
- ⚠️ Asume clusters esféricos

### CBLOF
- ✅ Usa clustering para contexto local
- ✅ Bueno para anomalías locales
- ⚠️ Requiere especificar número de clusters
- ⚠️ Más lento en grandes datasets

### Isolation Forest
- ✅ Basado en ensemble de árboles
- ✅ Rápido y escalable
- ✅ Funciona bien en alta dimensión
- ⚠️ Menos interpretable

## 📈 Flujo de Trabajo Recomendado

1. **Preparar datos**: Colocar el archivo `data.csv` en cada carpeta de algoritmo
2. **Ejecutar algoritmos**: Correr cada algoritmo individualmente
3. **Revisar métricas individuales**: Examinar los archivos de métricas y gráficas
4. **Ejecutar comparaciones**: Correr los scripts de comparación
5. **Analizar reportes**: Revisar los reportes generados y tomar decisiones

## 📝 Notas Importantes

- Todos los algoritmos están configurados para ser **100% no supervisados** (no requieren etiquetas)
- Los algoritmos optimizan automáticamente sus parámetros usando búsqueda en grid
- Para datasets muy grandes (>50,000 registros), se usa muestreo para optimización de parámetros
- Los modelos entrenados se guardan en formato `.pkl` y `.h5` para reutilización

## 🤝 Contribuciones

Para agregar nuevos algoritmos o métricas:
1. Seguir la estructura de carpetas establecida
2. Implementar las métricas estandarizadas en `metrics.csv`
3. Generar las gráficas con los nombres estándar
4. Actualizar los scripts de comparación si es necesario

## 📄 Licencia

[Especificar licencia del proyecto]

---

**Autor**: Nicolás Ceballos Brito 
**Fecha de última actualización**: Octubre 2025

