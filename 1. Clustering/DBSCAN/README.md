# DBSCAN - Density-Based Spatial Clustering of Applications with Noise

## 📋 Descripción

DBSCAN es un algoritmo de clustering basado en densidad que agrupa puntos que están estrechamente empaquetados y marca como outliers los puntos que están solos en regiones de baja densidad.

### Características Principales
- ✅ Detecta clusters de forma arbitraria (no solo esféricos)
- ✅ Identifica automáticamente outliers (puntos de ruido)
- ✅ No requiere especificar el número de clusters a priori
- ✅ Funciona bien con clusters de densidad variable
- ⚠️ Sensible a la elección de parámetros (eps, min_samples)

## 🎯 Objetivo

Identificar grupos naturales en datos de acelerómetro para Mantenimiento Predictivo, detectando automáticamente patrones anómalos sin supervisión.

## 📁 Estructura de Archivos

```
DBSCAN/
├── DBSCAN.py                          # Script principal
├── data.csv                           # Dataset de entrada
├── README.md                          # Este archivo
│
├── graficas_DBSCAN/                   # Gráficas generadas
│   ├── clusters_2d_pca.png           # Visualización 2D con PCA
│   ├── clusters_3d_pca.png           # Visualización 3D con PCA
│   └── k_distance_graph.png          # Gráfica K-distancias (para estimación de eps)
│
├── metricas_DBSCAN/                   # Métricas y resultados
│   ├── output.log                    # Log de ejecución completo
│   ├── metrics.txt                   # Métricas en formato texto
│   ├── metrics.csv                   # Métricas estandarizadas (para comparación)
│   ├── anomaly_scores.csv            # Scores de todos los puntos
│   └── anomalies.csv                 # Solo outliers detectados
│
└── modelos_entrenados_DBSCAN/        # Modelos guardados
    ├── dbscan_model.pkl              # Modelo en formato Pickle
    ├── dbscan_model.h5               # Modelo en formato HDF5
    └── scaler.pkl                    # Escalador para inferencia
```

## 🚀 Uso

### Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn joblib h5py
```

### Ejecución

```bash
python DBSCAN.py
```

El script ejecutará automáticamente:
1. ✅ Carga y preprocesamiento de datos
2. ✅ Optimización automática de parámetros (eps, min_samples)
3. ✅ Entrenamiento del modelo
4. ✅ Generación de visualizaciones
5. ✅ Cálculo de métricas de calidad
6. ✅ Detección y guardado de anomalías

## 📊 Métricas Generadas

### Métricas de Clustering
- **Silhouette Score**: Mide qué tan bien están separados los clusters (0-1, mayor es mejor)
- **Calinski-Harabasz Score**: Ratio de dispersión entre/dentro de clusters (mayor es mejor)
- **Davies-Bouldin Index**: Similitud promedio entre clusters (menor es mejor)

### Métricas de Detección de Anomalías
- **Porcentaje de Anomalías**: Porcentaje de puntos clasificados como ruido
- **Separación de Scores (P95-P50)**: Qué tan bien se distinguen anomalías de normales

## 📈 Visualizaciones

### 1. Gráfica de K-Distancias (`k_distance_graph.png`)
- Ayuda a estimar el parámetro `eps` óptimo
- El "codo" en la gráfica sugiere un buen valor de eps

### 2. Clusters 2D con PCA (`clusters_2d_pca.png`)
- Visualización de clusters en 2 dimensiones
- Puntos de ruido mostrados en negro (x)
- Cada cluster con color diferente

### 3. Clusters 3D con PCA (`clusters_3d_pca.png`)
- Visualización tridimensional de clusters
- Permite ver mejor la separación entre grupos

## 🔧 Parámetros del Algoritmo

El script optimiza automáticamente:

- **eps**: Radio de vecindad (distancia máxima entre puntos en el mismo cluster)
- **min_samples**: Número mínimo de puntos para formar un cluster denso

El algoritmo usa búsqueda en grid para encontrar los mejores parámetros basándose en el Silhouette Score.

## 📄 Archivos de Salida

### `metrics.csv` (Para Comparación)
Formato estandarizado con columnas:
- algoritmo, params_json, n_clusters, silhouette_score, calinski_harabasz_score, davies_bouldin_score, pct_anomalias, p95_minus_p50, mean_score

### `anomaly_scores.csv`
Todos los puntos con:
- fecha, acceleration_x, acceleration_y, acceleration_z, anomaly_score, is_outlier, cluster_id

### `anomalies.csv`
Solo puntos detectados como outliers (is_outlier = 1)

## ⚙️ Configuración Avanzada

Para modificar parámetros, edita las constantes en `DBSCAN.py`:

```python
# Rango de búsqueda para eps
eps_min = 0.1
eps_max = 2.0
n_eps = 8

# Rango de búsqueda para min_samples
min_samples_min = 3
min_samples_max = 4
```

## 🔍 Interpretación de Resultados

### Clusters Válidos
- El algoritmo identifica automáticamente el número de clusters
- Cada punto pertenece a un cluster o es ruido (-1)

### Outliers/Anomalías
- Puntos con `is_outlier = 1` son anomalías detectadas
- `anomaly_score` indica qué tan anómalo es el punto (mayor = más anómalo)
- Útil para mantenimiento predictivo: estos puntos pueden indicar fallos inminentes

## 📊 Comparación con Otros Algoritmos

Para comparar DBSCAN con K-Means:

```bash
cd ../Comparaciones
python comparar_algoritmos.py
```

Esto generará:
- Comparación visual lado a lado
- Gráficos comparativos de métricas
- Reporte detallado con análisis
- Determinación del mejor algoritmo

## 🎓 Ventajas y Desventajas

### ✅ Ventajas
- Detecta clusters de cualquier forma (no solo círculos/esferas)
- No necesita saber cuántos clusters hay de antemano
- Identifica outliers automáticamente
- Robusto a ruido en los datos

### ⚠️ Desventajas
- Sensible a la elección de eps y min_samples
- Puede tener problemas con clusters de densidad muy diferente
- Computacionalmente más costoso que K-Means
- Rendimiento puede degradarse en alta dimensionalidad

## 📚 Referencias

- Ester, M., Kriegel, H.P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise"
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#dbscan

## 🤝 Contribución al Proyecto

Este algoritmo es parte del proyecto de comparación de algoritmos de ML no supervisados para Mantenimiento Predictivo. Los resultados de DBSCAN se comparan directamente con K-Means para determinar el mejor algoritmo de clustering para esta aplicación específica.

---

**Última actualización**: Octubre 2025  
**Versión**: 2.0
