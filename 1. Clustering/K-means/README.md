# K-Means - Algoritmo de Clustering por Centroides **OPTIMIZADO**

## 📋 Descripción

K-Means es un algoritmo de clustering que particiona los datos en K grupos (clusters) minimizando la varianza dentro de cada grupo. **Versión optimizada** para datasets grandes (500k+ registros) con muestreo estratégico y visualizaciones 3D corregidas.

### ⚡ Optimizaciones Implementadas
- 🚀 **Muestreo estratégico**: Reducción inteligente a 15,000 muestras para optimización de parámetros
- 🎯 **Visualización optimizada**: Muestreo a 10,000 puntos para gráficas rápidas y claras
- 📊 **Gráficas 3D corregidas**: Proporciones adecuadas, sin deformación visual
- ⚡ **K reducido**: Búsqueda limitada a K=2-6 para ejecución rápida

### Características Principales
- ✅ **Muy rápido** - Optimizado para ejecución veloz
- ✅ **Escalable** - Maneja datasets de 500k+ registros eficientemente  
- ✅ **Visualizaciones mejoradas** - Gráficas 3D con proporciones correctas
- ✅ **Asignación determinística** de puntos a clusters
- ⚠️ Requiere especificar el número de clusters a priori
- ⚠️ Sensible a inicialización y outliers

## 🎯 Objetivo

Agrupar datos de acelerómetro en K clusters para identificar patrones de operación normal en equipos industriales, facilitando la detección de comportamientos anómalos mediante el análisis de distancia a centroides.

## 📁 Estructura de Archivos

```
K-means/
├── K-means.py                         # Script principal
├── data.csv                           # Dataset de entrada
├── README.md                          # Este archivo
│
├── graficas_KMeans/                   # Gráficas generadas
│   ├── clusters_2d_pca.png           # Visualización 2D con PCA
│   ├── clusters_3d_pca.png           # Visualización 3D con PCA
│   ├── elbow_method.png              # Método del codo (K óptimo)
│   ├── silhouette_scores.png         # Scores Silhouette por K
│   ├── calinski_harabasz_scores.png  # Scores Calinski-Harabasz por K
│   ├── davies_bouldin_scores.png     # Scores Davies-Bouldin por K
│   └── anomalies_pca.png             # Visualización de anomalías
│
├── metricas_KMeans/                   # Métricas y resultados
│   ├── output.log                    # Log de ejecución completo
│   ├── metrics.txt                   # Métricas en formato texto
│   ├── metrics.csv                   # Métricas estandarizadas (para comparación)
│   └── anomaly_scores.csv            # Scores de todos los puntos
│
└── modelos_entrenados_KMeans/        # Modelos guardados
    ├── kmeans_model.pkl              # Modelo en formato Pickle
    ├── kmeans_model.h5               # Modelo en formato HDF5
    └── scaler.pkl                    # Escalador para inferencia
```

## 🚀 Uso

### Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn joblib h5py
```

### Ejecución

```bash
python K-means.py
```

El script ejecutará automáticamente:
1. ✅ Carga y preprocesamiento de datos
2. ✅ Búsqueda del K óptimo (número de clusters)
3. ✅ Generación de gráficas de evaluación (codo, silhouette, etc.)
4. ✅ Entrenamiento del modelo final con K óptimo
5. ✅ Generación de visualizaciones de clusters
6. ✅ Cálculo de scores de anomalía (distancia a centroides)

## 📊 Métricas Generadas

### Métricas de Clustering
- **Silhouette Score**: Mide qué tan bien están separados los clusters (0-1, mayor es mejor)
- **Calinski-Harabasz Score**: Ratio de dispersión entre/dentro de clusters (mayor es mejor)
- **Davies-Bouldin Index**: Similitud promedio entre clusters (menor es mejor)
- **Inercia (SSE)**: Suma de distancias cuadráticas a centroides (menor es mejor)

### Métricas de Anomalía
- **Anomaly Score**: Distancia euclidiana de cada punto a su centroide asignado
- **Separación de Scores (P95-P50)**: Distinguibilidad de puntos alejados

## 📈 Visualizaciones

### 1. Método del Codo (`elbow_method.png`)
- Muestra la inercia vs número de clusters
- El "codo" indica el K óptimo

### 2. Gráficas de Métricas por K
- `silhouette_scores.png`: Silhouette Score para diferentes K
- `calinski_harabasz_scores.png`: Calinski-Harabasz para diferentes K
- `davies_bouldin_scores.png`: Davies-Bouldin para diferentes K

### 3. Clusters en 2D y 3D
- `clusters_2d_pca.png`: Visualización 2D con PCA
- `clusters_3d_pca.png`: Visualización 3D con PCA
- Cada cluster con color diferente

### 4. Anomalías (`anomalies_pca.png`)
- Mapa de calor de scores de anomalía
- Puntos más rojos = más alejados de su centroide

## 🔧 Parámetros del Algoritmo **OPTIMIZADOS**

El script busca automáticamente el K óptimo en el rango **reducido para velocidad**:

```python
K_MIN = 2
K_MAX = 6  # Reducido de 8 a 6 para ejecución rápida
MAX_MUESTRAS_OPTIMIZACION = 15000  # Muestreo estratégico 
MAX_MUESTRAS_VISUALIZATION = 10000  # Muestreo para gráficas
```

Selección basada en el **Silhouette Score máximo**.

## 📄 Archivos de Salida

### `metrics.csv` (Para Comparación)
Formato estandarizado con columnas:
- algoritmo, params_json, n_clusters, silhouette_score, calinski_harabasz_score, davies_bouldin_score, pct_anomalias, p95_minus_p50, mean_score

### `anomaly_scores.csv`
Todos los puntos con:
- fecha, acceleration_x, acceleration_y, acceleration_z, anomaly_score, is_outlier, cluster_id

**Nota**: K-Means no detecta outliers binarios, por lo que `is_outlier` siempre es 0. El `anomaly_score` indica la distancia al centroide.

## ⚙️ Configuración Avanzada

Para modificar el rango de búsqueda de K, edita en `K-means.py`:

```python
K_MIN = 2     # Mínimo número de clusters
K_MAX = 8     # Máximo número de clusters
```

Para grandes datasets, ajusta:

```python
MAX_MUESTRAS_OPTIMIZACION = 50000  # Máximo para búsqueda de K
```

## 🔍 Interpretación de Resultados

### Clusters Identificados
- Cada punto se asigna a exactamente un cluster
- `cluster_id` indica el cluster asignado (0, 1, 2, ...)
- Los centroides representan los "centros" de cada grupo

### Scores de Anomalía
- **Mayor distancia al centroide** = comportamiento más inusual dentro del cluster
- Útil para detección de cambios en patrones de operación
- No son outliers binarios, sino grados de "normalidad"

## 📊 Comparación con Otros Algoritmos

Para comparar K-Means con DBSCAN:

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
- Muy rápido y eficiente
- Fácil de entender e implementar
- Escalable a datasets muy grandes
- Funciona bien cuando los clusters son esféricos
- Resultados reproducibles (con semilla fija)

### ⚠️ Desventajas
- Requiere especificar K a priori
- Asume clusters de forma esférica y tamaño similar
- Sensible a outliers (pueden afectar los centroides)
- Sensible a la inicialización
- No identifica outliers automáticamente

## 🔄 Comparación con DBSCAN

| Característica | K-Means | DBSCAN |
|----------------|---------|---------|
| Necesita especificar K | ✅ Sí | ❌ No |
| Detecta outliers | ❌ No | ✅ Sí |
| Forma de clusters | Solo esféricos | Cualquier forma |
| Velocidad | ⚡ Muy rápido | 🐌 Más lento |
| Escalabilidad | ✅ Excelente | ⚠️ Limitada |

## 📚 Referencias

- Lloyd, S. (1982). "Least squares quantization in PCM"
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#k-means

## 🤝 Contribución al Proyecto

Este algoritmo es parte del proyecto de comparación de algoritmos de ML no supervisados para Mantenimiento Predictivo. Los resultados de K-Means se comparan directamente con DBSCAN para determinar el mejor algoritmo de clustering para esta aplicación específica.

---

**Última actualización**: Octubre 2025  
**Versión**: 3.0 - **OPTIMIZADA** ⚡
