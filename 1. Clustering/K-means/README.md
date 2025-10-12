# K-Means - Algoritmo de Clustering por Centroides

## 📋 Descripción

K-Means es un algoritmo de clustering que particiona los datos en K grupos (clusters) minimizando la varianza dentro de cada grupo. El algoritmo itera asignando cada punto al centroide más cercano y recalculando los centroides hasta convergencia.

### Características Principales
- ✅ Asignación determinística de puntos a clusters
- ✅ Búsqueda automática del número óptimo de clusters (K=2-6)
- ✅ Visualizaciones en 2D y 3D usando PCA
- ✅ Genera visualizaciones con proporciones correctas
- ⚠️ Requiere especificar el rango de clusters a evaluar
- ⚠️ Sensible a inicialización y outliers

## 🎯 Objetivo

Agrupar datos de acelerómetro en K clusters para identificar patrones de operación normal en equipos industriales, facilitando la detección de comportamientos anómalos mediante el análisis de distancia a centroides.

## 📁 Estructura de Archivos

```
K-means/
├── K-means.py                         # Script principal
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
1. ✅ Carga y preprocesamiento de datos (desde `data.csv` en la raíz del proyecto)
2. ✅ Búsqueda del K óptimo (número de clusters)
3. ✅ Generación de gráficas de evaluación (codo, silhouette, etc.)
4. ✅ Entrenamiento del modelo final con K óptimo
5. ✅ Generación de visualizaciones de clusters
6. ✅ Cálculo de scores de anomalía (distancia a centroides)

**Nota**: El archivo `data.csv` se encuentra centralizado en la raíz del proyecto y es compartido por todos los algoritmos.

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
MAX_MUESTRAS_OPTIMIZACION = 5000   # Muestreo estratégico 
MAX_MUESTRAS_VISUALIZATION = 3000  # Muestreo para gráficas
SILHOUETTE_SAMPLE = 5000           # Muestra para cálculo eficiente de Silhouette
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
SILHOUETTE_SAMPLE = 10000          # Muestra para cálculo de Silhouette en datasets grandes
```

**Nota**: El parámetro `SILHOUETTE_SAMPLE` optimiza el cálculo del Silhouette Score usando muestreo estratificado, reduciendo la complejidad O(n²) sin pérdida significativa de precisión.

## 📊 Resultados Obtenidos

### Parámetros Óptimos Encontrados
- **Número de Clusters (K)**: 2
- **Tiempo de Ejecución**: 9.60 segundos
- **Memoria Utilizada**: 184.78 MB

### Métricas de Calidad del Clustering
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Silhouette Score** | 0.3269 | Moderado - Los clusters están razonablemente separados |
| **Calinski-Harabasz** | 300079.19 | Excelente - Muy buena separación entre clusters |
| **Davies-Bouldin** | 1.2372 | Bueno - Los clusters están bien definidos (menor es mejor) |
| **Inercia (SSE)** | 1313348.75 | Suma de distancias dentro de clusters |

### Detección de Anomalías
- **Porcentaje de Anomalías**: 5.00% (usando percentil 95)
- **Separación P95-P50**: 0.3360
- **Score Promedio**: 0.5038

### Análisis de Resultados

#### ✅ Fortalezas Observadas
1. **Excelente Calinski-Harabasz Score**: El valor de 300,079 indica que los clusters están muy bien separados entre sí y son cohesivos internamente.

2. **Buena Velocidad**: Con 9.6 segundos para procesar 518,400 registros, K-Means demuestra su eficiencia computacional.

3. **Uso Eficiente de Memoria**: 184.78 MB es razonable para el volumen de datos procesados.

4. **Simplicidad Interpretativa**: Solo 2 clusters facilitan enormemente la interpretación - probablemente representan dos estados operacionales distintos del equipo.

5. **Buena Separación de Anomalías**: El P95-P50 de 0.336 indica que el algoritmo puede distinguir claramente entre comportamientos normales y atípicos.

#### ⚠️ Limitaciones Observadas
1. **Silhouette Score Moderado**: El valor de 0.3269 indica que hay cierto solapamiento entre clusters. Esto sugiere que los datos podrían tener una estructura más compleja que círculos simples.

2. **Solo 2 Clusters**: Aunque es bueno para interpretabilidad, podría estar oversimplificando la estructura real de los datos si existen múltiples modos de operación.

3. **Sensibilidad a Outliers**: Como se observa en las visualizaciones, los outliers pueden afectar la posición de los centroides.

## 🔍 Interpretación de Resultados

### Clusters Identificados
- **Cluster 0 y Cluster 1**: Los dos clusters probablemente representan dos estados operacionales distintos del equipo industrial:
  - Posiblemente: operación normal vs. operación bajo carga
  - O bien: diferentes velocidades de operación
- Cada punto se asigna de forma determinística a exactamente un cluster
- Los centroides representan los "centros" de masa de cada grupo

### Visualizaciones Clave

#### 1. **Gráfica 3D con PCA**
- Muestra clara separación entre los 2 clusters
- Los colores (azul/cian) distinguen los grupos
- PCA captura ~99.4% de la varianza en 3 componentes
- Se observa cierto solapamiento en la región central

#### 2. **Mapa de Anomalías**
- Gradiente de color amarillo-rojo indica el score de anomalía
- Puntos más oscuros (rojos) están más alejados de sus centroides
- Distribución de anomalías es relativamente uniforme espacialmente
- Las anomalías no se concentran en una región específica

### Scores de Anomalía
- **Mayor distancia al centroide** = comportamiento más inusual dentro del cluster
- Score normalizado entre 0-1 para facilitar interpretación
- El percentil 95 (score > ~0.84) marca el umbral de anomalía
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
- Muy rápido y eficiente (9.6s para 518k registros)
- Fácil de entender e implementar
- Escalable a datasets muy grandes
- Funciona bien cuando los clusters son esféricos
- Resultados reproducibles (con semilla fija)
- Excelente para identificar estados operacionales principales
- Bajo consumo de memoria relativo

### ⚠️ Desventajas
- Requiere especificar K a priori
- Asume clusters de forma esférica y tamaño similar
- Sensible a outliers (pueden afectar los centroides)
- Sensible a la inicialización
- No identifica outliers automáticamente
- Puede oversimplificar datos con estructura compleja

## 🎯 Casos de Uso Recomendados

### ✅ Ideal Para:
1. **Identificación de Modos Operacionales**: Perfecto cuando se necesita identificar estados discretos de operación (normal, bajo carga, alta velocidad, etc.)

2. **Análisis Rápido Exploratorio**: Cuando se necesita una primera aproximación rápida a la estructura de los datos.

3. **Datasets Grandes**: Su eficiencia computacional lo hace ideal para millones de registros.

4. **Monitoreo en Tiempo Real**: La velocidad de predicción permite aplicaciones en tiempo real.

5. **Cuando la Interpretabilidad es Crítica**: Los centroides son fáciles de explicar a stakeholders no técnicos.

### ❌ No Recomendado Para:
1. **Detección de Outliers como Objetivo Principal**: K-Means no está diseñado para esto (considerar DBSCAN o Isolation Forest).

2. **Datos con Clusters de Forma Irregular**: Si los clusters son elongados, en forma de anillo, o con densidad variable.

3. **Cuando no se sabe cuántos clusters buscar**: Aunque se puede iterar, DBSCAN es mejor en estos casos.

## 💡 Conclusiones y Recomendaciones

### Para Este Dataset Específico:
1. **K-Means funciona bien**: Con un Calinski-Harabasz de 300k, demuestra que los datos tienen una estructura natural de 2 grupos bien definidos.

2. **Los 2 clusters son significativos**: Probablemente representan dos regímenes operacionales distintos del equipo industrial.

3. **Buena herramienta de detección de anomalías complementaria**: Aunque no es su propósito principal, los scores de distancia al centroide son útiles.

4. **Ganador en la comparación de clustering**: Supera a DBSCAN en 2 de 3 métricas clave.

### Recomendaciones de Implementación:
1. **Usar K-Means para segmentación inicial** del comportamiento del equipo.
2. **Complementar con Isolation Forest** para detección específica de anomalías.
3. **Monitorear la evolución de los centroides** en el tiempo como indicador de degradación.
4. **Establecer alertas** cuando puntos normalmente en un cluster saltan al otro repetidamente.

## 🔄 Comparación con DBSCAN

| Característica | K-Means | DBSCAN |
|----------------|---------|---------|
| Necesita especificar K | ✅ Sí | ❌ No |
| Detecta outliers | ❌ No | ✅ Sí |
| Forma de clusters | Solo esféricos | Cualquier forma |
| Complejidad | O(n*k*i) | O(n²) o O(n log n) |
| Escalabilidad | Para datasets grandes | Para datasets medianos |

## 📚 Referencias

- Lloyd, S. (1982). "Least squares quantization in PCM"
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#k-means

## 🤝 Contribución al Proyecto

Este algoritmo es parte del proyecto de comparación de algoritmos de ML no supervisados para Mantenimiento Predictivo. Los resultados de K-Means se comparan directamente con DBSCAN para determinar el mejor algoritmo de clustering para esta aplicación específica.
