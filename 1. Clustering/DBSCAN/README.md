# DBSCAN - Density-Based Spatial Clustering

## 📋 Descripción

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo de clustering basado en densidad que agrupa puntos que están estrechamente empaquetados y marca como outliers los puntos que están solos en regiones de baja densidad.

### Características Principales
- ✅ Detecta clusters de forma arbitraria (no solo esféricos)  
- ✅ Identifica automáticamente outliers (puntos de ruido)
- ✅ No requiere especificar el número de clusters a priori
- ✅ Búsqueda automática de parámetros óptimos (eps y min_samples)
- ✅ Visualizaciones en 2D y 3D usando PCA
- ✅ Genera gráfico k-distance para análisis de parámetros
- ⚠️ Sensible a la elección de parámetros (eps, min_samples)

## 🎯 Objetivo

Identificar grupos naturales en datos de acelerómetro para Mantenimiento Predictivo, detectando automáticamente patrones anómalos sin supervisión.

## 📁 Estructura de Archivos

```
DBSCAN/
├── DBSCAN.py                          # Script principal
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
1. ✅ Carga y preprocesamiento de datos (desde `data.csv` en la raíz del proyecto)
2. ✅ Optimización automática de parámetros (eps, min_samples)
3. ✅ Entrenamiento del modelo
4. ✅ Generación de visualizaciones
5. ✅ Cálculo de métricas de calidad
6. ✅ Detección y guardado de anomalías

**Nota**: El archivo `data.csv` se encuentra centralizado en la raíz del proyecto y es compartido por todos los algoritmos.

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

## ⚙️ Configuración Avanzada **OPTIMIZADA**

Parámetros optimizados para ejecución rápida en `DBSCAN.py`:

```python
# Rango de búsqueda para eps (REDUCIDO)
eps_min = 0.2        # Aumentado de 0.1
eps_max = 1.5        # Reducido de 2.0  
n_eps = 4           # Reducido de 8 para velocidad

# Rango de búsqueda para min_samples
min_samples_min = 3
min_samples_max = 4

# Muestreo optimizado
MAX_MUESTRAS_OPTIMIZACION = 5000   # Optimización de parámetros
MAX_MUESTRAS_VISUALIZATION = 3000  # Visualizaciones
SILHOUETTE_SAMPLE = 5000           # Muestra para cálculo eficiente de Silhouette
```

**Nota**: El parámetro `SILHOUETTE_SAMPLE` optimiza el cálculo del Silhouette Score usando muestreo, reduciendo la complejidad O(n²) para datasets grandes.

## 📊 Resultados Obtenidos

### Parámetros Óptimos Encontrados
- **eps**: 0.0644
- **min_samples**: 6
- **Tiempo de Ejecución**: 29.07 segundos
- **Memoria Utilizada**: 222.13 MB

### Métricas de Calidad del Clustering
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Silhouette Score** | -0.7026 | ⚠️ **Problemático** - Indica fuerte solapamiento entre clusters |
| **Calinski-Harabasz** | 16.62 | Muy bajo - Pobre separación entre clusters |
| **Davies-Bouldin** | 0.9335 | Bueno - Clusters bien definidos (menor es mejor) |
| **Número de Clusters** | 1432 | Muy alto - Fragmentación extrema |
| **Puntos de Ruido** | 17387 (3.35%) | Cantidad razonable de outliers |

### Detección de Anomalías
- **Porcentaje de Anomalías**: 3.35% (puntos clasificados como ruido)
- **Separación P95-P50**: 0.2949
- **Score Promedio**: 0.0379

### ⚠️ Análisis Crítico de Resultados

#### Problemas Identificados

1. **Silhouette Score Negativo Severo (-0.7026)**:
   - **Significado**: Los puntos están, en promedio, más cerca de puntos en otros clusters que de puntos en su propio cluster.
   - **Causa**: Los parámetros óptimos encontrados (`eps=0.0644`, `min_samples=6`) crean micro-clusters muy pequeños y densos.
   - **Impacto**: La calidad del clustering es **muy pobre** para este dataset.

2. **Fragmentación Extrema (1432 Clusters)**:
   - Con 518,400 puntos y 1432 clusters, hay ~362 puntos por cluster en promedio.
   - Esto sugiere que DBSCAN está dividiendo los datos en fragmentos muy pequeños.
   - No captura la estructura macro de los datos.

3. **Discrepancia entre Métricas**:
   - Davies-Bouldin (0.93) sugiere clusters bien definidos
   - Silhouette (-0.70) y Calinski-Harabasz (16.62) indican lo contrario
   - Esta contradicción señala que el clustering no es confiable

4. **Optimización Paradójica**:
   - Durante la optimización con muestra reducida (10,000 puntos), el Silhouette Score era **0.9853** ✅
   - Al aplicar al dataset completo (518,400 puntos), cayó a **-0.7026** ❌
   - Esto indica **sobreajuste severo** a la muestra de optimización

#### ⚠️ Limitaciones Observadas

1. **Escalabilidad Pobre**: 
   - 29.07 segundos (3x más lento que K-Means)
   - 222.13 MB (20% más memoria que K-Means)

2. **Estructura de Datos No Adecuada**:
   - Los datos de acelerómetro no tienen una estructura de densidad variable clara
   - DBSCAN funciona mejor con clusters de diferente densidad (forma de "manchas")

3. **Parámetros Críticos**:
   - La dependencia de `eps` es extremadamente sensible
   - Pequeños cambios en `eps` producen resultados radicalmente diferentes

## 🔍 Interpretación de Resultados

### Visualizaciones Clave

#### 1. **Gráfica 3D con PCA**
- Muestra 83 clusters (en la visualización reducida a 3,000 puntos)
- Muchos micro-clusters pequeños
- Puntos de ruido (marcados con X) dispersos por todo el espacio
- No se observa una estructura natural de clusters separados

#### 2. **Mapa de Anomalías**
- Los scores de anomalía son generalmente bajos (promedio 0.0379)
- Indica que la mayoría de puntos no son considerados muy anómalos
- Los outliers detectados (3.35%) están distribuidos espacialmente

### Clusters e Interpretación

- **1432 Clusters**: Demasiados para ser interpretables
- **Micro-clustering**: Los clusters son demasiado pequeños para representar modos operacionales
- **Sin Significado Operacional**: Es difícil extraer insights útiles de tantos clusters fragmentados

### Outliers/Anomalías
- **3.35% de outliers** (17,387 puntos) es razonable
- Puntos con `is_outlier = 1` son anomalías detectadas
- `anomaly_score` indica qué tan anómalo es el punto (mayor = más anómalo)
- Sin embargo, la baja separación (P95-P50 = 0.29) indica dificultad para distinguir anomalías claras

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

### ✅ Ventajas (Teóricas)
- Detecta clusters de cualquier forma (no solo círculos/esferas)
- No necesita saber cuántos clusters hay de antemano
- Identifica outliers automáticamente
- Robusto a ruido en los datos
- Excelente para datos con estructura de densidad variable

### ⚠️ Desventajas (Observadas en Este Dataset)
- **Muy sensible** a la elección de eps y min_samples
- Puede tener problemas con clusters de densidad similar (**este caso**)
- **3x más lento** que K-Means (29s vs 10s)
- **20% más memoria** que K-Means
- Rendimiento se degrada en alta dimensionalidad
- **Sobreajuste severo** durante optimización de parámetros
- **Fragmentación extrema** - genera demasiados micro-clusters sin significado

### 🆚 Comparativa K-Means vs DBSCAN en Este Dataset

| Aspecto | K-Means | DBSCAN | Ganador |
|---------|---------|---------|---------|
| **Silhouette Score** | 0.3269 ✅ | -0.7026 ❌ | **K-Means** |
| **Calinski-Harabasz** | 300,079 ✅ | 16.62 ❌ | **K-Means** |
| **Davies-Bouldin** | 1.2372 | 0.9335 ✅ | **DBSCAN** |
| **Número de Clusters** | 2 ✅ (interpretable) | 1432 ❌ (fragmentado) | **K-Means** |
| **Velocidad** | 9.6s ✅ | 29.1s ❌ | **K-Means** |
| **Memoria** | 184.78 MB ✅ | 222.13 MB ❌ | **K-Means** |
| **Interpretabilidad** | Alta ✅ | Muy baja ❌ | **K-Means** |

**Resultado**: K-Means **domina claramente** en este dataset específico.

## 🎯 Casos de Uso Recomendados

### ✅ DBSCAN es Ideal Para:
1. **Datos con Clusters de Diferentes Formas**: Anillos, formas alargadas, distribuciones no-esféricas.
2. **Datos con Variabilidad de Densidad**: Cuando algunos clusters son densos y otros son dispersos.
3. **Detección de Outliers Espaciales**: Cuando los outliers forman grupos aislados.
4. **Datos Geoespaciales**: GPS, mapas, ubicaciones con agrupamientos naturales.
5. **Datasets Pequeños a Medianos**: < 100,000 puntos idealmente.

### ❌ DBSCAN NO es Adecuado Para (Este Caso):
1. **Datos de Acelerómetro con Densidad Uniforme**: Como en este dataset ❌
2. **Datasets Grandes**: > 500,000 puntos (complejidad O(n²)) ❌
3. **Cuando se Necesita Velocidad**: DBSCAN es 3x más lento ❌
4. **Cuando se Requiere Interpretabilidad**: 1432 clusters no son interpretables ❌
5. **Datos en Alta Dimensionalidad**: 4+ dimensiones sin reducción ❌

## 💡 Conclusiones y Recomendaciones

### Para Este Dataset Específico:

#### ❌ DBSCAN NO es Recomendado
1. **Silhouette Score catastrófico (-0.70)**: Indica que el clustering no captura la estructura real de los datos.
2. **Fragmentación extrema**: 1432 clusters no tienen valor práctico para mantenimiento predictivo.
3. **Sobreajuste severo**: Los parámetros óptimos en muestra pequeña no generalizan al dataset completo.
4. **Peor en todas las métricas clave** excepto Davies-Bouldin (que es inconsistente con las demás).

#### ✅ Usar K-Means en su Lugar
- **2/3 métricas superiores**
- **Mucho más rápido** (3x)
- **Más interpretable** (2 clusters significativos)
- **Menor consumo de recursos**

### Por Qué DBSCAN Falló Aquí:

1. **Estructura de Datos**: Los datos de acelerómetro tienen densidad relativamente uniforme, no la variabilidad que DBSCAN necesita.

2. **Escalabilidad**: Con 518,400 puntos, la complejidad O(n²) de DBSCAN se hace evidente.

3. **Parámetros Hipersensibles**: Un `eps` muy pequeño (0.0644) creó fragmentación; uno más grande habría fusionado todo en un cluster gigante.

4. **Muestreo No Representativo**: La muestra de 10,000 puntos usada para optimización no capturó la verdadera estructura del dataset completo.

### Recomendaciones:

1. **Para Clustering**: Usar **K-Means** exclusivamente en este dataset.

2. **Para Detección de Anomalías**: Usar **Isolation Forest** o **CBLOF**, no DBSCAN.

3. **Si se Insiste en DBSCAN**:
   - Aumentar muestra de optimización a 50,000+ puntos
   - Expandir rango de búsqueda de `eps` (0.5 - 2.0)
   - Reducir `min_samples` a 2-3
   - Aplicar reducción de dimensionalidad más agresiva (PCA a 2-3 componentes)

4. **Para Otros Datasets**: DBSCAN puede funcionar bien si:
   - Hay clara variabilidad de densidad
   - Los clusters tienen formas no-esféricas
   - El dataset es < 100,000 puntos
   - Se puede invertir tiempo en tuning manual de parámetros

## 📚 Referencias

- Ester, M., Kriegel, H.P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise"
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html#dbscan

## 🤝 Contribución al Proyecto

Este algoritmo es parte del proyecto de comparación de algoritmos de ML no supervisados para Mantenimiento Predictivo. Los resultados de DBSCAN se comparan directamente con K-Means para determinar el mejor algoritmo de clustering para esta aplicación específica.
