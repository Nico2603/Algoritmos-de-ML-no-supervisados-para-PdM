# CBLOF - Cluster-Based Local Outlier Factor

## 📋 Descripción

CBLOF es un algoritmo de detección de anomalías que utiliza clustering para proporcionar contexto local. Clasifica las anomalías basándose en qué tan lejos están los puntos de su cluster más cercano, considerando tanto el tamaño como la distancia de los clusters.

### Características Principales
- ✅ Utiliza clustering para contexto local
- ✅ Bueno para detectar anomalías locales dentro de clusters
- ✅ Considera la densidad y estructura de los datos
- ✅ Puede identificar diferentes tipos de anomalías
- ⚠️ Requiere especificar número de clusters
- ⚠️ Más sensible a la elección de parámetros

## 🎯 Objetivo

Detectar comportamientos anómalos en datos de acelerómetro para Mantenimiento Predictivo, identificando puntos que se desvían significativamente de los patrones de operación normal mediante análisis basado en clusters.

## 📁 Estructura de Archivos

```
CBLOF (Cluster-Based Local Outlier Factor)/
├── CBLOF.PY                           # Script principal
├── data.csv                           # Dataset de entrada
├── README.md                          # Este archivo
│
├── graficas_CBLOF/                    # Gráficas generadas
│   ├── anomaly_scores.png            # Distribución de scores de anomalía
│   └── anomalies_3d.png              # Visualización 3D de anomalías
│
├── metricas_CBLOF/                    # Métricas y resultados
│   ├── output.log                    # Log de ejecución completo
│   ├── metrics.txt                   # Métricas en formato texto
│   ├── metrics.csv                   # Métricas estandarizadas (para comparación)
│   ├── anomaly_scores.csv            # Scores de todos los puntos
│   └── anomalies.csv                 # Solo anomalías detectadas
│
└── modelos_entrenados_CBLOF/        # Modelos guardados
    ├── cblof_model.pkl              # Modelo en formato Pickle
    ├── cblof_model.h5               # Modelo en formato HDF5
    ├── scaler.pkl                   # Escalador para inferencia
    └── pca.pkl                      # PCA para inferencia
```

## 🚀 Uso

### Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn pyod joblib h5py
```

**Nota**: CBLOF requiere la librería `pyod` (Python Outlier Detection).

### Ejecución

```bash
python CBLOF.PY
```

El script ejecutará automáticamente:
1. ✅ Carga y preprocesamiento de datos
2. ✅ Reducción de dimensionalidad con PCA
3. ✅ Optimización automática de parámetros
4. ✅ Entrenamiento del modelo
5. ✅ Detección de anomalías
6. ✅ Generación de visualizaciones
7. ✅ Cálculo y guardado de métricas

## 📊 Métricas Generadas

### Métricas de Optimización
- **Separación de Scores (P95-P50)**: Qué tan bien se distinguen anomalías de normales (mayor es mejor)
- **Desviación estándar de scores**: Variabilidad en las puntuaciones
- **Media de scores**: Score promedio de todos los puntos

### Métricas de Detección
- **Número de anomalías detectadas**: Cantidad absoluta de anomalías
- **Porcentaje de anomalías**: % del total de datos
- **Media de puntuaciones de anomalía**: Score promedio

### Estadísticas de Scores
- Score mínimo, máximo y desviación estándar

## 📈 Visualizaciones

### 1. Distribución de Scores (`anomaly_scores.png`)
- Histograma de scores de anomalía
- Permite ver la separación entre normales y anomalías
- Mayor score = mayor probabilidad de anomalía

### 2. Visualización 3D (`anomalies_3d.png`)
- Puntos normales en azul
- Anomalías en rojo
- Basada en las 3 componentes principales de PCA
- Facilita la interpretación visual

## 🔧 Parámetros del Algoritmo

El script optimiza automáticamente mediante grid search:

- **n_clusters**: Número de clusters para contexto local (5, 10)
- **alpha**: Peso relativo de clusters pequeños vs grandes (0.7, 0.9)
- **beta**: Umbral para clasificar clusters como pequeños (5, 7)
- **use_weights**: Si se usan pesos en el cálculo (True, False)
- **contamination**: Proporción esperada de anomalías (default: 0.1)

## 📄 Archivos de Salida

### `metrics.csv` (Para Comparación)
Formato estandarizado con columnas:
- algoritmo, params_json, n_clusters, silhouette_score (None), calinski_harabasz_score (None), davies_bouldin_score (None), pct_anomalias, p95_minus_p50, mean_score

### `anomaly_scores.csv`
Todos los puntos con:
- fecha, acceleration_x, acceleration_y, acceleration_z, is_outlier, anomaly_score

**is_outlier**: 0 = normal, 1 = anomalía

### `anomalies.csv`
Solo puntos clasificados como anomalías (`is_outlier = 1`)

## ⚙️ Configuración Avanzada

Para modificar parámetros de búsqueda, edita en `CBLOF.PY`:

```python
PARAM_GRID = {
    'n_clusters': [5, 10],      # Número de clusters internos
    'alpha': [0.7, 0.9],        # Peso de clusters grandes
    'beta': [5, 7],             # Umbral para clusters pequeños
    'use_weights': [True, False]
}

CONTAMINACION_DEFAULT = 0.1     # Proporción esperada de anomalías
```

Para datasets grandes:

```python
MAX_MUESTRAS_OPTIMIZACION = 50000  # Límite para optimización
```

## 🔍 Interpretación de Resultados

### Clasificación Binaria
- **is_outlier = 0**: Punto normal (comportamiento esperado)
- **is_outlier = 1**: Anomalía detectada (comportamiento inusual)

### Anomaly Score
- Valor continuo que indica el grado de anomalía
- Mayor score = más anómalo
- Útil para priorizar investigaciones

### Separación (P95-P50)
- **Métrica crítica** para evaluar calidad de detección
- Mayor valor = mejor distinción entre anomalías y normales
- Indica confiabilidad del modelo

## 📊 Comparación con Otros Algoritmos

Para comparar CBLOF con Isolation Forest:

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
- **Contexto local**: Usa clustering para entender estructura local
- **Anomalías contextuales**: Detecta puntos anómalos dentro de subgrupos
- **Interpretabilidad**: Basado en distancias a clusters
- **Flexibilidad**: Múltiples parámetros para ajustar sensibilidad

### ⚠️ Desventajas
- **Necesita n_clusters**: Requiere especificar número de clusters
- **Sensible a parámetros**: Múltiples hiperparámetros a optimizar
- **Complejidad computacional**: Más lento que Isolation Forest
- **Escalabilidad limitada**: Puede ser lento en datasets muy grandes

## 🔄 Comparación con Isolation Forest

| Característica | CBLOF | Isolation Forest |
|----------------|-------|------------------|
| Necesita clusters | ✅ Sí | ❌ No |
| Contexto local | ✅ Sí | ❌ No |
| Velocidad | 🐌 Moderada | ⚡ Rápido |
| Escalabilidad | ⚠️ Limitada | ✅ Excelente |
| Interpretabilidad | ✅ Alta | ⚠️ Media |
| Anomalías locales | ✅ Excelente | ⚠️ Bueno |

## 🧮 Cómo Funciona CBLOF

1. **Clustering inicial**: Agrupa los datos en K clusters
2. **Clasificación de clusters**:
   - Clusters **grandes**: Representan patrones normales
   - Clusters **pequeños**: Pueden contener anomalías
3. **Cálculo de scores**:
   - Puntos en clusters grandes: Score basado en distancia al centroide
   - Puntos en clusters pequeños: Score basado en distancia al cluster grande más cercano
4. **Detección**: Puntos con scores más altos son anomalías

## 📚 Referencias

- He, Z., Xu, X., & Deng, S. (2003). "Discovering cluster-based local outliers"
- PyOD Documentation: https://pyod.readthedocs.io/
- Zhao, Y., Nasrullah, Z., & Li, Z. (2019). "PyOD: A Python Toolbox for Scalable Outlier Detection"

## 🤝 Contribución al Proyecto

Este algoritmo es parte del proyecto de comparación de algoritmos de ML no supervisados para Mantenimiento Predictivo. Los resultados de CBLOF se comparan directamente con Isolation Forest para determinar el mejor algoritmo de detección de anomalías para esta aplicación específica.

---

**Última actualización**: Octubre 2025  
**Versión**: 2.0
