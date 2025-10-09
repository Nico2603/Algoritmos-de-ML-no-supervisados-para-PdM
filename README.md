# Algoritmos de ML No Supervisados para Mantenimiento Predictivo (PdM)

## 🎯 Objetivo del Proyecto

Este proyecto implementa y compara **cuatro algoritmos** de Machine Learning no supervisado aplicados al **Mantenimiento Predictivo**, determinando cuál es el mejor algoritmo de **Clustering** y cuál es el mejor de **Detección de Anomalías** mediante análisis comparativo exhaustivo con **datasets de más de 500,000 registros**.

### ⚡ Optimizaciones Implementadas

- **Muestreo Estratégico**: Algoritmos optimizados para datasets grandes (500k+ registros)
- **Visualizaciones Mejoradas**: Gráficas 3D corregidas con proporciones adecuadas
- **Rendimiento Optimizado**: Ejecución rápida mediante muestreo inteligente
- **Análisis Automatizado**: Scripts de comparación con reportes completos

## 📊 Algoritmos Implementados

### 🔵 Clustering
1. **DBSCAN** - Density-Based Spatial Clustering
2. **K-Means** - Clustering por Centroides

### 🔴 Detección de Anomalías
3. **CBLOF** - Cluster-Based Local Outlier Factor
4. **Isolation Forest** - Detección basada en Árboles

## 📁 Estructura del Proyecto

```
Algoritmos-de-ML-no-supervisados-para-PdM/
│
├── 1. Clustering/
│   ├── DBSCAN/
│   │   ├── DBSCAN.py
│   │   ├── data.csv
│   │   ├── README.md
│   │   ├── graficas_DBSCAN/
│   │   ├── metricas_DBSCAN/
│   │   └── modelos_entrenados_DBSCAN/
│   │
│   ├── K-means/
│   │   ├── K-means.py
│   │   ├── data.csv
│   │   ├── README.md
│   │   ├── graficas_KMeans/
│   │   ├── metricas_KMeans/
│   │   └── modelos_entrenados_KMeans/
│   │
│   └── Comparaciones/
│       ├── comparar_algoritmos.py
│       └── [Resultados de comparación]
│
├── 2. Detección de Anomalías/
│   ├── CBLOF (Cluster-Based Local Outlier Factor)/
│   │   ├── CBLOF.PY
│   │   ├── data.csv
│   │   ├── README.md
│   │   ├── graficas_CBLOF/
│   │   ├── metricas_CBLOF/
│   │   └── modelos_entrenados_CBLOF/
│   │
│   ├── Isolation Forest/
│   │   ├── Isolation Forest.py
│   │   ├── data.csv
│   │   ├── README.md
│   │   ├── graficas_IForest/
│   │   ├── metricas_IForest/
│   │   └── modelos_entrenados_IForest/
│   │
│   └── Comparaciones/
│       ├── comparar_algoritmos.py
│       └── [Resultados de comparación]
│
├── README.md                          # Este archivo
├── README_COMPARACION.md              # Guía detallada de comparación
└── CAMBIOS_REALIZADOS.md              # Log de cambios
```

## 🚀 Guía de Uso Rápida

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
cd "1. Clustering/Comparaciones"
python comparar_algoritmos.py
```

**Outputs generados:**
- `comparacion_visual_2d.png` - Comparación lado a lado 2D
- `comparacion_visual_3d.png` - Comparación lado a lado 3D
- `comparacion_metricas_barras.png` - Gráfico de barras comparativo
- `comparacion_metricas_radar.png` - Gráfico de radar multidimensional
- `tabla_comparativa.csv` - Tabla de métricas
- `REPORTE_COMPARACION_CLUSTERING.txt` - Reporte completo con ganador

#### Comparar Algoritmos de Detección de Anomalías

```bash
cd "2. Detección de Anomalías/Comparaciones"
python comparar_algoritmos.py
```

**Outputs generados:**
- `comparacion_visual_scores.png` - Comparación de distribución de scores
- `comparacion_visual_3d.png` - Comparación 3D lado a lado
- `comparacion_metricas_barras.png` - Gráfico de barras comparativo
- `comparacion_porcentaje_anomalias.png` - Comparación de porcentajes
- `comparacion_separacion_scores.png` - Comparación de separación
- `tabla_comparativa.csv` - Tabla de métricas
- `REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt` - Reporte completo con ganador

## 📊 Outputs Estandarizados

Todos los algoritmos generan outputs consistentes para facilitar la comparación:

### Para Clustering (DBSCAN y K-Means)

**Gráficas:**
- `clusters_2d_pca.png` - Visualización 2D con PCA
- `clusters_3d_pca.png` - Visualización 3D con PCA

**Métricas:**
- `metrics.txt` - Métricas en formato texto
- `metrics.csv` - Métricas estandarizadas para comparación
- `anomaly_scores.csv` - Scores de todos los puntos

### Para Detección de Anomalías (CBLOF e Isolation Forest)

**Gráficas:**
- `anomaly_scores.png` - Distribución de scores
- `anomalies_3d.png` - Visualización 3D

**Métricas:**
- `metrics.txt` - Métricas en formato texto
- `metrics.csv` - Métricas estandarizadas para comparación
- `anomaly_scores.csv` - Scores de todos los puntos
- `anomalies.csv` - Solo anomalías detectadas

## 📈 Métricas de Comparación y Formato Estandarizado

### Clustering (DBSCAN y K-Means)
- **Silhouette Score** (0-1): Mide separación entre clusters. Mayor es mejor
- **Calinski-Harabasz Score**: Ratio dispersión inter/intra clusters. Mayor es mejor  
- **Davies-Bouldin Index**: Similitud promedio entre clusters. Menor es mejor
- **Número de Clusters**: Identificación automática de clusters óptimos

**Gráficas Generadas:**
- `clusters_2d_pca.png` - Visualización 2D optimizada con PCA
- `clusters_3d_pca.png` - Visualización 3D corregida con proporciones adecuadas

### Detección de Anomalías (CBLOF e Isolation Forest)  
- **Separación de Scores (P95-P50)**: Separación entre anomalías y normales. Mayor es mejor
- **Porcentaje de Anomalías**: % de puntos detectados como anomalías
- **Score Promedio**: Puntuación media de anomalías

**Gráficas Generadas:**
- `anomaly_scores.png` - Distribución de scores de anomalía
- `anomalies_3d.png` - Visualización 3D optimizada de anomalías

### 📊 Formato Estandarizado CSV

Todos los algoritmos generan un archivo `metrics.csv` con columnas consistentes:
```csv
algoritmo,params_json,n_clusters,silhouette_score,calinski_harabasz_score,davies_bouldin_score,pct_anomalias,p95_minus_p50,mean_score
```

## 🏆 Objetivo del Análisis

El objetivo es determinar:

1. **Mejor Algoritmo de Clustering**: DBSCAN o K-Means
2. **Mejor Algoritmo de Detección de Anomalías**: CBLOF o Isolation Forest

Los scripts de comparación analizan múltiples métricas y generan un reporte con el **ganador de cada categoría**.

## 📦 Requisitos

```bash
# Requisitos básicos
pip install numpy pandas matplotlib scikit-learn joblib h5py

# Para CBLOF (PyOD - Python Outlier Detection)
pip install pyod
```

## ⚡ Optimizaciones de Rendimiento

### Muestreo Estratégico
- **Optimización de parámetros**: Datasets reducidos a 8,000-15,000 muestras
- **Visualización**: Muestreo inteligente a 6,000-10,000 puntos para gráficas
- **Reproducibilidad**: Semilla fija (42) para resultados consistentes

### Gráficas 3D Mejoradas
- **Corrección de aspecto**: Ejes proporcionalmente escalados
- **Rangos normalizados**: Evita deformación visual
- **Muestreo para visualización**: Renderizado rápido sin pérdida de información

### Parámetros Optimizados
- **K-Means**: K máximo reducido a 6, muestreo a 15,000 puntos
- **DBSCAN**: Grid simplificado, 4 valores eps, muestreo a 8,000 puntos  
- **Isolation Forest**: n_estimators reducido, muestreo a 10,000 puntos
- **CBLOF**: Grid simplificado, parámetros fijos, muestreo a 8,000 puntos

## 🎓 Características de los Algoritmos

### DBSCAN vs K-Means

| Aspecto | DBSCAN | K-Means |
|---------|--------|---------|
| Especificar K | ❌ No | ✅ Sí |
| Detecta outliers | ✅ Sí | ❌ No |
| Forma de clusters | Cualquiera | Solo esféricos |
| Velocidad | 🐌 Moderada | ⚡ Muy rápido |
| Escalabilidad | ⚠️ Limitada | ✅ Excelente |

### CBLOF vs Isolation Forest

| Aspecto | CBLOF | Isolation Forest |
|---------|-------|------------------|
| Necesita clusters | ✅ Sí | ❌ No |
| Contexto local | ✅ Sí | ❌ No |
| Velocidad | 🐌 Moderada | ⚡ Muy rápido |
| Escalabilidad | ⚠️ Limitada | ✅ Excelente |
| Interpretabilidad | ✅ Alta | ⚠️ Media |

## 📝 Documentación

- **READMEs individuales**: Cada algoritmo tiene documentación detallada en su carpeta
- **Scripts de comparación**: Generan reportes automáticos con análisis completo
- **Métricas estandarizadas**: Formato CSV consistente para análisis adicional

## 🔍 Interpretación de Resultados

### Clustering
El mejor algoritmo es aquel que:
- Maximiza Silhouette Score
- Maximiza Calinski-Harabasz Score
- Minimiza Davies-Bouldin Index

### Detección de Anomalías
El mejor algoritmo es aquel que:
- Maximiza la separación entre anomalías y normales (P95-P50)
- Detecta un porcentaje razonable de anomalías (contexto-dependiente)

## 🤝 Flujo de Trabajo Recomendado

1. **Ejecutar los 4 algoritmos** con el mismo dataset
2. **Revisar outputs individuales** (gráficas y métricas)
3. **Ejecutar comparaciones** en cada categoría
4. **Analizar reportes** generados
5. **Seleccionar los mejores** para aplicación en producción
6. **Implementar sistema de monitoreo** usando los algoritmos ganadores

## 🎯 Aplicación en Mantenimiento Predictivo

Los algoritmos ganadores se pueden usar para:

### Clustering (Ganador)
- Identificar **modos de operación** del equipo
- Detectar **cambios en patrones** de vibración
- Agrupar **condiciones operativas** similares

### Detección de Anomalías (Ganador)
- Detectar **fallos inminentes** antes de que ocurran
- Identificar **comportamientos anómalos** en tiempo real
- Priorizar **mantenimientos** basados en scores de anomalía

## 📄 Licencia

[Especificar licencia del proyecto]

## 👥 Autores

[Tus datos]

## 📅 Última Actualización

Octubre 2025 - **Versión 3.0 - Optimizada**
- ✅ Optimizaciones de rendimiento para datasets grandes (500k+ registros)
- ✅ Gráficas 3D corregidas con proporciones adecuadas  
- ✅ Muestreo estratégico inteligente
- ✅ Documentación consolidada y actualizada

---

## 🚀 Inicio Rápido

```bash
# 1. Ejecutar DBSCAN
cd "1. Clustering/DBSCAN"
python DBSCAN.py

# 2. Ejecutar K-Means
cd "../K-means"
python K-means.py

# 3. Comparar clustering
cd "../Comparaciones"
python comparar_algoritmos.py

# 4. Ejecutar CBLOF
cd "../../2. Detección de Anomalías/CBLOF (Cluster-Based Local Outlier Factor)"
python CBLOF.PY

# 5. Ejecutar Isolation Forest
cd "../Isolation Forest"
python "Isolation Forest.py"

# 6. Comparar detección de anomalías
cd "../Comparaciones"
python comparar_algoritmos.py
```

## 📊 Resultados Esperados

Al final del proceso tendrás:
- ✅ 4 modelos entrenados
- ✅ Métricas detalladas de cada algoritmo
- ✅ Visualizaciones comparativas
- ✅ **Ganador de clustering** identificado
- ✅ **Ganador de detección de anomalías** identificado
- ✅ Reportes completos con análisis

**¡El mejor algoritmo de cada categoría estará claramente identificado en los reportes!**

