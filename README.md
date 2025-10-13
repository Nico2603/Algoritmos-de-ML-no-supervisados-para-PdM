# Algoritmos de ML No Supervisados para Mantenimiento Predictivo (PdM)

## 🎯 Objetivo del Proyecto

Este proyecto implementa y compara **cuatro algoritmos** de Machine Learning no supervisado aplicados al **Mantenimiento Predictivo**, determinando cuál es el mejor algoritmo de **Clustering** y cuál es el mejor de **Detección de Anomalías** mediante análisis comparativo exhaustivo con **datasets de más de 500,000 registros**.

## 📊 Algoritmos Implementados

### 🔵 Clustering
1. **K-Means** - Clustering por Centroides
2. **DBSCAN** - Density-Based Spatial Clustering

### 🔴 Detección de Anomalías
3. **Isolation Forest** - Detección basada en Árboles
4. **CBLOF** - Cluster-Based Local Outlier Factor

## 📁 Estructura del Proyecto

```
Algoritmos-de-ML-no-supervisados-para-PdM/
│
├── data.csv                           # Dataset centralizado (518,400 registros)
├── config.py                          # Configuración compartida y funciones utilitarias
├── requirements.txt                   # Dependencias del proyecto
├── run_all.ps1                        # Pipeline automatizado de ejecución
├── README.md                          # Este archivo
│
├── clustering/
│   ├── K-means/
│   │   ├── K-means.py
│   │   ├── graficas_KMeans/
│   │   ├── metricas_KMeans/
│   │   └── modelos_entrenados_KMeans/
│   │
│   ├── DBSCAN/
│   │   ├── DBSCAN.py
│   │   ├── graficas_DBSCAN/
│   │   ├── metricas_DBSCAN/
│   │   └── modelos_entrenados_DBSCAN/
│   │
│   └── Comparaciones/
│       ├── comparar_algoritmos.py
│       ├── REPORTE_COMPARACION_CLUSTERING.txt
│       ├── tabla_comparativa.csv
│       └── [Gráficas comparativas]
│
└── deteccion_anomalias/
    ├── Isolation Forest/
    │   ├── Isolation Forest.py
    │   ├── graficas_IForest/
    │   ├── metricas_IForest/
    │   └── modelos_entrenados_IForest/
    │
    ├── CBLOF/
    │   ├── CBLOF.PY
    │   ├── graficas_CBLOF/
    │   ├── metricas_CBLOF/
    │   └── modelos_entrenados_CBLOF/
    │
    └── Comparaciones/
        ├── comparar_algoritmos.py
        ├── REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt
        ├── tabla_comparativa.csv
        └── [Gráficas comparativas]
```

## 🏆 Resultados y Conclusiones

### Resumen Ejecutivo

Tras ejecutar y comparar exhaustivamente los 4 algoritmos en un dataset real de **518,400 registros** de acelerómetro, los resultados son claros:

#### 🥇 Ganadores

| Categoría | Algoritmo Ganador | Puntuación | Razón Principal |
|-----------|-------------------|------------|-----------------|
| **Clustering** | **K-Means** ✅ | 3/3 métricas | Excelente separación (Calinski-Harabasz: 266k), 5.8x más rápido |
| **Detección de Anomalías** | **Isolation Forest** ✅ | 2 puntos | Mejor separación de scores (0.3883), 11% más rápido, 30% menos memoria |

### Resultados Detallados por Categoría

#### 🔵 Clustering: K-Means vs DBSCAN

**K-Means Domina Absolutamente**

| Métrica | K-Means | DBSCAN | Ganador |
|---------|---------|--------|---------|
| Silhouette Score | 0.3086 ✅ | N/A ❌ | **K-Means** |
| Calinski-Harabasz | 266,441 ✅ | N/A ❌ | **K-Means** |
| Davies-Bouldin | 1.3107 ✅ | N/A ❌ | **K-Means** |
| Clusters | 2 (interpretable) | 1 (no clustering real) | **K-Means** |
| Anomalías Detectadas | 5.00% | 0.00% | **K-Means** |
| Tiempo | 10.28s ✅ | 59.44s ❌ | **K-Means** |
| Memoria | 184.76 MB ✅ | 193.94 MB | **K-Means** |

**Conclusión Clustering**: 
- ✅ **Usar K-Means** para datos de acelerómetro - rendimiento superior en todas las métricas
- ❌ **DBSCAN no es viable** en este dataset (no logró clustering significativo, detectó solo 1 cluster)

#### 🔴 Detección de Anomalías: Isolation Forest vs CBLOF

**Isolation Forest es Superior**

| Métrica | Isolation Forest | CBLOF | Ganador |
|---------|------------------|-------|---------|
| Separación P95-P50 | 0.3883 ✅ | 0.3643 | **Isolation Forest** |
| Score Promedio | 0.3842 ✅ | 0.4802 | **Isolation Forest** |
| Tiempo | 12.53s ✅ | 13.99s | **Isolation Forest** |
| Memoria | 64.13 MB ✅ | 92.59 MB | **Isolation Forest** |
| Anomalías | 10.00% | 10.00% | Empate |

**Conclusión Detección de Anomalías**:
- ✅ **Usar Isolation Forest** como primera opción - mejor separación y más eficiente
- ✅ CBLOF es válido pero inferior en todas las métricas clave (separación, tiempo, memoria)

### 💡 Recomendaciones Finales para Mantenimiento Predictivo

#### Estrategia Recomendada

1. **Para Segmentación Operacional**: Usar **K-Means**
   - Identifica 2 estados operacionales principales
   - Interpretación clara y directa
   - Velocidad óptima para monitoreo continuo

2. **Para Detección de Anomalías**: Usar **Isolation Forest**
   - Mejor separación entre normal y anómalo
   - Más eficiente computacionalmente
   - No requiere configuración compleja

3. **Estrategia Combinada** (Recomendado):
   - Aplicar K-Means para segmentar modos operacionales
   - Aplicar Isolation Forest dentro de cada segmento para detectar anomalías específicas
   - Monitorear evolución de centroides como indicador de degradación

#### Implementación en Producción

```python
# 1. Segmentar con K-Means
estado_operacional = kmeans.predict(datos_nuevos)

# 2. Detectar anomalías con Isolation Forest
score_anomalia = isolation_forest.score_samples(datos_nuevos)

# 3. Alertar si:
#    - Anomalía detectada (score < umbral)
#    - Cambio frecuente de estado operacional
#    - Desplazamiento de centroides
```

### 📊 Métricas de Rendimiento Global

| Algoritmo | Tiempo (s) | Memoria (MB) | Eficiencia | Uso Recomendado |
|-----------|------------|--------------|------------|-----------------|
| K-Means | 10.28 | 184.76 | ⭐⭐⭐⭐⭐ | **Clustering principal** |
| Isolation Forest | 12.53 | 64.13 | ⭐⭐⭐⭐⭐ | **Detección de anomalías principal** |
| CBLOF | 13.99 | 92.59 | ⭐⭐⭐ | Alternativa detección anomalías |
| DBSCAN | 59.44 | 193.94 | ⭐ | No recomendado para este dataset |

**Dataset**: 518,400 registros | 4 características (acceleration_x, acceleration_y, acceleration_z, magnitud_aceleracion)

## 🚀 Guía de Uso Rápida

### Opción 1: Pipeline Automatizado (Recomendado)

Ejecuta todos los algoritmos y comparaciones con un solo comando:

```powershell
# Desde PowerShell (Windows)
.\run_all.ps1

# O desde cualquier ubicación
powershell -ExecutionPolicy Bypass -File "ruta\al\proyecto\run_all.ps1"
```

**El pipeline ejecuta automáticamente:**
1. ✅ Verifica entorno virtual y dependencias
2. ✅ Ejecuta K-Means y DBSCAN (Clustering)
3. ✅ Ejecuta Isolation Forest y CBLOF (Detección de Anomalías)
4. ✅ Genera comparaciones de Clustering
5. ✅ Genera comparaciones de Detección de Anomalías
6. ✅ Crea reportes finales y logs detallados

**Logs generados:**
- `ejecucion_YYYY-MM-DD_HH-mm-ss.log` - Log completo con timestamps

### Opción 2: Ejecución Manual

#### Clustering

```bash
# K-Means
cd clustering/K-means
python K-means.py

# DBSCAN
cd clustering/DBSCAN
python DBSCAN.py
```

#### Detección de Anomalías

```bash
# Isolation Forest
cd deteccion_anomalias/"Isolation Forest"
python "Isolation Forest.py"

# CBLOF
cd deteccion_anomalias/CBLOF
python CBLOF.PY
```

#### Comparaciones

```bash
# Comparar Clustering
cd clustering/Comparaciones
python comparar_algoritmos.py

# Comparar Detección de Anomalías
cd deteccion_anomalias/Comparaciones
python comparar_algoritmos.py
```

### Outputs Generados por las Comparaciones

#### Comparación de Clustering

**Archivos generados:**
- `comparacion_visual_3d.png` - Comparación lado a lado 3D
- `comparacion_metricas_barras.png` - Gráfico de barras comparativo
- `comparacion_rendimiento.png` - Comparación de tiempo y memoria
- `tabla_comparativa.csv` - Tabla de métricas
- `REPORTE_COMPARACION_CLUSTERING.txt` - Reporte completo con ganador

#### Comparación de Detección de Anomalías

**Archivos generados:**
- `comparacion_visual_scores.png` - Comparación de distribución de scores
- `comparacion_visual_3d.png` - Comparación 3D lado a lado
- `comparacion_metricas_barras.png` - Gráfico de barras comparativo
- `comparacion_rendimiento.png` - Comparación de tiempo y memoria
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

## 📦 Instalación y Requisitos

### Requisitos del Sistema
- Python 3.9 o superior
- Windows (script automatizado optimizado para PowerShell)

### Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd Algoritmos-de-ML-no-supervisados-para-PdM
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
```

3. **Activar entorno virtual**
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Dependencias Principales

El archivo `requirements.txt` incluye:

```
numpy>=1.24.0,<3.0.0                  # Operaciones numéricas
pandas>=2.0.0,<3.0.0                  # Manipulación de datos
scikit-learn>=1.3.0,<2.0.0            # Algoritmos ML
matplotlib>=3.7.0,<4.0.0              # Visualizaciones
pyod>=1.1.0,<3.0.0                    # Detección de anomalías (CBLOF)
numba>=0.56.0,<1.0.0                  # Optimización (requerido por PyOD)
joblib>=1.3.0,<2.0.0                  # Persistencia de modelos
h5py>=3.9.0,<4.0.0                    # Almacenamiento de modelos
scipy>=1.11.0,<2.0.0                  # Utilidades científicas
kneed>=0.8.0                          # Detección de punto óptimo (método del codo)
```

## ⚡ Optimizaciones de Rendimiento

### Muestreo Estratégico Unificado
- **Optimización de parámetros**: 5,000 muestras consistentes entre todos los algoritmos
- **Visualización**: 3,000 puntos para gráficas con reproducibilidad garantizada
- **Cálculo de Silhouette**: 5,000 puntos muestreados para reducir complejidad O(n²)
- **Seeds consistentes**: Semilla fija (42) aplicada en TODOS los muestreos aleatorios

### Gráficas 3D Mejoradas
- **Corrección de aspecto**: Ejes proporcionalmente escalados
- **Rangos normalizados**: Evita deformación visual
- **Ángulos optimizados**: Vista inicial óptima (elev=20, azim=45)
- **Colores consistentes**: Esquema unificado entre algoritmos para comparaciones justas

### Parámetros Optimizados
- **K-Means**: K máximo reducido a 6, detección de outliers con percentil 95
- **DBSCAN**: Grid simplificado, optimización con NearestNeighbors
- **Isolation Forest**: n_estimators reducido, scores normalizados
- **CBLOF**: Grid simplificado, scores normalizados de PyOD

## 🚀 Características del Proyecto

### Pipeline Automatizado
- **✅ Script PowerShell `run_all.ps1`**: Ejecuta todos los algoritmos y comparaciones automáticamente
- **✅ Verificación de entorno**: Valida entorno virtual y dependencias antes de ejecutar
- **✅ Instalación automática**: Instala dependencias faltantes automáticamente
- **✅ Logs detallados**: Genera archivos de log con timestamps para seguimiento completo
- **✅ Validación de outputs**: Verifica que todos los archivos esperados fueron generados
- **✅ Manejo de errores**: Continúa con otros algoritmos si uno falla

### Configuración Centralizada
- **✅ `config.py`**: Archivo centralizado con todas las constantes globales y funciones compartidas
- **✅ `data.csv`**: Dataset único compartido (518,400 registros) en la raíz del proyecto
- **✅ Funciones compartidas**: 
  - `normalizar_scores_min_max()` - Normalización de scores [0, 1]
  - `validar_datos_entrada()` - Validación de datos con warnings
  - `aplicar_pca_consistente()` - PCA reproducible
  - `muestrear_datos_consistente()` - Muestreo con seed fija
- **✅ Constantes unificadas**: Seeds, tamaños de muestra, colores, parámetros de visualización
- **✅ Optimización de métricas**: `SILHOUETTE_SAMPLE = 5000` para cálculo eficiente

### Estandarización y Comparabilidad
- **✅ Normalización de scores**: Todos los algoritmos normalizan scores al rango [0, 1]
- **✅ Muestreo unificado**: 5,000 muestras para optimización, 3,000 para visualización
- **✅ Seeds consistentes**: Reproducibilidad garantizada con `RANDOM_STATE = 42`
- **✅ Nombres de archivos**: Estandarizados (`anomaly_scores.csv`, `metrics.csv`, `output.log`)
- **✅ Formato CSV unificado**: Mismas columnas en todos los algoritmos para comparación directa

### Tracking de Rendimiento
- **✅ Tiempo de ejecución**: Medición precisa con `time.time()`
- **✅ Uso de memoria**: Tracking con `tracemalloc` (memoria máxima en MB)
- **✅ Métricas extendidas**: CSV incluye `tiempo_ejecucion_s` y `memoria_max_mb`
- **✅ Comparaciones automáticas**: Scripts generan gráficos de rendimiento comparativo

### Visualizaciones Consistentes
- **✅ Colores estandarizados**: 
  - Clustering: `tab10` colormap para clusters
  - Detección de anomalías: Azul (`#1f77b4`) normal, Rojo (`#d62728`) anomalía
- **✅ Tamaños de puntos**: Normal (10), Anomalía (25), Ruido DBSCAN (30)
- **✅ Títulos informativos**: Incluyen número de clusters, anomalías y porcentajes
- **✅ Grids optimizados**: Alpha 0.3, linestyle '--', linewidth 0.5
- **✅ Ángulos 3D**: elev=20°, azim=45° para mejor perspectiva
- **✅ PCA consistente**: Visualizaciones 3D con reducción dimensional reproducible

### Validación y Robustez
- **✅ Validación de datos**: Verifica tamaño mínimo, varianza y outliers extremos
- **✅ Warnings informativos**: Alerta sobre outliers extremos (>10 std)
- **✅ Detección de outliers en K-Means**: Usando percentil 95
- **✅ Batching en DBSCAN**: Procesamiento eficiente para datasets grandes
- **✅ Métricas seguras**: Validación de NaN/None en comparaciones
- **✅ Manejo de errores**: Los scripts continúan ejecutándose ante errores no críticos

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

## 📝 Configuración y Archivos Clave

### `config.py` - Configuración Centralizada

Archivo central que unifica toda la configuración del proyecto:

**Constantes Globales:**
```python
RANDOM_STATE = 42                     # Seed para reproducibilidad
SAMPLE_OPT = 5000                     # Muestras para optimización
SAMPLE_VIS = 3000                     # Muestras para visualización
SILHOUETTE_SAMPLE = 5000              # Muestras para Silhouette Score
N_JOBS = 2                            # Paralelismo moderado
RUTA_DATOS_COMPARTIDA = 'data.csv'   # Dataset centralizado
```

**Parámetros de Visualización:**
```python
CMAP_CLUSTERING = 'tab10'            # Colormap para clustering
COLOR_NORMAL = '#1f77b4'             # Azul para datos normales
COLOR_ANOMALIA = '#d62728'           # Rojo para anomalías
VIEW_ELEV = 20                       # Elevación para gráficos 3D
VIEW_AZIM = 45                       # Azimut para gráficos 3D
```

**Funciones Compartidas:**
- `normalizar_scores_min_max()`: Normaliza scores al rango [0, 1]
- `validar_datos_entrada()`: Valida tamaño, varianza y outliers
- `aplicar_pca_consistente()`: PCA reproducible con visualización de varianza
- `muestrear_datos_consistente()`: Muestreo con seed fija

### `run_all.ps1` - Pipeline Automatizado

Script PowerShell que automatiza todo el proceso:

**Funcionalidades:**
1. Verifica entorno virtual en `.venv/`
2. Valida e instala dependencias faltantes
3. Ejecuta los 4 algoritmos en orden
4. Genera comparaciones automáticas
5. Valida outputs generados
6. Crea logs detallados con timestamps

**Módulos Verificados:**
- numpy, pandas, sklearn, matplotlib
- pyod, numba, joblib, h5py, scipy

**Logs Generados:**
- `ejecucion_YYYY-MM-DD_HH-mm-ss.log`
- Incluye timestamps, códigos de salida y errores

### Documentación

- **Scripts de comparación**: Generan reportes automáticos con análisis completo
- **Métricas estandarizadas**: Formato CSV consistente para análisis adicional
- **Reportes finales**: Identifican ganadores con análisis detallado

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

## 📊 Dataset

### Características del Dataset

**Archivo:** `data.csv` (centralizado en la raíz del proyecto)

**Tamaño:** 518,400 registros

**Características:**
- `acceleration_x`: Aceleración en eje X (m/s²)
- `acceleration_y`: Aceleración en eje Y (m/s²)
- `acceleration_z`: Aceleración en eje Z (m/s²)
- `fecha`: Timestamp de la medición

**Característica Derivada:**
- `magnitud_aceleracion`: Magnitud vectorial calculada como √(x² + y² + z²)

**Aplicación:**
- Datos de vibración de acelerómetro
- Monitoreo de condición de maquinaria
- Mantenimiento Predictivo (PdM)

**Preprocesamiento:**
- Normalización MinMax al rango [0, 1]
- PCA para visualización 3D
- Detección de outliers extremos (>10 std)

## 🤝 Flujo de Trabajo Recomendado

### Para Análisis Completo

1. **Configurar entorno**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Ejecutar pipeline automatizado**
   ```powershell
   .\run_all.ps1
   ```

3. **Revisar reportes generados**
   - `clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt`
   - `deteccion_anomalias\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt`

4. **Analizar visualizaciones**
   - Gráficas 3D comparativas
   - Distribución de scores
   - Métricas de rendimiento

5. **Seleccionar mejores algoritmos** según reportes

6. **Implementar en producción** usando los ganadores:
   - K-Means para clustering
   - Isolation Forest para detección de anomalías

### Para Desarrollo e Iteración

1. **Ejecutar algoritmos individuales** según necesidad
2. **Modificar parámetros en `config.py`** para experimentos
3. **Re-ejecutar comparaciones** después de cambios
4. **Validar mejoras** con métricas estandarizadas

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

---

## 🚀 Inicio Rápido

### Método Automatizado (Recomendado)

```powershell
# 1. Configurar entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Ejecutar pipeline completo
.\run_all.ps1

# 3. Revisar reportes generados
# - clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt
# - deteccion_anomalias\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt
```

### Método Manual (Paso a Paso)

```bash
# 1. Configurar entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Ejecutar K-Means
cd clustering/K-means
python K-means.py

# 3. Ejecutar DBSCAN
cd ../DBSCAN
python DBSCAN.py

# 4. Comparar clustering
cd ../Comparaciones
python comparar_algoritmos.py

# 5. Ejecutar Isolation Forest
cd ../../deteccion_anomalias/"Isolation Forest"
python "Isolation Forest.py"

# 6. Ejecutar CBLOF
cd ../CBLOF
python CBLOF.PY

# 7. Comparar detección de anomalías
cd ../Comparaciones
python comparar_algoritmos.py
```

## 📊 Resultados Esperados

Al final del proceso tendrás:

### Modelos Entrenados (4)
- ✅ `clustering/K-means/modelos_entrenados_KMeans/`
  - `kmeans_model.pkl` - Modelo K-Means
  - `scaler.pkl` - Escalador MinMax
- ✅ `clustering/DBSCAN/modelos_entrenados_DBSCAN/`
  - `dbscan_model.pkl` - Modelo DBSCAN
  - `scaler.pkl` - Escalador MinMax
- ✅ `deteccion_anomalias/Isolation Forest/modelos_entrenados_IForest/`
  - `isolation_forest_model.pkl` - Modelo Isolation Forest
  - `scaler.pkl` - Escalador MinMax
- ✅ `deteccion_anomalias/CBLOF/modelos_entrenados_CBLOF/`
  - `cblof_model.pkl` - Modelo CBLOF
  - `scaler.pkl` - Escalador MinMax
  - `pca.pkl` - Modelo PCA

### Métricas Detalladas
- ✅ Archivos `metrics.csv` para cada algoritmo
- ✅ Archivos `metrics.txt` con métricas legibles
- ✅ Archivos `anomaly_scores.csv` con scores de cada punto
- ✅ Logs de ejecución `output.log`

### Visualizaciones
- ✅ Gráficas 3D con PCA
- ✅ Comparaciones lado a lado
- ✅ Gráficos de barras de métricas
- ✅ Gráficos de rendimiento (tiempo/memoria)

### Reportes Finales
- ✅ **Ganador de clustering** identificado en `REPORTE_COMPARACION_CLUSTERING.txt`
- ✅ **Ganador de detección de anomalías** identificado en `REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt`
- ✅ Análisis completo con ventajas/desventajas de cada algoritmo
- ✅ Recomendaciones para producción

**¡El mejor algoritmo de cada categoría estará claramente identificado en los reportes!**

## ⚠️ Troubleshooting

### Error: Entorno Virtual No Encontrado

```bash
# Crear entorno virtual
python -m venv .venv

# Activar
.\.venv\Scripts\Activate.ps1  # PowerShell
# o
.venv\Scripts\activate.bat     # CMD
```

### Error: Dependencias Faltantes

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# O instalar manualmente
pip install numpy pandas scikit-learn matplotlib pyod numba joblib h5py scipy kneed
```

### Error: Permiso de Ejecución de Scripts (PowerShell)

```powershell
# Permitir ejecución de scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# O ejecutar con bypass
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

### Error: Memoria Insuficiente

Si encuentras errores de memoria:
1. Reduce `SAMPLE_OPT` y `SAMPLE_VIS` en `config.py`
2. Ejecuta algoritmos individuales en lugar del pipeline completo
3. Cierra otras aplicaciones para liberar RAM

### Error: Dataset No Encontrado

Verifica que `data.csv` esté en la raíz del proyecto:
```bash
ls data.csv  # Linux/Mac
dir data.csv # Windows
```

### Visualizaciones No Se Muestran

El proyecto usa backend `Agg` (sin GUI) para compatibilidad. Las imágenes se guardan automáticamente en las carpetas `graficas_*`.

## 📚 Referencias y Recursos

### Algoritmos Implementados
- **K-Means**: Clustering particional basado en centroides
- **DBSCAN**: Clustering basado en densidad (Density-Based Spatial Clustering of Applications with Noise)
- **Isolation Forest**: Detección de anomalías usando árboles de decisión aislados
- **CBLOF**: Cluster-Based Local Outlier Factor para detección de anomalías

### Bibliotecas Utilizadas
- **scikit-learn**: Algoritmos K-Means, DBSCAN, Isolation Forest
- **PyOD**: Implementación de CBLOF
- **NumPy/Pandas**: Procesamiento de datos
- **Matplotlib**: Visualizaciones

### Métricas
- **Silhouette Score**: Mide cohesión y separación de clusters
- **Calinski-Harabasz**: Ratio de dispersión inter/intra clusters
- **Davies-Bouldin**: Similitud promedio entre clusters
- **Separación P95-P50**: Diferencia entre percentiles para evaluar detección de anomalías

## 🎓 Conclusiones del Proyecto

### Hallazgos Clave

1. **K-Means domina en clustering** para datos de acelerómetro:
   - 5.8x más rápido que DBSCAN
   - Métricas superiores en todas las categorías
   - 2 clusters interpretables vs 1 cluster en DBSCAN

2. **Isolation Forest es superior** en detección de anomalías:
   - 11% más rápido que CBLOF
   - 30% menos uso de memoria
   - Mejor separación de scores (0.3883 vs 0.3643)

3. **DBSCAN no es adecuado** para este tipo de dataset:
   - No logró clustering significativo (solo 1 cluster)
   - 6x más lento que K-Means
   - No proporcionó valor agregado

4. **La estandarización es crucial**:
   - Seeds fijas garantizan reproducibilidad
   - Normalización consistente permite comparación justa
   - Configuración centralizada facilita mantenimiento

### Recomendaciones para Producción

**Sistema Combinado Recomendado:**
```python
# 1. Segmentación operacional con K-Means
cluster = kmeans.predict(nuevos_datos)

# 2. Detección de anomalías con Isolation Forest
score = isolation_forest.score_samples(nuevos_datos)
es_anomalia = score < umbral

# 3. Sistema de alertas
if es_anomalia:
    activar_alerta(prioridad="alta")
elif detectar_cambio_cluster_frecuente():
    activar_alerta(prioridad="media")
```

**Ventajas del Sistema Combinado:**
- Identifica modos operacionales (K-Means)
- Detecta anomalías dentro de cada modo (Isolation Forest)
- Monitorea tendencias de degradación
- Reduce falsos positivos

### Trabajo Futuro

- [ ] Implementar sistema de monitoreo en tiempo real
- [ ] Agregar más algoritmos de comparación (HDBSCAN, LOF, One-Class SVM)
- [ ] Optimizar hiperparámetros con Grid Search exhaustivo
- [ ] Implementar ensemble methods para mayor robustez
- [ ] Desarrollar dashboard interactivo para visualización
- [ ] Integrar con sistemas SCADA industriales

---

## 📄 Licencia y Créditos

**Proyecto:** Algoritmos de ML No Supervisados para Mantenimiento Predictivo (PdM)

**Versión:** 2.0

**Fecha:** Octubre 2025

**Dataset:** 518,400 registros de acelerómetro para mantenimiento predictivo

---

### 🚀 ¡Comienza Ahora!

```powershell
# Configuración inicial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ejecutar todo
.\run_all.ps1

# Revisar ganadores
notepad clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt
notepad deteccion_anomalias\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt
```

**¿Tienes preguntas?** Revisa los reportes generados o los logs detallados en `ejecucion_*.log`

