# Isolation Forest - Detección de Anomalías **OPTIMIZADA**

## 📋 Descripción

Isolation Forest es un algoritmo de detección de anomalías basado en ensemble de árboles de decisión. **Versión optimizada** para datasets grandes (500k+ registros) con muestreo estratégico y visualizaciones 3D corregidas.

### ⚡ Optimizaciones Implementadas
- 🚀 **Muestreo estratégico**: Reducción inteligente a 10,000 muestras para optimización
- 🎯 **Visualización optimizada**: Muestreo a 8,000 puntos para gráficas rápidas
- 📊 **Gráficas 3D corregidas**: Proporciones adecuadas, sin deformación visual
- ⚡ **Parámetros reducidos**: Grid simplificado para ejecución rápida

### Características Principales
- ✅ **Basado en ensemble de árboles** de decisión
- ✅ **Muy rápido y altamente escalable** (optimizado)
- ✅ **Excelente con alta dimensionalidad** 
- ✅ **No requiere clustering previo**
- ✅ **Eficiente en memoria**
- ⚠️ Puede ser sensible al parámetro de contaminación

## 🎯 Objetivo

Detectar comportamientos anómalos en datos de acelerómetro para Mantenimiento Predictivo mediante aislamiento eficiente de puntos que se desvían del comportamiento normal, utilizando la profundidad de partición en árboles aleatorios.

## 📁 Estructura de Archivos

```
Isolation Forest/
├── Isolation Forest.py                # Script principal
├── data.csv                           # Dataset de entrada
├── README.md                          # Este archivo
│
├── graficas_IForest/                  # Gráficas generadas
│   ├── anomaly_scores.png            # Distribución de scores de anomalía
│   └── anomalies_3d.png              # Visualización 3D de anomalías
│
├── metricas_IForest/                  # Métricas y resultados
│   ├── output.log                    # Log de ejecución completo
│   ├── metrics.txt                   # Métricas en formato texto
│   ├── metrics.csv                   # Métricas estandarizadas (para comparación)
│   ├── anomaly_scores.csv            # Scores de todos los puntos
│   └── anomalies.csv                 # Solo anomalías detectadas
│
└── modelos_entrenados_IForest/       # Modelos guardados
    ├── isolation_forest_model.pkl    # Modelo en formato Pickle
    ├── isolation_forest_model.h5     # Modelo en formato HDF5
    └── scaler.pkl                    # Escalador para inferencia
```

## 🚀 Uso

### Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn joblib h5py
```

**Nota**: Isolation Forest está incluido en scikit-learn, no requiere librerías adicionales.

### Ejecución

```bash
python "Isolation Forest.py"
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

- **n_estimators**: Número de árboles en el ensemble (100, 150)
- **max_samples**: Muestras para entrenar cada árbol ('auto', 0.8)
- **contamination**: Proporción esperada de anomalías (0.05, 0.1)
- **max_features**: Características para cada split (1.0)

El algoritmo selecciona la mejor combinación basándose en la **separación de scores**.

## 📄 Archivos de Salida

### `metrics.csv` (Para Comparación)
Formato estandarizado con columnas:
- algoritmo, params_json, n_clusters (None), silhouette_score (None), calinski_harabasz_score (None), davies_bouldin_score (None), pct_anomalias, p95_minus_p50, mean_score

### `anomaly_scores.csv`
Todos los puntos con:
- fecha, acceleration_x, acceleration_y, acceleration_z, is_outlier, anomaly_score

**is_outlier**: 0 = normal, 1 = anomalía

### `anomalies.csv`
Solo puntos clasificados como anomalías (`is_outlier = 1`)

## ⚙️ Configuración Avanzada **OPTIMIZADA**

Parámetros optimizados para ejecución rápida en `Isolation Forest.py`:

```python
PARAMETROS_BUSQUEDA = {
    'n_estimators': [50, 100],       # Reducido para velocidad
    'max_samples': ['auto'],         # Simplificado
    'contamination': [0.05, 0.1],    # Tasa de contaminación esperada  
    'max_features': [1.0]            # Características por split
}

# Muestreo optimizado
MAX_MUESTRAS_OPTIMIZACION = 10000  # Reducido para velocidad
MAX_MUESTRAS_VISUALIZATION = 8000  # Visualizaciones
```

## 🔍 Interpretación de Resultados

### Clasificación Binaria
- **is_outlier = 0**: Punto normal (requiere muchas particiones para aislar)
- **is_outlier = 1**: Anomalía detectada (se aísla rápidamente)

### Anomaly Score
- Basado en la profundidad promedio de aislamiento
- Mayor score = más anómalo (se aísla con menos particiones)
- Menor score = más normal (requiere más particiones)

### Separación (P95-P50)
- **Métrica crítica** para evaluar calidad de detección
- Mayor valor = mejor distinción entre anomalías y normales
- Indica confiabilidad del modelo

## 📊 Comparación con Otros Algoritmos

Para comparar Isolation Forest con CBLOF:

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
- **Muy rápido**: Entrenamiento e inferencia eficientes
- **Escalable**: Funciona bien con millones de puntos
- **Alta dimensionalidad**: No sufre la "maldición de la dimensionalidad"
- **Sin clustering**: No necesita agrupar datos previamente
- **Memoria eficiente**: Usa poco espacio de almacenamiento
- **Paralelizable**: Fácil de distribuir

### ⚠️ Desventajas
- **Menos interpretable**: Basado en profundidad de árboles (menos intuitivo)
- **Sensible a contaminación**: Requiere estimar proporción de anomalías
- **Anomalías locales**: Puede perder anomalías dentro de grupos densos
- **No contextual**: No usa estructura local de datos

## 🔄 Comparación con CBLOF

| Característica | Isolation Forest | CBLOF |
|----------------|------------------|-------|
| Necesita clusters | ❌ No | ✅ Sí |
| Contexto local | ❌ No | ✅ Sí |
| Velocidad | ⚡ Muy rápido | 🐌 Moderada |
| Escalabilidad | ✅ Excelente | ⚠️ Limitada |
| Interpretabilidad | ⚠️ Media | ✅ Alta |
| Alta dimensionalidad | ✅ Excelente | ⚠️ Limitada |

## 🧮 Cómo Funciona Isolation Forest

1. **Construcción de árboles**:
   - Selecciona aleatoriamente una característica
   - Selecciona aleatoriamente un valor de split
   - Particiona los datos recursivamente

2. **Profundidad de aislamiento**:
   - Anomalías requieren **menos particiones** (se aíslan rápido)
   - Puntos normales requieren **más particiones** (están en regiones densas)

3. **Score de anomalía**:
   - Basado en la profundidad promedio en todos los árboles
   - Normalizado para comparabilidad

4. **Detección**:
   - Puntos con scores más altos son anomalías

### Ejemplo Intuitivo
Imagina buscar a una persona en una multitud vs. buscar a alguien solo en una esquina. La persona aislada (anomalía) se encuentra más rápido con menos preguntas.

## 📚 Referencias

- Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). "Isolation Forest"
- Liu, F.T., Ting, K.M., & Zhou, Z.H. (2012). "Isolation-based Anomaly Detection"
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

## 🤝 Contribución al Proyecto

Este algoritmo es parte del proyecto de comparación de algoritmos de ML no supervisados para Mantenimiento Predictivo. Los resultados de Isolation Forest se comparan directamente con CBLOF para determinar el mejor algoritmo de detección de anomalías para esta aplicación específica.

## 🏆 Casos de Uso Ideales

Isolation Forest es especialmente adecuado para:
- ✅ Datasets muy grandes (millones de registros)
- ✅ Alta dimensionalidad (muchas características)
- ✅ Necesidad de detección rápida en tiempo real
- ✅ Anomalías globales (no contextuales)
- ✅ Recursos computacionales limitados

---

**Última actualización**: Octubre 2025  
**Versión**: 3.0 - **OPTIMIZADA** ⚡
