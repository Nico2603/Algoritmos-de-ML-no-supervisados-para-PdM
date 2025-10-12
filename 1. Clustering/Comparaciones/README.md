# Comparaciones - Algoritmos de Clustering

## 📋 Descripción

Esta carpeta contiene el script de comparación exhaustiva entre DBSCAN y K-Means. El script genera análisis visuales y métricas comparativas para evaluar el desempeño de ambos algoritmos de clustering en el contexto de Mantenimiento Predictivo.

## 🎯 Objetivo

Comparar los algoritmos DBSCAN y K-Means lado a lado para:
- Evaluar el rendimiento de cada algoritmo
- Visualizar diferencias en resultados
- Generar reporte detallado con análisis de métricas
- Facilitar la toma de decisiones basada en métricas objetivas

## 🚀 Uso

### Prerrequisitos

Antes de ejecutar la comparación, asegúrate de haber ejecutado:

```bash
# 1. DBSCAN
cd ../DBSCAN
python DBSCAN.py

# 2. K-Means
cd ../K-means
python K-means.py
```

### Ejecución

```bash
cd Comparaciones
python comparar_algoritmos.py
```

## 📊 Outputs que se Generarán

Al ejecutar el script, se generarán automáticamente:

### Comparaciones Visuales Lado a Lado

1. **`comparacion_visual_2d.png`**
   - DBSCAN y K-Means en 2D lado a lado
   - Facilita la comparación visual directa
   - Basado en PCA

2. **`comparacion_visual_3d.png`**
   - DBSCAN y K-Means en 3D lado a lado
   - Visualización tridimensional comparativa

### Gráficos Comparativos de Métricas

3. **`comparacion_metricas_barras.png`**
   - Gráfico de barras comparativo
   - Métricas: Silhouette, Calinski-Harabasz, Davies-Bouldin
   - Colores diferenciados para cada algoritmo
   - Valores numéricos sobre las barras

4. **`comparacion_metricas_radar.png`**
   - Gráfico de radar multidimensional
   - Visualización 360° de todas las métricas
   - Valores normalizados (0-1) para comparabilidad
   - Áreas coloreadas para cada algoritmo

### Tablas y Reportes

5. **`tabla_comparativa.csv`**
   - Tabla de métricas en formato CSV
   - Importable a Excel/Google Sheets
   - Todas las métricas lado a lado

6. **`REPORTE_COMPARACION_CLUSTERING.txt`**
   - **Reporte completo detallado**
   - Tabla comparativa de métricas
   - Análisis por métrica individual
   - Puntuación final
   - **🏆 GANADOR IDENTIFICADO**
   - Ventajas y desventajas de cada algoritmo
   - Recomendaciones de uso

## 📈 Métricas Comparadas

### Silhouette Score (0-1)
- Mide qué tan bien están separados los clusters
- **Mayor es mejor**
- Indica cohesión interna y separación entre clusters

### Calinski-Harabasz Score
- Ratio de dispersión entre clusters vs dentro de clusters
- **Mayor es mejor**
- Valores más altos = clusters mejor definidos

### Davies-Bouldin Index
- Promedio de similitud entre clusters
- **Menor es mejor**
- Valores más bajos = clusters mejor separados

### Número de Clusters
- Clusters identificados por cada algoritmo
- DBSCAN: Identificación automática
- K-Means: Optimización mediante grid search

### Separación de Scores (P95-P50)
- Diferencia entre percentil 95 y 50
- Mayor valor = mejor distinción de puntos alejados

### Porcentaje de Anomalías
- DBSCAN: Puntos de ruido detectados
- K-Means: 0% (no detecta anomalías binarias)

## 📊 Resultados de la Comparación

### Resumen Ejecutivo: K-Means Gana Decisivamente

Tras ejecutar y comparar ambos algoritmos en el dataset de 518,400 registros de acelerómetro, **K-Means es el claro ganador** con superioridad en 2 de 3 métricas principales.

### Tabla de Resultados

| Métrica | DBSCAN | K-Means | Ganador |
|---------|--------|---------|---------|
| **Silhouette Score** | -0.7026 ❌ | 0.3269 ✅ | **K-Means** |
| **Calinski-Harabasz** | 16.62 ❌ | 300,079.19 ✅ | **K-Means** |
| **Davies-Bouldin** | 0.9335 ✅ | 1.2372 ❌ | **DBSCAN** |
| **Número de Clusters** | 1432 | 2 | - |
| **Tiempo (s)** | 29.07 | 9.60 | **K-Means** |
| **Memoria (MB)** | 222.13 | 184.78 | **K-Means** |
| **Anomalías Detectadas** | 3.35% | 5.00% | - |

**Puntuación Final**: K-Means 2/3 métricas ✅ | DBSCAN 1/3 métricas ❌

### Análisis Detallado

#### 🏆 K-Means: Ganador Claro

**Fortalezas**:
- **Silhouette Score positivo** (0.33): Indica clusters razonablemente separados
- **Calinski-Harabasz excepcional** (300k): Excelente definición de clusters
- **3x más rápido**: 9.6s vs 29.1s
- **20% menos memoria**: 184MB vs 222MB
- **Solo 2 clusters**: Altamente interpretable para operaciones

**Debilidades**:
- Davies-Bouldin ligeramente peor (pero no crítico)
- No detecta outliers automáticamente

#### ❌ DBSCAN: Rendimiento Problemático

**Problemas Críticos**:
- **Silhouette Score muy negativo** (-0.70): Indica clustering de muy mala calidad
- **Calinski-Harabasz extremadamente bajo** (16.62): Mala separación entre clusters
- **1432 clusters**: Fragmentación excesiva sin significado operacional
- **Sobreajuste severo**: Parámetros óptimos en muestra pequeña no generalizan

**Única Fortaleza**:
- Davies-Bouldin mejor (0.93 vs 1.24)
- Pero inconsistente con las otras métricas

### Interpretación

#### Por Qué K-Means Gana

1. **Estructura de Datos Adecuada**: Los datos de acelerómetro tienen densidad uniforme, ideal para K-Means.
2. **Simplicidad Efectiva**: 2 clusters capturan bien los estados operacionales principales.
3. **Eficiencia Computacional**: Velocidad y memoria óptimas.
4. **Estabilidad**: Resultados consistentes y reproducibles.

#### Por Qué DBSCAN Falla

1. **Datos No Apropiados**: DBSCAN necesita variabilidad de densidad, que estos datos no tienen.
2. **Fragmentación Excesiva**: 1432 micro-clusters no son útiles.
3. **Optimización Fallida**: Los parámetros no generalizaron al dataset completo.
4. **Silhouette Catastrófico**: -0.70 es inaceptable para clustering.

## 🏆 Determinación del Ganador

El script determina automáticamente el ganador usando sistema de puntuación:

- **3 puntos posibles** (una por cada métrica principal)
- Gana el algoritmo que supera al otro en más métricas
- En caso de empate, se considera el contexto específico

**Resultado Actual**: K-Means gana con 2/3 métricas

El ganador se muestra claramente en:
- Consola durante la ejecución
- `REPORTE_COMPARACION_CLUSTERING.txt`

## 📝 Interpretación del Reporte

### Ejemplo de Sección del Reporte

```
4. ALGORITMO GANADOR
--------------------------------------------------
🏆 GANADOR: DBSCAN

DBSCAN demostró mejor rendimiento en la mayoría 
de las métricas de clustering.

PUNTUACIÓN FINAL
--------------------------------------------------
DBSCAN: 2/3 métricas
K-Means: 1/3 métricas
```

## 🔍 Análisis Detallado

El reporte incluye:

1. **Tabla Comparativa**: Todas las métricas lado a lado
2. **Análisis por Métrica**: Ganador en cada métrica
3. **Puntuación**: Score numérico
4. **Ganador Final**: Algoritmo recomendado
5. **Consideraciones**: Ventajas/desventajas de cada uno
6. **Detalles Adicionales**: Clusters, anomalías, etc.

## 🎓 Uso de Resultados

### Si gana DBSCAN:
- ✅ Usar para clusters de forma arbitraria
- ✅ Aprovechar detección automática de outliers
- ✅ Ideal cuando no se conoce número de clusters

### Si gana K-Means:
- ✅ Usar para rapidez y escalabilidad
- ✅ Cuando los clusters son esféricos
- ✅ Cuando se tiene estimación del número de clusters

## ⚙️ Personalización

Para modificar el análisis, edita `comparar_algoritmos.py`:

```python
# Cambiar colores de gráficos
colors = ['steelblue', 'coral']  # DBSCAN, K-Means

# Modificar sistema de puntuación
# Función: analizar_ganador()
```

## 📚 Archivos Necesarios

El script busca automáticamente:

```
../DBSCAN/metricas_DBSCAN/
  - metrics.csv
  - metrics.txt
  - anomaly_scores.csv

../DBSCAN/graficas_DBSCAN/
  - clusters_2d_pca.png
  - clusters_3d_pca.png

../K-means/metricas_KMeans/
  - metrics.csv
  - metrics.txt
  - anomaly_scores.csv

../K-means/graficas_KMeans/
  - clusters_2d_pca.png
  - clusters_3d_pca.png
```

## ❌ Solución de Problemas

### Error: "Archivos faltantes"
```bash
# Ejecuta primero los algoritmos
cd ../DBSCAN && python DBSCAN.py
cd ../K-means && python K-means.py
```

### Error: "No se encuentra matplotlib"
```bash
pip install matplotlib
```

### Gráficas no se muestran
- Las gráficas se guardan automáticamente
- No es necesario mostrarlas interactivamente

## 🔄 Flujo Completo

```bash
# 1. Ejecutar algoritmos
cd ../DBSCAN
python DBSCAN.py

cd ../K-means
python K-means.py

# 2. Ejecutar comparación
cd ../Comparaciones
python comparar_algoritmos.py

# 3. Revisar resultados
# - Abrir gráficas PNG
# - Leer REPORTE_COMPARACION_CLUSTERING.txt
# - Revisar tabla_comparativa.csv
```

## 📊 Ejemplo de Output

```
================================================================================
          COMPARACIÓN DE ALGORITMOS DE CLUSTERING
                         DBSCAN vs K-Means
================================================================================

✅ Todos los archivos necesarios están presentes.

📂 Cargando métricas...
✅ Métricas cargadas correctamente.

📊 Generando comparación visual de gráficas...
✅ Comparación 2D guardada: comparacion_visual_2d.png
✅ Comparación 3D guardada: comparacion_visual_3d.png

📋 Generando tabla comparativa...
[Tabla de métricas]

📈 Generando gráficos comparativos de métricas...
✅ Gráfico de barras guardado: comparacion_metricas_barras.png
✅ Gráfico de radar guardado: comparacion_metricas_radar.png

🔍 Analizando resultados...

🏆 ALGORITMO GANADOR: DBSCAN
   Puntuación: DBSCAN 2/3 - K-Means 1/3

📝 Generando reporte completo...
✅ Reporte completo guardado: REPORTE_COMPARACION_CLUSTERING.txt

================================================================================
              ✅ COMPARACIÓN COMPLETADA EXITOSAMENTE
              📁 Resultados guardados en: Comparaciones/
================================================================================
```

---

**Última actualización**: Octubre 2025  
**Versión**: 2.0

