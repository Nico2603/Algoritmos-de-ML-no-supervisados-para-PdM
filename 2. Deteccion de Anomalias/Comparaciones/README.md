# Comparaciones - Algoritmos de Detección de Anomalías

## 📋 Descripción

Esta carpeta contiene el script de comparación exhaustiva entre CBLOF e Isolation Forest. El script genera análisis visuales y métricas comparativas para evaluar el desempeño de ambos algoritmos de detección de anomalías en el contexto de Mantenimiento Predictivo.

## 🎯 Objetivo

Comparar los algoritmos CBLOF e Isolation Forest lado a lado para:
- Evaluar el rendimiento de cada algoritmo
- Visualizar diferencias en las detecciones
- Generar reporte detallado con análisis de métricas
- Facilitar la toma de decisiones basada en métricas objetivas

## 🚀 Uso

### Prerrequisitos

Antes de ejecutar la comparación, asegúrate de haber ejecutado:

```bash
# 1. CBLOF
cd "../CBLOF (Cluster-Based Local Outlier Factor)"
python CBLOF.PY

# 2. Isolation Forest
cd "../Isolation Forest"
python "Isolation Forest.py"
```

### Ejecución

```bash
cd Comparaciones
python comparar_algoritmos.py
```

## 📊 Outputs que se Generarán

Al ejecutar el script, se generarán automáticamente:

### Comparaciones Visuales Lado a Lado

1. **`comparacion_visual_scores.png`**
   - Distribución de scores de CBLOF e Isolation Forest
   - Lado a lado para comparación directa
   - Muestra separación entre anomalías y normales

2. **`comparacion_visual_3d.png`**
   - Visualización 3D de anomalías detectadas
   - CBLOF e Isolation Forest lado a lado
   - Puntos normales (azul) vs anomalías (rojo)

### Gráficos Comparativos de Métricas

3. **`comparacion_metricas_barras.png`**
   - Gráfico de barras comparativo
   - Métricas: % anomalías, separación, score promedio
   - Colores diferenciados para cada algoritmo
   - Valores numéricos sobre las barras

4. **`comparacion_porcentaje_anomalias.png`**
   - Gráfico horizontal de barras
   - Compara porcentaje de anomalías detectadas
   - Valores exactos mostrados

5. **`comparacion_separacion_scores.png`**
   - **Métrica más crítica**
   - Separación entre percentil 95 y 50
   - Mayor valor = mejor distinción anomalías/normales
   - Indicador de confiabilidad del modelo

### Tablas y Reportes

6. **`tabla_comparativa.csv`**
   - Tabla de métricas en formato CSV
   - Importable a Excel/Google Sheets
   - Todas las métricas lado a lado

7. **`REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt`**
   - **Reporte completo detallado**
   - Tabla comparativa de métricas
   - Análisis por métrica individual
   - Puntuación final (separación pesa doble)
   - **🏆 GANADOR IDENTIFICADO**
   - Ventajas y desventajas de cada algoritmo
   - Interpretación de métricas clave

## 📈 Métricas Comparadas

### Separación de Scores (P95-P50) ⭐ CRÍTICA
- **Métrica más importante**
- Diferencia entre percentil 95 y percentil 50
- **Mayor es mejor**
- Indica qué tan bien el algoritmo distingue anomalías de normales
- Mayor valor = mayor confiabilidad en las detecciones
- **Pesa doble en la puntuación final**

### Porcentaje de Anomalías
- % de puntos clasificados como anomalías
- Debe evaluarse en contexto del dominio
- No necesariamente "más es mejor"
- Ideal: coincidir con tasa esperada real

### Score Promedio
- Score medio de todos los puntos
- Indicador general del comportamiento del modelo
- Para contexto y referencia

## 📊 Resultados de la Comparación

### Resumen Ejecutivo: Isolation Forest Gana
**Isolation Forest** es el ganador claro con mejor separación de scores y eficiencia superior.

| Métrica | CBLOF | Isolation Forest | Ganador |
|---------|-------|------------------|---------|
| **Separación P95-P50** | 0.3239 | 0.3772 ✅ | **Isolation Forest** |
| **Tiempo (s)** | 14.53 | 10.94 ✅ | **Isolation Forest** |
| **Memoria (MB)** | 92.75 | 64.22 ✅ | **Isolation Forest** |

**Puntuación**: Isolation Forest 2 | CBLOF 0

## 🏆 Determinación del Ganador

El script determina automáticamente el ganador usando sistema de puntuación ponderado:

- **Separación de Scores**: Peso de 2 puntos (métrica crítica)
- **Porcentaje de Anomalías**: Peso de 1 punto (si diferencia >2%)
- **Total máximo**: 3 puntos

### Sistema de Puntuación

```
Si CBLOF tiene mejor separación: +2 puntos
Si CBLOF detecta significativamente más anomalías: +1 punto
Total posible: 3 puntos

(Mismo para Isolation Forest)
```

El ganador se muestra claramente en:
- Consola durante la ejecución
- `REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt`

## 📝 Interpretación del Reporte

### Ejemplo de Sección del Reporte

```
4. ALGORITMO GANADOR
--------------------------------------------------
🏆 GANADOR: Isolation Forest

Isolation Forest demostró mejor rendimiento, 
especialmente en la separación de anomalías.

PUNTUACIÓN FINAL (Separación pesa doble)
--------------------------------------------------
CBLOF: 0 puntos
Isolation Forest: 2 puntos
```

## 🔍 Análisis Detallado

El reporte incluye:

1. **Tabla Comparativa**: Todas las métricas lado a lado
2. **Análisis por Métrica**: Ganador en cada métrica
3. **Puntuación Ponderada**: Score con peso en separación
4. **Ganador Final**: Algoritmo recomendado
5. **Características**: Ventajas/desventajas de cada uno
6. **Interpretación de Métricas**: Guía de lectura
7. **Archivos Generados**: Lista completa de outputs

## 🎓 Uso de Resultados

### Si gana CBLOF:
- ✅ Usar para anomalías locales dentro de grupos
- ✅ Cuando se necesita contexto basado en clustering
- ✅ Para detección más interpretable
- ✅ Cuando la estructura de datos es importante

### Si gana Isolation Forest:
- ✅ Usar para velocidad y escalabilidad
- ✅ Datasets muy grandes (millones de registros)
- ✅ Alta dimensionalidad
- ✅ Detección en tiempo real
- ✅ Anomalías globales (no contextuales)

## ⚙️ Personalización

Para modificar el análisis, edita `comparar_algoritmos.py`:

```python
# Cambiar colores de gráficos
colors = ['mediumseagreen', 'mediumpurple']  # CBLOF, IForest

# Modificar peso de separación en puntuación
# Función: analizar_ganador()
# peso_separacion = 2  # Doble peso
```

## 📚 Archivos Necesarios

El script busca automáticamente:

```
../CBLOF (Cluster-Based Local Outlier Factor)/metricas_CBLOF/
  - metrics.csv
  - metrics.txt
  - anomaly_scores.csv
  - anomalies.csv

../CBLOF (Cluster-Based Local Outlier Factor)/graficas_CBLOF/
  - anomaly_scores.png
  - anomalies_3d.png

../Isolation Forest/metricas_IForest/
  - metrics.csv
  - metrics.txt
  - anomaly_scores.csv
  - anomalies.csv

../Isolation Forest/graficas_IForest/
  - anomaly_scores.png
  - anomalies_3d.png
```

## ❌ Solución de Problemas

### Error: "Archivos faltantes"
```bash
# Ejecuta primero los algoritmos
cd "../CBLOF (Cluster-Based Local Outlier Factor)"
python CBLOF.PY

cd "../Isolation Forest"
python "Isolation Forest.py"
```

### Error: "No module named 'pyod'"
```bash
pip install pyod
```

### Error: "matplotlib no encontrado"
```bash
pip install matplotlib
```

## 🔄 Flujo Completo

```bash
# 1. Ejecutar algoritmos
cd "../CBLOF (Cluster-Based Local Outlier Factor)"
python CBLOF.PY

cd "../Isolation Forest"
python "Isolation Forest.py"

# 2. Ejecutar comparación
cd ../Comparaciones
python comparar_algoritmos.py

# 3. Revisar resultados
# - Abrir gráficas PNG
# - Leer REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt
# - Revisar tabla_comparativa.csv
```

## 📊 Ejemplo de Output

```
================================================================================
      COMPARACIÓN DE ALGORITMOS DE DETECCIÓN DE ANOMALÍAS
                    CBLOF vs Isolation Forest
================================================================================

✅ Todos los archivos necesarios están presentes.

📂 Cargando métricas...
✅ Métricas cargadas correctamente.

📊 Generando comparación visual de gráficas...
✅ Comparación de scores guardada: comparacion_visual_scores.png
✅ Comparación 3D guardada: comparacion_visual_3d.png

📋 Generando tabla comparativa...
[Tabla de métricas]

📈 Generando gráficos comparativos de métricas...
✅ Gráfico de barras guardado: comparacion_metricas_barras.png
✅ Gráfico de porcentajes guardado: comparacion_porcentaje_anomalias.png
✅ Gráfico de separación guardado: comparacion_separacion_scores.png

🔍 Analizando resultados...

🏆 ALGORITMO GANADOR: Isolation Forest
   Puntuación: CBLOF 0 - Isolation Forest 2

📝 Generando reporte completo...
✅ Reporte completo guardado: REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt

================================================================================
           ✅ COMPARACIÓN COMPLETADA EXITOSAMENTE
         📁 Resultados guardados en: Comparaciones/
================================================================================
```

## 🎯 Métricas Clave para Decisión

### 1. Separación de Scores (P95-P50) - LA MÁS IMPORTANTE
- **Valor alto**: El algoritmo distingue claramente anomalías de normales
- **Valor bajo**: Las detecciones son menos confiables
- **Decisivo**: Esta métrica pesa doble en la puntuación

### 2. Porcentaje de Anomalías
- Debe coincidir con expectativas del dominio
- Si esperás 5% de anomalías y detecta 50%, revisar parámetros
- Contexto-dependiente

### 3. Visualizaciones
- Gráfico de scores: ¿Hay dos "picos" claros? (bueno)
- Gráfico 3D: ¿Las anomalías están visualmente separadas? (bueno)

## 🔬 Interpretación Avanzada

### Separación Alta (>0.5)
- ✅ Excelente distinción anomalías/normales
- ✅ Alta confiabilidad
- ✅ Fácil de establecer umbrales

### Separación Media (0.2-0.5)
- ⚠️ Distinción aceptable
- ⚠️ Puede requerir ajuste de parámetros
- ⚠️ Umbrales menos claros

### Separación Baja (<0.2)
- ❌ Poca distinción
- ❌ Revisar parámetros o considerar otro algoritmo
- ❌ Difícil establecer umbrales confiables

## 📋 Checklist de Análisis

- [ ] ¿Qué algoritmo tiene mejor separación (P95-P50)?
- [ ] ¿El porcentaje de anomalías es razonable?
- [ ] ¿Las visualizaciones muestran separación clara?
- [ ] ¿El ganador coincide con necesidades de velocidad/escalabilidad?
- [ ] ¿Se necesita interpretabilidad (favorece CBLOF)?
- [ ] ¿Se necesita velocidad (favorece Isolation Forest)?

---

**Última actualización**: Octubre 2025  
**Versión**: 2.0

