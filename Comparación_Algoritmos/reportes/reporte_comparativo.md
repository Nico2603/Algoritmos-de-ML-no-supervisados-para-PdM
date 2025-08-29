# Reporte Comparativo de Algoritmos de ML No Supervisado para PdM

## Resumen Ejecutivo

Este reporte presenta la comparación entre los 4 algoritmos implementados:

- **DBSCAN** (Clustering)
- **Isolation Forest** (Detección de Anomalías)
- **CBLOF** (Detección de Anomalías)

## Estadísticas por Algoritmo

### DBSCAN

- **Total de registros**: 50,000
- **Score de severidad promedio**: 0.0831
- **Desviación estándar**: 0.1416
- **Distribución por severidad**:
  - Normal: 40,000 (80.0%)
  - Leve: 5,000 (10.0%)
  - Moderada: 2,500 (5.0%)
  - Severa: 2,000 (4.0%)
  - Critica: 500 (1.0%)

### Isolation Forest

- **Total de registros**: 25,920
- **Score de severidad promedio**: 0.2073
- **Desviación estándar**: 0.1719
- **Distribución por severidad**:
  - Normal: 20,736 (80.0%)
  - Leve: 2,592 (10.0%)
  - Moderada: 1,296 (5.0%)
  - Severa: 1,036 (4.0%)
  - Critica: 260 (1.0%)

### CBLOF

- **Total de registros**: 51,840
- **Score de severidad promedio**: 0.2407
- **Desviación estándar**: 0.1890
- **Distribución por severidad**:
  - Normal: 41,472 (80.0%)
  - Leve: 5,184 (10.0%)
  - Moderada: 2,592 (5.0%)
  - Severa: 2,073 (4.0%)
  - Critica: 519 (1.0%)

## Recomendaciones Operativas

### Para Detección de Patrones (Clustering)
- **K-Means**: Recomendado para identificar patrones de operación normal
- **DBSCAN**: Mejor para detectar grupos de densidad variable

### Para Alertas de Anomalías
- **Isolation Forest**: Eficiente para anomalías globales
- **CBLOF**: Mejor para anomalías basadas en clusters

### Umbrales Recomendados
- **Alertas Operativas**: P95 (Severidad ≥ 95%)
- **Monitoreo Preventivo**: P90 (Severidad ≥ 90%)
- **Análisis Rutinario**: P80 (Severidad ≥ 80%)

