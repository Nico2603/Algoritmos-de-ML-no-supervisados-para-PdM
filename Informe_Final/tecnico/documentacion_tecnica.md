# Documentación Técnica - Sistema ML PdM

## 🏗️ Arquitectura del Sistema

### Componentes Principales
1. **Módulo de Preprocesamiento**: Limpieza y normalización de datos
2. **Motor de ML**: Algoritmos de clustering y detección de anomalías
3. **Sistema de Scoring**: Unificación de puntuaciones de severidad
4. **Motor de Alertas**: Generación de alertas basadas en umbrales
5. **Dashboard**: Visualización en tiempo real

## ⚙️ Especificaciones Técnicas

### Algoritmos Implementados

#### K-Means
- **Librería**: scikit-learn
- **Parámetros Clave**: n_clusters, random_state
- **Complejidad**: O(n*k*i)
- **Memoria**: O(n*d)
- **Caso de Uso**: Identificación de modos operativos

#### DBSCAN
- **Librería**: scikit-learn
- **Parámetros Clave**: eps, min_samples
- **Complejidad**: O(n*log(n))
- **Memoria**: O(n)
- **Caso de Uso**: Clustering basado en densidad

#### Isolation Forest
- **Librería**: scikit-learn
- **Parámetros Clave**: n_estimators, contamination
- **Complejidad**: O(n*log(n))
- **Memoria**: O(n)
- **Caso de Uso**: Detección de anomalías globales

#### CBLOF
- **Librería**: PyOD
- **Parámetros Clave**: n_clusters, alpha, beta
- **Complejidad**: O(n*k + n*log(n))
- **Memoria**: O(n)
- **Caso de Uso**: Anomalías basadas en clusters

### Pipeline de Datos

```python
# Ejemplo de pipeline
1. data_raw = load_sensor_data()
2. data_clean = preprocess_data(data_raw)
3. features = feature_engineering(data_clean)
4. features_scaled = scaler.transform(features)
5. anomaly_scores = model.predict(features_scaled)
6. alerts = generate_alerts(anomaly_scores, thresholds)
```

## 🔌 APIs y Endpoints

### Endpoint Principal
```
POST /api/v1/predict
Content-Type: application/json

{
  "acceleration_x": [1.2, 1.3, 1.1],
  "acceleration_y": [0.8, 0.9, 0.7],
  "acceleration_z": [9.8, 9.9, 9.7],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Respuesta
```json
{
  "anomaly_score": 0.85,
  "severity_level": "moderate",
  "alert_level": "P90",
  "recommended_action": "Schedule inspection",
  "confidence": 0.92
}
```

## 🚢 Configuración y Deployment

### Requisitos del Sistema
- **SO**: Linux Ubuntu 20.04+ / Windows 10+
- **Python**: 3.8+
- **RAM**: 8GB mínimo, 16GB recomendado
- **CPU**: 4 cores mínimo, 8 cores recomendado
- **Almacenamiento**: 100GB mínimo

### Dependencias
```bash
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install pyod>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
```

