"""
Configuración centralizada para algoritmos de ML no supervisados
Mantenimiento Predictivo (PdM)

Este módulo contiene todas las constantes globales, funciones de normalización,
validación de datos y utilidades compartidas entre todos los algoritmos.
"""

import numpy as np
import pandas as pd
import multiprocessing
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple
import warnings

# ============================================================================
# CONFIGURACIÓN DE REPRODUCIBILIDAD
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# CONFIGURACIÓN DE MUESTREO (UNIFICADA)
# ============================================================================
MAX_MUESTRAS_OPTIMIZACION = 10000
MAX_MUESTRAS_VISUALIZATION = 8000

# ============================================================================
# CONFIGURACIÓN DE PARALELIZACIÓN
# ============================================================================
try:
    N_JOBS_PARALELO = max(1, multiprocessing.cpu_count() - 1)
except:
    N_JOBS_PARALELO = 1

# ============================================================================
# CONFIGURACIÓN DE COLORES PARA VISUALIZACIONES
# ============================================================================
# Para clustering (DBSCAN, K-Means)
CMAP_CLUSTERING = 'tab10'

# Para detección de anomalías (Isolation Forest, CBLOF)
COLOR_NORMAL = '#1f77b4'      # Azul estándar
COLOR_ANOMALIA = '#d62728'    # Rojo estándar
ALPHA_NORMAL = 0.5
ALPHA_ANOMALIA = 0.8

# ============================================================================
# CONFIGURACIÓN DE TAMAÑOS DE PUNTOS EN SCATTER
# ============================================================================
SCATTER_SIZE_NORMAL = 10
SCATTER_SIZE_ANOMALIA = 25
SCATTER_SIZE_NOISE = 30  # Para ruido en DBSCAN

# ============================================================================
# CONFIGURACIÓN DE FIGURAS
# ============================================================================
FIGSIZE_2D = (10, 8)
FIGSIZE_3D = (12, 9)

# Ángulo de vista óptimo para gráficas 3D
VIEW_ELEV = 20
VIEW_AZIM = 45

# ============================================================================
# CONFIGURACIÓN DE CARACTERÍSTICAS
# ============================================================================
USECOLS = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'fecha']
CARACTERISTICAS_BASE = ['acceleration_x', 'acceleration_y', 'acceleration_z']
CARACTERISTICA_MAGNITUD = 'magnitud_aceleracion'

# ============================================================================
# CONFIGURACIÓN DE BATCHING (PARA DATASETS MUY GRANDES)
# ============================================================================
BATCH_SIZE = 50000

# ============================================================================
# FUNCIÓN DE NORMALIZACIÓN ESTÁNDAR
# ============================================================================
def normalizar_scores_min_max(scores: np.ndarray) -> np.ndarray:
    """
    Normaliza scores al rango [0, 1] usando Min-Max scaling.
    
    Esta función debe ser usada por TODOS los algoritmos para garantizar
    que los scores de anomalía sean comparables entre diferentes métodos.
    
    Args:
        scores: Array numpy con scores originales (cualquier rango)
        
    Returns:
        Array numpy con scores normalizados en el rango [0, 1]
        
    Ejemplo:
        >>> scores = np.array([10, 20, 30, 40, 50])
        >>> scores_norm = normalizar_scores_min_max(scores)
        >>> print(scores_norm)
        [0.   0.25 0.5  0.75 1.  ]
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # Caso especial: si todos los scores son iguales
    if max_score == min_score:
        return np.zeros_like(scores, dtype=np.float64)
    
    # Normalización Min-Max
    scores_normalizados = (scores - min_score) / (max_score - min_score)
    
    return scores_normalizados


# ============================================================================
# FUNCIÓN DE VALIDACIÓN DE DATOS
# ============================================================================
def validar_datos_entrada(datos: pd.DataFrame, caracteristicas: List[str]) -> None:
    """
    Valida que los datos cumplan con requisitos mínimos para análisis.
    
    Verifica:
    - Tamaño mínimo del dataset (>= 100 filas)
    - Varianza no cero en todas las características
    - Advierte sobre outliers extremos (> 10 desviaciones estándar)
    
    Args:
        datos: DataFrame con los datos a validar
        caracteristicas: Lista de nombres de columnas a validar
        
    Raises:
        ValueError: Si el dataset no cumple requisitos mínimos
        
    Ejemplo:
        >>> datos = pd.read_csv('data.csv')
        >>> validar_datos_entrada(datos, ['acceleration_x', 'acceleration_y', 'acceleration_z'])
    """
    # Verificar tamaño mínimo
    if len(datos) < 100:
        raise ValueError(
            f"Dataset muy pequeño: {len(datos)} filas. "
            f"Mínimo requerido: 100 filas"
        )
    
    # Verificar que todas las características existan
    columnas_faltantes = [col for col in caracteristicas if col not in datos.columns]
    if columnas_faltantes:
        raise ValueError(
            f"Características faltantes en el dataset: {columnas_faltantes}"
        )
    
    # Verificar varianza en cada característica
    for col in caracteristicas:
        varianza = datos[col].std()
        if varianza == 0 or np.isnan(varianza):
            raise ValueError(
                f"Característica '{col}' tiene varianza cero o inválida. "
                f"No se puede realizar análisis con características constantes."
            )
    
    # Advertir sobre outliers extremos (> 10 desviaciones estándar)
    for col in caracteristicas:
        mean = datos[col].mean()
        std = datos[col].std()
        
        if std > 0:
            outliers_extremos = np.abs(datos[col] - mean) > 10 * std
            n_outliers = outliers_extremos.sum()
            pct_outliers = (n_outliers / len(datos)) * 100
            
            if pct_outliers > 1.0:  # Más del 1% son outliers extremos
                warnings.warn(
                    f"⚠️ Característica '{col}': {n_outliers} outliers extremos "
                    f"({pct_outliers:.2f}% del dataset) detectados (>10 std). "
                    f"Considera revisar la calidad de los datos.",
                    UserWarning
                )
    
    print(f"✅ Validación de datos exitosa: {len(datos)} filas, "
          f"{len(caracteristicas)} características")


# ============================================================================
# FUNCIÓN DE PCA CONSISTENTE
# ============================================================================
def aplicar_pca_consistente(
    X_escalado: np.ndarray, 
    n_components: int,
    pca_guardado: Optional[PCA] = None
) -> Tuple[np.ndarray, PCA]:
    """
    Aplica PCA de forma consistente o reutiliza un modelo PCA existente.
    
    Esta función permite que múltiples algoritmos usen el MISMO espacio PCA,
    haciendo que las visualizaciones sean directamente comparables.
    
    Args:
        X_escalado: Datos normalizados (StandardScaler aplicado)
        n_components: Número de componentes principales (2 o 3 para visualización)
        pca_guardado: Modelo PCA previamente ajustado (opcional)
                      - Si es None: se ajusta un nuevo PCA
                      - Si se proporciona: se usa para transformar datos
        
    Returns:
        Tuple con:
        - X_pca: Datos transformados a espacio PCA
        - pca: Modelo PCA (nuevo o el proporcionado)
        
    Ejemplo:
        >>> # Primera ejecución (crear PCA)
        >>> X_pca_1, pca_model = aplicar_pca_consistente(X_scaled, 3, None)
        >>> 
        >>> # Segunda ejecución (reutilizar PCA)
        >>> X_pca_2, _ = aplicar_pca_consistente(X_scaled_2, 3, pca_model)
    """
    if pca_guardado is None:
        # Crear y ajustar nuevo modelo PCA
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_escalado)
        
        # Calcular varianza explicada
        varianza_explicada = pca.explained_variance_ratio_
        varianza_total = np.sum(varianza_explicada) * 100
        
        print(f"PCA aplicado: {n_components} componentes, "
              f"varianza explicada: {varianza_total:.2f}%")
    else:
        # Reutilizar modelo PCA existente
        pca = pca_guardado
        X_pca = pca.transform(X_escalado)
        
        print(f"PCA reutilizado: {n_components} componentes")
    
    return X_pca, pca


# ============================================================================
# FUNCIÓN DE APLICACIÓN CONSISTENTE DE SEEDS
# ============================================================================
def aplicar_seeds_reproducibilidad(seed: int = RANDOM_STATE) -> None:
    """
    Aplica seeds a todos los generadores de números aleatorios.
    
    Garantiza reproducibilidad en:
    - NumPy random
    - Python random
    - Cualquier operación estocástica posterior
    
    Args:
        seed: Semilla aleatoria (por defecto usa RANDOM_STATE global)
        
    Ejemplo:
        >>> aplicar_seeds_reproducibilidad(42)
    """
    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except:
        pass
    
    # Para algunos módulos de sklearn que usan random_state
    # ya se maneja directamente con el parámetro random_state=RANDOM_STATE


# ============================================================================
# FUNCIÓN DE MUESTREO ESTRATIFICADO CONSISTENTE
# ============================================================================
def muestrear_datos_consistente(
    X: np.ndarray, 
    max_muestras: int,
    seed: int = RANDOM_STATE
) -> np.ndarray:
    """
    Realiza muestreo aleatorio consistente de datos.
    
    Args:
        X: Datos a muestrear
        max_muestras: Número máximo de muestras a retornar
        seed: Semilla para reproducibilidad
        
    Returns:
        Índices de las muestras seleccionadas
        
    Ejemplo:
        >>> indices = muestrear_datos_consistente(X, 10000, 42)
        >>> X_muestreado = X[indices]
    """
    if len(X) <= max_muestras:
        return np.arange(len(X))
    
    # Aplicar seed antes de muestrear
    np.random.seed(seed)
    
    # Muestreo aleatorio simple
    indices = np.random.choice(len(X), size=max_muestras, replace=False)
    indices = np.sort(indices)  # Mantener orden original
    
    return indices


# ============================================================================
# INFORMACIÓN DEL MÓDULO
# ============================================================================
__version__ = '1.0.0'
__author__ = 'Proyecto ML PdM'
__all__ = [
    # Constantes
    'RANDOM_STATE',
    'MAX_MUESTRAS_OPTIMIZACION',
    'MAX_MUESTRAS_VISUALIZATION',
    'N_JOBS_PARALELO',
    'CMAP_CLUSTERING',
    'COLOR_NORMAL',
    'COLOR_ANOMALIA',
    'ALPHA_NORMAL',
    'ALPHA_ANOMALIA',
    'SCATTER_SIZE_NORMAL',
    'SCATTER_SIZE_ANOMALIA',
    'SCATTER_SIZE_NOISE',
    'FIGSIZE_2D',
    'FIGSIZE_3D',
    'VIEW_ELEV',
    'VIEW_AZIM',
    'USECOLS',
    'CARACTERISTICAS_BASE',
    'CARACTERISTICA_MAGNITUD',
    'BATCH_SIZE',
    
    # Funciones
    'normalizar_scores_min_max',
    'validar_datos_entrada',
    'aplicar_pca_consistente',
    'aplicar_seeds_reproducibilidad',
    'muestrear_datos_consistente',
]

