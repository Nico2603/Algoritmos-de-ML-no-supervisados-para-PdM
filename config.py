import numpy as np
import pandas as pd
import multiprocessing
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple
import warnings

RANDOM_STATE = 42
MAX_MUESTRAS_OPTIMIZACION = 5000
MAX_MUESTRAS_VISUALIZATION = 3000

try:
    N_JOBS_PARALELO = min(2, max(1, multiprocessing.cpu_count() // 2))
except:
    N_JOBS_PARALELO = 1

CMAP_CLUSTERING = 'tab10'

COLOR_NORMAL = '#1f77b4'
COLOR_ANOMALIA = '#d62728'
ALPHA_NORMAL = 0.5
ALPHA_ANOMALIA = 0.8

SCATTER_SIZE_NORMAL = 10
SCATTER_SIZE_ANOMALIA = 25
SCATTER_SIZE_NOISE = 30

FIGSIZE_2D = (10, 8)
FIGSIZE_3D = (12, 9)

VIEW_ELEV = 20
VIEW_AZIM = 45

USECOLS = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'fecha']
CARACTERISTICAS_BASE = ['acceleration_x', 'acceleration_y', 'acceleration_z']
CARACTERISTICA_MAGNITUD = 'magnitud_aceleracion'

BATCH_SIZE = 50000

def normalizar_scores_min_max(scores: np.ndarray) -> np.ndarray:
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.zeros_like(scores, dtype=np.float64)
    
    scores_normalizados = (scores - min_score) / (max_score - min_score)
    
    return scores_normalizados

def validar_datos_entrada(datos: pd.DataFrame, caracteristicas: List[str]) -> None:
    if len(datos) < 100:
        raise ValueError(
            f"Dataset muy pequeño: {len(datos)} filas. "
            f"Mínimo requerido: 100 filas"
        )
    
    columnas_faltantes = [col for col in caracteristicas if col not in datos.columns]
    if columnas_faltantes:
        raise ValueError(
            f"Características faltantes en el dataset: {columnas_faltantes}"
        )
    
    for col in caracteristicas:
        varianza = datos[col].std()
        if varianza == 0 or np.isnan(varianza):
            raise ValueError(
                f"Característica '{col}' tiene varianza cero o inválida. "
                f"No se puede realizar análisis con características constantes."
            )
    
    for col in caracteristicas:
        mean = datos[col].mean()
        std = datos[col].std()
        
        if std > 0:
            outliers_extremos = np.abs(datos[col] - mean) > 10 * std
            n_outliers = outliers_extremos.sum()
            pct_outliers = (n_outliers / len(datos)) * 100
            
            if pct_outliers > 1.0:
                warnings.warn(
                    f"[ALERTA] Característica '{col}': {n_outliers} outliers extremos "
                    f"({pct_outliers:.2f}% del dataset) detectados (>10 std). "
                    f"Considera revisar la calidad de los datos.",
                    UserWarning
                )
    
    print(f"[OK] Validación de datos exitosa: {len(datos)} filas, "
          f"{len(caracteristicas)} características")

def aplicar_pca_consistente(
    X_escalado: np.ndarray, 
    n_components: int,
    pca_guardado: Optional[PCA] = None
) -> Tuple[np.ndarray, PCA]:
    if pca_guardado is None:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_escalado)
        
        varianza_explicada = pca.explained_variance_ratio_
        varianza_total = np.sum(varianza_explicada) * 100
        
        print(f"PCA aplicado: {n_components} componentes, "
              f"varianza explicada: {varianza_total:.2f}%")
    else:
        pca = pca_guardado
        X_pca = pca.transform(X_escalado)
        
        print(f"PCA reutilizado: {n_components} componentes")
    
    return X_pca, pca

def aplicar_seeds_reproducibilidad(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except:
        pass

def muestrear_datos_consistente(
    X: np.ndarray, 
    max_muestras: int,
    seed: int = RANDOM_STATE
) -> np.ndarray:
    if len(X) <= max_muestras:
        return np.arange(len(X))
    
    np.random.seed(seed)
    
    indices = np.random.choice(len(X), size=max_muestras, replace=False)
    indices = np.sort(indices)
    
    return indices

__version__ = '1.0.0'
__author__ = 'Proyecto ML PdM'
__all__ = [
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
    'normalizar_scores_min_max',
    'validar_datos_entrada',
    'aplicar_pca_consistente',
    'aplicar_seeds_reproducibilidad',
    'muestrear_datos_consistente',
]
