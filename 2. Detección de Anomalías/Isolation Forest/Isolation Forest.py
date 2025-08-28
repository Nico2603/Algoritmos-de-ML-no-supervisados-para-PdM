#!/usr/bin/env python3
"""
Detección de Anomalías usando Isolation Forest

Este módulo implementa un sistema completo de detección de anomalías
utilizando el algoritmo Isolation Forest con optimización de parámetros
y visualización de resultados.

Autor: Sistema de Detección de Anomalías
Versión: 2.0
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import h5py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=FutureWarning)

# Constantes
CARACTERISTICAS_BASE = ['acceleration_x', 'acceleration_y', 'acceleration_z']
NOMBRE_ARCHIVO_DATOS = 'data.csv'
EXTENSION_MODELO_PKL = 'isolation_forest_model.pkl'
EXTENSION_MODELO_H5 = 'isolation_forest_model.h5'
ARCHIVO_METRICAS = 'metrics.txt'
ARCHIVO_LOG = 'output.log'
ARCHIVO_ANOMALIAS = 'anomalies.csv'
COMPONENTES_PCA = 3
RANDOM_STATE = 42

# Parámetros por defecto para búsqueda de hiperparámetros (optimizado para Google Colab)
PARAMETROS_BUSQUEDA = {
    'n_estimators': [100, 150],
    'max_samples': ['auto', 0.8],
    'contamination': [0.05, 0.1],
    'max_features': [1.0]
}


class DirectoriosProyecto:
    """Gestiona la creación y rutas de directorios del proyecto."""
    
    def __init__(self, directorio_base: str):
        self.directorio_base = Path(directorio_base)
        self.directorio_metricas = self.directorio_base / 'metricas_IForest'
        self.directorio_modelos = self.directorio_base / 'modelos_entrenados_IForest'
        self.directorio_graficas = self.directorio_base / 'graficas_IForest'
        
    def crear_directorios(self) -> None:
        """Crea todos los directorios necesarios."""
        for directorio in [self.directorio_metricas, self.directorio_modelos, self.directorio_graficas]:
            directorio.mkdir(exist_ok=True)


def configurar_logging(ruta_archivo_log: Path) -> logging.Logger:
    """
    Configura el sistema de logging para guardar en archivo y mostrar en consola.
    
    Args:
        ruta_archivo_log: Ruta donde guardar el archivo de log
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configurar formato
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler para archivo
    file_handler = logging.FileHandler(ruta_archivo_log, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def cargar_datos(ruta_datos: Path) -> pd.DataFrame:
    """
    Carga los datos desde archivo CSV con validación.
    
    Args:
        ruta_datos: Ruta al archivo de datos
        
    Returns:
        DataFrame con los datos cargados
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío o mal formateado
    """
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_datos}")
    
    try:
        datos = pd.read_csv(ruta_datos)
        if datos.empty:
            raise ValueError("El archivo de datos está vacío")
        return datos
    except Exception as e:
        raise ValueError(f"Error al cargar los datos: {e}")


def preprocesar_datos(datos: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocesa los datos: limpia valores faltantes y crea características derivadas.
    
    Args:
        datos: DataFrame original
        
    Returns:
        Tupla con los datos procesados y lista de características
    """
    # Eliminar valores faltantes
    datos_limpios = datos.dropna().copy()
    
    # Verificar que tenemos las características base
    for caracteristica in CARACTERISTICAS_BASE:
        if caracteristica not in datos_limpios.columns:
            raise ValueError(f"Característica requerida no encontrada: {caracteristica}")
    
    # Crear característica derivada: magnitud de aceleración
    datos_limpios['magnitud_aceleracion'] = np.sqrt(
        datos_limpios['acceleration_x']**2 +
        datos_limpios['acceleration_y']**2 +
        datos_limpios['acceleration_z']**2
    )
    
    caracteristicas = CARACTERISTICAS_BASE + ['magnitud_aceleracion']
    X = datos_limpios[caracteristicas].values
    
    return X, caracteristicas


def escalar_datos(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Escala los datos usando StandardScaler.
    
    Args:
        X: Datos a escalar
        
    Returns:
        Tupla con datos escalados y el escalador ajustado
    """
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    return X_escalado, escalador


def reducir_muestra_para_optimizacion(X: np.ndarray, max_muestras: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce el tamaño de la muestra para optimización de parámetros si es muy grande.
    
    Args:
        X: Matriz de características completa.
        max_muestras: Número máximo de muestras para optimización.
        
    Returns:
        Tupla con (muestra_reducida, indices_seleccionados).
    """
    if len(X) <= max_muestras:
        return X, np.arange(len(X))
    
    # Muestreo aleatorio estratificado
    np.random.seed(RANDOM_STATE)  # Para reproducibilidad
    indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
    indices_seleccionados = np.sort(indices_seleccionados)
    
    X_reducido = X[indices_seleccionados]
    
    print(f"Dataset reducido para optimización: {len(X)} -> {len(X_reducido)} muestras")
    return X_reducido, indices_seleccionados


def reducir_dimensionalidad(X: np.ndarray, n_componentes: int = COMPONENTES_PCA) -> Tuple[np.ndarray, PCA]:
    """
    Reduce la dimensionalidad usando PCA.
    
    Args:
        X: Datos a reducir
        n_componentes: Número de componentes principales
        
    Returns:
        Tupla con datos reducidos y el modelo PCA ajustado
    """
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def evaluar_modelo_isolation_forest(params: Dict[str, Any], X_escalado: np.ndarray, logger: logging.Logger) -> Optional[Tuple]:
    """
    Evalúa un modelo Isolation Forest con parámetros específicos.
    
    Args:
        params: Parámetros del modelo
        X_escalado: Datos escalados para entrenamiento
        logger: Logger para registrar información
        
    Returns:
        Tupla con métricas de evaluación y modelo entrenado, o None si falla
    """
    try:
        # Entrenar modelo
        clf = IsolationForest(
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            contamination=params['contamination'],
            max_features=params['max_features'],
            random_state=RANDOM_STATE
        )
        clf.fit(X_escalado)
        
        # Obtener predicciones
        scores = clf.decision_function(X_escalado)
        etiquetas = clf.predict(X_escalado)
        etiquetas = np.where(etiquetas == 1, 0, 1)  # 0: normal, 1: anomalía
        
        # Calcular métricas básicas
        n_anomalias = np.sum(etiquetas)
        porcentaje_anomalias = (n_anomalias / len(etiquetas)) * 100
        media_scores = np.mean(scores)
        
        logger.info(f"Parámetros: {params}")
        logger.info(f"Anomalías detectadas: {n_anomalias} ({porcentaje_anomalias:.2f}%)")
        logger.info(f"Media de puntuaciones: {media_scores:.4f}")
        
        # Calcular métricas de clustering si es posible
        metricas_clustering = calcular_metricas_clustering(X_escalado, etiquetas, logger)
        
        return (media_scores, params, clf, *metricas_clustering)
        
    except Exception as e:
        logger.error(f"Error evaluando parámetros {params}: {e}")
        return None


def calcular_metricas_clustering(X: np.ndarray, etiquetas: np.ndarray, logger: logging.Logger) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calcula métricas de clustering cuando es posible.
    
    Args:
        X: Datos para calcular métricas
        etiquetas: Etiquetas de clasificación
        logger: Logger para registrar información
        
    Returns:
        Tupla con silhouette, calinski-harabasz y davies-bouldin scores
    """
    if len(np.unique(etiquetas)) > 1:
        try:
            silhouette = silhouette_score(X, etiquetas)
            calinski = calinski_harabasz_score(X, etiquetas)
            davies = davies_bouldin_score(X, etiquetas)
            
            logger.info(f"Silhouette Score: {silhouette:.4f}")
            logger.info(f"Calinski-Harabasz Score: {calinski:.4f}")
            logger.info(f"Davies-Bouldin Score: {davies:.4f}")
            
            return silhouette, calinski, davies
        except Exception as e:
            logger.warning(f"Error calculando métricas de clustering: {e}")
    else:
        logger.info("No es posible calcular métricas de clustering con una sola clase")
    
    return None, None, None


def buscar_mejores_parametros(X_escalado: np.ndarray, logger: logging.Logger, param_grid: Dict = None) -> Tuple:
    """
    Busca los mejores parámetros para Isolation Forest usando búsqueda paralela.
    
    Args:
        X_escalado: Datos escalados para entrenamiento
        logger: Logger para registrar información
        param_grid: Grilla de parámetros a evaluar
        
    Returns:
        Tupla con el mejor modelo y sus métricas
        
    Raises:
        ValueError: Si no se encuentran parámetros válidos
    """
    if param_grid is None:
        param_grid = PARAMETROS_BUSQUEDA
    
    logger.info("Iniciando búsqueda de mejores parámetros...")
    
    # Evaluar parámetros en paralelo (limitado para Google Colab)
    resultados = Parallel(n_jobs=2)(
        delayed(evaluar_modelo_isolation_forest)(params, X_escalado, logger) 
        for params in ParameterGrid(param_grid)
    )
    
    # Filtrar resultados válidos
    resultados_validos = [res for res in resultados if res is not None]
    
    if not resultados_validos:
        raise ValueError("No se encontraron parámetros que produzcan resultados válidos")
    
    # Seleccionar el mejor resultado
    mejor_resultado = max(resultados_validos, key=lambda x: x[0])
    logger.info(f"Mejores parámetros encontrados: {mejor_resultado[1]}")
    
    return mejor_resultado


def guardar_metricas(ruta_archivo: Path, mejor_resultado: Tuple, n_anomalias: int, porcentaje_anomalias: float) -> None:
    """
    Guarda las métricas del modelo en un archivo de texto.
    
    Args:
        ruta_archivo: Ruta donde guardar las métricas
        mejor_resultado: Tupla con los mejores resultados
        n_anomalias: Número de anomalías detectadas
        porcentaje_anomalias: Porcentaje de anomalías
    """
    _, mejores_params, _, silhouette, calinski, davies = mejor_resultado
    
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write(f"Mejores parámetros: {mejores_params}\n")
        f.write(f"Número de anomalías detectadas: {n_anomalias}\n")
        f.write(f"Porcentaje de anomalías detectadas: {porcentaje_anomalias:.4f}%\n")
        f.write(f"Media de puntuaciones de anomalía: {mejor_resultado[0]:.4f}\n")
        
        if silhouette is not None:
            f.write(f"Silhouette Score: {silhouette:.4f}\n")
        if calinski is not None:
            f.write(f"Calinski-Harabasz Score: {calinski:.4f}\n")
        if davies is not None:
            f.write(f"Davies-Bouldin Score: {davies:.4f}\n")


def guardar_modelo(modelo: IsolationForest, scores: np.ndarray, etiquetas: np.ndarray, 
                  ruta_pkl: Path, ruta_h5: Path) -> None:
    """
    Guarda el modelo entrenado en formatos pickle y HDF5.
    
    Args:
        modelo: Modelo entrenado
        scores: Puntuaciones de anomalía
        etiquetas: Etiquetas predichas
        ruta_pkl: Ruta para archivo pickle
        ruta_h5: Ruta para archivo HDF5
    """
    # Guardar como pickle
    joblib.dump(modelo, ruta_pkl)
    
    # Guardar como HDF5
    with h5py.File(ruta_h5, 'w') as hf:
        hf.create_dataset('decision_scores', data=scores)
        hf.create_dataset('etiquetas', data=etiquetas)
        hf.attrs['n_estimators'] = modelo.n_estimators
        hf.attrs['max_samples'] = modelo.max_samples
        hf.attrs['contamination'] = modelo.contamination
        hf.attrs['max_features'] = modelo.max_features


def generar_graficos(scores: np.ndarray, X_pca: np.ndarray, etiquetas: np.ndarray, 
                    directorio_graficas: Path) -> None:
    """
    Genera y guarda los gráficos de visualización.
    
    Args:
        scores: Puntuaciones de anomalía
        X_pca: Datos reducidos con PCA
        etiquetas: Etiquetas de clasificación
        directorio_graficas: Directorio donde guardar los gráficos
    """
    # Gráfico de distribución de puntuaciones
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución de Puntuaciones de Anomalía (Isolation Forest)', fontsize=14)
    plt.xlabel('Puntuación de Anomalía')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    ruta_puntuaciones = directorio_graficas / 'anomaly_scores.png'
    plt.savefig(ruta_puntuaciones, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3D de anomalías
    X_anomalias = X_pca[etiquetas == 1]
    X_normales = X_pca[etiquetas == 0]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotear puntos normales y anomalías
    ax.scatter(X_normales[:, 0], X_normales[:, 1], X_normales[:, 2], 
              c='blue', label='Normales', s=20, alpha=0.6)
    ax.scatter(X_anomalias[:, 0], X_anomalias[:, 1], X_anomalias[:, 2], 
              c='red', label='Anomalías', s=30, alpha=0.8)
    
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    ax.set_title('Detección de Anomalías con Isolation Forest (3D)', fontsize=14)
    ax.legend()
    
    ruta_anomalias_3d = directorio_graficas / 'anomalies_3d.png'
    plt.savefig(ruta_anomalias_3d, dpi=300, bbox_inches='tight')
    plt.close()


def guardar_anomalias(datos_originales: pd.DataFrame, etiquetas: np.ndarray, 
                     scores: np.ndarray, ruta_archivo: Path) -> None:
    """
    Guarda las anomalías detectadas en un archivo CSV.
    
    Args:
        datos_originales: Datos originales
        etiquetas: Etiquetas de clasificación
        scores: Puntuaciones de anomalía
        ruta_archivo: Ruta donde guardar el archivo
    """
    anomalias = datos_originales[etiquetas == 1].copy()
    anomalias['puntuacion_anomalia'] = scores[etiquetas == 1]
    anomalias = anomalias.sort_values('puntuacion_anomalia', ascending=True)
    anomalias.to_csv(ruta_archivo, index=False)


def main() -> None:
    """Función principal del programa."""
    try:
        # Configuración inicial
        directorio_script = Path(__file__).parent
        directorios = DirectoriosProyecto(directorio_script)
        directorios.crear_directorios()
        
        # Configurar logging
        logger = configurar_logging(directorios.directorio_metricas / ARCHIVO_LOG)
        logger.info("Iniciando proceso de detección de anomalías con Isolation Forest...")
        
        # Cargar y preprocesar datos
        ruta_datos = directorio_script / NOMBRE_ARCHIVO_DATOS
        datos = cargar_datos(ruta_datos)
        logger.info(f"Datos cargados: {len(datos)} registros")
        
        X, caracteristicas = preprocesar_datos(datos)
        logger.info(f"Datos preprocesados: {X.shape}")
        logger.info(f"Características utilizadas: {caracteristicas}")
        
        # Escalar datos
        X_escalado, escalador = escalar_datos(X)
        logger.info("Datos escalados correctamente")
        
        # Reducción de dimensionalidad
        X_pca, pca = reducir_dimensionalidad(X_escalado)
        logger.info(f"Reducción de dimensionalidad completada: {X_pca.shape}")
        
        # Reducir muestra para optimización si es muy grande
        X_para_optimizacion, indices_muestra = reducir_muestra_para_optimizacion(X_escalado)
        
        # Búsqueda de mejores parámetros usando muestra reducida
        mejor_resultado = buscar_mejores_parametros(X_para_optimizacion, logger)
        
        # Aplicar el mejor modelo al dataset completo
        _, mejores_params, _, _, _, _ = mejor_resultado
        logger.info(f"Aplicando mejores parámetros al dataset completo...")
        
        modelo_final = IsolationForest(
            n_estimators=mejores_params['n_estimators'],
            max_samples=mejores_params['max_samples'],
            contamination=mejores_params['contamination'],
            max_features=mejores_params['max_features'],
            random_state=RANDOM_STATE
        )
        modelo_final.fit(X_escalado)
        
        # Hacer predicciones con el modelo final en el dataset completo
        scores_pred = modelo_final.decision_function(X_escalado)
        etiquetas_pred = modelo_final.predict(X_escalado)
        etiquetas_pred = np.where(etiquetas_pred == 1, 0, 1)
        
        n_anomalias = np.sum(etiquetas_pred)
        porcentaje_anomalias = (n_anomalias / len(etiquetas_pred)) * 100
        
        logger.info(f"Detección completada: {n_anomalias} anomalías ({porcentaje_anomalias:.2f}%)")
        
        # Guardar resultados
        ruta_metricas = directorios.directorio_metricas / ARCHIVO_METRICAS
        guardar_metricas(ruta_metricas, mejor_resultado, n_anomalias, porcentaje_anomalias)
        logger.info(f"Métricas guardadas en {ruta_metricas}")
        
        # Guardar modelo
        ruta_modelo_pkl = directorios.directorio_modelos / EXTENSION_MODELO_PKL
        ruta_modelo_h5 = directorios.directorio_modelos / EXTENSION_MODELO_H5
        guardar_modelo(modelo_final, scores_pred, etiquetas_pred, ruta_modelo_pkl, ruta_modelo_h5)
        logger.info(f"Modelo guardado en formatos pickle y HDF5")
        
        # Generar gráficos
        generar_graficos(scores_pred, X_pca, etiquetas_pred, directorios.directorio_graficas)
        logger.info("Gráficos generados y guardados")
        
        # Guardar anomalías detectadas
        ruta_anomalias = directorios.directorio_metricas / ARCHIVO_ANOMALIAS
        guardar_anomalias(datos, etiquetas_pred, scores_pred, ruta_anomalias)
        logger.info(f"Anomalías guardadas en {ruta_anomalias}")
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":
    main()