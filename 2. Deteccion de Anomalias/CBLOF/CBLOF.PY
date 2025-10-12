import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from pathlib import Path
from pyod.models.cblof import CBLOF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import h5py
import logging
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from typing import Tuple, Dict, Any, Optional, List
import warnings
import time
import tracemalloc
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

ARCHIVO_LOG = 'output.log'
ARCHIVO_METRICAS = 'metrics.txt'
ARCHIVO_ANOMALIAS = 'anomalies.csv'
ARCHIVO_MODELO_PKL = 'cblof_model.pkl'
ARCHIVO_MODELO_H5 = 'cblof_model.h5'

USECOLS = config.USECOLS
CARACTERISTICAS_BASE = config.CARACTERISTICAS_BASE
CARACTERISTICA_MAGNITUD = config.CARACTERISTICA_MAGNITUD
PARAMETROS_PCA = {'n_components': 3}
CONTAMINACION_DEFAULT = 0.1
RANDOM_STATE = config.RANDOM_STATE
N_JOBS = config.N_JOBS_PARALELO
COLOR_NORMAL = config.COLOR_NORMAL
COLOR_ANOMALIA = config.COLOR_ANOMALIA
ALPHA_NORMAL = config.ALPHA_NORMAL
ALPHA_ANOMALIA = config.ALPHA_ANOMALIA
SCATTER_SIZE_NORMAL = config.SCATTER_SIZE_NORMAL
SCATTER_SIZE_ANOMALIA = config.SCATTER_SIZE_ANOMALIA
FIGSIZE_2D = config.FIGSIZE_2D
FIGSIZE_3D = config.FIGSIZE_3D

PARAM_GRID = {
    'n_clusters': [5],
    'alpha': [0.8],
    'beta': [5],
    'use_weights': [True]
}

MAX_MUESTRAS_OPTIMIZACION = config.MAX_MUESTRAS_OPTIMIZACION
MAX_MUESTRAS_VISUALIZATION = config.MAX_MUESTRAS_VISUALIZATION


def configurar_logging(directorio_metricas: Path) -> None:
    """
    Configura el sistema de logging para archivo y consola.
    
    Args:
        directorio_metricas: Directorio donde guardar el archivo de log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    ruta_archivo_log = directorio_metricas / ARCHIVO_LOG
    file_handler = logging.FileHandler(ruta_archivo_log, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)


def crear_directorios(directorio_base: Path) -> Tuple[Path, Path, Path]:
    """
    Crea los directorios necesarios para el proyecto.
    
    Args:
        directorio_base: Directorio base del script
        
    Returns:
        Tuple con las rutas de modelos, gráficas y métricas
    """
    directorio_modelos = directorio_base / 'modelos_entrenados_CBLOF'
    directorio_graficas = directorio_base / 'graficas_CBLOF'
    directorio_metricas = directorio_base / 'metricas_CBLOF'
    
    for directorio in [directorio_modelos, directorio_graficas, directorio_metricas]:
        directorio.mkdir(exist_ok=True)
    
    return directorio_modelos, directorio_graficas, directorio_metricas


def cargar_y_validar_datos(ruta_datos: Path) -> pd.DataFrame:
    """
    Carga y valida los datos desde el archivo CSV (100% no-supervisado).
    
    Args:
        ruta_datos: Ruta al archivo de datos
        
    Returns:
        DataFrame con los datos cargados y validados
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si los datos no contienen las columnas requeridas
    """
    if not ruta_datos.exists():
        raise FileNotFoundError(f"El archivo de datos no existe: {ruta_datos}")
    
    try:
        try:
            datos = pd.read_csv(
                ruta_datos, 
                usecols=USECOLS, 
                parse_dates=['fecha'], 
                dayfirst=True,
                dtype={
                    'acceleration_x': 'float32',
                    'acceleration_y': 'float32', 
                    'acceleration_z': 'float32'
                },
                encoding='utf-8'
            )
            logging.info("Datos cargados correctamente con encoding: utf-8")
        except UnicodeDecodeError:
            datos = pd.read_csv(
                ruta_datos, 
                usecols=USECOLS, 
                parse_dates=['fecha'], 
                dayfirst=True,
                dtype={
                    'acceleration_x': 'float32',
                    'acceleration_y': 'float32', 
                    'acceleration_z': 'float32'
                },
                encoding='latin-1'
            )
            logging.info("Datos cargados correctamente con encoding: latin-1")
        
        datos = datos[['fecha'] + CARACTERISTICAS_BASE].copy()
        
        datos.sort_values('fecha', inplace=True)
        
        logging.info(f"Muestra de los datos (primeras 3 filas):")
        logging.info(f"{datos.head(3).to_string()}")
        
        if len(datos) < 100:
            logging.warning(f"Advertencia: Pocos datos disponibles: {len(datos)} registros")
        
        for col in CARACTERISTICAS_BASE:
            muestra_no_nula = datos[col].dropna().head(5)
            logging.info(f"Muestra de valores en {col}: {list(muestra_no_nula)}")
        
        return datos
        
    except Exception as e:
        logging.error(f"Error al cargar los datos: {e}")
        raise


def preprocesar_datos(datos: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Preprocesa los datos: limpieza, ingeniería de características y escalado.
    
    Args:
        datos: DataFrame con los datos originales
        
    Returns:
        Tuple con los datos escalados, el escalador ajustado y el DataFrame limpio
    """
    logging.info(f"Datos cargados: {len(datos)} filas, {len(datos.columns)} columnas")
    logging.info(f"Columnas disponibles: {list(datos.columns)}")
    
    datos_limpios = datos.dropna()
    filas_eliminadas = len(datos) - len(datos_limpios)
    
    if filas_eliminadas > 0:
        logging.info(f"Valores faltantes eliminados: {filas_eliminadas} filas")
    
    for col in CARACTERISTICAS_BASE:
        if datos_limpios[col].dtype == 'object':
            logging.info(f"Convirtiendo columna {col} de texto a numérico")
            datos_limpios[col] = pd.to_numeric(datos_limpios[col], errors='coerce')
    
    datos_limpios = datos_limpios.dropna(subset=CARACTERISTICAS_BASE)
    
    datos_limpios = datos_limpios.copy()
    datos_limpios[CARACTERISTICA_MAGNITUD] = np.sqrt(
        datos_limpios['acceleration_x']**2 +
        datos_limpios['acceleration_y']**2 +
        datos_limpios['acceleration_z']**2
    )
    logging.info(f"Característica '{CARACTERISTICA_MAGNITUD}' añadida.")
    
    caracteristicas = CARACTERISTICAS_BASE + [CARACTERISTICA_MAGNITUD]
    X = datos_limpios[caracteristicas].values
    
    logging.info(f"Estadísticas de los datos:")
    for i, col in enumerate(caracteristicas):
        logging.info(f"  {col}: min={X[:, i].min():.4f}, max={X[:, i].max():.4f}, media={X[:, i].mean():.4f}")
    
    if np.any(np.isinf(X)) or np.any(np.isnan(X)):
        logging.warning("Se encontraron valores infinitos o NaN después del preprocesamiento")
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    logging.info("Datos escalados correctamente.")
    
    return X_escalado, escalador, datos_limpios


def aplicar_pca(X_escalado: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """
    Aplica reducción de dimensionalidad usando PCA.
    
    Args:
        X_escalado: Datos escalados
        
    Returns:
        Tuple con los datos transformados por PCA y el objeto PCA ajustado
    """
    pca = PCA(**PARAMETROS_PCA)
    X_pca = pca.fit_transform(X_escalado)
    
    varianza_explicada = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA aplicado. Varianza explicada: {varianza_explicada:.4f}")
    
    return X_pca, pca


def reducir_muestra_para_optimizacion(X: np.ndarray, max_muestras: int = MAX_MUESTRAS_OPTIMIZACION) -> Tuple[np.ndarray, np.ndarray]:
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
    
    np.random.seed(RANDOM_STATE)
    indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
    indices_seleccionados = np.sort(indices_seleccionados)
    
    X_reducido = X[indices_seleccionados]
    
    logging.info(f"Dataset reducido para optimización: {len(X)} -> {len(X_reducido)} muestras")
    return X_reducido, indices_seleccionados


def reducir_muestra_para_visualizacion(X: np.ndarray, etiquetas: np.ndarray, max_muestras: int = MAX_MUESTRAS_VISUALIZATION) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce el tamaño de la muestra para visualización si es muy grande.
    
    Args:
        X: Matriz de características completa.
        etiquetas: Etiquetas correspondientes.
        max_muestras: Número máximo de muestras para visualización.
        
    Returns:
        Tupla con (muestra_reducida, etiquetas_reducidas).
    """
    if len(X) <= max_muestras:
        return X, etiquetas
    
    np.random.seed(RANDOM_STATE)
    indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
    indices_seleccionados = np.sort(indices_seleccionados)
    
    X_reducido = X[indices_seleccionados]
    etiquetas_reducidas = etiquetas[indices_seleccionados]
    
    logging.info(f"Dataset reducido para visualización: {len(X)} -> {len(X_reducido)} muestras")
    return X_reducido, etiquetas_reducidas


def evaluar_parametros_cblof(params: Dict[str, Any], X_escalado: np.ndarray) -> Optional[Tuple]:
    """
    Evalúa un conjunto de parámetros para el modelo CBLOF.
    
    Args:
        params: Diccionario con los parámetros a evaluar
        X_escalado: Datos escalados para entrenamiento
        
    Returns:
        Tuple con las métricas y el modelo entrenado, o None si falla
    """
    try:
        clf = CBLOF(
            n_clusters=params['n_clusters'],
            alpha=params['alpha'],
            beta=params['beta'],
            use_weights=params['use_weights'],
            contamination=CONTAMINACION_DEFAULT,
            random_state=RANDOM_STATE
        )
        
        clf.fit(X_escalado)
        anomaly_scores = clf.decision_scores_
        
        p95_score = np.percentile(anomaly_scores, 95)
        p50_score = np.percentile(anomaly_scores, 50)
        separacion_scores = p95_score - p50_score
        
        std_scores = np.std(anomaly_scores)
        mean_scores = np.mean(anomaly_scores)
        
        logging.info(f"Parámetros: {params}")
        logging.info(f"Separación scores (P95-P50): {separacion_scores:.4f}")
        logging.info(f"Std scores: {std_scores:.4f}")
        logging.info(f"Mean scores: {mean_scores:.4f}")
        
        return (separacion_scores, std_scores, mean_scores, params, clf)
        
    except Exception as e:
        logging.warning(f"Error evaluando parámetros {params}: {e}")
        return None


def buscar_mejores_parametros(X_escalado: np.ndarray) -> Tuple[CBLOF, Dict[str, float]]:
    """
    Busca los mejores parámetros para el modelo CBLOF usando búsqueda en grid.
    
    Args:
        X_escalado: Datos escalados para entrenamiento
        
    Returns:
        Tuple con el mejor modelo entrenado y las métricas
        
    Raises:
        ValueError: Si no se encuentran parámetros válidos
    """
    logging.info("Iniciando búsqueda de mejores parámetros...")
    
    resultados = Parallel(n_jobs=N_JOBS)(
        delayed(evaluar_parametros_cblof)(params, X_escalado) 
        for params in ParameterGrid(PARAM_GRID)
    )
    
    resultados_validos = [res for res in resultados if res is not None]
    
    if not resultados_validos:
        raise ValueError("No se encontraron parámetros que produzcan clusters válidos")
    
    mejor_resultado = max(resultados_validos, key=lambda x: x[0])
    mejor_separacion, mejor_std, mejor_mean, mejores_parametros, mejor_modelo = mejor_resultado
    
    metricas = {
        'separacion_scores': mejor_separacion,
        'std_scores': mejor_std,
        'mean_scores': mejor_mean
    }
    
    logging.info(f"\nMejores parámetros encontrados: {mejores_parametros}")
    logging.info(f"Mejor separación de scores (P95-P50): {mejor_separacion:.4f}")
    logging.info(f"Std de scores: {mejor_std:.4f}")
    logging.info(f"Mean de scores: {mejor_mean:.4f}")
    
    return mejor_modelo, metricas


def calcular_metricas_adicionales(modelo: CBLOF, X_escalado: np.ndarray) -> Dict[str, Any]:
    """
    Calcula métricas adicionales del modelo entrenado.
    
    Args:
        modelo: Modelo CBLOF entrenado
        X_escalado: Datos escalados
        
    Returns:
        Diccionario con las métricas calculadas
    """
    binary_labels = modelo.labels_
    anomaly_scores = modelo.decision_scores_
    
    n_anomalias = np.sum(binary_labels)
    porcentaje_anomalias = (n_anomalias / len(binary_labels)) * 100
    
    logging.info(f"Anomalías detectadas: {n_anomalias} ({porcentaje_anomalias:.2f}%)")
    logging.info(f"Rango de scores de anomalía: {np.min(anomaly_scores):.4f} - {np.max(anomaly_scores):.4f}")
    
    media_puntuaciones_anomalia = np.mean(anomaly_scores)
    
    return {
        'n_anomalias': n_anomalias,
        'porcentaje_anomalias': porcentaje_anomalias,
        'media_puntuaciones_anomalia': media_puntuaciones_anomalia,
        'min_score': np.min(anomaly_scores),
        'max_score': np.max(anomaly_scores),
        'std_score': np.std(anomaly_scores)
    }


def guardar_metricas(metricas: Dict[str, Any], metricas_adicionales: Dict[str, Any],
                    mejores_parametros: Dict[str, Any], directorio_metricas: Path) -> None:
    """
    Guarda todas las métricas en un archivo de texto.
    
    Args:
        metricas: Métricas principales del modelo
        metricas_adicionales: Métricas adicionales calculadas
        mejores_parametros: Mejores parámetros encontrados
        directorio_metricas: Directorio donde guardar las métricas
    """
    ruta_metricas = directorio_metricas / ARCHIVO_METRICAS
    
    with open(ruta_metricas, 'w', encoding='utf-8') as f:
        f.write("=== MÉTRICAS DEL MODELO CBLOF ===\n\n")
        f.write(f"Mejores parámetros: {mejores_parametros}\n\n")
        
        f.write("Métricas de optimización:\n")
        f.write(f"Separación de scores (P95-P50): {metricas['separacion_scores']:.4f}\n")
        f.write(f"Desviación estándar de scores: {metricas['std_scores']:.4f}\n")
        f.write(f"Media de scores: {metricas['mean_scores']:.4f}\n\n")
        
        f.write("Métricas de anomalías:\n")
        f.write(f"Número de anomalías detectadas: {metricas_adicionales['n_anomalias']}\n")
        f.write(f"Porcentaje de anomalías detectadas: {metricas_adicionales['porcentaje_anomalias']:.4f}%\n")
        f.write(f"Media de puntuaciones de anomalía: {metricas_adicionales['media_puntuaciones_anomalia']:.4f}\n\n")
        
        f.write("Estadísticas de puntuaciones:\n")
        f.write(f"Score mínimo: {metricas_adicionales['min_score']:.4f}\n")
        f.write(f"Score máximo: {metricas_adicionales['max_score']:.4f}\n")
        f.write(f"Desviación estándar: {metricas_adicionales['std_score']:.4f}\n")
    
    logging.info(f"Métricas guardadas en {ruta_metricas}")


def guardar_modelo(modelo: CBLOF, directorio_modelos: Path) -> None:
    """
    Guarda el modelo entrenado en formato pickle y HDF5.
    
    Args:
        modelo: Modelo CBLOF entrenado
        directorio_modelos: Directorio donde guardar los modelos
    """
    ruta_modelo_pkl = directorio_modelos / ARCHIVO_MODELO_PKL
    try:
        joblib.dump(modelo, ruta_modelo_pkl)
        logging.info(f"Modelo guardado como pickle en {ruta_modelo_pkl}")
    except Exception as e:
        logging.error(f"Error guardando modelo pickle: {e}")
    
    ruta_modelo_h5 = directorio_modelos / ARCHIVO_MODELO_H5
    try:
        with h5py.File(ruta_modelo_h5, 'w') as hf:
            hf.create_dataset('cluster_labels', data=modelo.labels_)
            hf.create_dataset('decision_scores', data=modelo.decision_scores_)
            hf.attrs['n_clusters'] = modelo.n_clusters
            hf.attrs['alpha'] = modelo.alpha
            hf.attrs['beta'] = modelo.beta
            hf.attrs['use_weights'] = modelo.use_weights
            hf.attrs['contamination'] = modelo.contamination
        logging.info(f"Modelo guardado como HDF5 en {ruta_modelo_h5}")
    except Exception as e:
        logging.error(f"Error guardando modelo HDF5: {e}")


def generar_visualizaciones(modelo: CBLOF, X_pca: np.ndarray, datos: pd.DataFrame,
                          directorio_graficas: Path) -> None:
    """
    Genera las visualizaciones del análisis de anomalías.
    
    Args:
        modelo: Modelo CBLOF entrenado
        X_pca: Datos transformados por PCA
        datos: Datos originales
        directorio_graficas: Directorio donde guardar las gráficas
    """
    anomaly_scores = modelo.decision_scores_
    
    plt.ioff()
    
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribución de las Puntuaciones de Anomalía (CBLOF)', fontsize=14)
        plt.xlabel('Puntuación de Anomalía', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        ruta_puntuaciones = directorio_graficas / 'anomaly_scores.png'
        plt.savefig(ruta_puntuaciones, dpi=200, bbox_inches='tight')
        plt.close("all")
        logging.info(f"Gráfico de puntuaciones guardado en {ruta_puntuaciones}")
    except Exception as e:
        logging.error(f"Error generando gráfico de puntuaciones: {e}")
    
    try:
        labels_visualizacion = np.where(modelo.labels_ == 0, 0, 1)
        
        X_vis, labels_vis = reducir_muestra_para_visualizacion(X_pca, labels_visualizacion)
        
        X_anomalias = X_vis[labels_vis == 1]
        X_normales = X_vis[labels_vis == 0]
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(X_normales) > 0:
            ax.scatter(X_normales[:, 0], X_normales[:, 1], X_normales[:, 2], 
                      c='blue', label=f'Normales ({len(X_normales)})', s=15, alpha=0.6)
        
        if len(X_anomalias) > 0:
            ax.scatter(X_anomalias[:, 0], X_anomalias[:, 1], X_anomalias[:, 2], 
                      c='red', label=f'Anomalías ({len(X_anomalias)})', s=20, alpha=0.8)
        
        ax.set_xlabel('Componente Principal 1', fontsize=12)
        ax.set_ylabel('Componente Principal 2', fontsize=12)
        ax.set_zlabel('Componente Principal 3', fontsize=12)
        ax.set_title('Detección de Anomalías con CBLOF en 3D', fontsize=14)
        
        if len(X_vis) > 0:
            x_range = np.ptp(X_vis[:, 0])
            y_range = np.ptp(X_vis[:, 1])
            z_range = np.ptp(X_vis[:, 2])
            max_range = max(x_range, y_range, z_range)
            
            x_center = np.mean(X_vis[:, 0])
            y_center = np.mean(X_vis[:, 1])
            z_center = np.mean(X_vis[:, 2])
            
            ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
            
            ax.set_box_aspect([1,1,1])
        
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        ruta_anomalias_3d = directorio_graficas / 'anomalies_3d.png'
        plt.savefig(ruta_anomalias_3d, dpi=200, bbox_inches='tight')
        plt.close("all")
        logging.info(f"Gráfico 3D guardado en {ruta_anomalias_3d}")
    except Exception as e:
        logging.error(f"Error generando gráfico 3D: {e}")


def guardar_anomalias_detectadas(modelo: CBLOF, datos: pd.DataFrame, 
                                directorio_metricas: Path) -> None:
    """
    Guarda las anomalías detectadas en un archivo CSV (salida estandarizada).
    
    Args:
        modelo: Modelo CBLOF entrenado
        datos: Datos originales
        directorio_metricas: Directorio donde guardar el archivo
    """
    try:
        datos_salida = datos.copy()
        datos_salida['anomaly_score'] = modelo.decision_scores_
        datos_salida['is_outlier'] = modelo.labels_
        
        ruta_salida_completa = directorio_metricas / 'anomaly_scores.csv'
        datos_salida[['fecha'] + CARACTERISTICAS_BASE + ['is_outlier', 'anomaly_score']].to_csv(ruta_salida_completa, index=False)
        logging.info(f"Salida completa guardada en {ruta_salida_completa}")
        
        anomalias = datos_salida[modelo.labels_ == 1].copy()
        
        if not anomalias.empty:
            if 'fecha' in anomalias.columns:
                anomalias = anomalias.sort_values(['fecha', 'anomaly_score'], ascending=[True, False])
            else:
                anomalias = anomalias.sort_values('anomaly_score', ascending=False)
            
            ruta_anomalias_csv = directorio_metricas / ARCHIVO_ANOMALIAS
            anomalias.to_csv(ruta_anomalias_csv, index=False)
            logging.info(f"Anomalías guardadas en {ruta_anomalias_csv}")
        else:
            logging.info("No se detectaron anomalías para guardar")
    except Exception as e:
        logging.error(f"Error guardando anomalías: {e}")


def main() -> None:
    """Función principal que ejecuta todo el pipeline de detección de anomalías."""
    tiempo_inicio = time.time()
    tracemalloc.start()
    
    try:
        directorio_script = Path(__file__).parent.absolute()
        directorio_modelos, directorio_graficas, directorio_metricas = crear_directorios(directorio_script)
        configurar_logging(directorio_metricas)
        
        logging.info("=== INICIANDO PROCESO DE DETECCIÓN DE ANOMALÍAS CON CBLOF ===")
        
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        # Usar archivo de datos centralizado en la raíz del proyecto
        ruta_datos = directorio_script.parent.parent / config.RUTA_DATOS_COMPARTIDA
        datos = cargar_y_validar_datos(ruta_datos)
        
        X_escalado, escalador, datos_limpios = preprocesar_datos(datos)
        logging.info(f"Datos preprocesados: {X_escalado.shape[0]} muestras, {X_escalado.shape[1]} características")
        
        X_pca, pca = aplicar_pca(X_escalado)
        
        X_para_optimizacion, indices_muestra = reducir_muestra_para_optimizacion(X_escalado)
        
        mejor_modelo_muestra, metricas = buscar_mejores_parametros(X_para_optimizacion)
        
        mejores_parametros = {
            'n_clusters': mejor_modelo_muestra.n_clusters,
            'alpha': mejor_modelo_muestra.alpha,
            'beta': mejor_modelo_muestra.beta,
            'use_weights': mejor_modelo_muestra.use_weights
        }
        
        logging.info(f"Aplicando mejores parámetros al dataset completo...")
        mejor_modelo = CBLOF(
            n_clusters=mejores_parametros['n_clusters'],
            alpha=mejores_parametros['alpha'],
            beta=mejores_parametros['beta'],
            use_weights=mejores_parametros['use_weights'],
            contamination=CONTAMINACION_DEFAULT,
            random_state=RANDOM_STATE
        )
        mejor_modelo.fit(X_escalado)
        
        metricas_adicionales = calcular_metricas_adicionales(mejor_modelo, X_escalado)
        
        
        guardar_metricas(metricas, metricas_adicionales, mejores_parametros, directorio_metricas)
        guardar_modelo(mejor_modelo, directorio_modelos)
        
        ruta_escalador = directorio_modelos / 'scaler.pkl'
        joblib.dump(escalador, ruta_escalador)
        
        ruta_pca = directorio_modelos / 'pca.pkl'
        joblib.dump(pca, ruta_pca)
        
        logging.info(f"Escalador guardado en {ruta_escalador}")
        logging.info(f"PCA guardado en {ruta_pca}")
        
        tiempo_total = time.time() - tiempo_inicio
        memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
        memoria_max = memoria_pico / 1024**2
        tracemalloc.stop()
        
        logging.info(f"Tiempo total de ejecucion: {tiempo_total:.2f} segundos")
        logging.info(f"Memoria maxima utilizada: {memoria_max:.2f} MB")
        
        generar_visualizaciones(mejor_modelo, X_pca, datos_limpios, directorio_graficas)
        
        guardar_anomalias_detectadas(mejor_modelo, datos_limpios, directorio_metricas)
        
        ruta_metricas_csv = directorio_metricas / 'metrics.csv'
        guardar_metricas_csv(mejor_modelo, datos_limpios, mejores_parametros, metricas, ruta_metricas_csv,
                            tiempo_total, memoria_max)
        logging.info(f"Métricas CSV guardadas en {ruta_metricas_csv}")
        
        logging.info("=== PROCESO COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        tracemalloc.stop()
        logging.error(f"Error en el proceso principal: {e}")
        raise


def guardar_metricas_csv(modelo: CBLOF, datos: pd.DataFrame, mejores_parametros: Dict[str, Any], 
                        metricas: Dict[str, float], ruta_archivo: Path,
                        tiempo_total: float = 0.0, memoria_max: float = 0.0) -> None:
    """
    Guarda métricas estandarizadas en formato CSV para comparación entre algoritmos.
    
    Args:
        modelo: Modelo CBLOF entrenado
        datos: DataFrame con datos originales
        mejores_parametros: Parámetros del mejor modelo
        metricas: Métricas del modelo
        ruta_archivo: Ruta donde guardar el CSV
    """
    anomaly_scores_raw = modelo.decision_scores_
    
    anomaly_scores = config.normalizar_scores_min_max(anomaly_scores_raw)
    
    etiquetas_binarias = modelo.labels_
    
    p95_score = np.percentile(anomaly_scores, 95)
    p50_score = np.percentile(anomaly_scores, 50)
    separacion_normalizada = p95_score - p50_score
    
    metricas_csv = {
        'algoritmo': 'CBLOF',
        'params_json': str(mejores_parametros),
        'n_clusters': mejores_parametros.get('n_clusters', None),
        'silhouette_score': None,
        'calinski_harabasz_score': None,
        'davies_bouldin_score': None,
        'pct_anomalias': np.mean(etiquetas_binarias) * 100,
        'p95_minus_p50': separacion_normalizada,
        'mean_score': np.mean(anomaly_scores),
        'tiempo_ejecucion_s': tiempo_total,
        'memoria_max_mb': memoria_max
    }
    

    
    df_metricas = pd.DataFrame([metricas_csv])
    df_metricas.to_csv(ruta_archivo, index=False)


if __name__ == "__main__":
    main()