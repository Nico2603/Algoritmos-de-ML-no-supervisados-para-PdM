import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import time
import tracemalloc
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo ANTES de importar pyplot
import matplotlib.pyplot as plt
plt.ioff()  # Desactivar modo interactivo
import joblib
import h5py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

# Configurar UTF-8 para salida estándar
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Importar configuración centralizada
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=FutureWarning)

# Usar constantes del config centralizado
USECOLS = config.USECOLS
CARACTERISTICAS_BASE = config.CARACTERISTICAS_BASE
NOMBRE_ARCHIVO_DATOS = 'data.csv'
EXTENSION_MODELO_PKL = 'isolation_forest_model.pkl'
EXTENSION_MODELO_H5 = 'isolation_forest_model.h5'
ARCHIVO_METRICAS = 'metrics.txt'
ARCHIVO_LOG = 'output.log'
ARCHIVO_ANOMALIAS = 'anomalies.csv'
COMPONENTES_PCA = 3
RANDOM_STATE = config.RANDOM_STATE
COLOR_NORMAL = config.COLOR_NORMAL
COLOR_ANOMALIA = config.COLOR_ANOMALIA
ALPHA_NORMAL = config.ALPHA_NORMAL
ALPHA_ANOMALIA = config.ALPHA_ANOMALIA
SCATTER_SIZE_NORMAL = config.SCATTER_SIZE_NORMAL
SCATTER_SIZE_ANOMALIA = config.SCATTER_SIZE_ANOMALIA
FIGSIZE_2D = config.FIGSIZE_2D
FIGSIZE_3D = config.FIGSIZE_3D

# Parámetros por defecto para búsqueda de hiperparámetros (optimizado para velocidad)
PARAMETROS_BUSQUEDA = {
    'n_estimators': [50, 100],  # Reducido para velocidad
    'max_samples': ['auto'],    # Solo auto para simplicidad
    'contamination': [0.05, 0.1],
    'max_features': [1.0]
}
# Optimización de muestreo - usar valores del config
MAX_MUESTRAS_OPTIMIZACION = config.MAX_MUESTRAS_OPTIMIZACION
MAX_MUESTRAS_VISUALIZATION = config.MAX_MUESTRAS_VISUALIZATION


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
    formatter = logging.Formatter('%(message)s')  # Formato simple sin timestamp
    
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
    Carga los datos desde archivo CSV con validación (100% no-supervisado).
    
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
        # BLINDADO: Cargar solo las 4 columnas necesarias con tipos optimizados
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
                }
            )
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
            print("Archivo cargado con encoding latin-1")
        
        if datos.empty:
            raise ValueError("El archivo de datos está vacío")
        
        # BLINDADO: Recortar DataFrame a solo las 4 columnas por si el CSV trae más
        datos = datos[['fecha'] + CARACTERISTICAS_BASE].copy()
        
        # Ordenar por fecha (recomendable)
        datos.sort_values('fecha', inplace=True)
        
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
    
    # Aplicar seeds consistentemente
    config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
    
    indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
    indices_seleccionados = np.sort(indices_seleccionados)
    
    X_reducido = X[indices_seleccionados]
    
    print(f"Dataset reducido para optimización: {len(X)} -> {len(X_reducido)} muestras")
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
    
    # Aplicar seeds consistentemente
    config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
    
    indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
    indices_seleccionados = np.sort(indices_seleccionados)
    
    X_reducido = X[indices_seleccionados]
    etiquetas_reducidas = etiquetas[indices_seleccionados]
    
    print(f"Dataset reducido para visualización: {len(X)} -> {len(X_reducido)} muestras")
    return X_reducido, etiquetas_reducidas


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
        # CORRIGIDO: Invertir el score para que mayor valor = más anómalo
        scores_raw = -clf.decision_function(X_escalado)  # Invertir para score consistente
        
        # Normalizar scores al rango [0, 1] usando función centralizada
        scores = config.normalizar_scores_min_max(scores_raw)
        
        etiquetas = clf.predict(X_escalado)
        etiquetas = np.where(etiquetas == 1, 0, 1)  # 0: normal, 1: anomalía
        
        # CORRIGIDO: Usar separación de percentiles en lugar de media
        # Calcular separación entre anomalías y normales
        p95_score = np.percentile(scores, 95)
        p50_score = np.percentile(scores, 50)
        separacion_scores = p95_score - p50_score
        
        # Métricas adicionales
        n_anomalias = np.sum(etiquetas)
        porcentaje_anomalias = (n_anomalias / len(etiquetas)) * 100
        std_scores = np.std(scores)
        
        logger.info(f"Parámetros: {params}")
        logger.info(f"Anomalías detectadas: {n_anomalias} ({porcentaje_anomalias:.2f}%)")
        logger.info(f"Separación scores (P95-P50): {separacion_scores:.4f}")
        logger.info(f"Std scores: {std_scores:.4f}")
        
        return (separacion_scores, params, clf)
        
    except Exception as e:
        logger.error(f"Error evaluando parámetros {params}: {e}")
        return None





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
    
    # Evaluar parámetros en paralelo (limitado para estabilidad)
    resultados = Parallel(n_jobs=1)(
        delayed(evaluar_modelo_isolation_forest)(params, X_escalado, logger) 
        for params in ParameterGrid(param_grid)
    )
    
    # Filtrar resultados válidos
    resultados_validos = [res for res in resultados if res is not None]
    
    if not resultados_validos:
        raise ValueError("No se encontraron parámetros que produzcan resultados válidos")
    
    # Seleccionar el mejor resultado por separación de scores (mayor es mejor)
    mejor_resultado = max(resultados_validos, key=lambda x: x[0])
    logger.info(f"Mejores parámetros encontrados: {mejor_resultado[1]}")
    logger.info(f"Mejor separación de scores: {mejor_resultado[0]:.4f}")
    
    return mejor_resultado


def guardar_metricas(ruta_archivo: Path, mejor_resultado: Tuple, n_anomalias: int, porcentaje_anomalias: float, 
                    scores: np.ndarray) -> None:
    """
    Guarda las métricas del modelo en un archivo de texto (formato estandarizado con CBLOF).
    
    Args:
        ruta_archivo: Ruta donde guardar las métricas
        mejor_resultado: Tupla con los mejores resultados
        n_anomalias: Número de anomalías detectadas
        porcentaje_anomalias: Porcentaje de anomalías
        scores: Scores de anomalía
    """
    separacion_scores, mejores_params, _ = mejor_resultado
    
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write("=== MÉTRICAS DEL MODELO ISOLATION FOREST ===\n\n")
        f.write(f"Mejores parámetros: {mejores_params}\n\n")
        
        f.write("Métricas de optimización:\n")
        f.write(f"Separación de scores (P95-P50): {separacion_scores:.4f}\n")
        f.write(f"Desviación estándar de scores: {np.std(scores):.4f}\n")
        f.write(f"Media de scores: {np.mean(scores):.4f}\n\n")
        
        f.write("Métricas de anomalías:\n")
        f.write(f"Número de anomalías detectadas: {n_anomalias}\n")
        f.write(f"Porcentaje de anomalías detectadas: {porcentaje_anomalias:.4f}%\n")
        f.write(f"Media de puntuaciones de anomalía: {np.mean(scores):.4f}\n\n")
        
        f.write("Estadísticas de puntuaciones:\n")
        f.write(f"Score mínimo: {np.min(scores):.4f}\n")
        f.write(f"Score máximo: {np.max(scores):.4f}\n")
        f.write(f"Desviación estándar: {np.std(scores):.4f}\n")


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
    n_anomalias = np.sum(etiquetas)
    pct_anomalias = (n_anomalias / len(etiquetas)) * 100
    
    plt.figure(figsize=FIGSIZE_2D)
    plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribución de Puntuaciones de Anomalía (Isolation Forest)\n'
             f'Anomalías: {n_anomalias} ({pct_anomalias:.2f}%) | Muestras: {len(scores):,}',
             fontsize=14, pad=15)
    plt.xlabel('Puntuación de Anomalía [0-1]')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ruta_puntuaciones = directorio_graficas / 'anomaly_scores.png'
    plt.savefig(ruta_puntuaciones, dpi=200, bbox_inches='tight')
    plt.close("all")
    
    # Gráfico 3D de anomalías - OPTIMIZADO
    # Aplicar muestreo para visualización si es necesario
    X_vis, etiquetas_vis = reducir_muestra_para_visualizacion(X_pca, etiquetas)
    
    X_anomalias = X_vis[etiquetas_vis == 1]
    X_normales = X_vis[etiquetas_vis == 0]
    
    n_anomalias_vis = len(X_anomalias)
    n_normales_vis = len(X_normales)
    pct_anomalias_vis = (n_anomalias_vis / len(X_vis)) * 100 if len(X_vis) > 0 else 0
    
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotear puntos normales y anomalías con colores consistentes
    if len(X_normales) > 0:
        ax.scatter(X_normales[:, 0], X_normales[:, 1], X_normales[:, 2], 
                  c=COLOR_NORMAL, label=f'Normales ({len(X_normales)})', 
                  s=SCATTER_SIZE_NORMAL, alpha=ALPHA_NORMAL)
    
    if len(X_anomalias) > 0:
        ax.scatter(X_anomalias[:, 0], X_anomalias[:, 1], X_anomalias[:, 2], 
                  c=COLOR_ANOMALIA, label=f'Anomalías ({len(X_anomalias)})', 
                  s=SCATTER_SIZE_ANOMALIA, alpha=ALPHA_ANOMALIA)
    
    # Título mejorado con información clave
    ax.set_title(f'Detección de Anomalías con Isolation Forest (3D)\n'
                f'Anomalías: {n_anomalias_vis} ({pct_anomalias_vis:.2f}%) | Muestras: {len(X_vis):,}',
                fontsize=14, pad=15)
    
    ax.set_xlabel('Componente Principal 1', fontsize=12)
    ax.set_ylabel('Componente Principal 2', fontsize=12)
    ax.set_zlabel('Componente Principal 3', fontsize=12)
    
    # CORREGIR ASPECTO: Ajustar proporciones de los ejes para evitar deformación
    if len(X_vis) > 0:
        # Calcular rangos de datos
        x_range = np.ptp(X_vis[:, 0])  # Peak to peak (max - min)
        y_range = np.ptp(X_vis[:, 1])
        z_range = np.ptp(X_vis[:, 2])
        max_range = max(x_range, y_range, z_range)
        
        # Centrar y escalar los ejes
        x_center = np.mean(X_vis[:, 0])
        y_center = np.mean(X_vis[:, 1])
        z_center = np.mean(X_vis[:, 2])
        
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        # Configurar aspecto igual
        ax.set_box_aspect([1,1,1])
    
    # Ángulo de vista optimizado
    ax.view_init(elev=config.VIEW_ELEV, azim=config.VIEW_AZIM)
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    ruta_anomalias_3d = directorio_graficas / 'anomalies_3d.png'
    plt.savefig(ruta_anomalias_3d, dpi=200, bbox_inches='tight')
    plt.close("all")


def guardar_anomalias(datos_originales: pd.DataFrame, etiquetas: np.ndarray, 
                     scores: np.ndarray, ruta_archivo: Path) -> None:
    """
    Guarda las anomalías detectadas en un archivo CSV (salida estandarizada).
    
    Args:
        datos_originales: Datos originales
        etiquetas: Etiquetas de clasificación
        scores: Puntuaciones de anomalía
        ruta_archivo: Ruta donde guardar el archivo
    """
    # Crear DataFrame con todos los datos y scores estandarizados
    datos_salida = datos_originales.copy()
    datos_salida['anomaly_score'] = scores
    datos_salida['is_outlier'] = etiquetas
    
    # Guardar salida completa con fecha, XYZ, is_outlier y anomaly_score
    ruta_salida_completa = ruta_archivo.parent / 'anomaly_scores.csv'
    datos_salida[['fecha'] + CARACTERISTICAS_BASE + ['is_outlier', 'anomaly_score']].to_csv(ruta_salida_completa, index=False)
    
    # Guardar solo anomalías para retrocompatibilidad
    anomalias = datos_salida[etiquetas == 1].copy()
    
    # Ordenar por fecha si está disponible, sino por score
    if 'fecha' in anomalias.columns:
        anomalias = anomalias.sort_values(['fecha', 'anomaly_score'], ascending=[True, False])
    else:
        anomalias = anomalias.sort_values('anomaly_score', ascending=False)  # Mayor score = más anómalo
    
    anomalias.to_csv(ruta_archivo, index=False)


def guardar_metricas_csv(datos: pd.DataFrame, etiquetas: np.ndarray, scores: np.ndarray, 
                        mejor_resultado: Tuple, ruta_archivo: Path, 
                        tiempo_total: float = 0.0, memoria_max: float = 0.0) -> None:
    """
    Guarda métricas estandarizadas en formato CSV para comparación entre algoritmos (100% no-supervisado).
    
    Args:
        datos: DataFrame con datos originales
        etiquetas: Etiquetas binarias de anomalías
        scores: Scores de anomalía
        mejor_resultado: Tupla con resultados del mejor modelo
        ruta_archivo: Ruta donde guardar el CSV
        tiempo_total: Tiempo total de ejecución en segundos
        memoria_max: Memoria máxima utilizada en MB
    """
    separacion_scores, mejores_params, _ = mejor_resultado
    
    # Métricas básicas (100% no-supervisado)
    metricas = {
        'algoritmo': 'Isolation_Forest',
        'params_json': str(mejores_params),
        'n_clusters': None,  # No aplica para Isolation Forest
        'silhouette_score': None,  # No válido para detección binaria
        'calinski_harabasz_score': None,
        'davies_bouldin_score': None,
        'pct_anomalias': np.mean(etiquetas) * 100,
        'p95_minus_p50': separacion_scores,
        'mean_score': np.mean(scores),
        'tiempo_ejecucion_s': tiempo_total,
        'memoria_max_mb': memoria_max
    }
    
    # Guardar como CSV
    df_metricas = pd.DataFrame([metricas])
    df_metricas.to_csv(ruta_archivo, index=False)


def main() -> None:
    """Función principal del programa."""
    # Iniciar tracking de tiempo y memoria
    tiempo_inicio = time.time()
    tracemalloc.start()
    
    try:
        # Configuración inicial
        directorio_script = Path(__file__).parent
        directorios = DirectoriosProyecto(directorio_script)
        directorios.crear_directorios()
        
        # Configurar logging
        logger = configurar_logging(directorios.directorio_metricas / ARCHIVO_LOG)
        logger.info("Iniciando proceso de detección de anomalías con Isolation Forest...")
        
        # Aplicar seeds para reproducibilidad
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        # Cargar y preprocesar datos
        ruta_datos = directorio_script / NOMBRE_ARCHIVO_DATOS
        datos = cargar_datos(ruta_datos)
        logger.info(f"Datos cargados: {len(datos)} registros")
        
        # Validar datos de entrada
        config.validar_datos_entrada(datos, CARACTERISTICAS_BASE)
        
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
        _, mejores_params, _ = mejor_resultado
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
        # CORRIGIDO: Invertir el score para que mayor valor = más anómalo
        scores_pred_raw = -modelo_final.decision_function(X_escalado)  # Invertir para score consistente
        
        # Normalizar scores al rango [0, 1] usando función centralizada
        scores_pred = config.normalizar_scores_min_max(scores_pred_raw)
        
        etiquetas_pred = modelo_final.predict(X_escalado)
        etiquetas_pred = np.where(etiquetas_pred == 1, 0, 1)
        
        n_anomalias = np.sum(etiquetas_pred)
        porcentaje_anomalias = (n_anomalias / len(etiquetas_pred)) * 100
        
        logger.info(f"Detección completada: {n_anomalias} anomalías ({porcentaje_anomalias:.2f}%)")
        
        # Guardar resultados
        ruta_metricas = directorios.directorio_metricas / ARCHIVO_METRICAS
        guardar_metricas(ruta_metricas, mejor_resultado, n_anomalias, porcentaje_anomalias, scores_pred)
        logger.info(f"Métricas guardadas en {ruta_metricas}")
        
        # Guardar modelo
        ruta_modelo_pkl = directorios.directorio_modelos / EXTENSION_MODELO_PKL
        ruta_modelo_h5 = directorios.directorio_modelos / EXTENSION_MODELO_H5
        guardar_modelo(modelo_final, scores_pred, etiquetas_pred, ruta_modelo_pkl, ruta_modelo_h5)
        
        # Guardar escalador para inferencia futura
        ruta_escalador = directorios.directorio_modelos / 'scaler.pkl'
        joblib.dump(escalador, ruta_escalador)
        
        logger.info(f"Modelo guardado en formatos pickle y HDF5")
        logger.info(f"Escalador guardado en {ruta_escalador}")
        
        # Calcular métricas de rendimiento
        tiempo_total = time.time() - tiempo_inicio
        memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
        memoria_max = memoria_pico / 1024**2  # Convertir a MB
        tracemalloc.stop()
        
        logger.info(f"Tiempo total de ejecucion: {tiempo_total:.2f} segundos")
        logger.info(f"Memoria maxima utilizada: {memoria_max:.2f} MB")
        
        # Generar gráficos
        generar_graficos(scores_pred, X_pca, etiquetas_pred, directorios.directorio_graficas)
        logger.info("Gráficos generados y guardados")
        
        # Guardar anomalías detectadas
        ruta_anomalias = directorios.directorio_metricas / ARCHIVO_ANOMALIAS
        guardar_anomalias(datos, etiquetas_pred, scores_pred, ruta_anomalias)
        logger.info(f"Anomalías guardadas en {ruta_anomalias}")
        
        # Guardar métricas estandarizadas CSV (incluir tiempo y memoria)
        ruta_metricas_csv = directorios.directorio_metricas / 'metrics.csv'
        guardar_metricas_csv(datos, etiquetas_pred, scores_pred, mejor_resultado, ruta_metricas_csv, 
                            tiempo_total, memoria_max)
        logger.info(f"Métricas CSV guardadas en {ruta_metricas_csv}")
        
        logger.info("Proceso completado exitosamente")
        
    except Exception as e:
        tracemalloc.stop()
        logger.error(f"Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":
    main()