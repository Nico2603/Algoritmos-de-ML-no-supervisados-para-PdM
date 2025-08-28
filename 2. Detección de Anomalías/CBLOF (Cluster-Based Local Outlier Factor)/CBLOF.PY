"""
Detección de Anomalías usando CBLOF (Cluster-Based Local Outlier Factor)

Este script implementa la detección de anomalías utilizando el algoritmo CBLOF
con búsqueda automática de hiperparámetros para optimizar el rendimiento.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
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

# Suprimir warnings no críticos
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Constantes del proyecto
ARCHIVO_DATOS = 'data.csv'
ARCHIVO_LOG = 'output.log'
ARCHIVO_METRICAS = 'metrics.txt'
ARCHIVO_ANOMALIAS = 'anomalies.csv'
ARCHIVO_MODELO_PKL = 'cblof_model.pkl'
ARCHIVO_MODELO_H5 = 'cblof_model.h5'

CARACTERISTICAS_BASE = ['acceleration_x', 'acceleration_y', 'acceleration_z']
CARACTERISTICA_MAGNITUD = 'magnitud_aceleracion'

# Parámetros por defecto
PARAMETROS_PCA = {'n_components': 3}
CONTAMINACION_DEFAULT = 0.1
RANDOM_STATE = 42
N_JOBS = 2  # Optimizado para Google Colab (limitación de memoria)

# Grid de parámetros para búsqueda (optimizado para Google Colab)
PARAM_GRID = {
    'n_clusters': [5, 10],
    'alpha': [0.7, 0.9],
    'beta': [5, 7],
    'use_weights': [True, False]
}


def configurar_logging(directorio_metricas: Path) -> None:
    """
    Configura el sistema de logging para archivo y consola.
    
    Args:
        directorio_metricas: Directorio donde guardar el archivo de log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    ruta_archivo_log = directorio_metricas / ARCHIVO_LOG
    file_handler = logging.FileHandler(ruta_archivo_log, mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
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
    Carga y valida los datos desde el archivo CSV.
    
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
        # Intentar diferentes encodings comunes
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        datos = None
        
        for encoding in encodings:
            try:
                datos = pd.read_csv(ruta_datos, encoding=encoding)
                logging.info(f"Datos cargados correctamente con encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if datos is None:
            raise ValueError("No se pudo leer el archivo CSV con ningún encoding estándar")
        
        # Mostrar muestra de los primeros datos
        logging.info(f"Muestra de los datos (primeras 3 filas):")
        logging.info(f"{datos.head(3).to_string()}")
        
        # Validar columnas requeridas
        columnas_requeridas = CARACTERISTICAS_BASE
        columnas_faltantes = [col for col in columnas_requeridas if col not in datos.columns]
        
        if columnas_faltantes:
            logging.error(f"Columnas disponibles: {list(datos.columns)}")
            raise ValueError(f"Columnas faltantes en los datos: {columnas_faltantes}")
        
        # Verificar que hay datos suficientes
        if len(datos) < 100:
            logging.warning(f"Advertencia: Pocos datos disponibles: {len(datos)} registros")
        
        # Verificar que las columnas de aceleración contienen datos numéricos
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
    # Mostrar información inicial de los datos
    logging.info(f"Datos cargados: {len(datos)} filas, {len(datos.columns)} columnas")
    logging.info(f"Columnas disponibles: {list(datos.columns)}")
    
    # Manejo de valores faltantes
    datos_limpios = datos.dropna()
    filas_eliminadas = len(datos) - len(datos_limpios)
    
    if filas_eliminadas > 0:
        logging.info(f"Valores faltantes eliminados: {filas_eliminadas} filas")
    
    # Verificar tipos de datos para las columnas de aceleración
    for col in CARACTERISTICAS_BASE:
        if datos_limpios[col].dtype == 'object':
            logging.info(f"Convirtiendo columna {col} de texto a numérico")
            datos_limpios[col] = pd.to_numeric(datos_limpios[col], errors='coerce')
    
    # Eliminar filas con valores no numéricos después de la conversión
    datos_limpios = datos_limpios.dropna(subset=CARACTERISTICAS_BASE)
    
    # Ingeniería de características
    datos_limpios = datos_limpios.copy()  # Evitar warning de SettingWithCopyWarning
    datos_limpios[CARACTERISTICA_MAGNITUD] = np.sqrt(
        datos_limpios['acceleration_x']**2 +
        datos_limpios['acceleration_y']**2 +
        datos_limpios['acceleration_z']**2
    )
    logging.info(f"Característica '{CARACTERISTICA_MAGNITUD}' añadida.")
    
    # Seleccionar características
    caracteristicas = CARACTERISTICAS_BASE + [CARACTERISTICA_MAGNITUD]
    X = datos_limpios[caracteristicas].values
    
    # Mostrar estadísticas básicas
    logging.info(f"Estadísticas de los datos:")
    for i, col in enumerate(caracteristicas):
        logging.info(f"  {col}: min={X[:, i].min():.4f}, max={X[:, i].max():.4f}, media={X[:, i].mean():.4f}")
    
    # Validar que no hay valores infinitos o NaN
    if np.any(np.isinf(X)) or np.any(np.isnan(X)):
        logging.warning("Se encontraron valores infinitos o NaN después del preprocesamiento")
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    
    # Escalar los datos
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
    
    logging.info(f"Dataset reducido para optimización: {len(X)} -> {len(X_reducido)} muestras")
    return X_reducido, indices_seleccionados


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
        labels = clf.labels_
        
        # Verificar que hay al menos 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return None
        
        # Calcular métricas
        silhouette_avg = silhouette_score(X_escalado, labels)
        calinski_harabasz = calinski_harabasz_score(X_escalado, labels)
        davies_bouldin = davies_bouldin_score(X_escalado, labels)
        
        logging.info(f"Parámetros: {params}")
        logging.info(f"Silhouette Score: {silhouette_avg:.4f}")
        logging.info(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        logging.info(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        
        return (silhouette_avg, calinski_harabasz, davies_bouldin, params, clf)
        
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
    
    # Paralelizar la búsqueda
    resultados = Parallel(n_jobs=N_JOBS)(
        delayed(evaluar_parametros_cblof)(params, X_escalado) 
        for params in ParameterGrid(PARAM_GRID)
    )
    
    # Filtrar resultados válidos
    resultados_validos = [res for res in resultados if res is not None]
    
    if not resultados_validos:
        raise ValueError("No se encontraron parámetros que produzcan clusters válidos")
    
    # Seleccionar el mejor modelo
    mejor_resultado = max(resultados_validos, key=lambda x: x[0])
    mejor_score, mejor_calinski, mejor_davies, mejores_parametros, mejor_modelo = mejor_resultado
    
    metricas = {
        'silhouette_score': mejor_score,
        'calinski_harabasz_score': mejor_calinski,
        'davies_bouldin_score': mejor_davies
    }
    
    logging.info(f"\nMejores parámetros encontrados: {mejores_parametros}")
    logging.info(f"Mejor Silhouette Score: {mejor_score:.4f}")
    logging.info(f"Mejor Calinski-Harabasz Score: {mejor_calinski:.4f}")
    logging.info(f"Mejor Davies-Bouldin Score: {mejor_davies:.4f}")
    
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
    # CORRIGIDO: En PyOD, labels_ es 0/1 (normal/outlier), no cluster IDs
    binary_labels = modelo.labels_  # 0: normal, 1: outlier
    anomaly_scores = modelo.decision_scores_
    
    # Calcular métricas de anomalías correctamente
    n_anomalias = np.sum(binary_labels)
    porcentaje_anomalias = (n_anomalias / len(binary_labels)) * 100
    
    # Para CBLOF, no podemos calcular métricas de clustering tradicionales
    # porque no tenemos acceso directo a los cluster IDs del clustering interno
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
        
        f.write("Métricas de clustering:\n")
        f.write(f"Silhouette Score: {metricas['silhouette_score']:.4f}\n")
        f.write(f"Calinski-Harabasz Score: {metricas['calinski_harabasz_score']:.4f}\n")
        f.write(f"Davies-Bouldin Score: {metricas['davies_bouldin_score']:.4f}\n\n")
        
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
    # Guardar como pickle
    ruta_modelo_pkl = directorio_modelos / ARCHIVO_MODELO_PKL
    try:
        joblib.dump(modelo, ruta_modelo_pkl)
        logging.info(f"Modelo guardado como pickle en {ruta_modelo_pkl}")
    except Exception as e:
        logging.error(f"Error guardando modelo pickle: {e}")
    
    # Guardar como HDF5
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
    
    # Configurar matplotlib para evitar problemas con display
    plt.ioff()  # Desactivar modo interactivo
    
    # 1. Distribución de puntuaciones de anomalía
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribución de las Puntuaciones de Anomalía (CBLOF)', fontsize=14)
        plt.xlabel('Puntuación de Anomalía', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        ruta_puntuaciones = directorio_graficas / 'anomaly_scores.png'
        plt.savefig(ruta_puntuaciones, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Gráfico de puntuaciones guardado en {ruta_puntuaciones}")
    except Exception as e:
        logging.error(f"Error generando gráfico de puntuaciones: {e}")
    
    # 2. Visualización 3D de clusters y anomalías
    try:
        labels_visualizacion = np.where(modelo.labels_ == 0, 0, 1)
        X_anomalias = X_pca[labels_visualizacion == 1]
        X_normales = X_pca[labels_visualizacion == 0]
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        if len(X_normales) > 0:
            ax.scatter(X_normales[:, 0], X_normales[:, 1], X_normales[:, 2], 
                      c='blue', label=f'Normales ({len(X_normales)})', s=20, alpha=0.6)
        
        if len(X_anomalias) > 0:
            ax.scatter(X_anomalias[:, 0], X_anomalias[:, 1], X_anomalias[:, 2], 
                      c='red', label=f'Anomalías ({len(X_anomalias)})', s=20, alpha=0.8)
        
        ax.set_xlabel('Componente Principal 1', fontsize=12)
        ax.set_ylabel('Componente Principal 2', fontsize=12)
        ax.set_zlabel('Componente Principal 3', fontsize=12)
        ax.set_title('Detección de Anomalías con CBLOF en 3D', fontsize=14)
        ax.legend()
        
        ruta_anomalias_3d = directorio_graficas / 'anomalies_3d.png'
        plt.savefig(ruta_anomalias_3d, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Gráfico 3D guardado en {ruta_anomalias_3d}")
    except Exception as e:
        logging.error(f"Error generando gráfico 3D: {e}")


def guardar_anomalias_detectadas(modelo: CBLOF, datos: pd.DataFrame, 
                                directorio_metricas: Path) -> None:
    """
    Guarda las anomalías detectadas en un archivo CSV.
    
    Args:
        modelo: Modelo CBLOF entrenado
        datos: Datos originales
        directorio_metricas: Directorio donde guardar el archivo
    """
    try:
        # En PyOD, labels_ ya es 0/1 (normal/outlier)
        anomalias = datos[modelo.labels_ == 1].copy()
        
        if not anomalias.empty:
            # Estandarizar nombres de columnas
            anomalias['anomaly_score'] = modelo.decision_scores_[modelo.labels_ == 1]
            anomalias['is_outlier'] = 1
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
    try:
        # 1. Configuración inicial
        directorio_script = Path(__file__).parent.absolute()
        directorio_modelos, directorio_graficas, directorio_metricas = crear_directorios(directorio_script)
        configurar_logging(directorio_metricas)
        
        logging.info("=== INICIANDO PROCESO DE DETECCIÓN DE ANOMALÍAS CON CBLOF ===")
        
        # 2. Cargar y validar datos
        ruta_datos = directorio_script / ARCHIVO_DATOS
        datos = cargar_y_validar_datos(ruta_datos)
        
        # 3. Preprocesamiento
        X_escalado, escalador, datos_limpios = preprocesar_datos(datos)
        logging.info(f"Datos preprocesados: {X_escalado.shape[0]} muestras, {X_escalado.shape[1]} características")
        
        # 4. Reducción de dimensionalidad
        X_pca, pca = aplicar_pca(X_escalado)
        
        # 5. Reducir muestra para optimización si es muy grande
        X_para_optimizacion, indices_muestra = reducir_muestra_para_optimizacion(X_escalado)
        
        # 6. Búsqueda de mejores parámetros usando muestra reducida
        mejor_modelo_muestra, metricas = buscar_mejores_parametros(X_para_optimizacion)
        
        # 7. Aplicar mejores parámetros al dataset completo
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
        
        # 8. Evaluación adicional
        metricas_adicionales = calcular_metricas_adicionales(mejor_modelo, X_escalado)
        
        # 9. Guardar resultados
        # Los mejores parámetros ya están definidos arriba
        
        guardar_metricas(metricas, metricas_adicionales, mejores_parametros, directorio_metricas)
        guardar_modelo(mejor_modelo, directorio_modelos)
        
        # 10. Generar visualizaciones
        generar_visualizaciones(mejor_modelo, X_pca, datos_limpios, directorio_graficas)
        
        # 11. Guardar anomalías detectadas
        guardar_anomalias_detectadas(mejor_modelo, datos_limpios, directorio_metricas)
        
        logging.info("=== PROCESO COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        logging.error(f"Error en el proceso principal: {e}")
        raise


if __name__ == "__main__":
    main()
