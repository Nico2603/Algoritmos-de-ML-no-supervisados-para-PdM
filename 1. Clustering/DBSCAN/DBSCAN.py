"""
Módulo para clustering DBSCAN con optimización automática de parámetros.
Incluye preprocesamiento, búsqueda de parámetros óptimos, visualización y detección de anomalías.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid
import joblib
import h5py
import logging
import gc
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Tuple, Any

# Configuración de warnings
warnings.filterwarnings('ignore')
plt.style.use('default')

# Constantes (100% no-supervisado)
USECOLS = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'fecha']
CARACTERISTICAS_BASE = ['acceleration_x', 'acceleration_y', 'acceleration_z']
MIN_CLUSTERS_VALIDOS = 2
N_JOBS_PARALELO = 1  # Reducido a 1 para evitar problemas de memoria
FORMATO_METRICAS = '.4f'
MAX_MUESTRAS_OPTIMIZACION = 10000  # Reducido para optimización de memoria

class ConfiguradorLogging:
    """Configurador del sistema de logging."""
    
    @staticmethod
    def configurar_logging(directorio_metricas: str) -> None:
        """
        Configura el sistema de logging para archivo y consola.
        
        Args:
            directorio_metricas: Directorio donde guardar el archivo de log.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Limpiar handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Handler para archivo
        ruta_archivo_log = os.path.join(directorio_metricas, 'output.log')
        file_handler = logging.FileHandler(ruta_archivo_log, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

class GestorDirectorios:
    """Gestor de directorios del proyecto."""
    
    def __init__(self, directorio_base: str):
        self.directorio_base = directorio_base
        self.directorio_modelos = os.path.join(directorio_base, 'modelos_entrenados_DBSCAN')
        self.directorio_graficas = os.path.join(directorio_base, 'graficas_DBSCAN')
        self.directorio_metricas = os.path.join(directorio_base, 'metricas_DBSCAN')
    
    def crear_directorios(self) -> None:
        """Crea todos los directorios necesarios."""
        directorios = [self.directorio_modelos, self.directorio_graficas, self.directorio_metricas]
        for directorio in directorios:
            os.makedirs(directorio, exist_ok=True)

class ProcesadorDatos:
    """Procesador y transformador de datos."""
    
    @staticmethod
    def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
        """
        Carga los datos desde un archivo CSV (100% no-supervisado).
        
        Args:
            ruta_archivo: Ruta al archivo CSV.
            
        Returns:
            DataFrame con los datos cargados.
            
        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el archivo está vacío o tiene formato incorrecto.
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"El archivo {ruta_archivo} no existe")
        
        try:
            # BLINDADO: Cargar solo las 4 columnas necesarias con tipos optimizados
            try:
                datos = pd.read_csv(
                    ruta_archivo, 
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
                    ruta_archivo, 
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
                logging.info("Archivo cargado con encoding latin-1")
            
            if datos.empty:
                raise ValueError("El archivo CSV está vacío")
            
            # BLINDADO: Recortar DataFrame a solo las 4 columnas por si el CSV trae más
            datos = datos[['fecha'] + CARACTERISTICAS_BASE].copy()
            
            # Ordenar por fecha (recomendable)
            datos.sort_values('fecha', inplace=True)
            
            # Mostrar información detallada sobre los datos cargados
            logging.info(f"Archivo CSV cargado exitosamente:")
            logging.info(f"  - Dimensiones: {datos.shape[0]} filas x {datos.shape[1]} columnas")
            logging.info(f"  - Columnas cargadas: {list(datos.columns)}")
            logging.info(f"  - Tipos de datos por columna:")
            for col in datos.columns:
                logging.info(f"    • {col}: {datos[col].dtype}")
            
            # Mostrar estadísticas básicas de las primeras filas
            logging.info(f"  - Primeras 3 filas de datos:")
            for i, (_, fila) in enumerate(datos.head(3).iterrows()):
                logging.info(f"    Fila {i+1}: {dict(fila)}")
            
            # Verificar valores faltantes por columna
            valores_faltantes = datos[CARACTERISTICAS_BASE].isnull().sum()
            if valores_faltantes.any():
                logging.info(f"  - Valores faltantes por columna:")
                for col, faltantes in valores_faltantes.items():
                    if faltantes > 0:
                        logging.info(f"    • {col}: {faltantes} valores faltantes")
            else:
                logging.info(f"  - No se encontraron valores faltantes")
            
            return datos
        except Exception as e:
            raise ValueError(f"Error al cargar el archivo CSV: {str(e)}")
    
    @staticmethod
    def validar_caracteristicas(datos: pd.DataFrame, caracteristicas: List[str]) -> None:
        """
        Valida que las características requeridas estén presentes en los datos.
        
        Args:
            datos: DataFrame con los datos.
            caracteristicas: Lista de características requeridas.
            
        Raises:
            ValueError: Si faltan características requeridas.
        """
        caracteristicas_faltantes = [col for col in caracteristicas if col not in datos.columns]
        if caracteristicas_faltantes:
            raise ValueError(f"Características faltantes: {caracteristicas_faltantes}")
    
    @staticmethod
    def preprocesar_datos(datos: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocesa los datos: elimina valores faltantes y añade características derivadas.
        
        Args:
            datos: DataFrame original.
            
        Returns:
            Tupla con (datos_procesados, matriz_caracteristicas).
        """
        # Validar características base
        ProcesadorDatos.validar_caracteristicas(datos, CARACTERISTICAS_BASE)
        logging.info(f"Características base validadas: {CARACTERISTICAS_BASE}")
        
        # Mostrar columnas adicionales disponibles
        columnas_adicionales = [col for col in datos.columns if col not in CARACTERISTICAS_BASE]
        if columnas_adicionales:
            logging.info(f"Columnas adicionales encontradas: {columnas_adicionales}")
        
        # Eliminar valores faltantes
        filas_originales = len(datos)
        datos_limpios = datos.dropna()
        filas_eliminadas = filas_originales - len(datos_limpios)
        
        logging.info(f"Limpieza de datos:")
        logging.info(f"  - Filas originales: {filas_originales}")
        logging.info(f"  - Filas eliminadas por valores faltantes: {filas_eliminadas}")
        logging.info(f"  - Filas restantes: {len(datos_limpios)}")
        
        if datos_limpios.empty:
            raise ValueError("No quedan datos después de eliminar valores faltantes")
        
        # Añadir magnitud de aceleración
        datos_limpios = datos_limpios.copy()
        datos_limpios['magnitud_aceleracion'] = np.sqrt(
            datos_limpios['acceleration_x']**2 +
            datos_limpios['acceleration_y']**2 +
            datos_limpios['acceleration_z']**2
        )
        logging.info("Característica derivada añadida: 'magnitud_aceleracion'")
        
        # Seleccionar características para clustering
        caracteristicas = CARACTERISTICAS_BASE + ['magnitud_aceleracion']
        matriz_caracteristicas = datos_limpios[caracteristicas].values
        
        # Mostrar estadísticas de las características
        logging.info(f"Estadísticas de características para clustering:")
        for i, col in enumerate(caracteristicas):
            valores = matriz_caracteristicas[:, i]
            logging.info(f"  - {col}:")
            logging.info(f"    • Min: {np.min(valores):.4f}")
            logging.info(f"    • Max: {np.max(valores):.4f}")
            logging.info(f"    • Media: {np.mean(valores):.4f}")
            logging.info(f"    • Desv. Estándar: {np.std(valores):.4f}")
        
        logging.info(f"Matriz de características creada: {matriz_caracteristicas.shape}")
        return datos_limpios, matriz_caracteristicas
    
    @staticmethod
    def escalar_datos(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Escala los datos usando StandardScaler.
        
        Args:
            X: Matriz de características.
            
        Returns:
            Tupla con (datos_escalados, escalador).
        """
        escalador = StandardScaler()
        X_escalado = escalador.fit_transform(X)
        logging.info("Datos escalados correctamente")
        return X_escalado, escalador
    
    @staticmethod
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
        
        # Muestreo aleatorio estratificado
        np.random.seed(42)  # Para reproducibilidad
        indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
        indices_seleccionados = np.sort(indices_seleccionados)
        
        X_reducido = X[indices_seleccionados]
        
        logging.info(f"Dataset reducido para optimización: {len(X)} -> {len(X_reducido)} muestras")
        return X_reducido, indices_seleccionados

class AnalizadorDistancias:
    """Analizador de distancias para estimar parámetros de DBSCAN."""
    
    @staticmethod
    def generar_grafica_k_distancias(X_escalado: np.ndarray, directorio_graficas: str, k: int = 5) -> None:
        """
        Genera la gráfica de k-distancias para estimar el parámetro eps.
        
        Args:
            X_escalado: Datos escalados.
            directorio_graficas: Directorio donde guardar la gráfica.
            k: Número de vecinos más cercanos.
        """
        try:
            vecinos = NearestNeighbors(n_neighbors=k)
            vecinos_ajustados = vecinos.fit(X_escalado)
            distancias, _ = vecinos_ajustados.kneighbors(X_escalado)
            distancias_ordenadas = np.sort(distancias[:, k-1], axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(distancias_ordenadas)
            plt.title(f'Gráfica de {k}-distancias para estimar eps')
            plt.xlabel('Puntos ordenados')
            plt.ylabel(f'Distancia al {k}to vecino más cercano')
            plt.grid(True, alpha=0.3)
            
            ruta_grafica = os.path.join(directorio_graficas, 'k_distance_graph.png')
            plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Gráfica de k-distancias guardada en {ruta_grafica}")
        except Exception as e:
            logging.error(f"Error al generar gráfica de k-distancias: {str(e)}")

class OptimizadorDBSCAN:
    """Optimizador de parámetros para DBSCAN."""
    
    @staticmethod
    def generar_grilla_parametros(eps_min: float = 0.1, eps_max: float = 2.0, 
                                 n_eps: int = 8, min_samples_min: int = 3, 
                                 min_samples_max: int = 4) -> List[Dict[str, Any]]:
        """
        Genera la grilla de parámetros para la búsqueda.
        
        Args:
            eps_min: Valor mínimo de eps.
            eps_max: Valor máximo de eps.
            n_eps: Número de valores de eps a probar (reducido para optimización de memoria).
            min_samples_min: Valor mínimo de min_samples.
            min_samples_max: Valor máximo de min_samples (reducido para optimización de memoria).
            
        Returns:
            Lista de diccionarios con combinaciones de parámetros.
        """
        valores_eps = np.linspace(eps_min, eps_max, n_eps)
        valores_min_samples = range(min_samples_min, min_samples_max + 1)
        
        param_grid = {'eps': valores_eps, 'min_samples': valores_min_samples}
        parametros = list(ParameterGrid(param_grid))
        
        logging.info(f"Grilla de parámetros generada: {len(parametros)} combinaciones")
        return parametros
    
    @staticmethod
    def evaluar_parametros_dbscan(parametros: Dict[str, Any], X_escalado: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Evalúa una combinación específica de parámetros para DBSCAN.
        
        Args:
            parametros: Diccionario con parámetros eps y min_samples.
            X_escalado: Datos escalados.
            
        Returns:
            Diccionario con resultados de la evaluación o None si no es válida.
        """
        try:
            eps = parametros['eps']
            min_samples = parametros['min_samples']
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            etiquetas = dbscan.fit_predict(X_escalado)
            
            n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
            n_ruido = list(etiquetas).count(-1)
            
            # Validar si hay suficientes clusters
            if n_clusters < MIN_CLUSTERS_VALIDOS:
                return None
            
            # Calcular métricas excluyendo ruido
            mascara_sin_ruido = etiquetas != -1
            if np.sum(mascara_sin_ruido) < MIN_CLUSTERS_VALIDOS:
                return None
            
            X_sin_ruido = X_escalado[mascara_sin_ruido]
            etiquetas_sin_ruido = etiquetas[mascara_sin_ruido]
            
            silhouette = silhouette_score(X_sin_ruido, etiquetas_sin_ruido)
            calinski_harabasz = calinski_harabasz_score(X_sin_ruido, etiquetas_sin_ruido)
            davies_bouldin = davies_bouldin_score(X_sin_ruido, etiquetas_sin_ruido)
            
            return {
                'parametros': parametros,
                'modelo': dbscan,
                'etiquetas': etiquetas,
                'n_clusters': n_clusters,
                'n_ruido': n_ruido,
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin
            }
        except Exception as e:
            logging.warning(f"Error evaluando parámetros {parametros}: {str(e)}")
            return None
    
    @staticmethod
    def buscar_mejores_parametros(X_escalado: np.ndarray, grilla_parametros: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Busca los mejores parámetros usando búsqueda optimizada para memoria.
        
        Args:
            X_escalado: Datos escalados.
            grilla_parametros: Lista de combinaciones de parámetros.
            
        Returns:
            Diccionario con el mejor resultado.
            
        Raises:
            ValueError: Si no se encuentran parámetros válidos.
        """
        logging.info(f"Iniciando búsqueda de mejores parámetros...")
        logging.info(f"Total de combinaciones a evaluar: {len(grilla_parametros)}")
        logging.info(f"Tamaño de datos para optimización: {X_escalado.shape}")
        logging.info(f"Número de trabajos paralelos: {N_JOBS_PARALELO}")
        
        if N_JOBS_PARALELO == 1:
            # Búsqueda secuencial para evitar problemas de memoria
            logging.info("Ejecutando búsqueda secuencial para optimizar memoria...")
            resultados = []
            for i, params in enumerate(grilla_parametros):
                if i % 5 == 0:  # Log progreso cada 5 iteraciones
                    logging.info(f"Evaluando combinación {i+1}/{len(grilla_parametros)}: {params}")
                resultado = OptimizadorDBSCAN.evaluar_parametros_dbscan(params, X_escalado)
                resultados.append(resultado)
                
                # Forzar liberación de memoria cada 10 iteraciones
                if i % 10 == 0:
                    gc.collect()
        else:
            # Búsqueda paralela (solo si n_jobs > 1)
            resultados = Parallel(n_jobs=N_JOBS_PARALELO)(
                delayed(OptimizadorDBSCAN.evaluar_parametros_dbscan)(params, X_escalado) 
                for params in grilla_parametros
            )
        
        # Filtrar resultados válidos
        resultados_validos = [res for res in resultados if res is not None]
        
        logging.info(f"Combinaciones válidas encontradas: {len(resultados_validos)}/{len(grilla_parametros)}")
        
        if not resultados_validos:
            raise ValueError("No se encontraron parámetros que generen clusters válidos")
        
        # Seleccionar el mejor basado en silhouette score
        mejor_resultado = max(resultados_validos, key=lambda x: x['silhouette'])
        
        logging.info(f"Mejor resultado encontrado:")
        logging.info(f"  - Parámetros: eps={mejor_resultado['parametros']['eps']:.3f}, min_samples={mejor_resultado['parametros']['min_samples']}")
        logging.info(f"  - Silhouette Score: {mejor_resultado['silhouette']:.4f}")
        logging.info("Búsqueda de parámetros completada")
        return mejor_resultado

class VisualizadorClusters:
    """Visualizador de clusters y resultados."""
    
    @staticmethod
    def visualizar_clusters_2d(X_escalado: np.ndarray, etiquetas: np.ndarray, 
                              directorio_graficas: str, titulo: str = "Clustering DBSCAN (2D)") -> None:
        """
        Visualiza clusters en 2D usando PCA.
        
        Args:
            X_escalado: Datos escalados.
            etiquetas: Etiquetas de clusters.
            directorio_graficas: Directorio donde guardar la gráfica.
            titulo: Título de la gráfica.
        """
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_escalado)
            
            etiquetas_unicas = set(etiquetas)
            colores = plt.cm.Spectral(np.linspace(0, 1, len(etiquetas_unicas)))
            
            plt.figure(figsize=(12, 8))
            for etiqueta, color in zip(etiquetas_unicas, colores):
                mascara = etiquetas == etiqueta
                puntos = X_pca[mascara]
                
                if etiqueta == -1:
                    plt.scatter(puntos[:, 0], puntos[:, 1], c='black', marker='x', 
                              s=50, alpha=0.7, label='Ruido')
                else:
                    plt.scatter(puntos[:, 0], puntos[:, 1], c=[color], marker='o', 
                              s=30, alpha=0.7, label=f'Cluster {etiqueta}')
            
            plt.title(titulo)
            plt.xlabel(f'Componente Principal 1 (Varianza: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'Componente Principal 2 (Varianza: {pca.explained_variance_ratio_[1]:.2%})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            ruta_grafica = os.path.join(directorio_graficas, 'clusters_2d_pca.png')
            plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Visualización 2D guardada en {ruta_grafica}")
        except Exception as e:
            logging.error(f"Error en visualización 2D: {str(e)}")
    
    @staticmethod
    def visualizar_clusters_3d(X_escalado: np.ndarray, etiquetas: np.ndarray, 
                              directorio_graficas: str, titulo: str = "Clustering DBSCAN (3D)") -> None:
        """
        Visualiza clusters en 3D usando PCA.
        
        Args:
            X_escalado: Datos escalados.
            etiquetas: Etiquetas de clusters.
            directorio_graficas: Directorio donde guardar la gráfica.
            titulo: Título de la gráfica.
        """
        try:
            pca_3d = PCA(n_components=3)
            X_pca_3d = pca_3d.fit_transform(X_escalado)
            
            etiquetas_unicas = set(etiquetas)
            colores = plt.cm.Spectral(np.linspace(0, 1, len(etiquetas_unicas)))
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            for etiqueta, color in zip(etiquetas_unicas, colores):
                mascara = etiquetas == etiqueta
                puntos = X_pca_3d[mascara]
                
                if etiqueta == -1:
                    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], 
                             c='black', marker='x', s=50, alpha=0.7, label='Ruido')
                else:
                    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], 
                             c=[color], marker='o', s=30, alpha=0.7, label=f'Cluster {etiqueta}')
            
            ax.set_title(titulo)
            ax.set_xlabel(f'PC1 (Var: {pca_3d.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 (Var: {pca_3d.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 (Var: {pca_3d.explained_variance_ratio_[2]:.2%})')
            ax.legend()
            
            ruta_grafica = os.path.join(directorio_graficas, 'clusters_3d_pca.png')
            plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Visualización 3D guardada en {ruta_grafica}")
        except Exception as e:
            logging.error(f"Error en visualización 3D: {str(e)}")

class GuardadorModelos:
    """Guardador de modelos y resultados."""
    
    @staticmethod
    def guardar_modelo_pkl(modelo: DBSCAN, ruta_archivo: str) -> None:
        """
        Guarda el modelo en formato pickle.
        
        Args:
            modelo: Modelo DBSCAN entrenado.
            ruta_archivo: Ruta donde guardar el archivo.
        """
        try:
            joblib.dump(modelo, ruta_archivo)
            logging.info(f"Modelo guardado en formato pickle: {ruta_archivo}")
        except Exception as e:
            logging.error(f"Error guardando modelo pickle: {str(e)}")
    
    @staticmethod
    def guardar_modelo_h5(modelo: DBSCAN, ruta_archivo: str) -> None:
        """
        Guarda los parámetros del modelo en formato HDF5.
        
        Args:
            modelo: Modelo DBSCAN entrenado.
            ruta_archivo: Ruta donde guardar el archivo.
        """
        try:
            with h5py.File(ruta_archivo, 'w') as archivo_h5:
                if hasattr(modelo, 'components_') and modelo.components_ is not None:
                    archivo_h5.create_dataset('components', data=modelo.components_)
                if hasattr(modelo, 'labels_') and modelo.labels_ is not None:
                    archivo_h5.create_dataset('labels', data=modelo.labels_)
                if hasattr(modelo, 'core_sample_indices_') and modelo.core_sample_indices_ is not None:
                    archivo_h5.create_dataset('core_sample_indices', data=modelo.core_sample_indices_)
            
            logging.info(f"Parámetros del modelo guardados en HDF5: {ruta_archivo}")
        except Exception as e:
            logging.error(f"Error guardando modelo HDF5: {str(e)}")
    
    @staticmethod
    def guardar_metricas(metricas: Dict[str, Any], ruta_archivo: str) -> None:
        """
        Guarda las métricas en un archivo de texto.
        
        Args:
            metricas: Diccionario con las métricas.
            ruta_archivo: Ruta donde guardar el archivo.
        """
        try:
            with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
                archivo.write("=== RESULTADOS CLUSTERING DBSCAN ===\n\n")
                archivo.write(f"Mejores parámetros:\n")
                archivo.write(f"  - eps: {metricas['eps']}\n")
                archivo.write(f"  - min_samples: {metricas['min_samples']}\n\n")
                archivo.write(f"Resultados del clustering:\n")
                archivo.write(f"  - Número de clusters: {metricas['n_clusters']}\n")
                archivo.write(f"  - Número de puntos de ruido: {metricas['n_ruido']}\n\n")
                archivo.write(f"Métricas de calidad (sin ruido):\n")
                archivo.write(f"  - Coeficiente Silhouette: {metricas['silhouette']:{FORMATO_METRICAS}}\n")
                archivo.write(f"  - Calinski-Harabasz Score: {metricas['calinski_harabasz']:{FORMATO_METRICAS}}\n")
                archivo.write(f"  - Davies-Bouldin Index: {metricas['davies_bouldin']:{FORMATO_METRICAS}}\n")
            
            logging.info(f"Métricas guardadas en: {ruta_archivo}")
        except Exception as e:
            logging.error(f"Error guardando métricas: {str(e)}")

class DetectorAnomalias:
    """Detector y analizador de anomalías."""
    
    @staticmethod
    def calcular_scores_todos_los_puntos(etiquetas: np.ndarray, X_escalado: np.ndarray, 
                                       modelo: DBSCAN) -> np.ndarray:
        """
        Calcula scores de anomalía para todos los puntos.
        
        Args:
            etiquetas: Etiquetas de clusters
            X_escalado: Datos escalados
            modelo: Modelo DBSCAN entrenado
            
        Returns:
            Array con scores de anomalía para todos los puntos
        """
        scores = np.zeros(len(etiquetas))
        
        if hasattr(modelo, 'core_sample_indices_') and len(modelo.core_sample_indices_) > 0:
            # OPTIMIZACIÓN CRÍTICA: Usar NearestNeighbors en lugar de pairwise_distances
            # para evitar colapso de memoria con 500k puntos
            from sklearn.neighbors import NearestNeighbors
            indices_nucleo = modelo.core_sample_indices_
            
            # Crear modelo de vecinos más cercanos con puntos núcleo
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(X_escalado[indices_nucleo])
            
            # Calcular distancia mínima a punto núcleo más cercano
            distancias_minimas, _ = nn_model.kneighbors(X_escalado)
            distancias_minimas = distancias_minimas.flatten()
            
            # Para puntos de ruido: distancia mínima a núcleos (alta = más anómalo)
            # Para puntos núcleo: distancia = 0 (menos anómalo)
            # Para puntos frontera: distancia baja a núcleos (menos anómalo que ruido)
            scores = distancias_minimas
            
            # Asignar score especial a puntos núcleo (score mínimo)
            scores[indices_nucleo] = 0.0
        else:
            # Si no hay puntos núcleo, asignar scores basados en etiquetas
            scores[etiquetas == -1] = 1.0  # Ruido = score alto
            scores[etiquetas != -1] = 0.0  # Clusters = score bajo
        
        return scores
    
    @staticmethod
    def identificar_anomalias(datos: pd.DataFrame, etiquetas: np.ndarray, 
                            X_escalado: np.ndarray, modelo: DBSCAN, 
                            directorio_metricas: str) -> None:
        """
        Identifica y guarda las anomalías detectadas.
        
        Args:
            datos: DataFrame con los datos originales.
            etiquetas: Etiquetas de clusters.
            X_escalado: Datos escalados.
            modelo: Modelo DBSCAN entrenado.
            directorio_metricas: Directorio donde guardar los resultados.
        """
        try:
            # Calcular scores para todos los puntos
            anomaly_scores = DetectorAnomalias.calcular_scores_todos_los_puntos(etiquetas, X_escalado, modelo)
            
            # Crear DataFrame con todos los datos y scores
            datos_con_scores = datos.copy()
            datos_con_scores['anomaly_score'] = anomaly_scores
            datos_con_scores['is_outlier'] = (etiquetas == -1).astype(int)
            datos_con_scores['cluster_id'] = etiquetas
            
            # Guardar todos los datos con scores
            ruta_scores = os.path.join(directorio_metricas, 'scores_dbscan.csv')
            datos_con_scores.to_csv(ruta_scores, index=False)
            logging.info(f"Scores de todos los puntos guardados en: {ruta_scores}")
            
            # Guardar métricas estandarizadas en CSV para comparación (100% no-supervisado)
            ruta_metricas_csv = os.path.join(directorio_metricas, 'metrics.csv')
            metricas_df = pd.DataFrame([{
                'algoritmo': 'DBSCAN',
                'params_json': f'{{"eps": {modelo.eps}, "min_samples": {modelo.min_samples}}}',
                'n_clusters': len(set(etiquetas)) - (1 if -1 in etiquetas else 0),
                'silhouette_score': None,  # Se calculará si hay clusters válidos
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None,
                'pct_anomalias': np.mean(etiquetas == -1) * 100,
                'p95_minus_p50': np.percentile(anomaly_scores, 95) - np.percentile(anomaly_scores, 50),
                'mean_score': np.mean(anomaly_scores)
            }])
            
            # Calcular métricas de clustering si hay clusters válidos
            mascara_sin_ruido = etiquetas != -1
            if np.sum(mascara_sin_ruido) >= 10 and len(set(etiquetas[mascara_sin_ruido])) >= 2:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                X_sin_ruido = X_escalado[mascara_sin_ruido]
                etiquetas_sin_ruido = etiquetas[mascara_sin_ruido]
                
                metricas_df.loc[0, 'silhouette_score'] = silhouette_score(X_sin_ruido, etiquetas_sin_ruido)
                metricas_df.loc[0, 'calinski_harabasz_score'] = calinski_harabasz_score(X_sin_ruido, etiquetas_sin_ruido)
                metricas_df.loc[0, 'davies_bouldin_score'] = davies_bouldin_score(X_sin_ruido, etiquetas_sin_ruido)
            
            metricas_df.to_csv(ruta_metricas_csv, index=False)
            logging.info(f"Métricas CSV guardadas en: {ruta_metricas_csv}")
                    
            # Identificar anomalías (ruido)
            anomalias = datos_con_scores[etiquetas == -1].copy()
            
            if len(anomalias) == 0:
                logging.info("No se detectaron anomalías")
                return
            
            # Guardar solo anomalías ordenadas por score
            anomalias_ordenadas = anomalias.sort_values('anomaly_score', ascending=False)
            ruta_anomalias = os.path.join(directorio_metricas, 'anomalies.csv')
            anomalias_ordenadas.to_csv(ruta_anomalias, index=False)
            logging.info(f"Anomalías guardadas en: {ruta_anomalias}")
            
            logging.info(f"Total de anomalías detectadas: {len(anomalias)}")
            logging.info(f"Score promedio de anomalías: {np.mean(anomalias['anomaly_score']):.4f}")
            logging.info(f"Score máximo de anomalía: {np.max(anomalias['anomaly_score']):.4f}")
            
        except Exception as e:
            logging.error(f"Error identificando anomalías: {str(e)}")


    


def main():
    """Función principal que ejecuta todo el pipeline de clustering DBSCAN."""
    try:
        # Configuración inicial
        directorio_script = os.path.dirname(os.path.abspath(__file__))
        gestor_directorios = GestorDirectorios(directorio_script)
        gestor_directorios.crear_directorios()
        
        # Configurar logging
        ConfiguradorLogging.configurar_logging(gestor_directorios.directorio_metricas)
        logging.info("=== INICIANDO PROCESO DE CLUSTERING DBSCAN ===")
        
        # Cargar y procesar datos
        ruta_datos = os.path.join(directorio_script, 'data.csv')
        datos_originales = ProcesadorDatos.cargar_datos(ruta_datos)
        logging.info(f"Datos cargados: {len(datos_originales)} filas")
        
        datos_procesados, matriz_caracteristicas = ProcesadorDatos.preprocesar_datos(datos_originales)
        X_escalado, escalador = ProcesadorDatos.escalar_datos(matriz_caracteristicas)
        
        # Generar gráfica de k-distancias
        AnalizadorDistancias.generar_grafica_k_distancias(X_escalado, gestor_directorios.directorio_graficas)
        
        # Reducir muestra para optimización si es muy grande (para evitar problemas de memoria)
        X_para_optimizacion, indices_muestra = ProcesadorDatos.reducir_muestra_para_optimizacion(X_escalado)
        
        # Optimizar parámetros usando la muestra reducida
        grilla_parametros = OptimizadorDBSCAN.generar_grilla_parametros()
        mejor_resultado = OptimizadorDBSCAN.buscar_mejores_parametros(X_para_optimizacion, grilla_parametros)
        
        # Aplicar los mejores parámetros al dataset completo
        mejores_parametros = mejor_resultado['parametros']
        logging.info(f"Aplicando mejores parámetros al dataset completo...")
        
        modelo_final = DBSCAN(eps=mejores_parametros['eps'], min_samples=mejores_parametros['min_samples'])
        etiquetas_finales = modelo_final.fit_predict(X_escalado)
        
        # Recalcular métricas para el dataset completo
        n_clusters_final = len(set(etiquetas_finales)) - (1 if -1 in etiquetas_finales else 0)
        n_ruido_final = list(etiquetas_finales).count(-1)
        
        # Calcular métricas excluyendo ruido para el dataset completo
        mascara_sin_ruido = etiquetas_finales != -1
        if np.sum(mascara_sin_ruido) >= MIN_CLUSTERS_VALIDOS:
            X_sin_ruido = X_escalado[mascara_sin_ruido]
            etiquetas_sin_ruido = etiquetas_finales[mascara_sin_ruido]
            
            silhouette_final = silhouette_score(X_sin_ruido, etiquetas_sin_ruido)
            calinski_harabasz_final = calinski_harabasz_score(X_sin_ruido, etiquetas_sin_ruido)
            davies_bouldin_final = davies_bouldin_score(X_sin_ruido, etiquetas_sin_ruido)
        else:
            silhouette_final = mejor_resultado['silhouette']
            calinski_harabasz_final = mejor_resultado['calinski_harabasz']
            davies_bouldin_final = mejor_resultado['davies_bouldin']
        
        # Crear diccionario de métricas usando los resultados del dataset completo
        metricas = {
            'eps': mejores_parametros['eps'],
            'min_samples': mejores_parametros['min_samples'],
            'n_clusters': n_clusters_final,
            'n_ruido': n_ruido_final,
            'silhouette': silhouette_final,
            'calinski_harabasz': calinski_harabasz_final,
            'davies_bouldin': davies_bouldin_final
        }
        
        # Mostrar resultados
        logging.info(f"\n=== MEJORES RESULTADOS ===")
        logging.info(f"Parámetros óptimos: eps={metricas['eps']:.3f}, min_samples={metricas['min_samples']}")
        logging.info(f"Clusters encontrados: {metricas['n_clusters']}")
        logging.info(f"Puntos de ruido: {metricas['n_ruido']}")
        logging.info(f"Coeficiente Silhouette: {metricas['silhouette']:{FORMATO_METRICAS}}")
        logging.info(f"Calinski-Harabasz: {metricas['calinski_harabasz']:{FORMATO_METRICAS}}")
        logging.info(f"Davies-Bouldin: {metricas['davies_bouldin']:{FORMATO_METRICAS}}")
        
        # Guardar métricas
        ruta_metricas = os.path.join(gestor_directorios.directorio_metricas, 'metrics.txt')
        GuardadorModelos.guardar_metricas(metricas, ruta_metricas)
        
        # Guardar modelos
        ruta_modelo_pkl = os.path.join(gestor_directorios.directorio_modelos, 'dbscan_model.pkl')
        GuardadorModelos.guardar_modelo_pkl(modelo_final, ruta_modelo_pkl)
        
        # Guardar escalador para inferencia futura
        ruta_escalador = os.path.join(gestor_directorios.directorio_modelos, 'scaler.pkl')
        joblib.dump(escalador, ruta_escalador)
        logging.info(f"Escalador guardado en {ruta_escalador}")
        
        ruta_modelo_h5 = os.path.join(gestor_directorios.directorio_modelos, 'dbscan_model.h5')
        GuardadorModelos.guardar_modelo_h5(modelo_final, ruta_modelo_h5)
        
        # Visualizar resultados
        VisualizadorClusters.visualizar_clusters_2d(X_escalado, etiquetas_finales, gestor_directorios.directorio_graficas)
        VisualizadorClusters.visualizar_clusters_3d(X_escalado, etiquetas_finales, gestor_directorios.directorio_graficas)
        
        # Detectar anomalías
        DetectorAnomalias.identificar_anomalias(datos_procesados, etiquetas_finales, X_escalado, modelo_final, gestor_directorios.directorio_metricas)
        
        logging.info("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()