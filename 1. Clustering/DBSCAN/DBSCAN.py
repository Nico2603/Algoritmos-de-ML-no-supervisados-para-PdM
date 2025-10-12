import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
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
from pathlib import Path
import time
import tracemalloc
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

warnings.filterwarnings('ignore')
plt.style.use('default')

USECOLS = config.USECOLS
CARACTERISTICAS_BASE = config.CARACTERISTICAS_BASE
MIN_CLUSTERS_VALIDOS = 2
N_JOBS_PARALELO = config.N_JOBS_PARALELO
FORMATO_METRICAS = '.4f'
MAX_MUESTRAS_OPTIMIZACION = config.MAX_MUESTRAS_OPTIMIZACION
MAX_MUESTRAS_VISUALIZATION = config.MAX_MUESTRAS_VISUALIZATION
SCATTER_SIZE = config.SCATTER_SIZE_NORMAL
SCATTER_SIZE_NOISE = config.SCATTER_SIZE_NOISE
CMAP_CLUSTERING = config.CMAP_CLUSTERING
RANDOM_STATE = config.RANDOM_STATE
BATCH_SIZE = config.BATCH_SIZE

class ConfiguradorLogging:
    
    @staticmethod
    def configurar_logging(directorio_metricas: Path) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        ruta_archivo_log = directorio_metricas / 'output.log'
        file_handler = logging.FileHandler(ruta_archivo_log, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

class GestorDirectorios:
    
    def __init__(self, directorio_base: Path):
        self.directorio_base = Path(directorio_base)
        self.directorio_modelos = self.directorio_base / 'modelos_entrenados_DBSCAN'
        self.directorio_graficas = self.directorio_base / 'graficas_DBSCAN'
        self.directorio_metricas = self.directorio_base / 'metricas_DBSCAN'
    
    def crear_directorios(self) -> None:
        directorios = [self.directorio_modelos, self.directorio_graficas, self.directorio_metricas]
        for directorio in directorios:
            directorio.mkdir(parents=True, exist_ok=True)

class ProcesadorDatos:
    
    @staticmethod
    def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"El archivo {ruta_archivo} no existe")
        
        try:
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
            
            datos = datos[['fecha'] + CARACTERISTICAS_BASE].copy()
            
            datos.sort_values('fecha', inplace=True)
            
            logging.info(f"Archivo CSV cargado exitosamente:")
            logging.info(f"  - Dimensiones: {datos.shape[0]} filas x {datos.shape[1]} columnas")
            logging.info(f"  - Columnas cargadas: {list(datos.columns)}")
            logging.info(f"  - Tipos de datos por columna:")
            for col in datos.columns:
                logging.info(f"    • {col}: {datos[col].dtype}")
            
            logging.info(f"  - Primeras 3 filas de datos:")
            for i, (_, fila) in enumerate(datos.head(3).iterrows()):
                logging.info(f"    Fila {i+1}: {dict(fila)}")
            
            valores_faltantes = datos[CARACTERISTICAS_BASE].isnull().sum()
            if valores_faltantes.any():
                logging.info(f"  - Valores faltantes por columna:")
                for col, faltantes in valores_faltantes.items():
                    if faltantes > 0:
                        logging.info(f"    • {col}: {faltantes} valores faltantes")
            else:
                logging.info(f"  - No se encontraron valores faltantes")
            
            config.validar_datos_entrada(datos, CARACTERISTICAS_BASE)
            
            return datos
        except Exception as e:
            raise ValueError(f"Error al cargar el archivo CSV: {str(e)}")
    
    @staticmethod
    def validar_caracteristicas(datos: pd.DataFrame, caracteristicas: List[str]) -> None:
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
        ProcesadorDatos.validar_caracteristicas(datos, CARACTERISTICAS_BASE)
        logging.info(f"Características base validadas: {CARACTERISTICAS_BASE}")
        
        columnas_adicionales = [col for col in datos.columns if col not in CARACTERISTICAS_BASE]
        if columnas_adicionales:
            logging.info(f"Columnas adicionales encontradas: {columnas_adicionales}")
        
        filas_originales = len(datos)
        datos_limpios = datos.dropna()
        filas_eliminadas = filas_originales - len(datos_limpios)
        
        logging.info(f"Limpieza de datos:")
        logging.info(f"  - Filas originales: {filas_originales}")
        logging.info(f"  - Filas eliminadas por valores faltantes: {filas_eliminadas}")
        logging.info(f"  - Filas restantes: {len(datos_limpios)}")
        
        if datos_limpios.empty:
            raise ValueError("No quedan datos después de eliminar valores faltantes")
        
        datos_limpios = datos_limpios.copy()
        datos_limpios['magnitud_aceleracion'] = np.sqrt(
            datos_limpios['acceleration_x']**2 +
            datos_limpios['acceleration_y']**2 +
            datos_limpios['acceleration_z']**2
        )
        logging.info("Característica derivada añadida: 'magnitud_aceleracion'")
        
        caracteristicas = CARACTERISTICAS_BASE + ['magnitud_aceleracion']
        matriz_caracteristicas = datos_limpios[caracteristicas].values
        
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
        escalador = StandardScaler()
        X_escalado = escalador.fit_transform(X)
        logging.info("Datos escalados correctamente")
        return X_escalado, escalador
    
    @staticmethod
    def reducir_muestra_para_optimizacion(X: np.ndarray, max_muestras: int = MAX_MUESTRAS_OPTIMIZACION) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= max_muestras:
            return X, np.arange(len(X))
        
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
        indices_seleccionados = np.sort(indices_seleccionados)
        
        X_reducido = X[indices_seleccionados]
        
        logging.info(f"Dataset reducido para optimización: {len(X)} -> {len(X_reducido)} muestras")
        return X_reducido, indices_seleccionados
    
    @staticmethod
    def reducir_muestra_para_visualizacion(X: np.ndarray, etiquetas: np.ndarray, max_muestras: int = MAX_MUESTRAS_VISUALIZATION) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= max_muestras:
            return X, etiquetas
        
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
        indices_seleccionados = np.sort(indices_seleccionados)
        
        X_reducido = X[indices_seleccionados]
        etiquetas_reducidas = etiquetas[indices_seleccionados]
        
        logging.info(f"Dataset reducido para visualización: {len(X)} -> {len(X_reducido)} muestras")
        return X_reducido, etiquetas_reducidas

class AnalizadorDistancias:
    
    @staticmethod
    def generar_grafica_k_distancias(X_escalado: np.ndarray, directorio_graficas: str, k: int = 5) -> None:
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
            
            ruta_grafica = directorio_graficas / 'k_distance_graph.png'
            plt.savefig(ruta_grafica, dpi=200, bbox_inches='tight')
            plt.close("all")
            
            logging.info(f"Gráfica de k-distancias guardada en {ruta_grafica}")
        except Exception as e:
            logging.error(f"Error al generar gráfica de k-distancias: {str(e)}")

class OptimizadorDBSCAN:
    
    @staticmethod
    def generar_grilla_parametros(eps_min: float = 0.2, eps_max: float = 1.5, 
                                 n_eps: int = 4, min_samples_min: int = 3, 
                                 min_samples_max: int = 4) -> List[Dict[str, Any]]:
        valores_eps = np.linspace(eps_min, eps_max, n_eps)
        valores_min_samples = range(min_samples_min, min_samples_max + 1)
        
        param_grid = {'eps': valores_eps, 'min_samples': valores_min_samples}
        parametros = list(ParameterGrid(param_grid))
        
        logging.info(f"Grilla de parámetros generada: {len(parametros)} combinaciones")
        return parametros
    
    @staticmethod
    def evaluar_parametros_dbscan(parametros: Dict[str, Any], X_escalado: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            eps = parametros['eps']
            min_samples = parametros['min_samples']
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            etiquetas = dbscan.fit_predict(X_escalado)
            
            n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
            n_ruido = list(etiquetas).count(-1)
            
            if n_clusters < MIN_CLUSTERS_VALIDOS:
                return None
            
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
        logging.info(f"Iniciando búsqueda de mejores parámetros...")
        logging.info(f"Total de combinaciones a evaluar: {len(grilla_parametros)}")
        logging.info(f"Tamaño de datos para optimización: {X_escalado.shape}")
        logging.info(f"Número de trabajos paralelos: {N_JOBS_PARALELO}")
        
        if N_JOBS_PARALELO == 1:
            logging.info("Ejecutando búsqueda secuencial para optimizar memoria...")
            resultados = []
            for i, params in enumerate(grilla_parametros):
                if i % 5 == 0:
                    logging.info(f"Evaluando combinación {i+1}/{len(grilla_parametros)}: {params}")
                resultado = OptimizadorDBSCAN.evaluar_parametros_dbscan(params, X_escalado)
                resultados.append(resultado)
                
                if i % 10 == 0:
                    gc.collect()
        else:
            resultados = Parallel(n_jobs=N_JOBS_PARALELO)(
                delayed(OptimizadorDBSCAN.evaluar_parametros_dbscan)(params, X_escalado) 
                for params in grilla_parametros
            )
        
        resultados_validos = [res for res in resultados if res is not None]
        
        logging.info(f"Combinaciones válidas encontradas: {len(resultados_validos)}/{len(grilla_parametros)}")
        
        if not resultados_validos:
            raise ValueError("No se encontraron parámetros que generen clusters válidos")
        
        mejor_resultado = max(resultados_validos, key=lambda x: x['silhouette'])
        
        logging.info(f"Mejor resultado encontrado:")
        logging.info(f"  - Parámetros: eps={mejor_resultado['parametros']['eps']:.3f}, min_samples={mejor_resultado['parametros']['min_samples']}")
        logging.info(f"  - Silhouette Score: {mejor_resultado['silhouette']:.4f}")
        logging.info("Búsqueda de parámetros completada")
        return mejor_resultado

class VisualizadorClusters:
    
    @staticmethod
    def visualizar_clusters_2d(X_escalado: np.ndarray, etiquetas: np.ndarray, 
                              directorio_graficas: str, titulo: str = "Clustering DBSCAN (2D)") -> None:
        try:
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(X_escalado)
            
            X_vis, etiquetas_vis = ProcesadorDatos.reducir_muestra_para_visualizacion(X_pca, etiquetas)
            
            n_clusters = len(set(etiquetas_vis)) - (1 if -1 in etiquetas_vis else 0)
            n_ruido = np.sum(etiquetas_vis == -1)
            
            etiquetas_unicas = set(etiquetas_vis)
            colores = plt.get_cmap(CMAP_CLUSTERING)(np.linspace(0, 1, len(etiquetas_unicas)))
            
            plt.figure(figsize=config.FIGSIZE_2D)
            for etiqueta, color in zip(etiquetas_unicas, colores):
                mascara = etiquetas_vis == etiqueta
                puntos = X_vis[mascara]
                
                if etiqueta == -1:
                    plt.scatter(puntos[:, 0], puntos[:, 1], c='black', marker='x', 
                              s=SCATTER_SIZE_NOISE, alpha=0.7, label='Ruido')
                else:
                    plt.scatter(puntos[:, 0], puntos[:, 1], c=[color], marker='o', 
                              s=SCATTER_SIZE, alpha=0.7, label=f'Cluster {etiqueta}')
            
            plt.title(f'{titulo}\nClusters: {n_clusters} | Ruido: {n_ruido} | Muestras: {len(X_vis):,}',
                     fontsize=14, pad=15)
            plt.xlabel(f'Componente Principal 1 (Varianza: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'Componente Principal 2 (Varianza: {pca.explained_variance_ratio_[1]:.2%})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            ruta_grafica = directorio_graficas / 'clusters_2d_pca.png'
            plt.savefig(ruta_grafica, dpi=200, bbox_inches='tight')
            plt.close("all")
            
            logging.info(f"Visualización 2D guardada en {ruta_grafica}")
        except Exception as e:
            logging.error(f"Error en visualización 2D: {str(e)}")
    
    @staticmethod
    def visualizar_clusters_3d(X_escalado: np.ndarray, etiquetas: np.ndarray, 
                              directorio_graficas: str, titulo: str = "Clustering DBSCAN (3D)") -> None:
        try:
            pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
            X_pca_3d = pca_3d.fit_transform(X_escalado)
            
            X_vis, etiquetas_vis = ProcesadorDatos.reducir_muestra_para_visualizacion(X_pca_3d, etiquetas)
            
            n_clusters = len(set(etiquetas_vis)) - (1 if -1 in etiquetas_vis else 0)
            n_ruido = np.sum(etiquetas_vis == -1)
            
            etiquetas_unicas = set(etiquetas_vis)
            colores = plt.get_cmap(CMAP_CLUSTERING)(np.linspace(0, 1, len(etiquetas_unicas)))
            
            fig = plt.figure(figsize=config.FIGSIZE_3D)
            ax = fig.add_subplot(111, projection='3d')
            
            for etiqueta, color in zip(etiquetas_unicas, colores):
                mascara = etiquetas_vis == etiqueta
                puntos = X_vis[mascara]
                
                if etiqueta == -1:
                    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], 
                             c='black', marker='x', s=SCATTER_SIZE_NOISE, alpha=0.7, label='Ruido')
                else:
                    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], 
                             c=[color], marker='o', s=SCATTER_SIZE, alpha=0.7, label=f'Cluster {etiqueta}')
            
            ax.set_title(f'{titulo}\nClusters: {n_clusters} | Ruido: {n_ruido} | Muestras: {len(X_vis):,}',
                        fontsize=14, pad=15)
            ax.set_xlabel(f'PC1 (Var: {pca_3d.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 (Var: {pca_3d.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 (Var: {pca_3d.explained_variance_ratio_[2]:.2%})')
            
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
            
            ax.view_init(elev=config.VIEW_ELEV, azim=config.VIEW_AZIM)
            
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
            
            ruta_grafica = directorio_graficas / 'clusters_3d_pca.png'
            plt.savefig(ruta_grafica, dpi=200, bbox_inches='tight')
            plt.close("all")
            
            logging.info(f"Visualización 3D guardada en {ruta_grafica}")
        except Exception as e:
            logging.error(f"Error en visualización 3D: {str(e)}")

class GuardadorModelos:
    
    @staticmethod
    def guardar_modelo_pkl(modelo: DBSCAN, ruta_archivo: str) -> None:
        try:
            joblib.dump(modelo, ruta_archivo)
            logging.info(f"Modelo guardado en formato pickle: {ruta_archivo}")
        except Exception as e:
            logging.error(f"Error guardando modelo pickle: {str(e)}")
    
    @staticmethod
    def guardar_modelo_h5(modelo: DBSCAN, ruta_archivo: str) -> None:
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
    
    @staticmethod
    def calcular_scores_todos_los_puntos(etiquetas: np.ndarray, X_escalado: np.ndarray, 
                                       modelo: DBSCAN) -> np.ndarray:
        scores = np.zeros(len(etiquetas))
        
        if hasattr(modelo, 'core_sample_indices_') and len(modelo.core_sample_indices_) > 0:
            from sklearn.neighbors import NearestNeighbors
            indices_nucleo = modelo.core_sample_indices_
            
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(X_escalado[indices_nucleo])
            
            if len(X_escalado) > BATCH_SIZE:
                logging.info(f"Dataset grande detectado ({len(X_escalado)} puntos), usando procesamiento por batches...")
                scores_list = []
                for i in range(0, len(X_escalado), BATCH_SIZE):
                    batch = X_escalado[i:i+BATCH_SIZE]
                    dist, _ = nn_model.kneighbors(batch)
                    scores_list.append(dist.flatten())
                distancias_minimas = np.concatenate(scores_list)
            else:
                distancias_minimas, _ = nn_model.kneighbors(X_escalado)
                distancias_minimas = distancias_minimas.flatten()
            
            scores = distancias_minimas
            scores[indices_nucleo] = 0.0
        else:
            scores[etiquetas == -1] = 1.0
            scores[etiquetas != -1] = 0.0
        
        scores_normalizados = config.normalizar_scores_min_max(scores)
        
        return scores_normalizados
    
    @staticmethod
    def identificar_anomalias(datos: pd.DataFrame, etiquetas: np.ndarray, 
                            X_escalado: np.ndarray, modelo: DBSCAN, 
                            directorio_metricas: str, tiempo_total: float = 0.0, 
                            memoria_max: float = 0.0) -> None:
        try:
            anomaly_scores = DetectorAnomalias.calcular_scores_todos_los_puntos(etiquetas, X_escalado, modelo)
            
            datos_con_scores = datos.copy()
            datos_con_scores['anomaly_score'] = anomaly_scores
            datos_con_scores['is_outlier'] = (etiquetas == -1).astype(int)
            datos_con_scores['cluster_id'] = etiquetas
            
            ruta_scores = os.path.join(directorio_metricas, 'anomaly_scores.csv')
            datos_con_scores.to_csv(ruta_scores, index=False)
            logging.info(f"Scores de todos los puntos guardados en: {ruta_scores}")
            
            ruta_metricas_csv = os.path.join(directorio_metricas, 'metrics.csv')
            metricas_df = pd.DataFrame([{
                'algoritmo': 'DBSCAN',
                'params_json': f'{{"eps": {modelo.eps}, "min_samples": {modelo.min_samples}}}',
                'n_clusters': len(set(etiquetas)) - (1 if -1 in etiquetas else 0),
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None,
                'pct_anomalias': np.mean(etiquetas == -1) * 100,
                'p95_minus_p50': np.percentile(anomaly_scores, 95) - np.percentile(anomaly_scores, 50),
                'mean_score': np.mean(anomaly_scores),
                'tiempo_ejecucion_s': tiempo_total,
                'memoria_max_mb': memoria_max
            }])
            
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
                    
            anomalias = datos_con_scores[etiquetas == -1].copy()
            
            if len(anomalias) == 0:
                logging.info("No se detectaron anomalías")
                return
            
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
    tiempo_inicio = time.time()
    tracemalloc.start()
    
    try:
        directorio_script = Path(__file__).parent
        gestor_directorios = GestorDirectorios(directorio_script)
        gestor_directorios.crear_directorios()
        
        ConfiguradorLogging.configurar_logging(gestor_directorios.directorio_metricas)
        logging.info("=== INICIANDO PROCESO DE CLUSTERING DBSCAN ===")
        
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        ruta_datos = directorio_script / 'data.csv'
        datos_originales = ProcesadorDatos.cargar_datos(ruta_datos)
        logging.info(f"Datos cargados: {len(datos_originales)} filas")
        
        datos_procesados, matriz_caracteristicas = ProcesadorDatos.preprocesar_datos(datos_originales)
        X_escalado, escalador = ProcesadorDatos.escalar_datos(matriz_caracteristicas)
        
        AnalizadorDistancias.generar_grafica_k_distancias(X_escalado, gestor_directorios.directorio_graficas)
        
        X_para_optimizacion, indices_muestra = ProcesadorDatos.reducir_muestra_para_optimizacion(X_escalado)
        
        grilla_parametros = OptimizadorDBSCAN.generar_grilla_parametros()
        mejor_resultado = OptimizadorDBSCAN.buscar_mejores_parametros(X_para_optimizacion, grilla_parametros)
        
        mejores_parametros = mejor_resultado['parametros']
        logging.info(f"Aplicando mejores parámetros al dataset completo...")
        
        modelo_final = DBSCAN(eps=mejores_parametros['eps'], min_samples=mejores_parametros['min_samples'])
        etiquetas_finales = modelo_final.fit_predict(X_escalado)
        
        n_clusters_final = len(set(etiquetas_finales)) - (1 if -1 in etiquetas_finales else 0)
        n_ruido_final = list(etiquetas_finales).count(-1)
        
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
        
        metricas = {
            'eps': mejores_parametros['eps'],
            'min_samples': mejores_parametros['min_samples'],
            'n_clusters': n_clusters_final,
            'n_ruido': n_ruido_final,
            'silhouette': silhouette_final,
            'calinski_harabasz': calinski_harabasz_final,
            'davies_bouldin': davies_bouldin_final
        }
        
        logging.info(f"\n=== MEJORES RESULTADOS ===")
        logging.info(f"Parámetros óptimos: eps={metricas['eps']:.3f}, min_samples={metricas['min_samples']}")
        logging.info(f"Clusters encontrados: {metricas['n_clusters']}")
        logging.info(f"Puntos de ruido: {metricas['n_ruido']}")
        logging.info(f"Coeficiente Silhouette: {metricas['silhouette']:{FORMATO_METRICAS}}")
        logging.info(f"Calinski-Harabasz: {metricas['calinski_harabasz']:{FORMATO_METRICAS}}")
        logging.info(f"Davies-Bouldin: {metricas['davies_bouldin']:{FORMATO_METRICAS}}")
        
        ruta_metricas = os.path.join(gestor_directorios.directorio_metricas, 'metrics.txt')
        GuardadorModelos.guardar_metricas(metricas, ruta_metricas)
        
        ruta_modelo_pkl = os.path.join(gestor_directorios.directorio_modelos, 'dbscan_model.pkl')
        GuardadorModelos.guardar_modelo_pkl(modelo_final, ruta_modelo_pkl)
        
        ruta_escalador = os.path.join(gestor_directorios.directorio_modelos, 'scaler.pkl')
        joblib.dump(escalador, ruta_escalador)
        logging.info(f"Escalador guardado en {ruta_escalador}")
        
        ruta_modelo_h5 = os.path.join(gestor_directorios.directorio_modelos, 'dbscan_model.h5')
        GuardadorModelos.guardar_modelo_h5(modelo_final, ruta_modelo_h5)
        
        tiempo_total = time.time() - tiempo_inicio
        memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
        memoria_max = memoria_pico / 1024**2
        tracemalloc.stop()
        
        logging.info(f"Tiempo total de ejecucion: {tiempo_total:.2f} segundos")
        logging.info(f"Memoria maxima utilizada: {memoria_max:.2f} MB")
        
        VisualizadorClusters.visualizar_clusters_2d(X_escalado, etiquetas_finales, gestor_directorios.directorio_graficas)
        VisualizadorClusters.visualizar_clusters_3d(X_escalado, etiquetas_finales, gestor_directorios.directorio_graficas)
        
        DetectorAnomalias.identificar_anomalias(datos_procesados, etiquetas_finales, X_escalado, modelo_final, 
                                               gestor_directorios.directorio_metricas, tiempo_total, memoria_max)
        
        logging.info("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
        
    except Exception as e:
        tracemalloc.stop()
        logging.error(f"Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()