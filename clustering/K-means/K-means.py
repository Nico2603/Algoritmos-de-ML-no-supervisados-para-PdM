import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.utils import check_random_state
import joblib
import h5py
import logging
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Any
from pathlib import Path
import warnings
import time
import tracemalloc
import random

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

warnings.filterwarnings('ignore')

# Paso 1: Carga y validación de datos
RANDOM_STATE = config.RANDOM_STATE
K_MIN = 2
K_MAX = 4
FIGSIZE_2D = config.FIGSIZE_2D
FIGSIZE_3D = config.FIGSIZE_3D
SCATTER_SIZE = config.SCATTER_SIZE_NORMAL
N_JOBS = config.N_JOBS
PCA_COMPONENTS_2D = 2
PCA_COMPONENTS_3D = 3
SAMPLE_OPT = config.SAMPLE_OPT  # Muestreo para optimización
SAMPLE_VIS = config.SAMPLE_VIS  # Muestreo para visualización
MAX_MUESTRAS_OPTIMIZACION = SAMPLE_OPT
MAX_MUESTRAS_VISUALIZATION = SAMPLE_VIS
SILHOUETTE_SAMPLE = config.SILHOUETTE_SAMPLE
USECOLS = config.USECOLS
CARACTERISTICAS_BASE = config.CARACTERISTICAS_BASE
CARACTERISTICA_MAGNITUD = config.CARACTERISTICA_MAGNITUD
CMAP_CLUSTERING = config.CMAP_CLUSTERING

class KMeansAnalyzer:
    
    def __init__(self, directorio_script: Path):
        self.directorio_script = Path(directorio_script)
        self.directorio_modelos = self.directorio_script / 'modelos_entrenados_KMeans'
        self.directorio_graficas = self.directorio_script / 'graficas_KMeans'
        self.directorio_metricas = self.directorio_script / 'metricas_KMeans'
        
        self._crear_directorios()
        self._configurar_logging()
        
        self.datos = None
        self.X = None
        self.X_escalado = None
        self.escalador = None
        self.kmeans_final = None
        
        self.tiempo_inicio = None
        self.tiempo_total = None
        self.memoria_max = None
    
    def _crear_directorios(self) -> None:
        for directorio in [self.directorio_modelos, self.directorio_graficas, self.directorio_metricas]:
            directorio.mkdir(parents=True, exist_ok=True)
    
    def _configurar_logging(self) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        ruta_archivo_log = self.directorio_metricas / 'output.log'
        file_handler = logging.FileHandler(ruta_archivo_log, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    
    def cargar_datos(self, ruta_datos: Path) -> None:
        try:
            if not ruta_datos.exists():
                raise FileNotFoundError(f"El archivo no existe en la ruta: {ruta_datos}")
            
            try:
                self.datos = pd.read_csv(
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
            except UnicodeDecodeError:
                self.datos = pd.read_csv(
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
                logging.info("Archivo cargado con encoding latin-1")
            
            self.datos = self.datos[['fecha'] + CARACTERISTICAS_BASE].copy()
            
            self.datos.sort_values('fecha', inplace=True)
            
            logging.info(f"Datos cargados correctamente desde: {ruta_datos}")
            logging.info(f"Forma del dataset: {self.datos.shape}")
            logging.info(f"Columnas cargadas: {list(self.datos.columns)}")
            
            logging.info("Primeras 3 filas del dataset:")
            for i, row in self.datos.head(3).iterrows():
                logging.info(f"  Fila {i}: {dict(row)}")
            
            logging.info("Tipos de datos por columna:")
            for col in self.datos.columns:
                logging.info(f"  {col}: {self.datos[col].dtype}")
            
            filas_originales = len(self.datos)
            
            valores_faltantes = self.datos[CARACTERISTICAS_BASE].isnull().sum()
            if valores_faltantes.sum() > 0:
                logging.info("Valores faltantes por columna:")
                for col, count in valores_faltantes.items():
                    if count > 0:
                        logging.info(f"  {col}: {count} valores faltantes")
            
            self.datos = self.datos.dropna(subset=CARACTERISTICAS_BASE)
            filas_eliminadas = filas_originales - len(self.datos)
            
            if filas_eliminadas > 0:
                logging.info(f"Se eliminaron {filas_eliminadas} filas con valores faltantes.")
            else:
                logging.info("No se encontraron valores faltantes.")
            
            if len(self.datos) < 10:
                raise ValueError(f"Datos insuficientes después de la limpieza: solo {len(self.datos)} filas")
            
            logging.info(f"Dataset final: {len(self.datos)} filas")
            
            logging.info("Estadísticas básicas de las características:")
            stats = self.datos[CARACTERISTICAS_BASE].describe()
            logging.info(f"\n{stats}")
            
            config.validar_datos_entrada(self.datos, CARACTERISTICAS_BASE)
            
            self._crear_caracteristicas()
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def _crear_caracteristicas(self) -> None:
        self.datos[CARACTERISTICA_MAGNITUD] = np.sqrt(
            self.datos['acceleration_x']**2 +
            self.datos['acceleration_y']**2 +
            self.datos['acceleration_z']**2
        )
        logging.info(f"Característica '{CARACTERISTICA_MAGNITUD}' añadida.")
        
        caracteristicas = CARACTERISTICAS_BASE + [CARACTERISTICA_MAGNITUD]
        self.X = self.datos[caracteristicas].values
    
    def escalar_datos(self) -> None:
        """Normaliza datos al rango [0, 1] usando MinMaxScaler"""
        self.X_escalado, self.escalador = config.normalizar_con_minmax(self.X)
        logging.info("Datos normalizados al rango [0, 1] correctamente.")
    
    def reducir_muestra_para_optimizacion(self, max_muestras: int = MAX_MUESTRAS_OPTIMIZACION) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.X_escalado) <= max_muestras:
            return self.X_escalado, np.arange(len(self.X_escalado))
        
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        indices_seleccionados = np.random.choice(len(self.X_escalado), max_muestras, replace=False)
        indices_seleccionados = np.sort(indices_seleccionados)
        
        X_reducido = self.X_escalado[indices_seleccionados]
        
        logging.info(f"Dataset reducido para optimización: {len(self.X_escalado)} -> {len(X_reducido)} muestras")
        return X_reducido, indices_seleccionados
    
    def reducir_muestra_para_visualizacion(self, X: np.ndarray, etiquetas: np.ndarray = None, max_muestras: int = MAX_MUESTRAS_VISUALIZATION) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) <= max_muestras:
            return X, etiquetas if etiquetas is not None else np.arange(len(X))
        
        config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
        
        indices_seleccionados = np.random.choice(len(X), max_muestras, replace=False)
        indices_seleccionados = np.sort(indices_seleccionados)
        
        X_reducido = X[indices_seleccionados]
        etiquetas_reducidas = etiquetas[indices_seleccionados] if etiquetas is not None else indices_seleccionados
        
        logging.info(f"Dataset reducido para visualización: {len(X)} -> {len(X_reducido)} muestras")
        return X_reducido, etiquetas_reducidas
    
    def _evaluar_k(self, k: int) -> Tuple[int, float, float, float, float, np.ndarray, np.ndarray]:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(self.X_escalado)
        labels = kmeans.labels_
        
        inertia_k = kmeans.inertia_
        
        # Usar muestra para Silhouette (evita O(n²) en datasets grandes)
        n_sil = min(SILHOUETTE_SAMPLE, len(self.X_escalado))
        silhouette_k = silhouette_score(self.X_escalado, labels, sample_size=n_sil, random_state=RANDOM_STATE)
        
        calinski_k = calinski_harabasz_score(self.X_escalado, labels)
        davies_k = davies_bouldin_score(self.X_escalado, labels)
        
        logging.info(f"K={k}, Inercia={inertia_k:.2f}, Silhouette={silhouette_k:.4f}, "
                    f"Calinski-Harabasz={calinski_k:.4f}, Davies-Bouldin={davies_k:.4f}")
        
        return (k, inertia_k, silhouette_k, calinski_k, davies_k, labels, kmeans.cluster_centers_)
    
    def encontrar_k_optimo(self) -> Tuple[int, Dict[str, List[float]]]:
        X_para_optimizacion, indices_muestra = self.reducir_muestra_para_optimizacion()
        
        X_escalado_original = self.X_escalado.copy()
        self.X_escalado = X_para_optimizacion
        
        K_range = range(K_MIN, K_MAX)
        
        resultados = Parallel(n_jobs=N_JOBS)(
            delayed(self._evaluar_k)(k) for k in K_range
        )
        
        self.X_escalado = X_escalado_original
        
        metricas = {
            'inercia': [],
            'silhouette': [],
            'calinski': [],
            'davies': []
        }
        
        for res in resultados:
            k, inertia_k, silhouette_k, calinski_k, davies_k, labels_k, centers_k = res
            metricas['inercia'].append(inertia_k)
            metricas['silhouette'].append(silhouette_k)
            metricas['calinski'].append(calinski_k)
            metricas['davies'].append(davies_k)
        
        k_optimo = K_range[metricas['silhouette'].index(max(metricas['silhouette']))]
        logging.info(f"Número óptimo de clusters seleccionado: K={k_optimo}")
        
        return k_optimo, metricas, resultados
    
    
    def entrenar_modelo_final(self, k_optimo: int, resultados: List[Tuple]) -> None:
        resultado_optimo = next(res for res in resultados if res[0] == k_optimo)
        
        self.kmeans_final = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init='auto')
        self.kmeans_final.fit(self.X_escalado)
        
        self.labels = self.kmeans_final.labels_
        self.cluster_centers = self.kmeans_final.cluster_centers_
        
        # Usar muestra para Silhouette en dataset completo
        n_sil = min(SILHOUETTE_SAMPLE, len(self.X_escalado))
        
        self.metricas_finales = {
            'k_optimo': k_optimo,
            'inertia': self.kmeans_final.inertia_,
            'silhouette': silhouette_score(self.X_escalado, self.labels, 
                                           sample_size=n_sil, random_state=RANDOM_STATE),
            'calinski_harabasz': calinski_harabasz_score(self.X_escalado, self.labels),
            'davies_bouldin': davies_bouldin_score(self.X_escalado, self.labels)
        }
        
        logging.info("Modelo K-Means entrenado con el número óptimo de clusters.")
        self._mostrar_metricas_finales()
    
    def _mostrar_metricas_finales(self) -> None:
        logging.info(f"Silhouette Score: {self.metricas_finales['silhouette']:.4f}")
        logging.info(f"Calinski-Harabasz Score: {self.metricas_finales['calinski_harabasz']:.4f}")
        logging.info(f"Davies-Bouldin Score: {self.metricas_finales['davies_bouldin']:.4f}")
        logging.info(f"Inercia (SSE): {self.metricas_finales['inertia']:.2f}")
    
    def calcular_puntuaciones_anomalia(self) -> None:
        distancias = np.linalg.norm(
            self.X_escalado - self.cluster_centers[self.labels], axis=1
        )
        
        distancias_normalizadas = config.normalizar_scores_min_max(distancias)
        
        threshold = np.percentile(distancias_normalizadas, 95)
        outliers_binarios = (distancias_normalizadas > threshold).astype(int)
        
        self.datos['anomaly_score'] = distancias_normalizadas
        self.datos['is_outlier'] = outliers_binarios
        self.datos['cluster_id'] = self.labels
        
        n_outliers = outliers_binarios.sum()
        pct_outliers = (n_outliers / len(outliers_binarios)) * 100
        
        logging.info(f"Puntuación de anomalías calculada y normalizada [0, 1].")
        logging.info(f"Outliers detectados: {n_outliers} ({pct_outliers:.2f}%) usando percentil 95")
    
    
    def guardar_resultados(self) -> None:
        ruta_metricas = self.directorio_metricas / 'metrics.txt'
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            f.write("=== RESULTADOS CLUSTERING K-MEANS ===\n\n")
            f.write(f"Mejores parámetros:\n")
            f.write(f"  - k_clusters: {self.metricas_finales['k_optimo']}\n\n")
            f.write(f"Resultados del clustering:\n")
            f.write(f"  - Número de clusters: {self.metricas_finales['k_optimo']}\n")
            f.write(f"  - Inercia (SSE): {self.metricas_finales['inertia']:.2f}\n\n")
            f.write(f"Métricas de calidad:\n")
            f.write(f"  - Coeficiente Silhouette: {self.metricas_finales['silhouette']:.4f}\n")
            f.write(f"  - Calinski-Harabasz Score: {self.metricas_finales['calinski_harabasz']:.4f}\n")
            f.write(f"  - Davies-Bouldin Index: {self.metricas_finales['davies_bouldin']:.4f}\n")
        
        ruta_scores = self.directorio_metricas / 'anomaly_scores.csv'
        datos_salida = self.datos[['fecha'] + CARACTERISTICAS_BASE + ['anomaly_score', 'is_outlier', 'cluster_id']].copy()
        datos_salida.to_csv(ruta_scores, index=False)
        
        ruta_metricas_csv = self.directorio_metricas / 'metrics.csv'
        
        pct_outliers = (self.datos['is_outlier'].sum() / len(self.datos)) * 100
        
        metricas_df = pd.DataFrame([{
            'algoritmo': 'K-Means',
            'params_json': f'{{"k_clusters": {self.metricas_finales["k_optimo"]}}}',
            'n_clusters': self.metricas_finales['k_optimo'],
            'silhouette_score': self.metricas_finales['silhouette'],
            'calinski_harabasz_score': self.metricas_finales['calinski_harabasz'],
            'davies_bouldin_score': self.metricas_finales['davies_bouldin'],
            'pct_anomalias': pct_outliers,
            'p95_minus_p50': np.percentile(self.datos['anomaly_score'], 95) - np.percentile(self.datos['anomaly_score'], 50),
            'mean_score': np.mean(self.datos['anomaly_score']),
            'tiempo_ejecucion_s': self.tiempo_total if self.tiempo_total else 0.0,
            'memoria_max_mb': self.memoria_max if self.memoria_max else 0.0
        }])
        metricas_df.to_csv(ruta_metricas_csv, index=False)
        
        ruta_modelo_pkl = self.directorio_modelos / 'kmeans_model.pkl'
        joblib.dump(self.kmeans_final, ruta_modelo_pkl)
        
        ruta_escalador = self.directorio_modelos / 'scaler.pkl'
        joblib.dump(self.escalador, ruta_escalador)
        
        ruta_modelo_h5 = self.directorio_modelos / 'kmeans_model.h5'
        with h5py.File(ruta_modelo_h5, 'w') as hf:
            hf.create_dataset('cluster_centers', data=self.cluster_centers)
            hf.create_dataset('labels', data=self.labels)
        
        logging.info(f"Métricas guardadas en {ruta_metricas}")
        logging.info(f"Scores guardados en {ruta_scores}")
        logging.info(f"Métricas CSV guardadas en {ruta_metricas_csv}")
        logging.info(f"Modelo guardado como pickle en {ruta_modelo_pkl}")
        logging.info(f"Escalador guardado en {ruta_escalador}")
        logging.info(f"Modelo guardado como h5 en {ruta_modelo_h5}")
    
    def crear_visualizaciones(self) -> None:
        pca_2d = PCA(n_components=PCA_COMPONENTS_2D)
        X_pca_2d = pca_2d.fit_transform(self.X_escalado)
        
        pca_3d = PCA(n_components=PCA_COMPONENTS_3D)
        X_pca_3d = pca_3d.fit_transform(self.X_escalado)
        
        distancias = self.datos['anomaly_score'].values
        
        self._crear_visualizacion_3d_clusters(X_pca_3d, pca_3d)
        
        self._crear_visualizacion_2d_anomalias(X_pca_2d, distancias)
    
    def _crear_visualizacion_2d_anomalias(self, X_pca: np.ndarray, distancias: np.ndarray) -> None:
        plt.figure(figsize=FIGSIZE_2D)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=distancias, 
                            cmap='hot', s=SCATTER_SIZE, alpha=0.7)
        plt.title('Puntuaciones de Anomalías (Distancia al Centroide)', fontsize=14, pad=15)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(scatter, label='Puntuación de Anomalía [0-1]')
        
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        ruta_anomalias_2d = self.directorio_graficas / 'anomalies_pca.png'
        plt.savefig(ruta_anomalias_2d, dpi=200, bbox_inches='tight')
        plt.close("all")
        
        logging.info(f"Visualización de anomalías 2D guardada en {ruta_anomalias_2d}")
    
    def _crear_visualizacion_3d_clusters(self, X_pca_3d: np.ndarray, pca_3d: PCA) -> None:
        X_vis, labels_vis = self.reducir_muestra_para_visualizacion(X_pca_3d, self.labels)
        
        n_clusters = len(np.unique(labels_vis))
        
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], 
                           c=labels_vis, cmap=CMAP_CLUSTERING, s=SCATTER_SIZE, alpha=0.7)
        
        ax.set_title(f'Clustering K-Means (3D con PCA)\n'
                    f'Clusters: {n_clusters} | Muestras: {len(X_vis):,}',
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
        
        legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax.add_artist(legend)
        
        ruta_clusters_3d = self.directorio_graficas / 'clusters_3d_pca.png'
        plt.savefig(ruta_clusters_3d, dpi=200, bbox_inches='tight')
        plt.close("all")
        
        logging.info(f"Visualización de clusters 3D guardada en {ruta_clusters_3d}")
    
    def ejecutar_analisis_completo(self) -> None:
        """
        ============================================================================
        PIPELINE DE CLUSTERING K-MEANS - FLUJO COMPLETO
        ============================================================================
        PASO 1: Carga y validación de datos
        PASO 2: Preprocesamiento y creación de características
        PASO 3: Normalización de datos (MinMaxScaler)
        PASO 4: Optimización de hiperparámetros (Método del Codo + Silhouette)
        PASO 5: Entrenamiento del modelo final con K óptimo
        PASO 6: Cálculo de métricas de clustering (Silhouette, Calinski, Davies-Bouldin)
        PASO 7: Detección de anomalías basada en distancia a centroides
        PASO 8: Guardado de modelos, escaladores y resultados
        PASO 9: Generación de visualizaciones (3D, anomalías)
        ============================================================================
        """
        logging.info("=== INICIANDO PROCESO DE CLUSTERING K-MEANS ===")
        
        self.tiempo_inicio = time.time()
        tracemalloc.start()
        
        try:
            config.aplicar_seeds_reproducibilidad(RANDOM_STATE)
            
            # ====================================================================
            # PASO 1: CARGA Y VALIDACIÓN DE DATOS
            # ====================================================================
            logging.info("\n[PASO 1] Carga y validación de datos...")
            ruta_datos = self.directorio_script.parent.parent / config.RUTA_DATOS_COMPARTIDA
            self.cargar_datos(ruta_datos)
            
            # ====================================================================
            # PASO 2+3: PREPROCESAMIENTO Y NORMALIZACIÓN
            # ====================================================================
            logging.info("\n[PASO 2-3] Preprocesamiento y normalización de datos...")
            self.escalar_datos()
            
            # ====================================================================
            # PASO 4: OPTIMIZACIÓN DE HIPERPARÁMETROS
            # ====================================================================
            logging.info("\n[PASO 4] Optimización de hiperparámetros (K óptimo)...")
            k_optimo, metricas, resultados = self.encontrar_k_optimo()
            
            # ====================================================================
            # PASO 5+6: ENTRENAMIENTO Y MÉTRICAS
            # ====================================================================
            logging.info("\n[PASO 5-6] Entrenamiento del modelo final y cálculo de métricas...")
            self.entrenar_modelo_final(k_optimo, resultados)
            
            # ====================================================================
            # PASO 7: DETECCIÓN DE ANOMALÍAS
            # ====================================================================
            logging.info("\n[PASO 7] Detección de anomalías basada en distancia...")
            self.calcular_puntuaciones_anomalia()
            
            self.tiempo_total = time.time() - self.tiempo_inicio
            memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
            self.memoria_max = memoria_pico / 1024**2
            tracemalloc.stop()
            
            logging.info(f"Tiempo total de ejecucion: {self.tiempo_total:.2f} segundos")
            logging.info(f"Memoria maxima utilizada: {self.memoria_max:.2f} MB")
            
            # ====================================================================
            # PASO 8: GUARDADO DE MODELOS Y RESULTADOS
            # ====================================================================
            logging.info("\n[PASO 8] Guardado de modelos y resultados...")
            self.guardar_resultados()
            
            # ====================================================================
            # PASO 9: GENERACIÓN DE VISUALIZACIONES
            # ====================================================================
            logging.info("\n[PASO 9] Generación de visualizaciones...")
            self.crear_visualizaciones()
            
            logging.info("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
            
        except Exception as e:
            tracemalloc.stop()
            logging.error(f"Error durante el análisis: {str(e)}")
            raise


def main():
    directorio_script = Path(__file__).parent
    
    analizador = KMeansAnalyzer(directorio_script)
    analizador.ejecutar_analisis_completo()


if __name__ == "__main__":
    main()
