"""
Algoritmo de Clustering K-Means
Análisis de datos de acelerómetro para detección de patrones y anomalías
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import h5py
import logging
from joblib import Parallel, delayed
from typing import Tuple, List, Dict, Any
import warnings

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore')

# Constantes de configuración
RANDOM_STATE = 42
K_MIN = 2
K_MAX = 8  # Reducido para optimizar memoria en Google Colab
FIGSIZE_2D = (10, 8)
FIGSIZE_3D = (12, 9)
SCATTER_SIZE = 30
N_JOBS = 2  # Optimizado para Google Colab (limitación de memoria)
PCA_COMPONENTS_2D = 2
PCA_COMPONENTS_3D = 3

# Configuración de características (100% no-supervisado)
USECOLS = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'fecha']
CARACTERISTICAS_BASE = ['acceleration_x', 'acceleration_y', 'acceleration_z']
CARACTERISTICA_MAGNITUD = 'magnitud_aceleracion'

class KMeansAnalyzer:
    """Clase para análisis de clustering K-Means con datos de acelerómetro"""
    
    def __init__(self, directorio_script: str):
        self.directorio_script = directorio_script
        self.directorio_modelos = os.path.join(directorio_script, 'modelos_entrenados_KMeans')
        self.directorio_graficas = os.path.join(directorio_script, 'graficas_KMeans')
        self.directorio_metricas = os.path.join(directorio_script, 'metricas_KMeans')
        
        self._crear_directorios()
        self._configurar_logging()
        
        # Atributos del modelo
        self.datos = None
        self.X = None
        self.X_escalado = None
        self.escalador = None
        self.kmeans_final = None
    
    def _crear_directorios(self) -> None:
        """Crear directorios necesarios para el análisis"""
        for directorio in [self.directorio_modelos, self.directorio_graficas, self.directorio_metricas]:
            if not os.path.exists(directorio):
                os.makedirs(directorio)
    
    def _configurar_logging(self) -> None:
        """Configurar sistema de logging"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Limpiar handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Handler para archivo
        ruta_archivo_log = os.path.join(self.directorio_metricas, 'output.log')
        file_handler = logging.FileHandler(ruta_archivo_log, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    
    def cargar_datos(self, ruta_datos: str) -> None:
        """Cargar y preprocesar los datos (100% no-supervisado)"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(ruta_datos):
                raise FileNotFoundError(f"El archivo no existe en la ruta: {ruta_datos}")
            
            # BLINDADO: Cargar solo las 4 columnas necesarias con tipos optimizados
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
            
            # BLINDADO: Recortar DataFrame a solo las 4 columnas por si el CSV trae más
            self.datos = self.datos[['fecha'] + CARACTERISTICAS_BASE].copy()
            
            # Ordenar por fecha (recomendable)
            self.datos.sort_values('fecha', inplace=True)
            
            logging.info(f"Datos cargados correctamente desde: {ruta_datos}")
            logging.info(f"Forma del dataset: {self.datos.shape}")
            logging.info(f"Columnas cargadas: {list(self.datos.columns)}")
            
            # Mostrar primeras filas para verificación
            logging.info("Primeras 3 filas del dataset:")
            for i, row in self.datos.head(3).iterrows():
                logging.info(f"  Fila {i}: {dict(row)}")
            
            # Verificar tipos de datos
            logging.info("Tipos de datos por columna:")
            for col in self.datos.columns:
                logging.info(f"  {col}: {self.datos[col].dtype}")
            
            # Manejo de valores faltantes
            filas_originales = len(self.datos)
            
            # Mostrar estadísticas de valores faltantes
            valores_faltantes = self.datos[CARACTERISTICAS_BASE].isnull().sum()
            if valores_faltantes.sum() > 0:
                logging.info("Valores faltantes por columna:")
                for col, count in valores_faltantes.items():
                    if count > 0:
                        logging.info(f"  {col}: {count} valores faltantes")
            
            # Eliminar filas con valores faltantes en las columnas de características
            self.datos = self.datos.dropna(subset=CARACTERISTICAS_BASE)
            filas_eliminadas = filas_originales - len(self.datos)
            
            if filas_eliminadas > 0:
                logging.info(f"Se eliminaron {filas_eliminadas} filas con valores faltantes.")
            else:
                logging.info("No se encontraron valores faltantes.")
            
            # Verificar que quedan datos suficientes
            if len(self.datos) < 10:
                raise ValueError(f"Datos insuficientes después de la limpieza: solo {len(self.datos)} filas")
            
            logging.info(f"Dataset final: {len(self.datos)} filas")
            
            # Mostrar estadísticas básicas
            logging.info("Estadísticas básicas de las características:")
            stats = self.datos[CARACTERISTICAS_BASE].describe()
            logging.info(f"\n{stats}")
            
            # Ingeniería de características
            self._crear_caracteristicas()
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def _crear_caracteristicas(self) -> None:
        """Crear características adicionales"""
        # Magnitud de aceleración
        self.datos[CARACTERISTICA_MAGNITUD] = np.sqrt(
            self.datos['acceleration_x']**2 +
            self.datos['acceleration_y']**2 +
            self.datos['acceleration_z']**2
        )
        logging.info(f"Característica '{CARACTERISTICA_MAGNITUD}' añadida.")
        
        # Seleccionar características finales
        caracteristicas = CARACTERISTICAS_BASE + [CARACTERISTICA_MAGNITUD]
        self.X = self.datos[caracteristicas].values
    
    def escalar_datos(self) -> None:
        """Escalar los datos utilizando StandardScaler"""
        self.escalador = StandardScaler()
        self.X_escalado = self.escalador.fit_transform(self.X)
        logging.info("Datos escalados correctamente.")
    
    def reducir_muestra_para_optimizacion(self, max_muestras: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce el tamaño de la muestra para optimización de parámetros si es muy grande.
        
        Args:
            max_muestras: Número máximo de muestras para optimización.
            
        Returns:
            Tupla con (muestra_reducida, indices_seleccionados).
        """
        if len(self.X_escalado) <= max_muestras:
            return self.X_escalado, np.arange(len(self.X_escalado))
        
        # Muestreo aleatorio estratificado
        np.random.seed(RANDOM_STATE)  # Para reproducibilidad
        indices_seleccionados = np.random.choice(len(self.X_escalado), max_muestras, replace=False)
        indices_seleccionados = np.sort(indices_seleccionados)
        
        X_reducido = self.X_escalado[indices_seleccionados]
        
        logging.info(f"Dataset reducido para optimización: {len(self.X_escalado)} -> {len(X_reducido)} muestras")
        return X_reducido, indices_seleccionados
    
    def _evaluar_k(self, k: int) -> Tuple[int, float, float, float, float, np.ndarray, np.ndarray]:
        """Evaluar un valor específico de K para clustering"""
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(self.X_escalado)
        labels = kmeans.labels_
        
        # Calcular métricas
        inertia_k = kmeans.inertia_
        silhouette_k = silhouette_score(self.X_escalado, labels)
        calinski_k = calinski_harabasz_score(self.X_escalado, labels)
        davies_k = davies_bouldin_score(self.X_escalado, labels)
        
        logging.info(f"K={k}, Inercia={inertia_k:.2f}, Silhouette={silhouette_k:.4f}, "
                    f"Calinski-Harabasz={calinski_k:.4f}, Davies-Bouldin={davies_k:.4f}")
        
        return (k, inertia_k, silhouette_k, calinski_k, davies_k, labels, kmeans.cluster_centers_)
    
    def encontrar_k_optimo(self) -> Tuple[int, Dict[str, List[float]]]:
        """Encontrar el número óptimo de clusters"""
        # Reducir muestra para optimización si es muy grande
        X_para_optimizacion, indices_muestra = self.reducir_muestra_para_optimizacion()
        
        # Guardar temporalmente los datos originales
        X_escalado_original = self.X_escalado.copy()
        self.X_escalado = X_para_optimizacion
        
        K_range = range(K_MIN, K_MAX)
        
        # Paralelizar el cálculo
        resultados = Parallel(n_jobs=N_JOBS)(
            delayed(self._evaluar_k)(k) for k in K_range
        )
        
        # Restaurar datos originales
        self.X_escalado = X_escalado_original
        
        # Organizar resultados
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
        
        # Seleccionar K óptimo basado en Silhouette Score
        k_optimo = K_range[metricas['silhouette'].index(max(metricas['silhouette']))]
        logging.info(f"Número óptimo de clusters seleccionado: K={k_optimo}")
        
        return k_optimo, metricas, resultados
    
    def _crear_grafico_metrica(self, k_range: range, valores: List[float], 
                              titulo: str, ylabel: str, nombre_archivo: str) -> None:
        """Crear gráfico para una métrica específica"""
        plt.figure(figsize=FIGSIZE_2D)
        plt.plot(k_range, valores, 'bx-', linewidth=2, markersize=8)
        plt.xlabel('Número de Clusters K')
        plt.ylabel(ylabel)
        plt.title(titulo)
        plt.grid(True, alpha=0.3)
        
        ruta_grafico = os.path.join(self.directorio_graficas, nombre_archivo)
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        logging.info(f"Gráfico guardado en {ruta_grafico}")
    
    def crear_graficos_evaluacion(self, metricas: Dict[str, List[float]]) -> None:
        """Crear gráficos de evaluación de métricas"""
        K_range = range(K_MIN, K_MAX)
        
        # Configuraciones de gráficos
        configuraciones = [
            (metricas['inercia'], 'Método del Codo para determinar el número óptimo de clusters',
             'Inercia', 'elbow_method.png'),
            (metricas['silhouette'], 'Coeficiente Silhouette para diferentes valores de K',
             'Coeficiente Silhouette', 'silhouette_scores.png'),
            (metricas['calinski'], 'Índice Calinski-Harabasz para diferentes valores de K',
             'Índice Calinski-Harabasz', 'calinski_harabasz_scores.png'),
            (metricas['davies'], 'Índice Davies-Bouldin para diferentes valores de K',
             'Índice Davies-Bouldin', 'davies_bouldin_scores.png')
        ]
        
        for valores, titulo, ylabel, nombre_archivo in configuraciones:
            self._crear_grafico_metrica(K_range, valores, titulo, ylabel, nombre_archivo)
    
    def entrenar_modelo_final(self, k_optimo: int, resultados: List[Tuple]) -> None:
        """Entrenar el modelo final con K óptimo"""
        # Obtener resultados para K óptimo
        resultado_optimo = next(res for res in resultados if res[0] == k_optimo)
        
        # Entrenar modelo final
        self.kmeans_final = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init='auto')
        self.kmeans_final.fit(self.X_escalado)
        
        # CORRIGIDO: Usar labels del modelo final entrenado en el dataset completo
        self.labels = self.kmeans_final.labels_
        self.cluster_centers = self.kmeans_final.cluster_centers_
        # Recalcular métricas para el modelo final en dataset completo
        self.metricas_finales = {
            'k_optimo': k_optimo,
            'inertia': self.kmeans_final.inertia_,
            'silhouette': silhouette_score(self.X_escalado, self.labels),
            'calinski_harabasz': calinski_harabasz_score(self.X_escalado, self.labels),
            'davies_bouldin': davies_bouldin_score(self.X_escalado, self.labels)
        }
        
        logging.info("Modelo K-Means entrenado con el número óptimo de clusters.")
        self._mostrar_metricas_finales()
    
    def _mostrar_metricas_finales(self) -> None:
        """Mostrar métricas del modelo final"""
        logging.info(f"Silhouette Score: {self.metricas_finales['silhouette']:.4f}")
        logging.info(f"Calinski-Harabasz Score: {self.metricas_finales['calinski_harabasz']:.4f}")
        logging.info(f"Davies-Bouldin Score: {self.metricas_finales['davies_bouldin']:.4f}")
        logging.info(f"Inercia (SSE): {self.metricas_finales['inertia']:.2f}")
    
    def calcular_puntuaciones_anomalia(self) -> None:
        """Calcular puntuaciones de anomalías"""
        # Distancia al centroide del cluster asignado
        distancias = np.linalg.norm(
            self.X_escalado - self.cluster_centers[self.labels], axis=1
        )
        # Estandarizar nombres de columnas
        self.datos['anomaly_score'] = distancias  # Score estándar
        self.datos['is_outlier'] = 0  # K-Means no detecta outliers binarios
        self.datos['cluster_id'] = self.labels  # ID del cluster asignado
        logging.info("Puntuación de anomalías calculada para cada punto de datos.")
    

    
    def guardar_resultados(self) -> None:
        """Guardar todos los resultados del análisis"""
        # Guardar métricas
        ruta_metricas = os.path.join(self.directorio_metricas, 'metrics.txt')
        with open(ruta_metricas, 'w', encoding='utf-8') as f:
            f.write(f"Número de clusters: {self.metricas_finales['k_optimo']}\n")
            f.write(f"Silhouette Score: {self.metricas_finales['silhouette']:.4f}\n")
            f.write(f"Calinski-Harabasz Score: {self.metricas_finales['calinski_harabasz']:.4f}\n")
            f.write(f"Davies-Bouldin Score: {self.metricas_finales['davies_bouldin']:.4f}\n")
            f.write(f"Inercia (SSE): {self.metricas_finales['inertia']:.2f}\n")
        
        # Guardar puntuaciones de anomalías con fecha, XYZ y scores (estandarizado)
        ruta_scores = os.path.join(self.directorio_metricas, 'anomaly_scores.csv')
        datos_salida = self.datos[['fecha'] + CARACTERISTICAS_BASE + ['anomaly_score', 'cluster_id']].copy()
        datos_salida.to_csv(ruta_scores, index=False)
        
        # Guardar métricas en CSV estándar (100% no-supervisado)
        ruta_metricas_csv = os.path.join(self.directorio_metricas, 'metrics.csv')
        metricas_df = pd.DataFrame([{
            'algoritmo': 'K-Means',
            'params_json': f'{{"k_clusters": {self.metricas_finales["k_optimo"]}}}',
            'n_clusters': self.metricas_finales['k_optimo'],
            'silhouette_score': self.metricas_finales['silhouette'],
            'calinski_harabasz_score': self.metricas_finales['calinski_harabasz'],
            'davies_bouldin_score': self.metricas_finales['davies_bouldin'],
            'pct_anomalias': 0.0,  # K-Means no detecta anomalías binarias
            'p95_minus_p50': np.percentile(self.datos['anomaly_score'], 95) - np.percentile(self.datos['anomaly_score'], 50),
            'mean_score': np.mean(self.datos['anomaly_score'])
        }])
        metricas_df.to_csv(ruta_metricas_csv, index=False)
        
        # Guardar modelo como pickle
        ruta_modelo_pkl = os.path.join(self.directorio_modelos, 'kmeans_model.pkl')
        joblib.dump(self.kmeans_final, ruta_modelo_pkl)
        
        # Guardar escalador para inferencia futura
        ruta_escalador = os.path.join(self.directorio_modelos, 'scaler.pkl')
        joblib.dump(self.escalador, ruta_escalador)
        
        # Guardar modelo como h5
        ruta_modelo_h5 = os.path.join(self.directorio_modelos, 'kmeans_model.h5')
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
        """Crear todas las visualizaciones"""
        # PCA para visualización 2D
        pca_2d = PCA(n_components=PCA_COMPONENTS_2D)
        X_pca_2d = pca_2d.fit_transform(self.X_escalado)
        
        # PCA para visualización 3D
        pca_3d = PCA(n_components=PCA_COMPONENTS_3D)
        X_pca_3d = pca_3d.fit_transform(self.X_escalado)
        
        # Distancias para anomalías
        distancias = self.datos['anomaly_score'].values
        
        # Crear visualizaciones
        self._crear_visualizacion_2d_clusters(X_pca_2d)
        self._crear_visualizacion_2d_anomalias(X_pca_2d, distancias)
        self._crear_visualizacion_3d_clusters(X_pca_3d)
    
    def _crear_visualizacion_2d_clusters(self, X_pca: np.ndarray) -> None:
        """Crear visualización 2D de clusters"""
        plt.figure(figsize=FIGSIZE_2D)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, 
                            cmap='viridis', s=SCATTER_SIZE, alpha=0.7)
        plt.title('Clustering K-Means (Visualización con PCA)', fontsize=14)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend(*scatter.legend_elements(), title="Clusters", loc='best')
        plt.grid(True, alpha=0.3)
        
        ruta_clusters_2d = os.path.join(self.directorio_graficas, 'clusters_pca.png')
        plt.savefig(ruta_clusters_2d, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        logging.info(f"Visualización de clusters 2D guardada en {ruta_clusters_2d}")
    
    def _crear_visualizacion_2d_anomalias(self, X_pca: np.ndarray, distancias: np.ndarray) -> None:
        """Crear visualización 2D de anomalías"""
        plt.figure(figsize=FIGSIZE_2D)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=distancias, 
                            cmap='hot', s=SCATTER_SIZE, alpha=0.7)
        plt.title('Puntuaciones de Anomalías (Distancia al Centroide)', fontsize=14)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(scatter, label='Puntuación de Anomalía')
        plt.grid(True, alpha=0.3)
        
        ruta_anomalias_2d = os.path.join(self.directorio_graficas, 'anomalies_pca.png')
        plt.savefig(ruta_anomalias_2d, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        logging.info(f"Visualización de anomalías 2D guardada en {ruta_anomalias_2d}")
    
    def _crear_visualizacion_3d_clusters(self, X_pca_3d: np.ndarray) -> None:
        """Crear visualización 3D de clusters"""
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                           c=self.labels, cmap='viridis', s=SCATTER_SIZE, alpha=0.7)
        
        ax.set_title('Clustering K-Means en 3D (Visualización con PCA)', fontsize=14)
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        
        legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc='best')
        ax.add_artist(legend)
        
        ruta_clusters_3d = os.path.join(self.directorio_graficas, 'clusters_3d.png')
        plt.savefig(ruta_clusters_3d, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        logging.info(f"Visualización de clusters 3D guardada en {ruta_clusters_3d}")
    
    def ejecutar_analisis_completo(self) -> None:
        """Ejecutar el análisis completo de K-Means"""
        logging.info("Iniciando proceso de clustering con K-Means...")
        
        try:
            # Cargar y preprocesar datos
            ruta_datos = os.path.join(self.directorio_script, 'data.csv')
            self.cargar_datos(ruta_datos)
            self.escalar_datos()
            
            # Encontrar K óptimo
            k_optimo, metricas, resultados = self.encontrar_k_optimo()
            
            # Crear gráficos de evaluación
            self.crear_graficos_evaluacion(metricas)
            
            # Entrenar modelo final
            self.entrenar_modelo_final(k_optimo, resultados)
            
            # Calcular anomalías
            self.calcular_puntuaciones_anomalia()
            
            # Guardar resultados
            self.guardar_resultados()
            
            # Crear visualizaciones
            self.crear_visualizaciones()
            
            logging.info("Proceso completado exitosamente.")
            
        except Exception as e:
            logging.error(f"Error durante el análisis: {str(e)}")
            raise


def main():
    """Función principal para ejecutar el análisis"""
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # Crear analizador y ejecutar
    analizador = KMeansAnalyzer(directorio_script)
    analizador.ejecutar_analisis_completo()


if __name__ == "__main__":
    main()
