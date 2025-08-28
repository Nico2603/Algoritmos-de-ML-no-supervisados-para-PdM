"""
Análisis de Estabilidad con Bootstrap para Algoritmos de ML No Supervisado

Este módulo implementa análisis de estabilidad usando técnicas de bootstrap
para evaluar la robustez y consistencia de los algoritmos:

1. Bootstrap de muestras para evaluar estabilidad
2. Análisis de sensibilidad a parámetros
3. Métricas de estabilidad (ARI, estabilidad de clusters)
4. Evaluación de robustez temporal

Autor: Sistema PdM
Versión: 1.0
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.cblof import CBLOF
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# Constantes
N_BOOTSTRAP = 50  # Número de muestras bootstrap (reducido para eficiencia)
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.8  # Fracción de datos para cada bootstrap

class AnalizadorEstabilidad:
    """Analizador de estabilidad con bootstrap para algoritmos de ML no supervisado."""
    
    def __init__(self, directorio_proyecto: str):
        """
        Inicializa el analizador.
        
        Args:
            directorio_proyecto: Directorio raíz del proyecto
        """
        self.directorio_proyecto = Path(directorio_proyecto)
        self.datos_originales = None
        self.X_escalado = None
        self.escalador = None
        
        # Configurar directorios
        self.directorio_estabilidad = self.directorio_proyecto / 'Analisis_Estabilidad'
        self.directorio_graficas = self.directorio_estabilidad / 'graficas'
        self.directorio_reportes = self.directorio_estabilidad / 'reportes'
        
        self._crear_directorios()
        self._configurar_logging()
        
        # Resultados
        self.resultados_estabilidad = {}
    
    def _crear_directorios(self) -> None:
        """Crea los directorios necesarios."""
        for directorio in [self.directorio_estabilidad, self.directorio_graficas, self.directorio_reportes]:
            directorio.mkdir(exist_ok=True)
    
    def _configurar_logging(self) -> None:
        """Configura el sistema de logging."""
        log_file = self.directorio_reportes / 'estabilidad.log'
        
        # Limpiar handlers existentes
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
    
    def cargar_datos(self, ruta_datos: str = None) -> None:
        """
        Carga los datos para análisis de estabilidad.
        
        Args:
            ruta_datos: Ruta específica al archivo de datos
        """
        if ruta_datos is None:
            # Buscar data.csv en los directorios de algoritmos
            posibles_rutas = [
                self.directorio_proyecto / "1. Clustering/K-means/data.csv",
                self.directorio_proyecto / "1. Clustering/DBSCAN/data.csv",
                self.directorio_proyecto / "2. Detección de Anomalías/Isolation Forest/data.csv",
                self.directorio_proyecto / "2. Detección de Anomalías/CBLOF (Cluster-Based Local Outlier Factor)/data.csv"
            ]
            
            for ruta in posibles_rutas:
                if ruta.exists():
                    ruta_datos = ruta
                    break
            
            if ruta_datos is None:
                raise FileNotFoundError("No se encontró archivo data.csv en ningún directorio de algoritmos")
        
        logging.info(f"Cargando datos desde: {ruta_datos}")
        
        try:
            self.datos_originales = pd.read_csv(ruta_datos)
            logging.info(f"Datos cargados: {len(self.datos_originales)} registros")
            
            # Preprocesar datos
            self._preprocesar_datos()
            
        except Exception as e:
            logging.error(f"Error cargando datos: {e}")
            raise
    
    def _preprocesar_datos(self) -> None:
        """Preprocesa los datos para análisis."""
        # Eliminar valores faltantes
        datos_limpios = self.datos_originales.dropna()
        
        # Verificar características base
        caracteristicas_base = ['acceleration_x', 'acceleration_y', 'acceleration_z']
        for col in caracteristicas_base:
            if col not in datos_limpios.columns:
                raise ValueError(f"Característica requerida no encontrada: {col}")
        
        # Crear característica de magnitud
        datos_limpios = datos_limpios.copy()
        datos_limpios['magnitud_aceleracion'] = np.sqrt(
            datos_limpios['acceleration_x']**2 +
            datos_limpios['acceleration_y']**2 +
            datos_limpios['acceleration_z']**2
        )
        
        # Seleccionar características
        caracteristicas = caracteristicas_base + ['magnitud_aceleracion']
        X = datos_limpios[caracteristicas].values
        
        # Escalar datos
        self.escalador = StandardScaler()
        self.X_escalado = self.escalador.fit_transform(X)
        
        logging.info(f"Datos preprocesados: {self.X_escalado.shape}")
    
    def generar_muestra_bootstrap(self, indice_bootstrap: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera una muestra bootstrap.
        
        Args:
            indice_bootstrap: Índice de la muestra bootstrap
            
        Returns:
            Tupla con (datos_bootstrap, indices_seleccionados)
        """
        np.random.seed(RANDOM_STATE + indice_bootstrap)
        n_muestras = int(len(self.X_escalado) * SAMPLE_FRACTION)
        indices = np.random.choice(len(self.X_escalado), n_muestras, replace=True)
        
        return self.X_escalado[indices], indices
    
    def evaluar_estabilidad_kmeans(self, k_optimo: int = None) -> Dict[str, Any]:
        """
        Evalúa la estabilidad de K-Means usando bootstrap.
        
        Args:
            k_optimo: Número óptimo de clusters (se detecta automáticamente si no se proporciona)
            
        Returns:
            Diccionario con métricas de estabilidad
        """
        logging.info("Evaluando estabilidad de K-Means...")
        
        if k_optimo is None:
            # Detectar K óptimo usando silhouette score
            k_optimo = self._detectar_k_optimo()
        
        def evaluar_bootstrap_kmeans(i):
            X_boot, indices = self.generar_muestra_bootstrap(i)
            
            # Entrenar modelo
            kmeans = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init='auto')
            labels = kmeans.fit_predict(X_boot)
            
            # Calcular métricas
            silhouette = silhouette_score(X_boot, labels)
            calinski = calinski_harabasz_score(X_boot, labels)
            
            return {
                'bootstrap_id': i,
                'silhouette': silhouette,
                'calinski': calinski,
                'labels': labels,
                'indices': indices
            }
        
        # Ejecutar bootstrap en paralelo
        resultados = Parallel(n_jobs=2)(
            delayed(evaluar_bootstrap_kmeans)(i) for i in range(N_BOOTSTRAP)
        )
        
        # Calcular métricas de estabilidad
        estabilidad = self._calcular_metricas_estabilidad_clustering(resultados, 'K-Means')
        estabilidad['k_optimo'] = k_optimo
        
        self.resultados_estabilidad['K-Means'] = estabilidad
        logging.info(f"✓ K-Means: Estabilidad promedio ARI = {estabilidad['ari_promedio']:.4f}")
        
        return estabilidad
    
    def evaluar_estabilidad_dbscan(self, eps: float = None, min_samples: int = None) -> Dict[str, Any]:
        """
        Evalúa la estabilidad de DBSCAN usando bootstrap.
        
        Args:
            eps: Parámetro epsilon (se detecta automáticamente si no se proporciona)
            min_samples: Parámetro min_samples (se detecta automáticamente si no se proporciona)
            
        Returns:
            Diccionario con métricas de estabilidad
        """
        logging.info("Evaluando estabilidad de DBSCAN...")
        
        if eps is None or min_samples is None:
            eps, min_samples = self._detectar_parametros_dbscan()
        
        def evaluar_bootstrap_dbscan(i):
            X_boot, indices = self.generar_muestra_bootstrap(i)
            
            # Entrenar modelo
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_boot)
            
            # Verificar que hay clusters válidos
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                return None
            
            # Calcular métricas excluyendo ruido
            mask_no_noise = labels != -1
            if np.sum(mask_no_noise) < 10:
                return None
            
            X_no_noise = X_boot[mask_no_noise]
            labels_no_noise = labels[mask_no_noise]
            
            silhouette = silhouette_score(X_no_noise, labels_no_noise)
            calinski = calinski_harabasz_score(X_no_noise, labels_no_noise)
            
            return {
                'bootstrap_id': i,
                'silhouette': silhouette,
                'calinski': calinski,
                'labels': labels,
                'indices': indices,
                'n_clusters': n_clusters,
                'n_noise': np.sum(labels == -1)
            }
        
        # Ejecutar bootstrap
        resultados = Parallel(n_jobs=2)(
            delayed(evaluar_bootstrap_dbscan)(i) for i in range(N_BOOTSTRAP)
        )
        
        # Filtrar resultados válidos
        resultados_validos = [r for r in resultados if r is not None]
        
        if len(resultados_validos) < 10:
            logging.warning("Pocos resultados válidos para DBSCAN, estabilidad puede no ser confiable")
        
        # Calcular métricas de estabilidad
        estabilidad = self._calcular_metricas_estabilidad_clustering(resultados_validos, 'DBSCAN')
        estabilidad['eps'] = eps
        estabilidad['min_samples'] = min_samples
        estabilidad['resultados_validos'] = len(resultados_validos)
        
        self.resultados_estabilidad['DBSCAN'] = estabilidad
        logging.info(f"✓ DBSCAN: Estabilidad promedio ARI = {estabilidad['ari_promedio']:.4f}")
        
        return estabilidad
    
    def evaluar_estabilidad_isolation_forest(self) -> Dict[str, Any]:
        """
        Evalúa la estabilidad de Isolation Forest usando bootstrap.
        
        Returns:
            Diccionario con métricas de estabilidad
        """
        logging.info("Evaluando estabilidad de Isolation Forest...")
        
        def evaluar_bootstrap_iforest(i):
            X_boot, indices = self.generar_muestra_bootstrap(i)
            
            # Entrenar modelo
            iforest = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=RANDOM_STATE + i
            )
            iforest.fit(X_boot)
            
            # Obtener scores
            scores = -iforest.decision_function(X_boot)  # Invertir para consistencia
            labels = iforest.predict(X_boot)
            labels = np.where(labels == 1, 0, 1)  # 0: normal, 1: anomalía
            
            return {
                'bootstrap_id': i,
                'scores': scores,
                'labels': labels,
                'indices': indices,
                'n_anomalias': np.sum(labels),
                'score_promedio': np.mean(scores)
            }
        
        # Ejecutar bootstrap
        resultados = Parallel(n_jobs=2)(
            delayed(evaluar_bootstrap_iforest)(i) for i in range(N_BOOTSTRAP)
        )
        
        # Calcular métricas de estabilidad
        estabilidad = self._calcular_metricas_estabilidad_anomalias(resultados, 'Isolation Forest')
        
        self.resultados_estabilidad['Isolation Forest'] = estabilidad
        logging.info(f"✓ Isolation Forest: Estabilidad de detección = {estabilidad['estabilidad_deteccion']:.4f}")
        
        return estabilidad
    
    def evaluar_estabilidad_cblof(self) -> Dict[str, Any]:
        """
        Evalúa la estabilidad de CBLOF usando bootstrap.
        
        Returns:
            Diccionario con métricas de estabilidad
        """
        logging.info("Evaluando estabilidad de CBLOF...")
        
        def evaluar_bootstrap_cblof(i):
            X_boot, indices = self.generar_muestra_bootstrap(i)
            
            # Entrenar modelo
            cblof = CBLOF(
                n_clusters=8,
                contamination=0.1,
                random_state=RANDOM_STATE + i
            )
            cblof.fit(X_boot)
            
            # Obtener scores
            scores = cblof.decision_scores_
            labels = cblof.labels_  # 0: normal, 1: anomalía
            
            return {
                'bootstrap_id': i,
                'scores': scores,
                'labels': labels,
                'indices': indices,
                'n_anomalias': np.sum(labels),
                'score_promedio': np.mean(scores)
            }
        
        # Ejecutar bootstrap
        resultados = Parallel(n_jobs=2)(
            delayed(evaluar_bootstrap_cblof)(i) for i in range(N_BOOTSTRAP)
        )
        
        # Calcular métricas de estabilidad
        estabilidad = self._calcular_metricas_estabilidad_anomalias(resultados, 'CBLOF')
        
        self.resultados_estabilidad['CBLOF'] = estabilidad
        logging.info(f"✓ CBLOF: Estabilidad de detección = {estabilidad['estabilidad_deteccion']:.4f}")
        
        return estabilidad
    
    def _detectar_k_optimo(self) -> int:
        """Detecta el número óptimo de clusters para K-Means."""
        k_range = range(2, 9)
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
            labels = kmeans.fit_predict(self.X_escalado)
            score = silhouette_score(self.X_escalado, labels)
            silhouette_scores.append(score)
        
        k_optimo = k_range[np.argmax(silhouette_scores)]
        logging.info(f"K óptimo detectado: {k_optimo}")
        return k_optimo
    
    def _detectar_parametros_dbscan(self) -> Tuple[float, int]:
        """Detecta parámetros óptimos para DBSCAN."""
        from sklearn.neighbors import NearestNeighbors
        
        # Estimar eps usando k-distance
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.X_escalado)
        distances, _ = nbrs.kneighbors(self.X_escalado)
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Usar percentil 90 como estimación de eps
        eps = np.percentile(distances, 90)
        min_samples = k
        
        logging.info(f"Parámetros DBSCAN detectados: eps={eps:.3f}, min_samples={min_samples}")
        return eps, min_samples
    
    def _calcular_metricas_estabilidad_clustering(self, resultados: List[Dict], algoritmo: str) -> Dict[str, Any]:
        """Calcula métricas de estabilidad para algoritmos de clustering."""
        # Extraer métricas
        silhouettes = [r['silhouette'] for r in resultados]
        calinskis = [r['calinski'] for r in resultados]
        
        # Calcular ARI entre pares de muestras bootstrap
        aris = []
        labels_list = [r['labels'] for r in resultados]
        
        for i in range(len(labels_list)):
            for j in range(i+1, min(i+10, len(labels_list))):  # Comparar con las siguientes 10 muestras
                try:
                    # Encontrar indices comunes
                    indices_i = set(resultados[i]['indices'])
                    indices_j = set(resultados[j]['indices'])
                    indices_comunes = list(indices_i.intersection(indices_j))
                    
                    if len(indices_comunes) > 10:
                        # Mapear a posiciones en cada muestra
                        pos_i = [list(resultados[i]['indices']).index(idx) for idx in indices_comunes]
                        pos_j = [list(resultados[j]['indices']).index(idx) for idx in indices_comunes]
                        
                        labels_i_common = labels_list[i][pos_i]
                        labels_j_common = labels_list[j][pos_j]
                        
                        ari = adjusted_rand_score(labels_i_common, labels_j_common)
                        aris.append(ari)
                except:
                    continue
        
        return {
            'silhouette_promedio': np.mean(silhouettes),
            'silhouette_std': np.std(silhouettes),
            'calinski_promedio': np.mean(calinskis),
            'calinski_std': np.std(calinskis),
            'ari_promedio': np.mean(aris) if aris else 0.0,
            'ari_std': np.std(aris) if aris else 0.0,
            'n_comparaciones_ari': len(aris),
            'estabilidad_silhouette': 1 - (np.std(silhouettes) / np.mean(silhouettes)) if np.mean(silhouettes) > 0 else 0
        }
    
    def _calcular_metricas_estabilidad_anomalias(self, resultados: List[Dict], algoritmo: str) -> Dict[str, Any]:
        """Calcula métricas de estabilidad para algoritmos de detección de anomalías."""
        # Extraer métricas
        n_anomalias = [r['n_anomalias'] for r in resultados]
        score_promedios = [r['score_promedio'] for r in resultados]
        
        # Calcular estabilidad de detección (consistencia en el número de anomalías)
        cv_anomalias = np.std(n_anomalias) / np.mean(n_anomalias) if np.mean(n_anomalias) > 0 else 0
        estabilidad_deteccion = 1 - cv_anomalias
        
        # Estabilidad de scores
        cv_scores = np.std(score_promedios) / np.mean(score_promedios) if np.mean(score_promedios) > 0 else 0
        estabilidad_scores = 1 - cv_scores
        
        return {
            'n_anomalias_promedio': np.mean(n_anomalias),
            'n_anomalias_std': np.std(n_anomalias),
            'score_promedio': np.mean(score_promedios),
            'score_std': np.std(score_promedios),
            'estabilidad_deteccion': max(0, estabilidad_deteccion),
            'estabilidad_scores': max(0, estabilidad_scores),
            'cv_anomalias': cv_anomalias,
            'cv_scores': cv_scores
        }
    
    def generar_visualizaciones_estabilidad(self) -> None:
        """Genera visualizaciones de estabilidad."""
        logging.info("Generando visualizaciones de estabilidad...")
        
        # 1. Gráfico de barras de estabilidad general
        self._plot_estabilidad_general()
        
        # 2. Distribuciones de métricas bootstrap
        self._plot_distribuciones_bootstrap()
        
        # 3. Evolución de estabilidad
        self._plot_evolucion_estabilidad()
        
        logging.info("✓ Visualizaciones de estabilidad generadas")
    
    def _plot_estabilidad_general(self) -> None:
        """Gráfico de barras de estabilidad general."""
        algoritmos = list(self.resultados_estabilidad.keys())
        
        # Extraer métricas de estabilidad principales
        estabilidades = []
        for algoritmo in algoritmos:
            if algoritmo in ['K-Means', 'DBSCAN']:
                estabilidad = self.resultados_estabilidad[algoritmo]['ari_promedio']
            else:  # Algoritmos de anomalías
                estabilidad = self.resultados_estabilidad[algoritmo]['estabilidad_deteccion']
            estabilidades.append(estabilidad)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(algoritmos, estabilidades, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        
        ax.set_title('Estabilidad de Algoritmos (Bootstrap)', fontsize=14)
        ax.set_ylabel('Índice de Estabilidad')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, estabilidades):
            height = bar.get_height()
            ax.annotate(f'{valor:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.directorio_graficas / 'estabilidad_general.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribuciones_bootstrap(self) -> None:
        """Gráficos de distribuciones de métricas bootstrap."""
        # Este es un placeholder - implementaría gráficos detallados de distribuciones
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Distribuciones Bootstrap\n(Por implementar)', 
                ha='center', va='center', fontsize=16)
        ax.set_title('Distribuciones de Métricas Bootstrap')
        ax.axis('off')
        plt.savefig(self.directorio_graficas / 'distribuciones_bootstrap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_evolucion_estabilidad(self) -> None:
        """Gráfico de evolución de estabilidad."""
        # Placeholder para análisis de evolución temporal
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Evolución de Estabilidad\n(Por implementar)', 
                ha='center', va='center', fontsize=16)
        ax.set_title('Evolución de Estabilidad en el Tiempo')
        ax.axis('off')
        plt.savefig(self.directorio_graficas / 'evolucion_estabilidad.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generar_reporte_estabilidad(self) -> None:
        """Genera el reporte de estabilidad."""
        logging.info("Generando reporte de estabilidad...")
        
        ruta_reporte = self.directorio_reportes / 'reporte_estabilidad.md'
        
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Estabilidad - Algoritmos de ML No Supervisado\n\n")
            
            f.write("## Resumen Ejecutivo\n\n")
            f.write(f"Análisis de estabilidad basado en {N_BOOTSTRAP} muestras bootstrap.\n")
            f.write(f"Fracción de muestra por bootstrap: {SAMPLE_FRACTION:.0%}\n\n")
            
            # Resultados por algoritmo
            for algoritmo, resultados in self.resultados_estabilidad.items():
                f.write(f"### {algoritmo}\n\n")
                
                if algoritmo in ['K-Means', 'DBSCAN']:
                    f.write(f"- **ARI Promedio**: {resultados['ari_promedio']:.4f} ± {resultados['ari_std']:.4f}\n")
                    f.write(f"- **Silhouette Promedio**: {resultados['silhouette_promedio']:.4f} ± {resultados['silhouette_std']:.4f}\n")
                    f.write(f"- **Estabilidad Silhouette**: {resultados['estabilidad_silhouette']:.4f}\n")
                else:
                    f.write(f"- **Estabilidad de Detección**: {resultados['estabilidad_deteccion']:.4f}\n")
                    f.write(f"- **Estabilidad de Scores**: {resultados['estabilidad_scores']:.4f}\n")
                    f.write(f"- **Anomalías Promedio**: {resultados['n_anomalias_promedio']:.1f} ± {resultados['n_anomalias_std']:.1f}\n")
                
                f.write("\n")
            
            # Recomendaciones
            f.write("## Recomendaciones\n\n")
            
            # Encontrar algoritmo más estable
            max_estabilidad = 0
            algoritmo_mas_estable = ""
            
            for algoritmo, resultados in self.resultados_estabilidad.items():
                if algoritmo in ['K-Means', 'DBSCAN']:
                    estabilidad = resultados['ari_promedio']
                else:
                    estabilidad = resultados['estabilidad_deteccion']
                
                if estabilidad > max_estabilidad:
                    max_estabilidad = estabilidad
                    algoritmo_mas_estable = algoritmo
            
            f.write(f"- **Algoritmo más estable**: {algoritmo_mas_estable} (Estabilidad: {max_estabilidad:.4f})\n")
            f.write("- **Criterios de estabilidad**: ARI > 0.7 para clustering, Estabilidad > 0.8 para anomalías\n")
            f.write("- **Recomendación**: Usar algoritmos con alta estabilidad para aplicaciones críticas\n\n")
        
        logging.info(f"✓ Reporte de estabilidad guardado en {ruta_reporte}")
    
    def ejecutar_analisis_completo(self) -> None:
        """Ejecuta el análisis completo de estabilidad."""
        logging.info("=== INICIANDO ANÁLISIS DE ESTABILIDAD ===")
        
        try:
            # 1. Cargar datos
            self.cargar_datos()
            
            # 2. Evaluar estabilidad de cada algoritmo
            self.evaluar_estabilidad_kmeans()
            self.evaluar_estabilidad_dbscan()
            self.evaluar_estabilidad_isolation_forest()
            self.evaluar_estabilidad_cblof()
            
            # 3. Generar visualizaciones
            self.generar_visualizaciones_estabilidad()
            
            # 4. Generar reporte
            self.generar_reporte_estabilidad()
            
            logging.info("=== ANÁLISIS DE ESTABILIDAD COMPLETADO ===")
            
        except Exception as e:
            logging.error(f"Error en análisis de estabilidad: {e}")
            raise


def main():
    """Función principal."""
    # Obtener directorio del proyecto
    directorio_proyecto = Path(__file__).parent
    
    # Ejecutar análisis
    analizador = AnalizadorEstabilidad(directorio_proyecto)
    analizador.ejecutar_analisis_completo()


if __name__ == "__main__":
    main()
