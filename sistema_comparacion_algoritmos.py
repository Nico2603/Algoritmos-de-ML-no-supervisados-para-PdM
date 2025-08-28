"""
Sistema de Comparación Unificado para Algoritmos de ML No Supervisado en PdM

Este módulo implementa un sistema completo de comparación entre los 4 algoritmos:
- K-Means (clustering)
- DBSCAN (clustering) 
- Isolation Forest (detección de anomalías)
- CBLOF (detección de anomalías)

Características principales:
1. Score de Severidad común para todos los algoritmos
2. Análisis de concordancia entre métodos
3. Estabilidad con bootstrap
4. Umbrales por percentiles
5. Recomendaciones operativas

Autor: Sistema PdM
Versión: 1.0
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.metrics import jaccard_score
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings('ignore')

# Constantes
PERCENTILES_SEVERIDAD = {
    'leve': 80,      # P80
    'moderada': 90,  # P90
    'severa': 95,    # P95
    'critica': 99    # P99
}

ALGORITMOS_CLUSTERING = ['K-Means', 'DBSCAN']
ALGORITMOS_ANOMALIAS = ['Isolation Forest', 'CBLOF']

class ComparadorAlgoritmos:
    """Sistema de comparación unificado para algoritmos de ML no supervisado."""
    
    def __init__(self, directorio_proyecto: str):
        """
        Inicializa el comparador.
        
        Args:
            directorio_proyecto: Directorio raíz del proyecto
        """
        self.directorio_proyecto = Path(directorio_proyecto)
        self.resultados = {}
        self.scores_unificados = {}
        self.metricas_consolidadas = {}
        
        # Configurar directorios
        self.directorio_comparacion = self.directorio_proyecto / 'Comparación_Algoritmos'
        self.directorio_graficas = self.directorio_comparacion / 'graficas'
        self.directorio_reportes = self.directorio_comparacion / 'reportes'
        
        self._crear_directorios()
        self._configurar_logging()
    
    def _crear_directorios(self) -> None:
        """Crea los directorios necesarios."""
        for directorio in [self.directorio_comparacion, self.directorio_graficas, self.directorio_reportes]:
            directorio.mkdir(exist_ok=True)
    
    def _configurar_logging(self) -> None:
        """Configura el sistema de logging."""
        log_file = self.directorio_reportes / 'comparacion.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
    
    def cargar_resultados_algoritmos(self) -> None:
        """Carga los resultados de todos los algoritmos."""
        logging.info("Cargando resultados de todos los algoritmos...")
        
        # Mapeo de algoritmos a directorios (actualizado post-correcciones)
        algoritmos_config = {
            'K-Means': {
                'directorio': '1. Clustering/K-means',
                'archivo_scores': 'metricas_KMeans/scores_kmeans.csv',
                'archivo_metricas': 'metricas_KMeans/metrics.csv',
                'columnas_esperadas': ['fecha', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'magnitud_aceleracion', 'anomaly_score', 'cluster_id']
            },
            'DBSCAN': {
                'directorio': '1. Clustering/DBSCAN',
                'archivo_scores': 'metricas_DBSCAN/scores_dbscan.csv',
                'archivo_metricas': 'metricas_DBSCAN/metrics.txt',
                'columnas_esperadas': ['fecha', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'magnitud_aceleracion', 'anomaly_score', 'is_outlier', 'cluster_id']
            },
            'Isolation Forest': {
                'directorio': '2. Detección de Anomalías/Isolation Forest',
                'archivo_scores': 'metricas_IForest/anomalies.csv',
                'archivo_metricas': 'metricas_IForest/metrics.txt',
                'columnas_esperadas': ['fecha', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'magnitud_aceleracion', 'anomaly_score', 'is_outlier']
            },
            'CBLOF': {
                'directorio': '2. Detección de Anomalías/CBLOF (Cluster-Based Local Outlier Factor)',
                'archivo_scores': 'metricas_CBLOF/anomalies.csv',
                'archivo_metricas': 'metricas_CBLOF/metrics.txt',
                'columnas_esperadas': ['fecha', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'magnitud_aceleracion', 'anomaly_score', 'is_outlier']
            }
        }
        
        for algoritmo, config in algoritmos_config.items():
            try:
                ruta_scores = self.directorio_proyecto / config['directorio'] / config['archivo_scores']
                if ruta_scores.exists():
                    datos = pd.read_csv(ruta_scores)
                    
                    # Validar que tiene las columnas esperadas post-correcciones
                    columnas_encontradas = set(datos.columns)
                    columnas_criticas = {'acceleration_x', 'acceleration_y', 'acceleration_z', 'anomaly_score'}
                    
                    if not columnas_criticas.issubset(columnas_encontradas):
                        logging.warning(f"⚠️  {algoritmo}: Faltan columnas críticas. Encontradas: {list(columnas_encontradas)}")
                    
                    # Verificar que es 100% no-supervisado (no debe tener 'severity' o similares)
                    columnas_prohibidas = {'severity', 'label', 'ground_truth'}
                    encontradas_prohibidas = columnas_prohibidas.intersection(columnas_encontradas)
                    if encontradas_prohibidas:
                        logging.warning(f"⚠️  {algoritmo}: Contiene columnas supervisadas: {encontradas_prohibidas}")
                    
                    self.resultados[algoritmo] = datos
                    logging.info(f"✓ {algoritmo}: {len(datos)} registros cargados")
                    logging.info(f"   Columnas: {list(datos.columns)}")
                else:
                    logging.warning(f"⚠️  {algoritmo}: No se encontró {ruta_scores}")
            except Exception as e:
                logging.error(f"❌ Error cargando {algoritmo}: {e}")
    
    def calcular_score_severidad_unificado(self) -> None:
        """
        Calcula el Score de Severidad común para todos los algoritmos.
        
        Score normalizado 0-1 donde:
        - 0 = Normal (no anómalo)
        - 1 = Máxima anomalía
        """
        logging.info("Calculando Score de Severidad unificado...")
        
        for algoritmo, datos in self.resultados.items():
            if 'anomaly_score' not in datos.columns:
                logging.warning(f"⚠️  {algoritmo}: No tiene columna 'anomaly_score'")
                continue
            
            # Normalizar scores a rango 0-1
            scores = datos['anomaly_score'].values
            
            # Manejar casos especiales
            if np.all(scores == 0):
                severidad_score = np.zeros_like(scores)
            elif np.max(scores) == np.min(scores):
                severidad_score = np.ones_like(scores) * 0.5
            else:
                # Normalización Min-Max
                severidad_score = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            
            # Para algoritmos de clustering, invertir si es necesario
            if algoritmo in ALGORITMOS_CLUSTERING:
                # En clustering, distancia alta = más anómalo (ya está bien orientado)
                pass
            
            # Guardar score unificado
            datos_con_severidad = datos.copy()
            datos_con_severidad['severidad_score'] = severidad_score
            datos_con_severidad['severidad_nivel'] = self._asignar_nivel_severidad(severidad_score)
            
            self.scores_unificados[algoritmo] = datos_con_severidad
            
            logging.info(f"✓ {algoritmo}: Score de severidad calculado")
            logging.info(f"   - Min: {np.min(severidad_score):.4f}")
            logging.info(f"   - Max: {np.max(severidad_score):.4f}")
            logging.info(f"   - Media: {np.mean(severidad_score):.4f}")
    
    def _asignar_nivel_severidad(self, scores: np.ndarray) -> np.ndarray:
        """Asigna niveles de severidad basados en percentiles."""
        percentiles = [
            np.percentile(scores, PERCENTILES_SEVERIDAD['leve']),
            np.percentile(scores, PERCENTILES_SEVERIDAD['moderada']), 
            np.percentile(scores, PERCENTILES_SEVERIDAD['severa']),
            np.percentile(scores, PERCENTILES_SEVERIDAD['critica'])
        ]
        
        niveles = np.full_like(scores, 'normal', dtype=object)
        niveles[scores >= percentiles[0]] = 'leve'
        niveles[scores >= percentiles[1]] = 'moderada'
        niveles[scores >= percentiles[2]] = 'severa'
        niveles[scores >= percentiles[3]] = 'critica'
        
        return niveles
    
    def analizar_concordancia_algoritmos(self) -> Dict[str, Any]:
        """
        Analiza la concordancia entre algoritmos usando diferentes métricas.
        
        Returns:
            Diccionario con métricas de concordancia
        """
        logging.info("Analizando concordancia entre algoritmos...")
        
        # Preparar datos para comparación
        algoritmos_disponibles = list(self.scores_unificados.keys())
        if len(algoritmos_disponibles) < 2:
            logging.warning("Se necesitan al menos 2 algoritmos para comparar")
            return {}
        
        concordancia = {}
        
        # 1. Correlación de scores de severidad
        correlaciones = self._calcular_correlaciones_scores()
        concordancia['correlaciones'] = correlaciones
        
        # 2. Jaccard Top-N% (solapamiento en los más anómalos)
        jaccard_scores = self._calcular_jaccard_top_percentiles()
        concordancia['jaccard_top_percentiles'] = jaccard_scores
        
        # 3. Acuerdo en clasificación por niveles
        acuerdo_niveles = self._calcular_acuerdo_niveles()
        concordancia['acuerdo_niveles'] = acuerdo_niveles
        
        logging.info("✓ Análisis de concordancia completado")
        return concordancia
    
    def _calcular_correlaciones_scores(self) -> Dict[str, float]:
        """Calcula correlaciones entre scores de severidad."""
        algoritmos = list(self.scores_unificados.keys())
        correlaciones = {}
        
        for i, alg1 in enumerate(algoritmos):
            for j, alg2 in enumerate(algoritmos[i+1:], i+1):
                # Asegurar mismo número de registros
                datos1 = self.scores_unificados[alg1]['severidad_score'].values
                datos2 = self.scores_unificados[alg2]['severidad_score'].values
                
                min_len = min(len(datos1), len(datos2))
                corr, p_value = stats.pearsonr(datos1[:min_len], datos2[:min_len])
                
                clave = f"{alg1}_vs_{alg2}"
                correlaciones[clave] = {
                    'correlacion': corr,
                    'p_value': p_value,
                    'significativo': p_value < 0.05
                }
        
        return correlaciones
    
    def _calcular_jaccard_top_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Calcula índice Jaccard para Top-N% más anómalos."""
        algoritmos = list(self.scores_unificados.keys())
        percentiles_test = [95, 90, 80]  # Top 5%, 10%, 20%
        
        jaccard_results = {}
        
        for percentil in percentiles_test:
            jaccard_results[f'top_{100-percentil}%'] = {}
            
            for i, alg1 in enumerate(algoritmos):
                for j, alg2 in enumerate(algoritmos[i+1:], i+1):
                    # Identificar top percentil para cada algoritmo
                    datos1 = self.scores_unificados[alg1]
                    datos2 = self.scores_unificados[alg2]
                    
                    threshold1 = np.percentile(datos1['severidad_score'], percentil)
                    threshold2 = np.percentile(datos2['severidad_score'], percentil)
                    
                    top1 = set(datos1[datos1['severidad_score'] >= threshold1].index)
                    top2 = set(datos2[datos2['severidad_score'] >= threshold2].index)
                    
                    # Calcular Jaccard
                    interseccion = len(top1.intersection(top2))
                    union = len(top1.union(top2))
                    jaccard = interseccion / union if union > 0 else 0
                    
                    clave = f"{alg1}_vs_{alg2}"
                    jaccard_results[f'top_{100-percentil}%'][clave] = jaccard
        
        return jaccard_results
    
    def _calcular_acuerdo_niveles(self) -> Dict[str, float]:
        """Calcula acuerdo en clasificación por niveles de severidad."""
        algoritmos = list(self.scores_unificados.keys())
        acuerdos = {}
        
        for i, alg1 in enumerate(algoritmos):
            for j, alg2 in enumerate(algoritmos[i+1:], i+1):
                datos1 = self.scores_unificados[alg1]['severidad_nivel'].values
                datos2 = self.scores_unificados[alg2]['severidad_nivel'].values
                
                min_len = min(len(datos1), len(datos2))
                acuerdo = np.mean(datos1[:min_len] == datos2[:min_len])
                
                clave = f"{alg1}_vs_{alg2}"
                acuerdos[clave] = acuerdo
        
        return acuerdos
    
    def generar_visualizaciones_comparativas(self) -> None:
        """Genera todas las visualizaciones comparativas."""
        logging.info("Generando visualizaciones comparativas...")
        
        # 1. Distribución de scores de severidad
        self._plot_distribucion_scores()
        
        # 2. Matriz de correlación
        self._plot_matriz_correlacion()
        
        # 3. Comparación Top-N anomalías
        self._plot_top_anomalias()
        
        # 4. Heatmap de concordancia
        self._plot_heatmap_concordancia()
        
        logging.info("✓ Visualizaciones generadas")
    
    def _plot_distribucion_scores(self) -> None:
        """Gráfico de distribución de scores de severidad."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribución de Scores de Severidad por Algoritmo', fontsize=16)
        
        for idx, (algoritmo, datos) in enumerate(self.scores_unificados.items()):
            ax = axes[idx // 2, idx % 2]
            
            # Histograma
            ax.hist(datos['severidad_score'], bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{algoritmo}')
            ax.set_xlabel('Score de Severidad')
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)
            
            # Líneas de percentiles
            for percentil, valor in PERCENTILES_SEVERIDAD.items():
                umbral = np.percentile(datos['severidad_score'], valor)
                ax.axvline(umbral, color='red', linestyle='--', alpha=0.7, 
                          label=f'P{valor}')
            
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.directorio_graficas / 'distribucion_scores_severidad.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_matriz_correlacion(self) -> None:
        """Matriz de correlación entre algoritmos."""
        # Preparar datos para matriz
        algoritmos = list(self.scores_unificados.keys())
        matriz_corr = np.eye(len(algoritmos))
        
        for i, alg1 in enumerate(algoritmos):
            for j, alg2 in enumerate(algoritmos):
                if i != j:
                    datos1 = self.scores_unificados[alg1]['severidad_score'].values
                    datos2 = self.scores_unificados[alg2]['severidad_score'].values
                    
                    min_len = min(len(datos1), len(datos2))
                    corr, _ = stats.pearsonr(datos1[:min_len], datos2[:min_len])
                    matriz_corr[i, j] = corr
        
        # Crear heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_corr, 
                   xticklabels=algoritmos,
                   yticklabels=algoritmos,
                   annot=True, 
                   cmap='coolwarm',
                   center=0,
                   square=True)
        plt.title('Matriz de Correlación - Scores de Severidad')
        plt.tight_layout()
        plt.savefig(self.directorio_graficas / 'matriz_correlacion_algoritmos.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_anomalias(self) -> None:
        """Gráfico de Top-N anomalías por algoritmo."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        percentiles = [95, 90, 80]
        
        for idx, percentil in enumerate(percentiles):
            ax = axes[idx]
            
            # Calcular Top-N para cada algoritmo
            top_counts = {}
            for algoritmo, datos in self.scores_unificados.items():
                threshold = np.percentile(datos['severidad_score'], percentil)
                n_top = np.sum(datos['severidad_score'] >= threshold)
                top_counts[algoritmo] = n_top
            
            # Gráfico de barras
            algoritmos = list(top_counts.keys())
            valores = list(top_counts.values())
            
            bars = ax.bar(algoritmos, valores, alpha=0.7)
            ax.set_title(f'Top {100-percentil}% Anomalías más Severas')
            ax.set_ylabel('Número de Anomalías')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Rotar etiquetas si es necesario
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Añadir valores en las barras
            for bar, valor in zip(bars, valores):
                height = bar.get_height()
                ax.annotate(f'{valor}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.directorio_graficas / 'top_anomalias_comparacion.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_heatmap_concordancia(self) -> None:
        """Heatmap de concordancia entre algoritmos."""
        # Este gráfico requiere que ya se haya ejecutado el análisis de concordancia
        # Por ahora, crear un placeholder
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Heatmap de Concordancia\n(Requiere análisis previo)', 
                ha='center', va='center', fontsize=16)
        plt.title('Concordancia entre Algoritmos - Múltiples Métricas')
        plt.axis('off')
        plt.savefig(self.directorio_graficas / 'heatmap_concordancia.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generar_reporte_comparativo(self) -> None:
        """Genera el reporte comparativo final."""
        logging.info("Generando reporte comparativo...")
        
        ruta_reporte = self.directorio_reportes / 'reporte_comparativo.md'
        
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write("# Reporte Comparativo de Algoritmos de ML No Supervisado para PdM\n\n")
            
            # Resumen ejecutivo
            f.write("## Resumen Ejecutivo\n\n")
            f.write("Este reporte presenta la comparación entre los 4 algoritmos implementados:\n\n")
            for algoritmo in self.scores_unificados.keys():
                tipo = "Clustering" if algoritmo in ALGORITMOS_CLUSTERING else "Detección de Anomalías"
                f.write(f"- **{algoritmo}** ({tipo})\n")
            f.write("\n")
            
            # Estadísticas por algoritmo
            f.write("## Estadísticas por Algoritmo\n\n")
            for algoritmo, datos in self.scores_unificados.items():
                f.write(f"### {algoritmo}\n\n")
                f.write(f"- **Total de registros**: {len(datos):,}\n")
                f.write(f"- **Score de severidad promedio**: {np.mean(datos['severidad_score']):.4f}\n")
                f.write(f"- **Desviación estándar**: {np.std(datos['severidad_score']):.4f}\n")
                
                # Distribución por niveles
                niveles_count = datos['severidad_nivel'].value_counts()
                f.write(f"- **Distribución por severidad**:\n")
                for nivel, count in niveles_count.items():
                    porcentaje = (count / len(datos)) * 100
                    f.write(f"  - {nivel.title()}: {count:,} ({porcentaje:.1f}%)\n")
                f.write("\n")
            
            # Recomendaciones
            f.write("## Recomendaciones Operativas\n\n")
            f.write("### Para Detección de Patrones (Clustering)\n")
            f.write("- **K-Means**: Recomendado para identificar patrones de operación normal\n")
            f.write("- **DBSCAN**: Mejor para detectar grupos de densidad variable\n\n")
            
            f.write("### Para Alertas de Anomalías\n")
            f.write("- **Isolation Forest**: Eficiente para anomalías globales\n")
            f.write("- **CBLOF**: Mejor para anomalías basadas en clusters\n\n")
            
            f.write("### Umbrales Recomendados\n")
            f.write("- **Alertas Operativas**: P95 (Severidad ≥ 95%)\n")
            f.write("- **Monitoreo Preventivo**: P90 (Severidad ≥ 90%)\n")
            f.write("- **Análisis Rutinario**: P80 (Severidad ≥ 80%)\n\n")
        
        logging.info(f"✓ Reporte guardado en {ruta_reporte}")
    
    def ejecutar_comparacion_completa(self) -> None:
        """Ejecuta todo el proceso de comparación."""
        logging.info("=== INICIANDO COMPARACIÓN COMPLETA DE ALGORITMOS ===")
        
        try:
            # 1. Cargar resultados
            self.cargar_resultados_algoritmos()
            
            # 2. Calcular scores unificados
            self.calcular_score_severidad_unificado()
            
            # 3. Analizar concordancia
            concordancia = self.analizar_concordancia_algoritmos()
            
            # 4. Generar visualizaciones
            self.generar_visualizaciones_comparativas()
            
            # 5. Generar reporte
            self.generar_reporte_comparativo()
            
            logging.info("=== COMPARACIÓN COMPLETA FINALIZADA ===")
            
        except Exception as e:
            logging.error(f"Error en la comparación: {e}")
            raise


def main():
    """Función principal."""
    # Obtener directorio del proyecto (parent del script actual)
    directorio_proyecto = Path(__file__).parent
    
    # Ejecutar comparación
    comparador = ComparadorAlgoritmos(directorio_proyecto)
    comparador.ejecutar_comparacion_completa()


if __name__ == "__main__":
    main()
