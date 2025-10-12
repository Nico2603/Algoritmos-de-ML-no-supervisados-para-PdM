import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Función para obtener métricas de forma segura
def obtener_metrica_segura(df: pd.DataFrame, columna: str, default: float = -1.0) -> float:
    """
    Obtiene una métrica de forma segura, manejando NaN y None.
    
    Args:
        df: DataFrame con las métricas
        columna: Nombre de la columna
        default: Valor por defecto si la métrica no está disponible
        
    Returns:
        Valor de la métrica o default si no está disponible
    """
    try:
        valor = df[columna].values[0]
        if pd.isna(valor) or valor is None:
            return default
        return float(valor)
    except (KeyError, IndexError):
        return default

DIRECTORIO_BASE = Path(__file__).parent.parent
DIRECTORIO_COMPARACIONES = Path(__file__).parent

RUTAS = {
    'DBSCAN': {
        'metricas_csv': DIRECTORIO_BASE / 'DBSCAN' / 'metricas_DBSCAN' / 'metrics.csv',
        'metricas_txt': DIRECTORIO_BASE / 'DBSCAN' / 'metricas_DBSCAN' / 'metrics.txt',
        'scores_csv': DIRECTORIO_BASE / 'DBSCAN' / 'metricas_DBSCAN' / 'anomaly_scores.csv',
        'grafica_2d': DIRECTORIO_BASE / 'DBSCAN' / 'graficas_DBSCAN' / 'clusters_2d_pca.png',
        'grafica_3d': DIRECTORIO_BASE / 'DBSCAN' / 'graficas_DBSCAN' / 'clusters_3d_pca.png',
    },
    'K-Means': {
        'metricas_csv': DIRECTORIO_BASE / 'K-means' / 'metricas_KMeans' / 'metrics.csv',
        'metricas_txt': DIRECTORIO_BASE / 'K-means' / 'metricas_KMeans' / 'metrics.txt',
        'scores_csv': DIRECTORIO_BASE / 'K-means' / 'metricas_KMeans' / 'anomaly_scores.csv',
        'grafica_2d': DIRECTORIO_BASE / 'K-means' / 'graficas_KMeans' / 'clusters_2d_pca.png',
        'grafica_3d': DIRECTORIO_BASE / 'K-means' / 'graficas_KMeans' / 'clusters_3d_pca.png',
    }
}


def verificar_archivos() -> bool:
    archivos_faltantes = []
    
    for algoritmo, rutas in RUTAS.items():
        for nombre_archivo, ruta in rutas.items():
            if not ruta.exists():
                archivos_faltantes.append(f"{algoritmo} - {nombre_archivo}: {ruta}")
    
    if archivos_faltantes:
        print("❌ ERROR: Archivos faltantes:")
        for archivo in archivos_faltantes:
            print(f"  - {archivo}")
        print("\n⚠️  Asegúrate de ejecutar primero los algoritmos DBSCAN y K-Means.")
        return False
    
    return True


def cargar_metricas() -> Tuple[pd.DataFrame, pd.DataFrame]:
    metricas_dbscan = pd.read_csv(RUTAS['DBSCAN']['metricas_csv'])
    metricas_kmeans = pd.read_csv(RUTAS['K-Means']['metricas_csv'])
    return metricas_dbscan, metricas_kmeans


def crear_comparacion_imagenes_lado_a_lado() -> None:
    print("\n📊 Generando comparación visual de gráficas...")
    
    # Comparación 2D
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    # DBSCAN 2D
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['DBSCAN']['grafica_2d'])
    ax1.imshow(img1)
    ax1.set_title('DBSCAN - Clustering 2D con PCA', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # K-Means 2D
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['K-Means']['grafica_2d'])
    ax2.imshow(img2)
    ax2.set_title('K-Means - Clustering 2D con PCA', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: DBSCAN vs K-Means (2D)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_2d = DIRECTORIO_COMPARACIONES / 'comparacion_visual_2d.png'
    plt.savefig(ruta_comparacion_2d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparación 2D guardada: {ruta_comparacion_2d.name}")
    
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    # DBSCAN 3D
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['DBSCAN']['grafica_3d'])
    ax1.imshow(img1)
    ax1.set_title('DBSCAN - Clustering 3D con PCA', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # K-Means 3D
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['K-Means']['grafica_3d'])
    ax2.imshow(img2)
    ax2.set_title('K-Means - Clustering 3D con PCA', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: DBSCAN vs K-Means (3D)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_3d = DIRECTORIO_COMPARACIONES / 'comparacion_visual_3d.png'
    plt.savefig(ruta_comparacion_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparación 3D guardada: {ruta_comparacion_3d.name}")


def crear_tabla_comparativa_metricas(metricas_dbscan: pd.DataFrame, 
                                     metricas_kmeans: pd.DataFrame) -> pd.DataFrame:
    tiempo_dbscan = obtener_metrica_segura(metricas_dbscan, 'tiempo_ejecucion_s', 0)
    tiempo_kmeans = obtener_metrica_segura(metricas_kmeans, 'tiempo_ejecucion_s', 0)
    memoria_dbscan = obtener_metrica_segura(metricas_dbscan, 'memoria_max_mb', 0)
    memoria_kmeans = obtener_metrica_segura(metricas_kmeans, 'memoria_max_mb', 0)
    
    comparacion = {
        'Métrica': [
            'Algoritmo',
            'Número de Clusters',
            'Silhouette Score',
            'Calinski-Harabasz Score',
            'Davies-Bouldin Index',
            'Porcentaje de Anomalías (%)',
            'Separación de Scores (P95-P50)',
            'Score Promedio',
            'Tiempo de Ejecución (s)',
            'Memoria Máxima (MB)'
        ],
        'DBSCAN': [
            metricas_dbscan['algoritmo'].values[0],
            metricas_dbscan['n_clusters'].values[0] if not pd.isna(metricas_dbscan['n_clusters'].values[0]) else 'N/A',
            f"{metricas_dbscan['silhouette_score'].values[0]:.4f}" if not pd.isna(metricas_dbscan['silhouette_score'].values[0]) else 'N/A',
            f"{metricas_dbscan['calinski_harabasz_score'].values[0]:.4f}" if not pd.isna(metricas_dbscan['calinski_harabasz_score'].values[0]) else 'N/A',
            f"{metricas_dbscan['davies_bouldin_score'].values[0]:.4f}" if not pd.isna(metricas_dbscan['davies_bouldin_score'].values[0]) else 'N/A',
            f"{metricas_dbscan['pct_anomalias'].values[0]:.2f}",
            f"{metricas_dbscan['p95_minus_p50'].values[0]:.4f}",
            f"{metricas_dbscan['mean_score'].values[0]:.4f}",
            f"{tiempo_dbscan:.2f}" if tiempo_dbscan > 0 else 'N/A',
            f"{memoria_dbscan:.2f}" if memoria_dbscan > 0 else 'N/A'
        ],
        'K-Means': [
            metricas_kmeans['algoritmo'].values[0],
            metricas_kmeans['n_clusters'].values[0] if not pd.isna(metricas_kmeans['n_clusters'].values[0]) else 'N/A',
            f"{metricas_kmeans['silhouette_score'].values[0]:.4f}" if not pd.isna(metricas_kmeans['silhouette_score'].values[0]) else 'N/A',
            f"{metricas_kmeans['calinski_harabasz_score'].values[0]:.4f}" if not pd.isna(metricas_kmeans['calinski_harabasz_score'].values[0]) else 'N/A',
            f"{metricas_kmeans['davies_bouldin_score'].values[0]:.4f}" if not pd.isna(metricas_kmeans['davies_bouldin_score'].values[0]) else 'N/A',
            f"{metricas_kmeans['pct_anomalias'].values[0]:.2f}",
            f"{metricas_kmeans['p95_minus_p50'].values[0]:.4f}",
            f"{metricas_kmeans['mean_score'].values[0]:.4f}",
            f"{tiempo_kmeans:.2f}" if tiempo_kmeans > 0 else 'N/A',
            f"{memoria_kmeans:.2f}" if memoria_kmeans > 0 else 'N/A'
        ]
    }
    
    df_comparacion = pd.DataFrame(comparacion)
    return df_comparacion


def generar_graficos_metricas(metricas_dbscan: pd.DataFrame, 
                              metricas_kmeans: pd.DataFrame) -> None:
    print("\n📈 Generando gráficos comparativos de métricas...")
    
    # Gráfico de barras comparativo
    metricas_nombres = ['Silhouette\nScore', 'Calinski-\nHarabasz\n(x1000)', 'Davies-\nBouldin\n(Inverso)']
    
    sil_dbscan = metricas_dbscan['silhouette_score'].values[0] if not pd.isna(metricas_dbscan['silhouette_score'].values[0]) else 0
    sil_kmeans = metricas_kmeans['silhouette_score'].values[0] if not pd.isna(metricas_kmeans['silhouette_score'].values[0]) else 0
    
    cal_dbscan = metricas_dbscan['calinski_harabasz_score'].values[0] / 1000 if not pd.isna(metricas_dbscan['calinski_harabasz_score'].values[0]) else 0
    cal_kmeans = metricas_kmeans['calinski_harabasz_score'].values[0] / 1000 if not pd.isna(metricas_kmeans['calinski_harabasz_score'].values[0]) else 0
    
    dav_dbscan = 1 / metricas_dbscan['davies_bouldin_score'].values[0] if not pd.isna(metricas_dbscan['davies_bouldin_score'].values[0]) and metricas_dbscan['davies_bouldin_score'].values[0] > 0 else 0
    dav_kmeans = 1 / metricas_kmeans['davies_bouldin_score'].values[0] if not pd.isna(metricas_kmeans['davies_bouldin_score'].values[0]) and metricas_kmeans['davies_bouldin_score'].values[0] > 0 else 0
    
    dbscan_values = [sil_dbscan, cal_dbscan, dav_dbscan]
    kmeans_values = [sil_kmeans, cal_kmeans, dav_kmeans]
    
    x = np.arange(len(metricas_nombres))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, dbscan_values, width, label='DBSCAN', 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    rects2 = ax.bar(x + width/2, kmeans_values, width, label='K-Means', 
                    color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Métricas de Clustering', fontsize=13, fontweight='bold')
    ax.set_ylabel('Valor Normalizado', fontsize=13, fontweight='bold')
    ax.set_title('Comparación de Métricas de Calidad: DBSCAN vs K-Means', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nombres, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Añadir valores sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    ruta_grafico = DIRECTORIO_COMPARACIONES / 'comparacion_metricas_barras.png'
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de barras guardado: {ruta_grafico.name}")
    
    crear_grafico_radar(metricas_dbscan, metricas_kmeans)


def crear_grafico_radar(metricas_dbscan: pd.DataFrame, metricas_kmeans: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='polar')
    
    categorias = ['Silhouette\nScore', 'Calinski-Harabasz\n(Normalizado)', 
                 'Davies-Bouldin\n(Inverso)', 'Separación\nde Scores', 'N° de Clusters']
    
    def normalizar(valor, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return (valor - min_val) / (max_val - min_val)
    
    sil_dbscan = metricas_dbscan['silhouette_score'].values[0] if not pd.isna(metricas_dbscan['silhouette_score'].values[0]) else 0
    sil_kmeans = metricas_kmeans['silhouette_score'].values[0] if not pd.isna(metricas_kmeans['silhouette_score'].values[0]) else 0
    
    cal_dbscan = metricas_dbscan['calinski_harabasz_score'].values[0] / 1000 if not pd.isna(metricas_dbscan['calinski_harabasz_score'].values[0]) else 0
    cal_kmeans = metricas_kmeans['calinski_harabasz_score'].values[0] / 1000 if not pd.isna(metricas_kmeans['calinski_harabasz_score'].values[0]) else 0
    
    dav_dbscan = 1 / metricas_dbscan['davies_bouldin_score'].values[0] if not pd.isna(metricas_dbscan['davies_bouldin_score'].values[0]) and metricas_dbscan['davies_bouldin_score'].values[0] > 0 else 0
    dav_kmeans = 1 / metricas_kmeans['davies_bouldin_score'].values[0] if not pd.isna(metricas_kmeans['davies_bouldin_score'].values[0]) and metricas_kmeans['davies_bouldin_score'].values[0] > 0 else 0
    
    sep_dbscan = metricas_dbscan['p95_minus_p50'].values[0]
    sep_kmeans = metricas_kmeans['p95_minus_p50'].values[0]
    
    n_clust_dbscan = metricas_dbscan['n_clusters'].values[0] if not pd.isna(metricas_dbscan['n_clusters'].values[0]) else 0
    n_clust_kmeans = metricas_kmeans['n_clusters'].values[0] if not pd.isna(metricas_kmeans['n_clusters'].values[0]) else 0
    
    valores_dbscan = [
        normalizar(sil_dbscan, min(sil_dbscan, sil_kmeans), max(sil_dbscan, sil_kmeans)),
        normalizar(cal_dbscan, min(cal_dbscan, cal_kmeans), max(cal_dbscan, cal_kmeans)),
        normalizar(dav_dbscan, min(dav_dbscan, dav_kmeans), max(dav_dbscan, dav_kmeans)),
        normalizar(sep_dbscan, min(sep_dbscan, sep_kmeans), max(sep_dbscan, sep_kmeans)),
        normalizar(n_clust_dbscan, min(n_clust_dbscan, n_clust_kmeans), max(n_clust_dbscan, n_clust_kmeans))
    ]
    
    valores_kmeans = [
        normalizar(sil_kmeans, min(sil_dbscan, sil_kmeans), max(sil_dbscan, sil_kmeans)),
        normalizar(cal_kmeans, min(cal_dbscan, cal_kmeans), max(cal_dbscan, cal_kmeans)),
        normalizar(dav_kmeans, min(dav_dbscan, dav_kmeans), max(dav_dbscan, dav_kmeans)),
        normalizar(sep_kmeans, min(sep_dbscan, sep_kmeans), max(sep_dbscan, sep_kmeans)),
        normalizar(n_clust_kmeans, min(n_clust_dbscan, n_clust_kmeans), max(n_clust_dbscan, n_clust_kmeans))
    ]
    
    valores_dbscan += valores_dbscan[:1]
    valores_kmeans += valores_kmeans[:1]
    
    angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
    angulos += angulos[:1]
    
    ax.plot(angulos, valores_dbscan, 'o-', linewidth=2.5, label='DBSCAN', color='steelblue', markersize=8)
    ax.fill(angulos, valores_dbscan, alpha=0.25, color='steelblue')
    
    ax.plot(angulos, valores_kmeans, 'o-', linewidth=2.5, label='K-Means', color='coral', markersize=8)
    ax.fill(angulos, valores_kmeans, alpha=0.25, color='coral')
    
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(categorias, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Análisis Multidimensional de Clustering\n(Valores Normalizados 0-1)', 
                size=15, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    
    plt.tight_layout()
    ruta_radar = DIRECTORIO_COMPARACIONES / 'comparacion_metricas_radar.png'
    plt.savefig(ruta_radar, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de radar guardado: {ruta_radar.name}")


def crear_grafico_rendimiento(metricas_dbscan: pd.DataFrame, metricas_kmeans: pd.DataFrame) -> None:
    """
    Genera gráficos comparativos de rendimiento (tiempo y memoria).
    """
    print("\n⏱️  Generando gráficos de rendimiento...")
    
    tiempo_dbscan = obtener_metrica_segura(metricas_dbscan, 'tiempo_ejecucion_s', 0)
    tiempo_kmeans = obtener_metrica_segura(metricas_kmeans, 'tiempo_ejecucion_s', 0)
    memoria_dbscan = obtener_metrica_segura(metricas_dbscan, 'memoria_max_mb', 0)
    memoria_kmeans = obtener_metrica_segura(metricas_kmeans, 'memoria_max_mb', 0)
    
    # Crear gráfico con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de tiempo de ejecución
    algoritmos = ['DBSCAN', 'K-Means']
    tiempos = [tiempo_dbscan, tiempo_kmeans]
    colores = ['steelblue', 'coral']
    
    bars1 = ax1.bar(algoritmos, tiempos, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Tiempo de Ejecución', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Añadir valores sobre las barras
    for bar, tiempo in zip(bars1, tiempos):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{tiempo:.2f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Gráfico de uso de memoria
    memorias = [memoria_dbscan, memoria_kmeans]
    bars2 = ax2.bar(algoritmos, memorias, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Memoria (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Uso de Memoria', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Añadir valores sobre las barras
    for bar, memoria in zip(bars2, memorias):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{memoria:.2f} MB',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Métricas de Rendimiento: DBSCAN vs K-Means', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    ruta_rendimiento = DIRECTORIO_COMPARACIONES / 'comparacion_rendimiento.png'
    plt.savefig(ruta_rendimiento, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de rendimiento guardado: {ruta_rendimiento.name}")


def analizar_ganador(metricas_dbscan: pd.DataFrame, metricas_kmeans: pd.DataFrame) -> Dict[str, Any]:
    analisis = {}
    
    sil_dbscan = metricas_dbscan['silhouette_score'].values[0] if not pd.isna(metricas_dbscan['silhouette_score'].values[0]) else -1
    sil_kmeans = metricas_kmeans['silhouette_score'].values[0] if not pd.isna(metricas_kmeans['silhouette_score'].values[0]) else -1
    
    cal_dbscan = metricas_dbscan['calinski_harabasz_score'].values[0] if not pd.isna(metricas_dbscan['calinski_harabasz_score'].values[0]) else -1
    cal_kmeans = metricas_kmeans['calinski_harabasz_score'].values[0] if not pd.isna(metricas_kmeans['calinski_harabasz_score'].values[0]) else -1
    
    dav_dbscan = metricas_dbscan['davies_bouldin_score'].values[0] if not pd.isna(metricas_dbscan['davies_bouldin_score'].values[0]) else float('inf')
    dav_kmeans = metricas_kmeans['davies_bouldin_score'].values[0] if not pd.isna(metricas_kmeans['davies_bouldin_score'].values[0]) else float('inf')
    
    puntos_dbscan = 0
    puntos_kmeans = 0
    
    if sil_dbscan > sil_kmeans:
        puntos_dbscan += 1
        analisis['mejor_silhouette'] = 'DBSCAN'
    elif sil_kmeans > sil_dbscan:
        puntos_kmeans += 1
        analisis['mejor_silhouette'] = 'K-Means'
    else:
        analisis['mejor_silhouette'] = 'Empate'
    
    if cal_dbscan > cal_kmeans:
        puntos_dbscan += 1
        analisis['mejor_calinski'] = 'DBSCAN'
    elif cal_kmeans > cal_dbscan:
        puntos_kmeans += 1
        analisis['mejor_calinski'] = 'K-Means'
    else:
        analisis['mejor_calinski'] = 'Empate'
    
    if dav_dbscan < dav_kmeans:
        puntos_dbscan += 1
        analisis['mejor_davies'] = 'DBSCAN'
    elif dav_kmeans < dav_dbscan:
        puntos_kmeans += 1
        analisis['mejor_davies'] = 'K-Means'
    else:
        analisis['mejor_davies'] = 'Empate'
    
    if puntos_dbscan > puntos_kmeans:
        analisis['ganador'] = 'DBSCAN'
        analisis['puntos_dbscan'] = puntos_dbscan
        analisis['puntos_kmeans'] = puntos_kmeans
    elif puntos_kmeans > puntos_dbscan:
        analisis['ganador'] = 'K-Means'
        analisis['puntos_dbscan'] = puntos_dbscan
        analisis['puntos_kmeans'] = puntos_kmeans
    else:
        analisis['ganador'] = 'Empate'
        analisis['puntos_dbscan'] = puntos_dbscan
        analisis['puntos_kmeans'] = puntos_kmeans
    
    return analisis


def guardar_reporte_completo(tabla_comparativa: pd.DataFrame, analisis: Dict[str, Any],
                             metricas_dbscan: pd.DataFrame, metricas_kmeans: pd.DataFrame) -> None:
    print("\n📝 Generando reporte completo...")
    
    ruta_reporte = DIRECTORIO_COMPARACIONES / 'REPORTE_COMPARACION_CLUSTERING.txt'
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 25 + "REPORTE DE COMPARACIÓN DE ALGORITMOS DE CLUSTERING\n")
        f.write(" " * 40 + "DBSCAN vs K-Means\n")
        f.write("=" * 100 + "\n\n")
        
        # Tabla comparativa
        f.write("1. TABLA COMPARATIVA DE MÉTRICAS\n")
        f.write("-" * 100 + "\n")
        f.write(tabla_comparativa.to_string(index=False))
        f.write("\n\n")
        
        f.write("2. ANÁLISIS DE RESULTADOS POR MÉTRICA\n")
        f.write("-" * 100 + "\n")
        f.write(f"✓ Mejor Silhouette Score: {analisis['mejor_silhouette']}\n")
        f.write(f"✓ Mejor Calinski-Harabasz Score: {analisis['mejor_calinski']}\n")
        f.write(f"✓ Mejor Davies-Bouldin Index: {analisis['mejor_davies']}\n\n")
        
        f.write("3. PUNTUACIÓN FINAL\n")
        f.write("-" * 100 + "\n")
        f.write(f"DBSCAN: {analisis['puntos_dbscan']}/3 métricas\n")
        f.write(f"K-Means: {analisis['puntos_kmeans']}/3 métricas\n\n")
        
        f.write("4. ALGORITMO GANADOR\n")
        f.write("-" * 100 + "\n")
        f.write(f"🏆 GANADOR: {analisis['ganador']}\n\n")
        
        if analisis['ganador'] == 'DBSCAN':
            f.write("DBSCAN demostró mejor rendimiento en la mayoría de las métricas de clustering.\n")
        elif analisis['ganador'] == 'K-Means':
            f.write("K-Means demostró mejor rendimiento en la mayoría de las métricas de clustering.\n")
        else:
            f.write("Ambos algoritmos mostraron rendimiento equivalente. La elección depende del contexto.\n")
        
        f.write("\n")
        
        f.write("5. CARACTERÍSTICAS Y CONSIDERACIONES\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("DBSCAN:\n")
        f.write("  Ventajas:\n")
        f.write("    ✓ Detecta clusters de forma arbitraria (no solo esféricos)\n")
        f.write("    ✓ Identifica automáticamente outliers (puntos de ruido)\n")
        f.write("    ✓ No requiere especificar el número de clusters a priori\n")
        f.write("    ✓ Funciona bien con clusters de densidad variable\n")
        f.write("  Desventajas:\n")
        f.write("    ✗ Sensible a la elección de parámetros (eps, min_samples)\n")
        f.write("    ✗ Puede tener problemas con clusters de densidad muy diferente\n")
        f.write("    ✗ Rendimiento puede degradarse en alta dimensionalidad\n\n")
        
        f.write("K-Means:\n")
        f.write("  Ventajas:\n")
        f.write("    ✓ Algoritmo simple y rápido\n")
        f.write("    ✓ Escalable a grandes datasets\n")
        f.write("    ✓ Funciona bien cuando los clusters son esféricos y de tamaño similar\n")
        f.write("    ✓ Asignación determinística de puntos a clusters\n")
        f.write("  Desventajas:\n")
        f.write("    ✗ Requiere especificar el número de clusters a priori\n")
        f.write("    ✗ Sensible a inicialización y outliers\n")
        f.write("    ✗ Asume clusters de forma esférica\n")
        f.write("    ✗ No identifica outliers automáticamente\n\n")
        
        f.write("6. DETALLES ADICIONALES\n")
        f.write("-" * 100 + "\n")
        f.write(f"Número de Clusters:\n")
        f.write(f"  DBSCAN: {metricas_dbscan['n_clusters'].values[0]}\n")
        f.write(f"  K-Means: {metricas_kmeans['n_clusters'].values[0]}\n\n")
        
        f.write(f"Porcentaje de Anomalías Detectadas:\n")
        f.write(f"  DBSCAN: {metricas_dbscan['pct_anomalias'].values[0]:.2f}%\n")
        f.write(f"  K-Means: {metricas_kmeans['pct_anomalias'].values[0]:.2f}% (Detecta outliers con percentil 95)\n\n")
        
        tiempo_dbscan = obtener_metrica_segura(metricas_dbscan, 'tiempo_ejecucion_s', 0)
        tiempo_kmeans = obtener_metrica_segura(metricas_kmeans, 'tiempo_ejecucion_s', 0)
        memoria_dbscan = obtener_metrica_segura(metricas_dbscan, 'memoria_max_mb', 0)
        memoria_kmeans = obtener_metrica_segura(metricas_kmeans, 'memoria_max_mb', 0)
        
        f.write(f"Métricas de Rendimiento:\n")
        if tiempo_dbscan > 0 and tiempo_kmeans > 0:
            f.write(f"  Tiempo de Ejecución:\n")
            f.write(f"    DBSCAN: {tiempo_dbscan:.2f}s\n")
            f.write(f"    K-Means: {tiempo_kmeans:.2f}s\n")
            mas_rapido = "K-Means" if tiempo_kmeans < tiempo_dbscan else "DBSCAN"
            f.write(f"    ⚡ {mas_rapido} es más rápido\n\n")
        
        if memoria_dbscan > 0 and memoria_kmeans > 0:
            f.write(f"  Uso de Memoria Máxima:\n")
            f.write(f"    DBSCAN: {memoria_dbscan:.2f} MB\n")
            f.write(f"    K-Means: {memoria_kmeans:.2f} MB\n")
            menos_memoria = "K-Means" if memoria_kmeans < memoria_dbscan else "DBSCAN"
            f.write(f"    💾 {menos_memoria} usa menos memoria\n\n")
        
        f.write("7. ARCHIVOS GENERADOS EN ESTA COMPARACIÓN\n")
        f.write("-" * 100 + "\n")
        f.write("  • comparacion_visual_2d.png - Comparación lado a lado de gráficas 2D\n")
        f.write("  • comparacion_visual_3d.png - Comparación lado a lado de gráficas 3D\n")
        f.write("  • comparacion_metricas_barras.png - Gráfico de barras comparativo\n")
        f.write("  • comparacion_metricas_radar.png - Gráfico de radar multidimensional\n")
        f.write("  • comparacion_rendimiento.png - Gráfico de rendimiento (tiempo y memoria)\n")
        f.write("  • tabla_comparativa.csv - Tabla de métricas en formato CSV\n")
        f.write("  • REPORTE_COMPARACION_CLUSTERING.txt - Este archivo\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("Fecha de generación: Octubre 2025\n")
        f.write("=" * 100 + "\n")
    
    print(f"✅ Reporte completo guardado: {ruta_reporte.name}")


def main():
    print("=" * 100)
    print(" " * 25 + "COMPARACIÓN DE ALGORITMOS DE CLUSTERING")
    print(" " * 40 + "DBSCAN vs K-Means")
    print("=" * 100)
    
    if not verificar_archivos():
        return
    
    print("\n✅ Todos los archivos necesarios están presentes.")
    
    print("\n📂 Cargando métricas...")
    metricas_dbscan, metricas_kmeans = cargar_metricas()
    print("✅ Métricas cargadas correctamente.")
    
    crear_comparacion_imagenes_lado_a_lado()
    
    print("\n📋 Generando tabla comparativa...")
    tabla_comparativa = crear_tabla_comparativa_metricas(metricas_dbscan, metricas_kmeans)
    print("\n" + tabla_comparativa.to_string(index=False))
    
    ruta_tabla_csv = DIRECTORIO_COMPARACIONES / 'tabla_comparativa.csv'
    tabla_comparativa.to_csv(ruta_tabla_csv, index=False, encoding='utf-8')
    print(f"\n✅ Tabla guardada: {ruta_tabla_csv.name}")
    
    generar_graficos_metricas(metricas_dbscan, metricas_kmeans)
    
    crear_grafico_rendimiento(metricas_dbscan, metricas_kmeans)
    
    print("\n🔍 Analizando resultados...")
    analisis = analizar_ganador(metricas_dbscan, metricas_kmeans)
    print(f"\n🏆 ALGORITMO GANADOR: {analisis['ganador']}")
    print(f"   Puntuación: DBSCAN {analisis['puntos_dbscan']}/3 - K-Means {analisis['puntos_kmeans']}/3")
    
    guardar_reporte_completo(tabla_comparativa, analisis, metricas_dbscan, metricas_kmeans)
    
    print("\n" + "=" * 100)
    print(" " * 30 + "✅ COMPARACIÓN COMPLETADA EXITOSAMENTE")
    print(" " * 20 + f"📁 Resultados guardados en: {DIRECTORIO_COMPARACIONES}")
    print("=" * 100)


if __name__ == "__main__":
    main()

