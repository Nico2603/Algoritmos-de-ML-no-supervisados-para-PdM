import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

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

# Paso 1: Configuración de rutas
DIRECTORIO_BASE = Path(__file__).parent.parent
DIRECTORIO_COMPARACIONES = Path(__file__).parent

RUTAS = {
    'DBSCAN': {
        'metricas_csv': DIRECTORIO_BASE / 'DBSCAN' / 'metricas_DBSCAN' / 'metrics.csv',
        'metricas_txt': DIRECTORIO_BASE / 'DBSCAN' / 'metricas_DBSCAN' / 'metrics.txt',
        'scores_csv': DIRECTORIO_BASE / 'DBSCAN' / 'metricas_DBSCAN' / 'anomaly_scores.csv',
        'grafica_3d': DIRECTORIO_BASE / 'DBSCAN' / 'graficas_DBSCAN' / 'clusters_3d_pca.png',
    },
    'K-Means': {
        'metricas_csv': DIRECTORIO_BASE / 'K-means' / 'metricas_KMeans' / 'metrics.csv',
        'metricas_txt': DIRECTORIO_BASE / 'K-means' / 'metricas_KMeans' / 'metrics.txt',
        'scores_csv': DIRECTORIO_BASE / 'K-means' / 'metricas_KMeans' / 'anomaly_scores.csv',
        'grafica_3d': DIRECTORIO_BASE / 'K-means' / 'graficas_KMeans' / 'clusters_3d_pca.png',
    }
}

# Columnas esperadas en metrics.csv
COLUMNAS_ESPERADAS = [
    'algoritmo', 'params_json', 'n_clusters', 'silhouette_score',
    'calinski_harabasz_score', 'davies_bouldin_score', 'pct_anomalias',
    'p95_minus_p50', 'mean_score', 'tiempo_ejecucion_s', 'memoria_max_mb'
]


def verificar_archivos() -> bool:
    archivos_faltantes = []
    
    for algoritmo, rutas in RUTAS.items():
        for nombre_archivo, ruta in rutas.items():
            if not ruta.exists():
                archivos_faltantes.append(f"{algoritmo} - {nombre_archivo}: {ruta}")
    
    if archivos_faltantes:
        print("[ERROR] Archivos faltantes:")
        for archivo in archivos_faltantes:
            print(f"  - {archivo}")
        print("\n[ALERTA] Asegurate de ejecutar primero los algoritmos DBSCAN y K-Means.")
        return False
    
    return True


def validar_columnas_metricas(df: pd.DataFrame, algoritmo: str) -> bool:
    """
    Valida que el DataFrame tenga las columnas esperadas.
    
    Args:
        df: DataFrame con las métricas
        algoritmo: Nombre del algoritmo para mensajes
        
    Returns:
        True si todas las columnas están presentes, False en caso contrario
    """
    columnas_faltantes = [col for col in COLUMNAS_ESPERADAS if col not in df.columns]
    
    if columnas_faltantes:
        print(f"[ERROR] Columnas faltantes en metrics.csv de {algoritmo}:")
        for col in columnas_faltantes:
            print(f"  - {col}")
        return False
    
    return True

def cargar_metricas() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y valida métricas de ambos algoritmos"""
    metricas_dbscan = pd.read_csv(RUTAS['DBSCAN']['metricas_csv'])
    metricas_kmeans = pd.read_csv(RUTAS['K-Means']['metricas_csv'])
    
    # Paso 2: Validar columnas esperadas
    if not validar_columnas_metricas(metricas_dbscan, 'DBSCAN'):
        print("[ERROR] El archivo metrics.csv de DBSCAN no tiene el formato esperado.")
        exit(1)
    
    if not validar_columnas_metricas(metricas_kmeans, 'K-Means'):
        print("[ERROR] El archivo metrics.csv de K-Means no tiene el formato esperado.")
        exit(1)
    
    return metricas_dbscan, metricas_kmeans


def crear_comparacion_imagenes_lado_a_lado() -> None:
    print("\nGenerando comparacion visual de graficas...")
    
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['DBSCAN']['grafica_3d'])
    ax1.imshow(img1)
    ax1.set_title('DBSCAN - Clustering 3D con PCA', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['K-Means']['grafica_3d'])
    ax2.imshow(img2)
    ax2.set_title('K-Means - Clustering 3D con PCA', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: DBSCAN vs K-Means (3D)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_3d = DIRECTORIO_COMPARACIONES / 'comparacion_visual_3d.png'
    plt.savefig(ruta_comparacion_3d, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Comparación 3D guardada: {ruta_comparacion_3d.name}")




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
    print("\nGenerando graficos comparativos de metricas...")
    
    # Extraer valores reales de las métricas
    sil_dbscan = obtener_metrica_segura(metricas_dbscan, 'silhouette_score', -999)
    sil_kmeans = obtener_metrica_segura(metricas_kmeans, 'silhouette_score', -999)
    
    cal_dbscan = obtener_metrica_segura(metricas_dbscan, 'calinski_harabasz_score', -999)
    cal_kmeans = obtener_metrica_segura(metricas_kmeans, 'calinski_harabasz_score', -999)
    
    dav_dbscan = obtener_metrica_segura(metricas_dbscan, 'davies_bouldin_score', -999)
    dav_kmeans = obtener_metrica_segura(metricas_kmeans, 'davies_bouldin_score', -999)
    
    # Verificar número de clusters
    n_clusters_dbscan = metricas_dbscan['n_clusters'].values[0] if not pd.isna(metricas_dbscan['n_clusters'].values[0]) else 0
    n_clusters_kmeans = metricas_kmeans['n_clusters'].values[0] if not pd.isna(metricas_kmeans['n_clusters'].values[0]) else 0
    
    # Detectar métricas N/A (cuando hay menos de 2 clusters)
    dbscan_na = (sil_dbscan <= -900 or n_clusters_dbscan < 2)
    kmeans_na = (sil_kmeans <= -900 or n_clusters_kmeans < 2)
    
    # Crear figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    algoritmos = ['DBSCAN', 'K-Means']
    colores_base = ['steelblue', 'coral']
    
    # Subplot 1: Silhouette Score (rango [-1, 1], mayor es mejor)
    if not dbscan_na and not kmeans_na:
        sil_values = [sil_dbscan, sil_kmeans]
        bars1 = ax1.bar(algoritmos, sil_values, color=colores_base, alpha=0.8, edgecolor='black', linewidth=1.5)
    elif dbscan_na and not kmeans_na:
        bars1 = ax1.bar(['K-Means'], [sil_kmeans], color=['coral'], alpha=0.8, edgecolor='black', linewidth=1.5)
        # Añadir barra rayada para N/A
        ax1.bar(['DBSCAN'], [0], color='lightgray', alpha=0.3, edgecolor='red', linewidth=2, linestyle='--', hatch='//')
        ax1.text(0, -0.5, f'N/A\n({int(n_clusters_dbscan)} cluster)', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    
    ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax1.set_title('Silhouette Score\n(Rango: -1 a 1, Mayor = Mejor)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylim(-1, 1)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_xticks(range(len(algoritmos)))
    ax1.set_xticklabels(algoritmos)
    
    if not dbscan_na:
        ax1.text(0, sil_dbscan, f'{sil_dbscan:.4f}', ha='center', va='bottom' if sil_dbscan > 0 else 'top', fontsize=11, fontweight='bold')
    if not kmeans_na:
        pos_x = 1 if not dbscan_na else 0
        ax1.text(pos_x, sil_kmeans, f'{sil_kmeans:.4f}', ha='center', va='bottom' if sil_kmeans > 0 else 'top', fontsize=11, fontweight='bold')
    
    # Subplot 2: Calinski-Harabasz Score (mayor es mejor)
    if not dbscan_na and not kmeans_na:
        cal_values = [cal_dbscan, cal_kmeans]
        bars2 = ax2.bar(algoritmos, cal_values, color=colores_base, alpha=0.8, edgecolor='black', linewidth=1.5)
    elif dbscan_na and not kmeans_na:
        bars2 = ax2.bar(['K-Means'], [cal_kmeans], color=['coral'], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.bar(['DBSCAN'], [cal_kmeans * 0.05], color='lightgray', alpha=0.3, edgecolor='red', linewidth=2, linestyle='--', hatch='//')
        ax2.text(0, cal_kmeans * 0.025, f'N/A\n({int(n_clusters_dbscan)} cluster)', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    
    ax2.set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
    ax2.set_title('Calinski-Harabasz Score\n(Mayor = Mejor)', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_xticks(range(len(algoritmos)))
    ax2.set_xticklabels(algoritmos)
    
    if not dbscan_na:
        ax2.text(0, cal_dbscan, f'{cal_dbscan:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    if not kmeans_na:
        pos_x = 1 if not dbscan_na else 0
        ax2.text(pos_x, cal_kmeans, f'{cal_kmeans:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Subplot 3: Davies-Bouldin Index (menor es mejor)
    if not dbscan_na and not kmeans_na:
        dav_values = [dav_dbscan, dav_kmeans]
        bars3 = ax3.bar(algoritmos, dav_values, color=colores_base, alpha=0.8, edgecolor='black', linewidth=1.5)
    elif dbscan_na and not kmeans_na:
        bars3 = ax3.bar(['K-Means'], [dav_kmeans], color=['coral'], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.bar(['DBSCAN'], [dav_kmeans * 0.5], color='lightgray', alpha=0.3, edgecolor='red', linewidth=2, linestyle='--', hatch='//')
        ax3.text(0, dav_kmeans * 0.25, f'N/A\n({int(n_clusters_dbscan)} cluster)', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    
    ax3.set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
    ax3.set_title('Davies-Bouldin Index\n(Menor = Mejor)', fontsize=13, fontweight='bold', pad=12)
    ax3.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax3.set_xticks(range(len(algoritmos)))
    ax3.set_xticklabels(algoritmos)
    
    if not dbscan_na:
        ax3.text(0, dav_dbscan, f'{dav_dbscan:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    if not kmeans_na:
        pos_x = 1 if not dbscan_na else 0
        ax3.text(pos_x, dav_kmeans, f'{dav_kmeans:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Título con nota explicativa si hay N/A
    titulo = 'Comparación de Métricas de Clustering: DBSCAN vs K-Means\n(Valores Reales Sin Normalizar)'
    if dbscan_na or kmeans_na:
        titulo += '\n⚠ N/A indica que el algoritmo no pudo calcular métricas (< 2 clusters)'
    
    plt.suptitle(titulo, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    ruta_grafico = DIRECTORIO_COMPARACIONES / 'comparacion_metricas_barras.png'
    plt.savefig(ruta_grafico, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Gráfico de barras guardado: {ruta_grafico.name}")


def crear_grafico_rendimiento(metricas_dbscan: pd.DataFrame, metricas_kmeans: pd.DataFrame) -> None:
    """
    Genera gráficos comparativos de rendimiento (tiempo y memoria).
    """
    print("\n[TIEMPO]  Generando gráficos de rendimiento...")
    
    tiempo_dbscan = obtener_metrica_segura(metricas_dbscan, 'tiempo_ejecucion_s', 0)
    tiempo_kmeans = obtener_metrica_segura(metricas_kmeans, 'tiempo_ejecucion_s', 0)
    memoria_dbscan = obtener_metrica_segura(metricas_dbscan, 'memoria_max_mb', 0)
    memoria_kmeans = obtener_metrica_segura(metricas_kmeans, 'memoria_max_mb', 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    algoritmos = ['DBSCAN', 'K-Means']
    tiempos = [tiempo_dbscan, tiempo_kmeans]
    colores = ['steelblue', 'coral']
    
    bars1 = ax1.bar(algoritmos, tiempos, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Tiempo de Ejecución', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
    for bar, tiempo in zip(bars1, tiempos):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{tiempo:.2f}s',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    memorias = [memoria_dbscan, memoria_kmeans]
    bars2 = ax2.bar(algoritmos, memorias, color=colores, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Memoria (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Uso de Memoria', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
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
    plt.savefig(ruta_rendimiento, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Gráfico de rendimiento guardado: {ruta_rendimiento.name}")


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
    print("\nGenerando reporte completo...")
    
    ruta_reporte = DIRECTORIO_COMPARACIONES / 'REPORTE_COMPARACION_CLUSTERING.txt'
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 25 + "REPORTE DE COMPARACIÓN DE ALGORITMOS DE CLUSTERING\n")
        f.write(" " * 40 + "DBSCAN vs K-Means\n")
        f.write("=" * 100 + "\n\n")
        
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
            f.write(f"    [MEMORIA] {menos_memoria} usa menos memoria\n\n")
        
        f.write("7. ARCHIVOS GENERADOS EN ESTA COMPARACIÓN\n")
        f.write("-" * 100 + "\n")
        f.write("  • comparacion_visual_3d.png - Comparación lado a lado de gráficas 3D\n")
        f.write("  • comparacion_metricas_barras.png - Gráfico de barras comparativo\n")
        f.write("  • comparacion_rendimiento.png - Gráfico de rendimiento (tiempo y memoria)\n")
        f.write("  • tabla_comparativa.csv - Tabla de métricas en formato CSV\n")
        f.write("  • REPORTE_COMPARACION_CLUSTERING.txt - Este archivo\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("Fecha de generación: Octubre 2025\n")
        f.write("=" * 100 + "\n")
    
    print(f"[OK] Reporte completo guardado: {ruta_reporte.name}")


def main():
    print("=" * 100)
    print(" " * 25 + "COMPARACIÓN DE ALGORITMOS DE CLUSTERING")
    print(" " * 40 + "DBSCAN vs K-Means")
    print("=" * 100)
    
    if not verificar_archivos():
        return
    
    print("\n[OK] Todos los archivos necesarios estan presentes.")
    
    print("\nCargando metricas...")
    metricas_dbscan, metricas_kmeans = cargar_metricas()
    print("[OK] Metricas cargadas correctamente.")
    
    crear_comparacion_imagenes_lado_a_lado()
    
    print("\nGenerando tabla comparativa...")
    tabla_comparativa = crear_tabla_comparativa_metricas(metricas_dbscan, metricas_kmeans)
    print("\n" + tabla_comparativa.to_string(index=False))
    
    ruta_tabla_csv = DIRECTORIO_COMPARACIONES / 'tabla_comparativa.csv'
    tabla_comparativa.to_csv(ruta_tabla_csv, index=False, encoding='utf-8')
    print(f"\n[OK] Tabla guardada: {ruta_tabla_csv.name}")
    
    generar_graficos_metricas(metricas_dbscan, metricas_kmeans)
    
    crear_grafico_rendimiento(metricas_dbscan, metricas_kmeans)
    
    print("\nAnalizando resultados...")
    analisis = analizar_ganador(metricas_dbscan, metricas_kmeans)
    print(f"\nALGORITMO GANADOR: {analisis['ganador']}")
    print(f"   Puntuacion: DBSCAN {analisis['puntos_dbscan']}/3 - K-Means {analisis['puntos_kmeans']}/3")
    
    guardar_reporte_completo(tabla_comparativa, analisis, metricas_dbscan, metricas_kmeans)
    
    print("\n" + "=" * 100)
    print(" " * 30 + "[OK] COMPARACION COMPLETADA EXITOSAMENTE")
    print(" " * 20 + f"Resultados guardados en: {DIRECTORIO_COMPARACIONES}")
    print("=" * 100)


if __name__ == "__main__":
    main()