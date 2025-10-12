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

DIRECTORIO_BASE = Path(__file__).parent.parent
DIRECTORIO_COMPARACIONES = Path(__file__).parent

RUTAS = {
    'CBLOF': {
        'metricas_csv': DIRECTORIO_BASE / 'CBLOF (Cluster-Based Local Outlier Factor)' / 'metricas_CBLOF' / 'metrics.csv',
        'metricas_txt': DIRECTORIO_BASE / 'CBLOF (Cluster-Based Local Outlier Factor)' / 'metricas_CBLOF' / 'metrics.txt',
        'scores_csv': DIRECTORIO_BASE / 'CBLOF (Cluster-Based Local Outlier Factor)' / 'metricas_CBLOF' / 'anomaly_scores.csv',
        'anomalias_csv': DIRECTORIO_BASE / 'CBLOF (Cluster-Based Local Outlier Factor)' / 'metricas_CBLOF' / 'anomalies.csv',
        'grafica_scores': DIRECTORIO_BASE / 'CBLOF (Cluster-Based Local Outlier Factor)' / 'graficas_CBLOF' / 'anomaly_scores.png',
        'grafica_3d': DIRECTORIO_BASE / 'CBLOF (Cluster-Based Local Outlier Factor)' / 'graficas_CBLOF' / 'anomalies_3d.png',
    },
    'Isolation Forest': {
        'metricas_csv': DIRECTORIO_BASE / 'Isolation Forest' / 'metricas_IForest' / 'metrics.csv',
        'metricas_txt': DIRECTORIO_BASE / 'Isolation Forest' / 'metricas_IForest' / 'metrics.txt',
        'scores_csv': DIRECTORIO_BASE / 'Isolation Forest' / 'metricas_IForest' / 'anomaly_scores.csv',
        'anomalias_csv': DIRECTORIO_BASE / 'Isolation Forest' / 'metricas_IForest' / 'anomalies.csv',
        'grafica_scores': DIRECTORIO_BASE / 'Isolation Forest' / 'graficas_IForest' / 'anomaly_scores.png',
        'grafica_3d': DIRECTORIO_BASE / 'Isolation Forest' / 'graficas_IForest' / 'anomalies_3d.png',
    }
}


def verificar_archivos() -> bool:
    archivos_faltantes = []
    
    for algoritmo, rutas in RUTAS.items():
        for nombre_archivo, ruta in rutas.items():
            if not ruta.exists():
                archivos_faltantes.append(f"{algoritmo} - {nombre_archivo}: {ruta}")
    
    if archivos_faltantes:
        print("ERROR: Archivos faltantes:")
        for archivo in archivos_faltantes:
            print(f"  - {archivo}")
        print("\nAsegurate de ejecutar primero los algoritmos CBLOF e Isolation Forest.")
        return False
    
    return True


def cargar_metricas() -> Tuple[pd.DataFrame, pd.DataFrame]:
    metricas_cblof = pd.read_csv(RUTAS['CBLOF']['metricas_csv'])
    metricas_iforest = pd.read_csv(RUTAS['Isolation Forest']['metricas_csv'])
    return metricas_cblof, metricas_iforest


def crear_comparacion_imagenes_lado_a_lado() -> None:
    print("\nGenerando comparacion visual de graficas...")
    
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['CBLOF']['grafica_scores'])
    ax1.imshow(img1)
    ax1.set_title('CBLOF - Distribución de Scores de Anomalía', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['Isolation Forest']['grafica_scores'])
    ax2.imshow(img2)
    ax2.set_title('Isolation Forest - Distribución de Scores de Anomalía', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: CBLOF vs Isolation Forest (Distribución de Scores)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_scores = DIRECTORIO_COMPARACIONES / 'comparacion_visual_scores.png'
    plt.savefig(ruta_comparacion_scores, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Comparacion de scores guardada: {ruta_comparacion_scores.name}")
    
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['CBLOF']['grafica_3d'])
    ax1.imshow(img1)
    ax1.set_title('CBLOF - Detección de Anomalías 3D', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['Isolation Forest']['grafica_3d'])
    ax2.imshow(img2)
    ax2.set_title('Isolation Forest - Detección de Anomalías 3D', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: CBLOF vs Isolation Forest (Visualización 3D)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_3d = DIRECTORIO_COMPARACIONES / 'comparacion_visual_3d.png'
    plt.savefig(ruta_comparacion_3d, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Comparacion 3D guardada: {ruta_comparacion_3d.name}")


def crear_tabla_comparativa_metricas(metricas_cblof: pd.DataFrame, 
                                     metricas_iforest: pd.DataFrame) -> pd.DataFrame:
    tiempo_cblof = obtener_metrica_segura(metricas_cblof, 'tiempo_ejecucion_s', 0)
    tiempo_iforest = obtener_metrica_segura(metricas_iforest, 'tiempo_ejecucion_s', 0)
    memoria_cblof = obtener_metrica_segura(metricas_cblof, 'memoria_max_mb', 0)
    memoria_iforest = obtener_metrica_segura(metricas_iforest, 'memoria_max_mb', 0)
    
    comparacion = {
        'Métrica': [
            'Algoritmo',
            'Porcentaje de Anomalías (%)',
            'Separación de Scores (P95-P50)',
            'Score Promedio',
            'Número de Clusters (si aplica)',
            'Tiempo de Ejecución (s)',
            'Memoria Máxima (MB)'
        ],
        'CBLOF': [
            metricas_cblof['algoritmo'].values[0],
            f"{metricas_cblof['pct_anomalias'].values[0]:.2f}",
            f"{metricas_cblof['p95_minus_p50'].values[0]:.4f}",
            f"{metricas_cblof['mean_score'].values[0]:.4f}",
            metricas_cblof['n_clusters'].values[0] if not pd.isna(metricas_cblof['n_clusters'].values[0]) else 'N/A',
            f"{tiempo_cblof:.2f}" if tiempo_cblof > 0 else 'N/A',
            f"{memoria_cblof:.2f}" if memoria_cblof > 0 else 'N/A'
        ],
        'Isolation Forest': [
            metricas_iforest['algoritmo'].values[0],
            f"{metricas_iforest['pct_anomalias'].values[0]:.2f}",
            f"{metricas_iforest['p95_minus_p50'].values[0]:.4f}",
            f"{metricas_iforest['mean_score'].values[0]:.4f}",
            metricas_iforest['n_clusters'].values[0] if not pd.isna(metricas_iforest['n_clusters'].values[0]) else 'N/A',
            f"{tiempo_iforest:.2f}" if tiempo_iforest > 0 else 'N/A',
            f"{memoria_iforest:.2f}" if memoria_iforest > 0 else 'N/A'
        ]
    }
    
    df_comparacion = pd.DataFrame(comparacion)
    return df_comparacion


def generar_graficos_metricas(metricas_cblof: pd.DataFrame, 
                              metricas_iforest: pd.DataFrame) -> None:
    print("\nGenerando graficos comparativos de metricas...")
    
    metricas_nombres = ['Porcentaje de\nAnomalías (%)', 'Separación\nde Scores\n(P95-P50)', 'Score\nPromedio']
    
    cblof_values = [
        metricas_cblof['pct_anomalias'].values[0],
        metricas_cblof['p95_minus_p50'].values[0],
        metricas_cblof['mean_score'].values[0]
    ]
    
    iforest_values = [
        metricas_iforest['pct_anomalias'].values[0],
        metricas_iforest['p95_minus_p50'].values[0],
        metricas_iforest['mean_score'].values[0]
    ]
    
    x = np.arange(len(metricas_nombres))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, cblof_values, width, label='CBLOF', 
                    color='mediumseagreen', alpha=0.8, edgecolor='black', linewidth=1.5)
    rects2 = ax.bar(x + width/2, iforest_values, width, label='Isolation Forest', 
                    color='mediumpurple', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Métricas de Detección de Anomalías', fontsize=13, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=13, fontweight='bold')
    ax.set_title('Comparación de Métricas: CBLOF vs Isolation Forest', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nombres, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    
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
    plt.savefig(ruta_grafico, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Grafico de barras guardado: {ruta_grafico.name}")
    
    crear_graficos_individuales(metricas_cblof, metricas_iforest)


def crear_graficos_individuales(metricas_cblof: pd.DataFrame, 
                                metricas_iforest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    
    algoritmos = ['CBLOF', 'Isolation Forest']
    porcentajes = [
        metricas_cblof['pct_anomalias'].values[0],
        metricas_iforest['pct_anomalias'].values[0]
    ]
    
    colors = ['mediumseagreen', 'mediumpurple']
    bars = ax.barh(algoritmos, porcentajes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Porcentaje de Anomalías Detectadas (%)', fontsize=13, fontweight='bold')
    ax.set_title('Comparación de Porcentaje de Anomalías Detectadas', fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
    
    for i, (bar, valor) in enumerate(zip(bars, porcentajes)):
        ax.text(valor + 0.3, i, f'{valor:.2f}%', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    ruta_porcentajes = DIRECTORIO_COMPARACIONES / 'comparacion_porcentaje_anomalias.png'
    plt.savefig(ruta_porcentajes, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Grafico de porcentajes guardado: {ruta_porcentajes.name}")
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    separaciones = [
        metricas_cblof['p95_minus_p50'].values[0],
        metricas_iforest['p95_minus_p50'].values[0]
    ]
    
    bars = ax.barh(algoritmos, separaciones, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Separación de Scores (P95 - P50)', fontsize=13, fontweight='bold')
    ax.set_title('Comparación de Separación entre Anomalías y Datos Normales', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
    
    for i, (bar, valor) in enumerate(zip(bars, separaciones)):
        ax.text(valor + 0.05, i, f'{valor:.4f}', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    ruta_separacion = DIRECTORIO_COMPARACIONES / 'comparacion_separacion_scores.png'
    plt.savefig(ruta_separacion, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Grafico de separacion guardado: {ruta_separacion.name}")


def crear_grafico_rendimiento(metricas_cblof: pd.DataFrame, metricas_iforest: pd.DataFrame) -> None:
    """
    Genera gráficos comparativos de rendimiento (tiempo y memoria).
    """
    print("\nGenerando graficos de rendimiento...")
    
    tiempo_cblof = obtener_metrica_segura(metricas_cblof, 'tiempo_ejecucion_s', 0)
    tiempo_iforest = obtener_metrica_segura(metricas_iforest, 'tiempo_ejecucion_s', 0)
    memoria_cblof = obtener_metrica_segura(metricas_cblof, 'memoria_max_mb', 0)
    memoria_iforest = obtener_metrica_segura(metricas_iforest, 'memoria_max_mb', 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    algoritmos = ['CBLOF', 'Isolation Forest']
    tiempos = [tiempo_cblof, tiempo_iforest]
    colores = ['mediumseagreen', 'mediumpurple']
    
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
    
    memorias = [memoria_cblof, memoria_iforest]
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
    
    plt.suptitle('Métricas de Rendimiento: CBLOF vs Isolation Forest', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    ruta_rendimiento = DIRECTORIO_COMPARACIONES / 'comparacion_rendimiento.png'
    plt.savefig(ruta_rendimiento, dpi=200, bbox_inches='tight')
    plt.close("all")
    print(f"[OK] Grafico de rendimiento guardado: {ruta_rendimiento.name}")


def analizar_ganador(metricas_cblof: pd.DataFrame, metricas_iforest: pd.DataFrame) -> Dict[str, Any]:
    analisis = {}
    
    sep_cblof = metricas_cblof['p95_minus_p50'].values[0]
    sep_iforest = metricas_iforest['p95_minus_p50'].values[0]
    
    if sep_cblof > sep_iforest:
        analisis['mejor_separacion'] = 'CBLOF'
        peso_separacion = 2
    elif sep_iforest > sep_cblof:
        analisis['mejor_separacion'] = 'Isolation Forest'
        peso_separacion = 2
    else:
        analisis['mejor_separacion'] = 'Empate'
        peso_separacion = 0
    
    pct_cblof = metricas_cblof['pct_anomalias'].values[0]
    pct_iforest = metricas_iforest['pct_anomalias'].values[0]
    
    if abs(pct_cblof - pct_iforest) > 2:
        if pct_cblof > pct_iforest:
            analisis['mayor_deteccion'] = 'CBLOF'
            peso_deteccion = 1
        else:
            analisis['mayor_deteccion'] = 'Isolation Forest'
            peso_deteccion = 1
    else:
        analisis['mayor_deteccion'] = 'Empate (similar)'
        peso_deteccion = 0
    
    puntos_cblof = 0
    puntos_iforest = 0
    
    if analisis['mejor_separacion'] == 'CBLOF':
        puntos_cblof += peso_separacion
    elif analisis['mejor_separacion'] == 'Isolation Forest':
        puntos_iforest += peso_separacion
    
    if analisis['mayor_deteccion'] == 'CBLOF':
        puntos_cblof += peso_deteccion
    elif analisis['mayor_deteccion'] == 'Isolation Forest':
        puntos_iforest += peso_deteccion
    
    if puntos_cblof > puntos_iforest:
        analisis['ganador'] = 'CBLOF'
    elif puntos_iforest > puntos_cblof:
        analisis['ganador'] = 'Isolation Forest'
    else:
        analisis['ganador'] = 'Empate'
    
    analisis['puntos_cblof'] = puntos_cblof
    analisis['puntos_iforest'] = puntos_iforest
    analisis['pct_anomalias_cblof'] = float(pct_cblof)
    analisis['pct_anomalias_iforest'] = float(pct_iforest)
    analisis['separacion_cblof'] = float(sep_cblof)
    analisis['separacion_iforest'] = float(sep_iforest)
    
    return analisis


def guardar_reporte_completo(tabla_comparativa: pd.DataFrame, analisis: Dict[str, Any],
                             metricas_cblof: pd.DataFrame, metricas_iforest: pd.DataFrame) -> None:
    print("\nGenerando reporte completo...")
    
    ruta_reporte = DIRECTORIO_COMPARACIONES / 'REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt'
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 18 + "REPORTE DE COMPARACIÓN DE ALGORITMOS DE DETECCIÓN DE ANOMALÍAS\n")
        f.write(" " * 35 + "CBLOF vs Isolation Forest\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("1. TABLA COMPARATIVA DE MÉTRICAS\n")
        f.write("-" * 100 + "\n")
        f.write(tabla_comparativa.to_string(index=False))
        f.write("\n\n")
        
        f.write("2. ANÁLISIS DE RESULTADOS POR MÉTRICA\n")
        f.write("-" * 100 + "\n")
        f.write(f"✓ Mejor Separación de Scores (P95-P50): {analisis['mejor_separacion']}\n")
        f.write(f"  Separación CBLOF: {analisis['separacion_cblof']:.4f}\n")
        f.write(f"  Separación Isolation Forest: {analisis['separacion_iforest']:.4f}\n\n")
        
        f.write(f"✓ Detección de Anomalías: {analisis['mayor_deteccion']}\n")
        f.write(f"  CBLOF: {analisis['pct_anomalias_cblof']:.2f}%\n")
        f.write(f"  Isolation Forest: {analisis['pct_anomalias_iforest']:.2f}%\n\n")
        
        f.write("3. PUNTUACIÓN FINAL (Separación pesa doble)\n")
        f.write("-" * 100 + "\n")
        f.write(f"CBLOF: {analisis['puntos_cblof']} puntos\n")
        f.write(f"Isolation Forest: {analisis['puntos_iforest']} puntos\n\n")
        
        f.write("4. ALGORITMO GANADOR\n")
        f.write("-" * 100 + "\n")
        f.write(f"🏆 GANADOR: {analisis['ganador']}\n\n")
        
        if analisis['ganador'] == 'CBLOF':
            f.write("CBLOF demostró mejor rendimiento, especialmente en la separación de anomalías.\n")
        elif analisis['ganador'] == 'Isolation Forest':
            f.write("Isolation Forest demostró mejor rendimiento, especialmente en la separación de anomalías.\n")
        else:
            f.write("Ambos algoritmos mostraron rendimiento equivalente.\n")
        
        f.write("\n")
        
        f.write("5. CARACTERÍSTICAS Y CONSIDERACIONES\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("CBLOF (Cluster-Based Local Outlier Factor):\n")
        f.write("  Ventajas:\n")
        f.write("    ✓ Utiliza clustering para contexto local\n")
        f.write("    ✓ Bueno para detectar anomalías locales dentro de clusters\n")
        f.write("    ✓ Considera la densidad y estructura de los datos\n")
        f.write("    ✓ Puede identificar diferentes tipos de anomalías\n")
        f.write("  Desventajas:\n")
        f.write("    ✗ Requiere especificar número de clusters\n")
        f.write("    ✗ Más sensible a la elección de parámetros\n")
        f.write("    ✗ Puede ser más lento en grandes datasets\n")
        f.write("    ✗ Complejidad computacional mayor\n\n")
        
        f.write("Isolation Forest:\n")
        f.write("  Ventajas:\n")
        f.write("    ✓ Basado en ensemble de árboles de decisión\n")
        f.write("    ✓ Rápido y escalable a grandes volúmenes de datos\n")
        f.write("    ✓ Funciona bien con datos de alta dimensionalidad\n")
        f.write("    ✓ No requiere clustering previo\n")
        f.write("  Desventajas:\n")
        f.write("    ✗ Puede ser sensible al parámetro de contaminación\n")
        f.write("    ✗ Menos interpretable que métodos basados en distancia\n")
        f.write("    ✗ Puede tener problemas con distribuciones no uniformes\n\n")
        
        f.write("6. INTERPRETACIÓN DE MÉTRICAS CLAVE\n")
        f.write("-" * 100 + "\n")
        f.write("Separación de Scores (P95-P50):\n")
        f.write("  • Mide la diferencia entre el percentil 95 y el percentil 50 de scores\n")
        f.write("  • Mayor valor = mejor separación entre anomalías y datos normales\n")
        f.write("  • Métrica CRÍTICA para confiabilidad de detecciones\n")
        f.write("  • Indica qué tan claramente el algoritmo distingue anomalías\n\n")
        
        f.write("Porcentaje de Anomalías:\n")
        f.write("  • Porcentaje de puntos clasificados como anomalías\n")
        f.write("  • Debe evaluarse en contexto del dominio específico\n")
        f.write("  • No necesariamente 'más es mejor'\n")
        f.write("  • Ideal: coincide con la tasa real esperada de anomalías\n\n")
        
        tiempo_cblof = obtener_metrica_segura(metricas_cblof, 'tiempo_ejecucion_s', 0)
        tiempo_iforest = obtener_metrica_segura(metricas_iforest, 'tiempo_ejecucion_s', 0)
        memoria_cblof = obtener_metrica_segura(metricas_cblof, 'memoria_max_mb', 0)
        memoria_iforest = obtener_metrica_segura(metricas_iforest, 'memoria_max_mb', 0)
        
        if tiempo_cblof > 0 or tiempo_iforest > 0 or memoria_cblof > 0 or memoria_iforest > 0:
            f.write("Métricas de Rendimiento:\n")
            if tiempo_cblof > 0 and tiempo_iforest > 0:
                f.write(f"  Tiempo de Ejecución:\n")
                f.write(f"    CBLOF: {tiempo_cblof:.2f}s\n")
                f.write(f"    Isolation Forest: {tiempo_iforest:.2f}s\n")
                mas_rapido = "Isolation Forest" if tiempo_iforest < tiempo_cblof else "CBLOF"
                f.write(f"    ⚡ {mas_rapido} es más rápido\n\n")
            
            if memoria_cblof > 0 and memoria_iforest > 0:
                f.write(f"  Uso de Memoria Máxima:\n")
                f.write(f"    CBLOF: {memoria_cblof:.2f} MB\n")
                f.write(f"    Isolation Forest: {memoria_iforest:.2f} MB\n")
                menos_memoria = "Isolation Forest" if memoria_iforest < memoria_cblof else "CBLOF"
                f.write(f"    [MEMORIA] {menos_memoria} usa menos memoria\n\n")
        
        f.write("7. ARCHIVOS GENERADOS EN ESTA COMPARACIÓN\n")
        f.write("-" * 100 + "\n")
        f.write("  • comparacion_visual_scores.png - Comparación de distribución de scores\n")
        f.write("  • comparacion_visual_3d.png - Comparación de visualización 3D\n")
        f.write("  • comparacion_metricas_barras.png - Gráfico de barras comparativo\n")
        f.write("  • comparacion_porcentaje_anomalias.png - Comparación de porcentajes\n")
        f.write("  • comparacion_separacion_scores.png - Comparación de separación\n")
        f.write("  • comparacion_rendimiento.png - Gráfico de rendimiento (tiempo y memoria)\n")
        f.write("  • tabla_comparativa.csv - Tabla de métricas en formato CSV\n")
        f.write("  • REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt - Este archivo\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("Fecha de generación: Octubre 2025\n")
        f.write("=" * 100 + "\n")
    
    print(f"[OK] Reporte completo guardado: {ruta_reporte.name}")


def validar_scores_normalizados() -> None:
    """
    Valida que los anomaly scores estén normalizados en el rango [0, 1].
    """
    for algoritmo, rutas in RUTAS.items():
        scores_csv = rutas['scores_csv']
        if scores_csv.exists():
            try:
                df_scores = pd.read_csv(scores_csv)
                if 'anomaly_score' in df_scores.columns:
                    scores = df_scores['anomaly_score'].values
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    
                    if min_score < -0.01 or max_score > 1.01:
                        print(f"ADVERTENCIA: Scores de {algoritmo} NO estan completamente normalizados")
                        print(f"   Rango: [{min_score:.4f}, {max_score:.4f}]")
                    else:
                        print(f"[OK] {algoritmo}: Scores normalizados correctamente [0, 1]")
            except Exception as e:
                print(f"Error al validar scores de {algoritmo}: {e}")


def main():
    print("=" * 100)
    print(" " * 18 + "COMPARACIÓN DE ALGORITMOS DE DETECCIÓN DE ANOMALÍAS")
    print(" " * 35 + "CBLOF vs Isolation Forest")
    print("=" * 100)
    
    if not verificar_archivos():
        return
    
    print("\n[OK] Todos los archivos necesarios estan presentes.")
    
    print("\nCargando metricas...")
    metricas_cblof, metricas_iforest = cargar_metricas()
    print("[OK] Metricas cargadas correctamente.")
    
    crear_comparacion_imagenes_lado_a_lado()
    
    print("\nGenerando tabla comparativa...")
    tabla_comparativa = crear_tabla_comparativa_metricas(metricas_cblof, metricas_iforest)
    print("\n" + tabla_comparativa.to_string(index=False))
    
    ruta_tabla_csv = DIRECTORIO_COMPARACIONES / 'tabla_comparativa.csv'
    tabla_comparativa.to_csv(ruta_tabla_csv, index=False, encoding='utf-8')
    print(f"\n[OK] Tabla guardada: {ruta_tabla_csv.name}")
    
    generar_graficos_metricas(metricas_cblof, metricas_iforest)
    
    crear_grafico_rendimiento(metricas_cblof, metricas_iforest)
    
    print("\nValidando normalizacion de scores...")
    validar_scores_normalizados()
    
    print("\nAnalizando resultados...")
    analisis = analizar_ganador(metricas_cblof, metricas_iforest)
    print(f"\nALGORITMO GANADOR: {analisis['ganador']}")
    print(f"   Puntuacion: CBLOF {analisis['puntos_cblof']} - Isolation Forest {analisis['puntos_iforest']}")
    
    guardar_reporte_completo(tabla_comparativa, analisis, metricas_cblof, metricas_iforest)
    
    print("\n" + "=" * 100)
    print(" " * 30 + "[OK] COMPARACION COMPLETADA EXITOSAMENTE")
    print(" " * 20 + f"Resultados guardados en: {DIRECTORIO_COMPARACIONES}")
    print("=" * 100)


if __name__ == "__main__":
    main()