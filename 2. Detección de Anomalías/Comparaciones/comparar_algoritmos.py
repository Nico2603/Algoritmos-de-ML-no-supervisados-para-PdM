"""
Script de Comparación de Algoritmos de Detección de Anomalías
CBLOF vs Isolation Forest

Este script realiza una comparación exhaustiva de los algoritmos de detección de anomalías
generando visualizaciones comparativas, métricas lado a lado y análisis detallado.

Autor: Sistema de Análisis de Detección de Anomalías
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configuración
DIRECTORIO_BASE = Path(__file__).parent.parent
DIRECTORIO_COMPARACIONES = Path(__file__).parent

# Rutas de datos
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
    """Verifica que todos los archivos necesarios existan."""
    archivos_faltantes = []
    
    for algoritmo, rutas in RUTAS.items():
        for nombre_archivo, ruta in rutas.items():
            if not ruta.exists():
                archivos_faltantes.append(f"{algoritmo} - {nombre_archivo}: {ruta}")
    
    if archivos_faltantes:
        print("❌ ERROR: Archivos faltantes:")
        for archivo in archivos_faltantes:
            print(f"  - {archivo}")
        print("\n⚠️  Asegúrate de ejecutar primero los algoritmos CBLOF e Isolation Forest.")
        return False
    
    return True


def cargar_metricas() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga las métricas de ambos algoritmos."""
    metricas_cblof = pd.read_csv(RUTAS['CBLOF']['metricas_csv'])
    metricas_iforest = pd.read_csv(RUTAS['Isolation Forest']['metricas_csv'])
    return metricas_cblof, metricas_iforest


def crear_comparacion_imagenes_lado_a_lado() -> None:
    """Crea una comparación visual de las imágenes generadas lado a lado."""
    print("\n📊 Generando comparación visual de gráficas...")
    
    # Comparación de distribución de scores
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    # CBLOF scores
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['CBLOF']['grafica_scores'])
    ax1.imshow(img1)
    ax1.set_title('CBLOF - Distribución de Scores de Anomalía', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Isolation Forest scores
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['Isolation Forest']['grafica_scores'])
    ax2.imshow(img2)
    ax2.set_title('Isolation Forest - Distribución de Scores de Anomalía', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: CBLOF vs Isolation Forest (Distribución de Scores)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_scores = DIRECTORIO_COMPARACIONES / 'comparacion_visual_scores.png'
    plt.savefig(ruta_comparacion_scores, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparación de scores guardada: {ruta_comparacion_scores.name}")
    
    # Comparación 3D de anomalías
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, figure=fig)
    
    # CBLOF 3D
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = mpimg.imread(RUTAS['CBLOF']['grafica_3d'])
    ax1.imshow(img1)
    ax1.set_title('CBLOF - Detección de Anomalías 3D', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Isolation Forest 3D
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = mpimg.imread(RUTAS['Isolation Forest']['grafica_3d'])
    ax2.imshow(img2)
    ax2.set_title('Isolation Forest - Detección de Anomalías 3D', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    
    plt.suptitle('Comparación Visual: CBLOF vs Isolation Forest (Visualización 3D)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    ruta_comparacion_3d = DIRECTORIO_COMPARACIONES / 'comparacion_visual_3d.png'
    plt.savefig(ruta_comparacion_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparación 3D guardada: {ruta_comparacion_3d.name}")


def crear_tabla_comparativa_metricas(metricas_cblof: pd.DataFrame, 
                                     metricas_iforest: pd.DataFrame) -> pd.DataFrame:
    """Crea tabla comparativa de métricas."""
    comparacion = {
        'Métrica': [
            'Algoritmo',
            'Porcentaje de Anomalías (%)',
            'Separación de Scores (P95-P50)',
            'Score Promedio',
            'Número de Clusters (si aplica)'
        ],
        'CBLOF': [
            metricas_cblof['algoritmo'].values[0],
            f"{metricas_cblof['pct_anomalias'].values[0]:.2f}",
            f"{metricas_cblof['p95_minus_p50'].values[0]:.4f}",
            f"{metricas_cblof['mean_score'].values[0]:.4f}",
            metricas_cblof['n_clusters'].values[0] if not pd.isna(metricas_cblof['n_clusters'].values[0]) else 'N/A'
        ],
        'Isolation Forest': [
            metricas_iforest['algoritmo'].values[0],
            f"{metricas_iforest['pct_anomalias'].values[0]:.2f}",
            f"{metricas_iforest['p95_minus_p50'].values[0]:.4f}",
            f"{metricas_iforest['mean_score'].values[0]:.4f}",
            metricas_iforest['n_clusters'].values[0] if not pd.isna(metricas_iforest['n_clusters'].values[0]) else 'N/A'
        ]
    }
    
    df_comparacion = pd.DataFrame(comparacion)
    return df_comparacion


def generar_graficos_metricas(metricas_cblof: pd.DataFrame, 
                              metricas_iforest: pd.DataFrame) -> None:
    """Genera gráficos comparativos de métricas."""
    print("\n📈 Generando gráficos comparativos de métricas...")
    
    # Gráfico de barras comparativo
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
    
    # Gráficos individuales de porcentaje y separación
    crear_graficos_individuales(metricas_cblof, metricas_iforest)


def crear_graficos_individuales(metricas_cblof: pd.DataFrame, 
                                metricas_iforest: pd.DataFrame) -> None:
    """Crea gráficos individuales para métricas específicas."""
    
    # Gráfico de porcentaje de anomalías
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
    
    # Añadir valores en las barras
    for i, (bar, valor) in enumerate(zip(bars, porcentajes)):
        ax.text(valor + 0.3, i, f'{valor:.2f}%', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    ruta_porcentajes = DIRECTORIO_COMPARACIONES / 'comparacion_porcentaje_anomalias.png'
    plt.savefig(ruta_porcentajes, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de porcentajes guardado: {ruta_porcentajes.name}")
    
    # Gráfico de separación de scores
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
    
    # Añadir valores en las barras
    for i, (bar, valor) in enumerate(zip(bars, separaciones)):
        ax.text(valor + 0.05, i, f'{valor:.4f}', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    ruta_separacion = DIRECTORIO_COMPARACIONES / 'comparacion_separacion_scores.png'
    plt.savefig(ruta_separacion, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de separación guardado: {ruta_separacion.name}")


def analizar_ganador(metricas_cblof: pd.DataFrame, metricas_iforest: pd.DataFrame) -> Dict[str, Any]:
    """Determina el ganador basándose en las métricas."""
    analisis = {}
    
    # Separación de scores (mayor es mejor - más crítico)
    sep_cblof = metricas_cblof['p95_minus_p50'].values[0]
    sep_iforest = metricas_iforest['p95_minus_p50'].values[0]
    
    if sep_cblof > sep_iforest:
        analisis['mejor_separacion'] = 'CBLOF'
        peso_separacion = 2  # Doble peso
    elif sep_iforest > sep_cblof:
        analisis['mejor_separacion'] = 'Isolation Forest'
        peso_separacion = 2
    else:
        analisis['mejor_separacion'] = 'Empate'
        peso_separacion = 0
    
    # Porcentaje de anomalías
    pct_cblof = metricas_cblof['pct_anomalias'].values[0]
    pct_iforest = metricas_iforest['pct_anomalias'].values[0]
    
    if abs(pct_cblof - pct_iforest) > 2:  # Diferencia significativa
        if pct_cblof > pct_iforest:
            analisis['mayor_deteccion'] = 'CBLOF'
            peso_deteccion = 1
        else:
            analisis['mayor_deteccion'] = 'Isolation Forest'
            peso_deteccion = 1
    else:
        analisis['mayor_deteccion'] = 'Empate (similar)'
        peso_deteccion = 0
    
    # Calcular puntuación total
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
    
    # Determinar ganador
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
    """Guarda reporte completo de la comparación."""
    print("\n📝 Generando reporte completo...")
    
    ruta_reporte = DIRECTORIO_COMPARACIONES / 'REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt'
    
    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(" " * 18 + "REPORTE DE COMPARACIÓN DE ALGORITMOS DE DETECCIÓN DE ANOMALÍAS\n")
        f.write(" " * 35 + "CBLOF vs Isolation Forest\n")
        f.write("=" * 100 + "\n\n")
        
        # Tabla comparativa
        f.write("1. TABLA COMPARATIVA DE MÉTRICAS\n")
        f.write("-" * 100 + "\n")
        f.write(tabla_comparativa.to_string(index=False))
        f.write("\n\n")
        
        # Análisis de resultados
        f.write("2. ANÁLISIS DE RESULTADOS POR MÉTRICA\n")
        f.write("-" * 100 + "\n")
        f.write(f"✓ Mejor Separación de Scores (P95-P50): {analisis['mejor_separacion']}\n")
        f.write(f"  Separación CBLOF: {analisis['separacion_cblof']:.4f}\n")
        f.write(f"  Separación Isolation Forest: {analisis['separacion_iforest']:.4f}\n\n")
        
        f.write(f"✓ Detección de Anomalías: {analisis['mayor_deteccion']}\n")
        f.write(f"  CBLOF: {analisis['pct_anomalias_cblof']:.2f}%\n")
        f.write(f"  Isolation Forest: {analisis['pct_anomalias_iforest']:.2f}%\n\n")
        
        # Puntuación
        f.write("3. PUNTUACIÓN FINAL (Separación pesa doble)\n")
        f.write("-" * 100 + "\n")
        f.write(f"CBLOF: {analisis['puntos_cblof']} puntos\n")
        f.write(f"Isolation Forest: {analisis['puntos_iforest']} puntos\n\n")
        
        # Ganador
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
        
        # Características
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
        
        # Interpretación de métricas
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
        
        # Archivos generados
        f.write("7. ARCHIVOS GENERADOS EN ESTA COMPARACIÓN\n")
        f.write("-" * 100 + "\n")
        f.write("  • comparacion_visual_scores.png - Comparación de distribución de scores\n")
        f.write("  • comparacion_visual_3d.png - Comparación de visualización 3D\n")
        f.write("  • comparacion_metricas_barras.png - Gráfico de barras comparativo\n")
        f.write("  • comparacion_porcentaje_anomalias.png - Comparación de porcentajes\n")
        f.write("  • comparacion_separacion_scores.png - Comparación de separación\n")
        f.write("  • tabla_comparativa.csv - Tabla de métricas en formato CSV\n")
        f.write("  • REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt - Este archivo\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("Fecha de generación: Octubre 2025\n")
        f.write("=" * 100 + "\n")
    
    print(f"✅ Reporte completo guardado: {ruta_reporte.name}")


def main():
    """Función principal del script de comparación."""
    print("=" * 100)
    print(" " * 18 + "COMPARACIÓN DE ALGORITMOS DE DETECCIÓN DE ANOMALÍAS")
    print(" " * 35 + "CBLOF vs Isolation Forest")
    print("=" * 100)
    
    # Verificar archivos
    if not verificar_archivos():
        return
    
    print("\n✅ Todos los archivos necesarios están presentes.")
    
    # Cargar métricas
    print("\n📂 Cargando métricas...")
    metricas_cblof, metricas_iforest = cargar_metricas()
    print("✅ Métricas cargadas correctamente.")
    
    # Crear comparaciones visuales
    crear_comparacion_imagenes_lado_a_lado()
    
    # Crear tabla comparativa
    print("\n📋 Generando tabla comparativa...")
    tabla_comparativa = crear_tabla_comparativa_metricas(metricas_cblof, metricas_iforest)
    print("\n" + tabla_comparativa.to_string(index=False))
    
    # Guardar tabla
    ruta_tabla_csv = DIRECTORIO_COMPARACIONES / 'tabla_comparativa.csv'
    tabla_comparativa.to_csv(ruta_tabla_csv, index=False, encoding='utf-8')
    print(f"\n✅ Tabla guardada: {ruta_tabla_csv.name}")
    
    # Generar gráficos de métricas
    generar_graficos_metricas(metricas_cblof, metricas_iforest)
    
    # Analizar ganador
    print("\n🔍 Analizando resultados...")
    analisis = analizar_ganador(metricas_cblof, metricas_iforest)
    print(f"\n🏆 ALGORITMO GANADOR: {analisis['ganador']}")
    print(f"   Puntuación: CBLOF {analisis['puntos_cblof']} - Isolation Forest {analisis['puntos_iforest']}")
    
    # Guardar reporte completo
    guardar_reporte_completo(tabla_comparativa, analisis, metricas_cblof, metricas_iforest)
    
    print("\n" + "=" * 100)
    print(" " * 30 + "✅ COMPARACIÓN COMPLETADA EXITOSAMENTE")
    print(" " * 20 + f"📁 Resultados guardados en: {DIRECTORIO_COMPARACIONES}")
    print("=" * 100)


if __name__ == "__main__":
    main()

