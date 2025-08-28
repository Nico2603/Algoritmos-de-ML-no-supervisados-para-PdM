"""
Script Maestro para Sistema Completo de ML No Supervisado en PdM

Este script ejecuta todo el pipeline completo:
1. Ejecuta los 4 algoritmos corregidos
2. Realiza el análisis de comparación
3. Ejecuta el análisis de estabilidad
4. Genera el informe final ejecutivo

Uso:
    python ejecutar_sistema_completo.py

Autor: Sistema PdM
Versión: 1.0
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import time

class EjecutorSistemaCompleto:
    """Ejecutor del sistema completo de ML no supervisado."""
    
    def __init__(self):
        self.directorio_proyecto = Path(__file__).parent
        self.log_file = self.directorio_proyecto / 'ejecucion_completa.log'
        self._configurar_logging()
    
    def _configurar_logging(self):
        """Configura el sistema de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
    
    def ejecutar_comando(self, comando: str, directorio: str = None) -> bool:
        """
        Ejecuta un comando del sistema.
        
        Args:
            comando: Comando a ejecutar
            directorio: Directorio donde ejecutar (opcional)
            
        Returns:
            True si el comando se ejecutó exitosamente
        """
        try:
            if directorio:
                comando_completo = f"cd {directorio} && {comando}"
            else:
                comando_completo = comando
            
            logging.info(f"Ejecutando: {comando_completo}")
            
            # Ejecutar comando
            resultado = subprocess.run(
                comando_completo,
                shell=True,
                cwd=directorio,
                capture_output=True,
                text=True
            )
            
            if resultado.returncode == 0:
                logging.info(f"✓ Comando ejecutado exitosamente")
                if resultado.stdout:
                    logging.info(f"Output: {resultado.stdout}")
                return True
            else:
                logging.error(f"❌ Error ejecutando comando")
                logging.error(f"Error: {resultado.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Excepción ejecutando comando: {e}")
            return False
    
    def verificar_datos(self) -> bool:
        """Verifica que existan los archivos de datos necesarios."""
        logging.info("Verificando archivos de datos...")
        
        archivos_datos = [
            "1. Clustering/K-means/data.csv",
            "1. Clustering/DBSCAN/data.csv",
            "2. Detección de Anomalías/Isolation Forest/data.csv",
            "2. Detección de Anomalías/CBLOF (Cluster-Based Local Outlier Factor)/data.csv"
        ]
        
        datos_encontrados = 0
        for archivo in archivos_datos:
            ruta_completa = self.directorio_proyecto / archivo
            if ruta_completa.exists():
                logging.info(f"✓ Encontrado: {archivo}")
                datos_encontrados += 1
            else:
                logging.warning(f"⚠️  No encontrado: {archivo}")
        
        if datos_encontrados > 0:
            logging.info(f"✓ Se encontraron {datos_encontrados}/4 archivos de datos")
            return True
        else:
            logging.error("❌ No se encontraron archivos de datos")
            return False
    
    def ejecutar_algoritmos(self) -> bool:
        """Ejecuta todos los algoritmos de ML."""
        logging.info("=== EJECUTANDO ALGORITMOS DE ML ===")
        
        algoritmos = [
            {
                'nombre': 'K-Means',
                'directorio': '1. Clustering/K-means',
                'script': 'K-means.py'
            },
            {
                'nombre': 'DBSCAN',
                'directorio': '1. Clustering/DBSCAN',
                'script': 'DBSCAN.py'
            },
            {
                'nombre': 'Isolation Forest',
                'directorio': '2. Detección de Anomalías/Isolation Forest',
                'script': 'Isolation Forest.py'
            },
            {
                'nombre': 'CBLOF',
                'directorio': '2. Detección de Anomalías/CBLOF (Cluster-Based Local Outlier Factor)',
                'script': 'CBLOF.PY'
            }
        ]
        
        algoritmos_exitosos = 0
        
        for algoritmo in algoritmos:
            logging.info(f"\n--- Ejecutando {algoritmo['nombre']} ---")
            
            directorio_completo = self.directorio_proyecto / algoritmo['directorio']
            script_completo = directorio_completo / algoritmo['script']
            
            if not script_completo.exists():
                logging.error(f"❌ Script no encontrado: {script_completo}")
                continue
            
            # Ejecutar algoritmo
            comando = f"python \"{algoritmo['script']}\""
            if self.ejecutar_comando(comando, str(directorio_completo)):
                algoritmos_exitosos += 1
                logging.info(f"✓ {algoritmo['nombre']} completado exitosamente")
            else:
                logging.error(f"❌ Error ejecutando {algoritmo['nombre']}")
        
        logging.info(f"\n✓ Algoritmos completados: {algoritmos_exitosos}/4")
        return algoritmos_exitosos > 0
    
    def ejecutar_comparacion(self) -> bool:
        """Ejecuta el análisis de comparación."""
        logging.info("\n=== EJECUTANDO ANÁLISIS DE COMPARACIÓN ===")
        
        script_comparacion = self.directorio_proyecto / 'sistema_comparacion_algoritmos.py'
        
        if not script_comparacion.exists():
            logging.error(f"❌ Script de comparación no encontrado: {script_comparacion}")
            return False
        
        comando = f"python \"{script_comparacion.name}\""
        if self.ejecutar_comando(comando, str(self.directorio_proyecto)):
            logging.info("✓ Análisis de comparación completado")
            return True
        else:
            logging.error("❌ Error en análisis de comparación")
            return False
    
    def ejecutar_estabilidad(self) -> bool:
        """Ejecuta el análisis de estabilidad."""
        logging.info("\n=== EJECUTANDO ANÁLISIS DE ESTABILIDAD ===")
        
        script_estabilidad = self.directorio_proyecto / 'analisis_estabilidad_bootstrap.py'
        
        if not script_estabilidad.exists():
            logging.error(f"❌ Script de estabilidad no encontrado: {script_estabilidad}")
            return False
        
        comando = f"python \"{script_estabilidad.name}\""
        if self.ejecutar_comando(comando, str(self.directorio_proyecto)):
            logging.info("✓ Análisis de estabilidad completado")
            return True
        else:
            logging.error("❌ Error en análisis de estabilidad")
            return False
    
    def generar_informe_final(self) -> bool:
        """Genera el informe final."""
        logging.info("\n=== GENERANDO INFORME FINAL ===")
        
        script_informe = self.directorio_proyecto / 'generador_informe_final.py'
        
        if not script_informe.exists():
            logging.error(f"❌ Script de informe no encontrado: {script_informe}")
            return False
        
        comando = f"python \"{script_informe.name}\""
        if self.ejecutar_comando(comando, str(self.directorio_proyecto)):
            logging.info("✓ Informe final generado")
            return True
        else:
            logging.error("❌ Error generando informe final")
            return False
    
    def mostrar_resumen_final(self):
        """Muestra el resumen final de la ejecución."""
        logging.info("\n" + "="*60)
        logging.info("🎉 EJECUCIÓN COMPLETA FINALIZADA")
        logging.info("="*60)
        
        # Verificar directorios creados
        directorios_esperados = [
            'Comparación_Algoritmos',
            'Analisis_Estabilidad', 
            'Informe_Final'
        ]
        
        logging.info("\n📂 Directorios generados:")
        for directorio in directorios_esperados:
            ruta_dir = self.directorio_proyecto / directorio
            if ruta_dir.exists():
                archivos = list(ruta_dir.rglob('*'))
                logging.info(f"✓ {directorio}/ ({len(archivos)} archivos)")
            else:
                logging.info(f"❌ {directorio}/ (no encontrado)")
        
        # Mostrar archivos principales
        logging.info("\n📄 Archivos principales generados:")
        archivos_principales = [
            'Informe_Final/ejecutivo/informe_ejecutivo.md',
            'Informe_Final/ejecutivo/dashboard_ejecutivo.png',
            'Informe_Final/tecnico/documentacion_tecnica.md',
            'Comparación_Algoritmos/reportes/reporte_comparativo.md',
            'Analisis_Estabilidad/reportes/reporte_estabilidad.md'
        ]
        
        for archivo in archivos_principales:
            ruta_archivo = self.directorio_proyecto / archivo
            if ruta_archivo.exists():
                logging.info(f"✓ {archivo}")
            else:
                logging.info(f"❌ {archivo} (no encontrado)")
        
        logging.info(f"\n📋 Log completo disponible en: {self.log_file}")
        logging.info("\n🚀 ¡Sistema listo para revisión e implementación!")
    
    def ejecutar_sistema_completo(self):
        """Ejecuta todo el sistema completo."""
        inicio = time.time()
        
        logging.info("🚀 INICIANDO EJECUCIÓN COMPLETA DEL SISTEMA")
        logging.info(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"📂 Directorio del proyecto: {self.directorio_proyecto}")
        
        try:
            # 1. Verificar datos
            if not self.verificar_datos():
                logging.error("❌ No se pueden ejecutar algoritmos sin datos")
                return False
            
            # 2. Ejecutar algoritmos
            if not self.ejecutar_algoritmos():
                logging.error("❌ Error ejecutando algoritmos")
                return False
            
            # 3. Ejecutar comparación
            self.ejecutar_comparacion()  # No crítico si falla
            
            # 4. Ejecutar estabilidad
            self.ejecutar_estabilidad()  # No crítico si falla
            
            # 5. Generar informe final
            self.generar_informe_final()  # No crítico si falla
            
            # 6. Mostrar resumen
            self.mostrar_resumen_final()
            
            fin = time.time()
            duracion = fin - inicio
            logging.info(f"\n⏱️  Duración total: {duracion:.1f} segundos")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error crítico durante la ejecución: {e}")
            return False


def main():
    """Función principal."""
    print("🤖 Sistema Completo de ML No Supervisado para PdM")
    print("=" * 50)
    
    ejecutor = EjecutorSistemaCompleto()
    
    # Confirmar ejecución
    respuesta = input("\n¿Desea ejecutar el sistema completo? (s/n): ").lower().strip()
    
    if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n🚀 Iniciando ejecución...")
        exito = ejecutor.ejecutar_sistema_completo()
        
        if exito:
            print("\n✅ ¡Ejecución completada exitosamente!")
            print("📋 Revise los logs para más detalles.")
        else:
            print("\n❌ Ejecución completada con errores.")
            print("📋 Revise los logs para diagnóstico.")
    else:
        print("\n🛑 Ejecución cancelada por el usuario.")


if __name__ == "__main__":
    main()
