# ============================================================================
# Script de Ejecución Automatizada - Algoritmos ML No Supervisados para PdM
# ============================================================================
# Ejecuta todos los algoritmos de clustering y detección de anomalías
# en orden secuencial con logs detallados.
#
# Uso:
#   .\run_all.ps1
#
# O desde cualquier ubicación:
#   powershell -ExecutionPolicy Bypass -File "C:\ruta\al\proyecto\run_all.ps1"
# ============================================================================

# Configuración de encoding UTF-8 para evitar errores con caracteres especiales
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Colores para output
$ColorExito = "Green"
$ColorError = "Red"
$ColorInfo = "Cyan"
$ColorAdvertencia = "Yellow"

# Obtener ruta del script y establecer como directorio de trabajo
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=" * 100 -ForegroundColor $ColorInfo
Write-Host " EJECUCION AUTOMATIZADA - ALGORITMOS ML NO SUPERVISADOS PARA PdM" -ForegroundColor $ColorInfo
Write-Host "=" * 100 -ForegroundColor $ColorInfo
Write-Host ""

# Timestamp para logs
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = ".\ejecucion_$Timestamp.log"

# Función para logging
function Write-Log {
    param(
        [string]$Mensaje,
        [string]$Nivel = "INFO"
    )
    $LogEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Nivel] $Mensaje"
    Add-Content -Path $LogFile -Value $LogEntry
    
    switch ($Nivel) {
        "ERROR" { Write-Host $Mensaje -ForegroundColor $ColorError }
        "SUCCESS" { Write-Host $Mensaje -ForegroundColor $ColorExito }
        "WARNING" { Write-Host $Mensaje -ForegroundColor $ColorAdvertencia }
        default { Write-Host $Mensaje -ForegroundColor $ColorInfo }
    }
}

Write-Log "Iniciando proceso de ejecucion automatizada" "INFO"
Write-Log "Directorio de trabajo: $ScriptDir" "INFO"
Write-Log "Archivo de log: $LogFile" "INFO"
Write-Host ""

# ============================================================================
# 1. VERIFICAR Y ACTIVAR ENTORNO VIRTUAL
# ============================================================================
Write-Host "[1/6] Verificando entorno virtual..." -ForegroundColor $ColorInfo

$VenvPath = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-Not (Test-Path $VenvPath)) {
    Write-Log "ERROR: No se encontro el entorno virtual en .venv\" "ERROR"
    Write-Log "Por favor, crea el entorno virtual primero:" "ERROR"
    Write-Log "  python -m venv .venv" "ERROR"
    Write-Log "  .\.venv\Scripts\Activate.ps1" "ERROR"
    Write-Log "  pip install -r requirements.txt" "ERROR"
    exit 1
}

Write-Log "Entorno virtual encontrado: $VenvPath" "SUCCESS"
Write-Host ""

# ============================================================================
# 2. VERIFICAR DEPENDENCIAS
# ============================================================================
Write-Host "[2/6] Verificando dependencias..." -ForegroundColor $ColorInfo

$RequiredModules = @("numpy", "pandas", "scikit-learn", "matplotlib", "pyod", "joblib", "h5py")
$MissingModules = @()

foreach ($Module in $RequiredModules) {
    $CheckCmd = "& '$VenvPath' -c 'import $Module' 2>&1"
    $Result = Invoke-Expression $CheckCmd
    
    if ($LASTEXITCODE -ne 0) {
        $MissingModules += $Module
        Write-Log "  [FALTA] $Module" "WARNING"
    } else {
        Write-Log "  [OK] $Module" "SUCCESS"
    }
}

if ($MissingModules.Count -gt 0) {
    Write-Log "ADVERTENCIA: Modulos faltantes detectados. Instalando..." "WARNING"
    Write-Log "Ejecutando: pip install -r requirements.txt" "INFO"
    
    & "$ScriptDir\.venv\Scripts\pip.exe" install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: Fallo la instalacion de dependencias" "ERROR"
        exit 1
    }
    Write-Log "Dependencias instaladas correctamente" "SUCCESS"
} else {
    Write-Log "Todas las dependencias estan instaladas" "SUCCESS"
}
Write-Host ""

# ============================================================================
# 3. EJECUTAR ALGORITMOS DE CLUSTERING
# ============================================================================
Write-Host "[3/6] Ejecutando Algoritmos de Clustering..." -ForegroundColor $ColorInfo
Write-Host ""

# 3.1 K-Means
Write-Host "  >> Ejecutando K-Means..." -ForegroundColor $ColorInfo
Write-Log "Iniciando K-Means" "INFO"

$KMeansPath = ".\1. Clustering\K-means\K-means.py"
& $VenvPath $KMeansPath 2>&1 | Tee-Object -Variable KMeansOutput

if ($LASTEXITCODE -eq 0) {
    Write-Log "K-Means completado exitosamente" "SUCCESS"
} else {
    Write-Log "ERROR en K-Means (Exit Code: $LASTEXITCODE)" "ERROR"
    Write-Log "Continuando con el siguiente algoritmo..." "WARNING"
}
Write-Host ""

# 3.2 DBSCAN
Write-Host "  >> Ejecutando DBSCAN..." -ForegroundColor $ColorInfo
Write-Log "Iniciando DBSCAN" "INFO"

$DBSCANPath = ".\1. Clustering\DBSCAN\DBSCAN.py"
& $VenvPath $DBSCANPath 2>&1 | Tee-Object -Variable DBSCANOutput

if ($LASTEXITCODE -eq 0) {
    Write-Log "DBSCAN completado exitosamente" "SUCCESS"
} else {
    Write-Log "ERROR en DBSCAN (Exit Code: $LASTEXITCODE)" "ERROR"
    Write-Log "Continuando con el siguiente algoritmo..." "WARNING"
}
Write-Host ""

# ============================================================================
# 4. EJECUTAR ALGORITMOS DE DETECCION DE ANOMALIAS
# ============================================================================
Write-Host "[4/6] Ejecutando Algoritmos de Deteccion de Anomalias..." -ForegroundColor $ColorInfo
Write-Host ""

# 4.1 Isolation Forest
Write-Host "  >> Ejecutando Isolation Forest..." -ForegroundColor $ColorInfo
Write-Log "Iniciando Isolation Forest" "INFO"

$IForestPath = ".\2. Deteccion de Anomalias\Isolation Forest\Isolation Forest.py"
& $VenvPath $IForestPath 2>&1 | Tee-Object -Variable IForestOutput

if ($LASTEXITCODE -eq 0) {
    Write-Log "Isolation Forest completado exitosamente" "SUCCESS"
} else {
    Write-Log "ERROR en Isolation Forest (Exit Code: $LASTEXITCODE)" "ERROR"
    Write-Log "Continuando con el siguiente algoritmo..." "WARNING"
}
Write-Host ""

# 4.2 CBLOF
Write-Host "  >> Ejecutando CBLOF..." -ForegroundColor $ColorInfo
Write-Log "Iniciando CBLOF" "INFO"

$CBLOFPath = ".\2. Deteccion de Anomalias\CBLOF (Cluster-Based Local Outlier Factor)\CBLOF.PY"
& $VenvPath $CBLOFPath 2>&1 | Tee-Object -Variable CBLOFOutput

if ($LASTEXITCODE -eq 0) {
    Write-Log "CBLOF completado exitosamente" "SUCCESS"
} else {
    Write-Log "ERROR en CBLOF (Exit Code: $LASTEXITCODE)" "ERROR"
    Write-Log "Continuando con el siguiente algoritmo..." "WARNING"
}
Write-Host ""

# ============================================================================
# 5. EJECUTAR COMPARACIONES
# ============================================================================
Write-Host "[5/6] Ejecutando Comparaciones..." -ForegroundColor $ColorInfo
Write-Host ""

# 5.1 Comparacion Clustering
Write-Host "  >> Comparando Algoritmos de Clustering (K-Means vs DBSCAN)..." -ForegroundColor $ColorInfo
Write-Log "Iniciando comparacion de Clustering" "INFO"

$CompClusteringPath = ".\1. Clustering\Comparaciones\comparar_algoritmos.py"
& $VenvPath $CompClusteringPath 2>&1 | Tee-Object -Variable CompClusteringOutput

if ($LASTEXITCODE -eq 0) {
    Write-Log "Comparacion de Clustering completada exitosamente" "SUCCESS"
} else {
    Write-Log "ERROR en Comparacion de Clustering (Exit Code: $LASTEXITCODE)" "ERROR"
}
Write-Host ""

# 5.2 Comparacion Deteccion de Anomalias
Write-Host "  >> Comparando Algoritmos de Deteccion de Anomalias (Isolation Forest vs CBLOF)..." -ForegroundColor $ColorInfo
Write-Log "Iniciando comparacion de Deteccion de Anomalias" "INFO"

$CompAnomaliesPath = ".\2. Deteccion de Anomalias\Comparaciones\comparar_algoritmos.py"
& $VenvPath $CompAnomaliesPath 2>&1 | Tee-Object -Variable CompAnomaliesOutput

if ($LASTEXITCODE -eq 0) {
    Write-Log "Comparacion de Deteccion de Anomalias completada exitosamente" "SUCCESS"
} else {
    Write-Log "ERROR en Comparacion de Deteccion de Anomalias (Exit Code: $LASTEXITCODE)" "ERROR"
}
Write-Host ""

# ============================================================================
# 6. RESUMEN FINAL
# ============================================================================
Write-Host "[6/6] Generando Resumen Final..." -ForegroundColor $ColorInfo
Write-Host ""

Write-Host "=" * 100 -ForegroundColor $ColorExito
Write-Host " RESUMEN DE EJECUCION" -ForegroundColor $ColorExito
Write-Host "=" * 100 -ForegroundColor $ColorExito
Write-Host ""

# Verificar archivos generados
$ArchivosEsperados = @(
    ".\1. Clustering\K-means\metricas_KMeans\metrics.csv",
    ".\1. Clustering\DBSCAN\metricas_DBSCAN\metrics.csv",
    ".\2. Deteccion de Anomalias\Isolation Forest\metricas_IForest\metrics.csv",
    ".\2. Deteccion de Anomalias\CBLOF (Cluster-Based Local Outlier Factor)\metricas_CBLOF\metrics.csv",
    ".\1. Clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt",
    ".\2. Deteccion de Anomalias\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt"
)

$ArchivosGenerados = 0
$ArchivosFaltantes = 0

Write-Host "Verificando archivos generados:" -ForegroundColor $ColorInfo
foreach ($Archivo in $ArchivosEsperados) {
    if (Test-Path $Archivo) {
        Write-Host "  [OK] $Archivo" -ForegroundColor $ColorExito
        $ArchivosGenerados++
    } else {
        Write-Host "  [FALTA] $Archivo" -ForegroundColor $ColorAdvertencia
        $ArchivosFaltantes++
    }
}

Write-Host ""
Write-Host "Archivos generados: $ArchivosGenerados / $($ArchivosEsperados.Count)" -ForegroundColor $(if ($ArchivosGenerados -eq $ArchivosEsperados.Count) { $ColorExito } else { $ColorAdvertencia })
Write-Host ""

Write-Host "=" * 100 -ForegroundColor $ColorExito
Write-Host " EJECUCION COMPLETADA" -ForegroundColor $ColorExito
Write-Host "=" * 100 -ForegroundColor $ColorExito
Write-Host ""
Write-Log "Proceso de ejecucion automatizada finalizado" "SUCCESS"
Write-Host "Log completo guardado en: $LogFile" -ForegroundColor $ColorInfo
Write-Host ""

# Mostrar ubicaciones de reportes finales
Write-Host "REPORTES FINALES GENERADOS:" -ForegroundColor $ColorExito
Write-Host "  1. Comparacion Clustering: .\1. Clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt" -ForegroundColor $ColorInfo
Write-Host "  2. Comparacion Deteccion de Anomalias: .\2. Deteccion de Anomalias\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt" -ForegroundColor $ColorInfo
Write-Host ""

Write-Host ""
Write-Host "Ejecucion completada. Revisa los logs y reportes en las carpetas correspondientes." -ForegroundColor $ColorExito

