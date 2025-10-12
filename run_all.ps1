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

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$env:MPLBACKEND = "Agg"
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"

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

# Inicializar log en UTF-8 para evitar problemas de codificación
Set-Content -Path $LogFile -Value "" -Encoding utf8

# Función para logging
function Write-Log {
    param(
        [string]$Mensaje,
        [string]$Nivel = "INFO"
    )
    $LogEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Nivel] $Mensaje"
    Add-Content -Path $LogFile -Value $LogEntry -Encoding utf8
    
    switch ($Nivel) {
        "ERROR" { Write-Host $Mensaje -ForegroundColor $ColorError }
        "SUCCESS" { Write-Host $Mensaje -ForegroundColor $ColorExito }
        "WARNING" { Write-Host $Mensaje -ForegroundColor $ColorAdvertencia }
        default { Write-Host $Mensaje -ForegroundColor $ColorInfo }
    }
}

$global:PasosFallidos = @()

function RunStep {
    param(
        [string]$Title,
        [string]$ScriptPath
    )
    
    if (-not (Test-Path $ScriptPath)) {
        Write-Log "SCRIPT NO ENCONTRADO: $ScriptPath" "ERROR"
        $global:PasosFallidos += $Title
        return 1
    }
    
    Write-Log "Iniciando $Title" "INFO"
    
    $cmd = "chcp 65001 >NUL & ""$VenvPath"" ""$ScriptPath"" >> ""$LogFile"" 2>&1"
    cmd /c $cmd
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Log "$Title completado exitosamente" "SUCCESS"
    } else {
        Write-Log "ERROR en $Title (Exit Code: $exitCode)" "ERROR"
        Write-Log "Continuando con el siguiente algoritmo..." "WARNING"
        $global:PasosFallidos += $Title
    }
    
    return $exitCode
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

$RequiredModules = @("numpy", "pandas", "sklearn", "matplotlib", "pyod", "joblib", "h5py", "scipy", "numba")
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
$KMeansPath = Join-Path $ScriptDir "1. Clustering\K-means\K-means.py"
RunStep "K-Means" $KMeansPath
Write-Host ""

# 3.2 DBSCAN
Write-Host "  >> Ejecutando DBSCAN..." -ForegroundColor $ColorInfo
$DBSCANPath = Join-Path $ScriptDir "1. Clustering\DBSCAN\DBSCAN.py"
RunStep "DBSCAN" $DBSCANPath
Write-Host ""

# ============================================================================
# 4. EJECUTAR ALGORITMOS DE DETECCION DE ANOMALIAS
# ============================================================================
Write-Host "[4/6] Ejecutando Algoritmos de Deteccion de Anomalias..." -ForegroundColor $ColorInfo
Write-Host ""

# 4.1 Isolation Forest
Write-Host "  >> Ejecutando Isolation Forest..." -ForegroundColor $ColorInfo
$IForestPath = Join-Path $ScriptDir "2. Detección de Anomalías\Isolation Forest\Isolation Forest.py"
RunStep "Isolation Forest" $IForestPath
Write-Host ""

# 4.2 CBLOF
Write-Host "  >> Ejecutando CBLOF..." -ForegroundColor $ColorInfo
$CBLOFPath = Join-Path $ScriptDir "2. Detección de Anomalías\CBLOF (Cluster-Based Local Outlier Factor)\CBLOF.PY"
RunStep "CBLOF" $CBLOFPath
Write-Host ""

# ============================================================================
# 5. EJECUTAR COMPARACIONES
# ============================================================================
Write-Host "[5/6] Ejecutando Comparaciones..." -ForegroundColor $ColorInfo
Write-Host ""

# 5.1 Comparacion Clustering
Write-Host "  >> Comparando Algoritmos de Clustering (K-Means vs DBSCAN)..." -ForegroundColor $ColorInfo
$CompClusteringPath = Join-Path $ScriptDir "1. Clustering\Comparaciones\comparar_algoritmos.py"
RunStep "Comparacion de Clustering" $CompClusteringPath
Write-Host ""

# 5.2 Comparacion Deteccion de Anomalias
Write-Host "  >> Comparando Algoritmos de Deteccion de Anomalias (Isolation Forest vs CBLOF)..." -ForegroundColor $ColorInfo
$CompAnomaliesPath = Join-Path $ScriptDir "2. Detección de Anomalías\Comparaciones\comparar_algoritmos.py"
RunStep "Comparacion de Deteccion de Anomalias" $CompAnomaliesPath
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
    ".\2. Detección de Anomalías\Isolation Forest\metricas_IForest\metrics.csv",
    ".\2. Detección de Anomalías\CBLOF (Cluster-Based Local Outlier Factor)\metricas_CBLOF\metrics.csv",
    ".\1. Clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt",
    ".\2. Detección de Anomalías\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt"
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
        Write-Log "Archivo faltante: $Archivo" "ERROR"
        $ArchivosFaltantes++
    }
}

Write-Host ""
Write-Host "Archivos generados: $ArchivosGenerados / $($ArchivosEsperados.Count)" -ForegroundColor $(if ($ArchivosGenerados -eq $ArchivosEsperados.Count) { $ColorExito } else { $ColorAdvertencia })
Write-Host ""

$ExitCodeFinal = 0

if ($global:PasosFallidos.Count -gt 0) {
    Write-Host "=" * 100 -ForegroundColor $ColorError
    Write-Host " EJECUCION COMPLETADA CON ERRORES" -ForegroundColor $ColorError
    Write-Host "=" * 100 -ForegroundColor $ColorError
    Write-Log "Los siguientes pasos fallaron:" "ERROR"
    foreach ($Paso in $global:PasosFallidos) {
        Write-Log "  - $Paso" "ERROR"
    }
    $ExitCodeFinal = 1
} elseif ($ArchivosFaltantes -gt 0) {
    Write-Host "=" * 100 -ForegroundColor $ColorAdvertencia
    Write-Host " EJECUCION COMPLETADA CON ARCHIVOS FALTANTES" -ForegroundColor $ColorAdvertencia
    Write-Host "=" * 100 -ForegroundColor $ColorAdvertencia
    Write-Log "Faltan $ArchivosFaltantes archivos esperados" "WARNING"
    $ExitCodeFinal = 1
} else {
    Write-Host "=" * 100 -ForegroundColor $ColorExito
    Write-Host " EJECUCION COMPLETADA EXITOSAMENTE" -ForegroundColor $ColorExito
    Write-Host "=" * 100 -ForegroundColor $ColorExito
}

Write-Host ""
Write-Log "Proceso de ejecucion automatizada finalizado" $(if ($ExitCodeFinal -eq 0) { "SUCCESS" } else { "ERROR" })
Write-Host "Log completo guardado en: $LogFile" -ForegroundColor $ColorInfo
Write-Host ""

Write-Host "REPORTES FINALES GENERADOS:" -ForegroundColor $(if ($ExitCodeFinal -eq 0) { $ColorExito } else { $ColorAdvertencia })
Write-Host "  1. Comparacion Clustering: .\1. Clustering\Comparaciones\REPORTE_COMPARACION_CLUSTERING.txt" -ForegroundColor $ColorInfo
Write-Host "  2. Comparacion Deteccion de Anomalias: .\2. Detección de Anomalías\Comparaciones\REPORTE_COMPARACION_DETECCION_ANOMALIAS.txt" -ForegroundColor $ColorInfo
Write-Host ""

Write-Host ""
if ($ExitCodeFinal -eq 0) {
    Write-Host "Ejecucion completada exitosamente. Revisa los logs y reportes." -ForegroundColor $ColorExito
} else {
    Write-Host "Ejecucion completada con errores. Revisa el log en $LogFile" -ForegroundColor $ColorError
}
Write-Host ""

exit $ExitCodeFinal

