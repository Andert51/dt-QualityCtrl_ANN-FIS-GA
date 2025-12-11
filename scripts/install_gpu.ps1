# 🚀 GPU Setup for dt-QualityCtrl System

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " GPU Setup for dt-QualityCtrl System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# [1/4] Verificar GPU
Write-Host "[1/4] Verificando GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GPU detectada: $gpuInfo" -ForegroundColor Green
    } else {
        throw "No GPU"
    }
} catch {
    Write-Host "❌ ERROR: No se detecta GPU NVIDIA o driver no instalado" -ForegroundColor Red
    Write-Host "Descarga drivers desde: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""

# [2/4] Instalar PyTorch con CUDA
Write-Host "[2/4] Instalando PyTorch con CUDA..." -ForegroundColor Yellow
Write-Host "⏳ Descarga: ~2.4 GB - Esto tomará 10-15 minutos" -ForegroundColor Cyan
Write-Host ""

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Falló la instalación de PyTorch con CUDA" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""

# [3/4] Verificar instalación de CUDA
Write-Host "[3/4] Verificando instalación de CUDA..." -ForegroundColor Yellow
python -c "import torch; print(f'\nPyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No detectada\"}')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: PyTorch no reconoce CUDA" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""

# [4/4] Instalar dependencias restantes
Write-Host "[4/4] Instalando dependencias restantes..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Falló la instalación de dependencias" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " ✅ Instalación Completada!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Para ejecutar el sistema:" -ForegroundColor Cyan
Write-Host "  python main.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "Selecciona '🚀 GPU (Recomendado)' cuando aparezca el prompt" -ForegroundColor Cyan
Write-Host ""
Read-Host "Presiona Enter para continuar"
