@echo off
echo ========================================
echo  GPU Setup for dt-QualityCtrl System
echo ========================================
echo.

echo [1/4] Verificando GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if %errorlevel% neq 0 (
    echo ERROR: No se detecta GPU NVIDIA o driver no instalado
    echo Descarga drivers desde: https://www.nvidia.com/Download/index.aspx
    pause
    exit /b 1
)

echo.
echo [2/4] Instalando PyTorch con CUDA...
echo Descarga: ~2.4 GB - Esto tomara 10-15 minutos
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion de PyTorch con CUDA
    pause
    exit /b 1
)

echo.
echo [3/4] Verificando instalacion de CUDA...
python -c "import torch; print(f'\nPyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No detectada\"}')"
if %errorlevel% neq 0 (
    echo ERROR: PyTorch no reconoce CUDA
    pause
    exit /b 1
)

echo.
echo [4/4] Instalando dependencias restantes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion de dependencias
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Instalacion Completada!
echo ========================================
echo.
echo Para ejecutar el sistema:
echo   python main.py
echo.
echo Selecciona "GPU (Recomendado)" cuando aparezca el prompt
echo.
pause
