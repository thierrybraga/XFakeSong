@echo off
REM ============================================================================
REM  XFakeSong - Setup GPU para Windows
REM ============================================================================
REM
REM  Diagnostica a situação atual e mostra as opções para habilitar GPU
REM  NVIDIA em Windows. Não modifica nada — apenas informa.
REM
REM  Uso:
REM    scripts\setup_gpu_windows.bat
REM
REM ============================================================================

chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ===============================================================================
echo  XFakeSong - Diagnostico e Setup de GPU NVIDIA (Windows)
echo ===============================================================================
echo.

REM --- 1. Hardware NVIDIA presente? ---
echo [1/4] Verificando hardware NVIDIA...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [X] nvidia-smi nao encontrado no PATH.
    echo       - Voce tem uma GPU NVIDIA fisica?
    echo       - Drivers NVIDIA instalados? (https://www.nvidia.com/drivers)
    echo       Se nao tem GPU NVIDIA, a aplicacao funciona em CPU.
    goto :end
)

for /f "delims=" %%a in ('nvidia-smi -L 2^>nul') do (
    echo   [v] %%a
)
echo.

REM --- 2. Estamos em Windows nativo ou WSL? ---
echo [2/4] Verificando ambiente...
if defined WSL_DISTRO_NAME (
    echo   [v] WSL detectado: %WSL_DISTRO_NAME%
    set IS_WSL=1
) else (
    echo   [i] Windows nativo (sem WSL)
    set IS_WSL=0
)
echo.

REM --- 3. Python e TensorFlow ---
echo [3/4] Verificando Python + TensorFlow...
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

%PY% --version 2>nul
if errorlevel 1 (
    echo   [X] Python nao encontrado.
    goto :end
)

%PY% -c "import tensorflow as tf; print('   TF', tf.__version__, '| built_with_cuda=', tf.test.is_built_with_cuda(), '| GPUs visiveis:', len(tf.config.list_physical_devices('GPU')))" 2>nul
if errorlevel 1 (
    echo   [X] TensorFlow nao esta instalado no Python ativo.
    echo       Rode: %PY% -m pip install -r requirements.txt
    goto :end
)
echo.

REM --- 4. Opções para habilitar GPU ---
echo [4/4] Opcoes para habilitar GPU NVIDIA em Windows
echo ----------------------------------------------------------
echo.

if "%IS_WSL%"=="0" (
    echo  TF ^>=2.11 NAO suporta GPU em Windows nativo.
    echo  Voce tem 3 opcoes:
    echo.
    echo   [A] WSL2 + Ubuntu  ^(RECOMENDADO - melhor performance^)
    echo       1. Instale WSL2:    wsl --install -d Ubuntu
    echo       2. Atualize driver NVIDIA Windows ^>= 525.x ^(suporta WSL2 CUDA^)
    echo       3. Dentro do Ubuntu WSL:
    echo            sudo apt update ^&^& sudo apt install python3-pip
    echo            pip install tensorflow[and-cuda]
    echo            python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    echo       4. Clone o repo dentro do WSL e rode normalmente.
    echo.
    echo   [B] tensorflow-directml-plugin  ^(Windows nativo, mais lento^)
    echo       REQUER Python 3.7-3.10 e TensorFlow 2.10 ^(versoes antigas^).
    echo       Como o projeto ja usa TF 2.21, esta opcao requer downgrade.
    echo       Nao recomendado para uso em producao.
    echo.
    echo   [C] Aceitar CPU  ^(funcional, mas treino ~5-20x mais lento^)
    echo       Sem mudancas necessarias. Apenas use start.bat normalmente.
    echo       Para inferencia ^(detectar^), CPU ja e aceitavel ^(^<1s por audio^).
    echo.
) else (
    echo  Voce esta em WSL. Para habilitar GPU CUDA:
    echo.
    echo   1. Verifique driver NVIDIA Windows ^>= 525.x:
    echo        nvidia-smi  ^# deve listar a GPU
    echo.
    echo   2. Reinstale TensorFlow com suporte CUDA:
    echo        pip uninstall -y tensorflow tensorflow-cpu
    echo        pip install 'tensorflow[and-cuda]'
    echo.
    echo   3. Reinicie o processo e cheque:
    echo        python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    echo.
)

echo  Diagnostico detalhado disponivel no Dashboard ^(apos iniciar a app^):
echo    1. start.bat
echo    2. Abra http://localhost:7860
echo    3. Aba "Dashboard" ^-^> "Diagnostico de GPU"
echo.

:end
echo ===============================================================================
endlocal
