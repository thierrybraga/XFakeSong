@echo off
REM Retreino dos modelos AJUSTADOS (Windows) apos diagnostico do run
REM clean_benchmark_full_20260626. Roda apenas os modelos ajustados, com GPU.
REM
REM Uso:  scripts\retrain_ajustado.bat
REM
REM Pre-requisitos: dataset em app\datasets\benchmark_audio_raw_balanced_20k.npz,
REM ambiente com TensorFlow/PyTorch + GPU (ver docs\10_TREINAMENTO.md).
setlocal
cd /d "%~dp0.."

set "DATASET=app/datasets/benchmark_audio_raw_balanced_20k.npz"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set STAMP=%%i
set "OUT=results/retune_ajustado_%STAMP%"

echo == Dataset : %DATASET%
echo == Saida   : %OUT%

python scripts\run_models_sequential.py ^
  --dataset "%DATASET%" ^
  --models "RawGAT-ST" "AASIST" "Ensemble" "Hybrid CNN-Transformer" "EfficientNet-LSTM" "MultiscaleCNN" "RandomForest" "SVM" ^
  --out "%OUT%" ^
  --epochs 120 ^
  --snr 30 20 10 ^
  --device-profile gpu ^
  --resume

echo.
echo == Retreino concluido. Resultados em %OUT%
echo == Depois: consolidate_results.py / validate_artifacts.py / sync_completed_benchmark_artifacts.py
endlocal
