@echo off
REM Retreino dos modelos AJUSTADOS (Windows) apos diagnostico do run
REM clean_benchmark_full_20260626. Roda apenas os modelos ajustados, com GPU.
REM
REM Uso:  scripts\training\retrain_ajustado.bat            (todos os 8 ajustados)
REM       scripts\training\retrain_ajustado.bat tcc-pending (so os 4 pendentes do TCC:
REM           RawGAT-ST, AASIST, WavLM Original, HuBERT Original)
REM
REM IMPORTANTE: se rodar via Docker, reconstrua a imagem antes (o retreino de
REM 30/06 usou imagem desatualizada e treinou com hparams antigos). Valide com
REM `python scripts\benchmark\run_benchmark.py --plan-only` antes de treinar.
REM
REM NAO usa --speaker-split por padrao: o TCC documenta particionamento
REM estratificado 70/15/15 para os 11 modelos da tabela. Split por locutor
REM muda o protocolo (teste menor/desbalanceado) e gera numero NAO comparavel
REM ao baseline -- confirmado em 2026-07-01 (RawGAT-ST: n=2250 balanceado ->
REM n=863 com 525/338). Passe "with-speaker-split" como 2o argumento para o
REM protocolo exploratorio disjunto por locutor (fora da tabela oficial).
REM
REM Pre-requisitos: dataset em data\datasets\benchmark_audio_raw_balanced_15k.npz,
REM ambiente com TensorFlow/PyTorch + GPU (ver docs\10_TREINAMENTO.md).
setlocal
cd /d "%~dp0..\.."

set "DATASET=data/datasets/benchmark_audio_raw_balanced_15k.npz"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set STAMP=%%i
set "OUT=results/retune_ajustado_%STAMP%"
REM 480min: AASIST/RawGAT-ST precisam de ~160min so de treino (120 epocas);
REM sem timeout explicito o script usa o default de 60min e o modelo estoura
REM antes de terminar (visto em 2026-07-01 com AASIST: timeout aos 43/120).
set "TIMEOUT_MIN=480"

if not exist "%DATASET%" (
  echo ERRO: dataset nao encontrado: %DATASET%
  exit /b 1
)

set MODELS="RawGAT-ST" "AASIST" "Ensemble" "Hybrid CNN-Transformer" "EfficientNet-LSTM" "MultiscaleCNN" "RandomForest" "SVM"
if /i "%~1"=="tcc-pending" set MODELS="RawGAT-ST" "AASIST" "WavLM Original" "HuBERT Original"

set "SPEAKER_SPLIT_FLAG="
if /i "%~1"=="with-speaker-split" set "SPEAKER_SPLIT_FLAG=--speaker-split"
if /i "%~2"=="with-speaker-split" set "SPEAKER_SPLIT_FLAG=--speaker-split"

echo == Dataset : %DATASET%
echo == Saida   : %OUT%
echo == Modelos : %MODELS%
echo == Timeout/modelo : %TIMEOUT_MIN%min

python scripts\benchmark\run_models_sequential.py ^
  --dataset "%DATASET%" ^
  --models %MODELS% ^
  --out "%OUT%" ^
  --epochs 120 ^
  --snr 30 20 10 ^
  --device-profile gpu ^
  --timeout-min %TIMEOUT_MIN% ^
  %SPEAKER_SPLIT_FLAG% ^
  --resume

echo.
echo == Retreino concluido. Resultados em %OUT%
echo == Depois: consolidate_results.py / validate_artifacts.py / sync_completed_benchmark_artifacts.py
echo == (sincronize apenas se melhorar o baseline, atencao a robustez a 10 dB)
endlocal
