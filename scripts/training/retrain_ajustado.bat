@echo off
REM Retreino dos modelos AJUSTADOS (Windows) apos diagnostico do run
REM clean_benchmark_15k (2026-08-06): Conformer e RawGAT-ST. Os outros nove do
REM escopo oficial foram auditados e NAO precisam de retreino.
REM
REM Uso:  scripts\training\retrain_ajustado.bat                  (Conformer + RawGAT-ST)
REM       scripts\training\retrain_ajustado.bat legacy-20260626  (recorte do diagnostico anterior)
REM       scripts\training\retrain_ajustado.bat extended         (Ensemble + EfficientNet-LSTM)
REM
REM IMPORTANTE: se rodar via Docker, reconstrua a imagem antes (o retreino de
REM 30/06 usou imagem desatualizada e treinou com hparams antigos). Valide com
REM `python scripts\benchmark\run_benchmark.py --plan-only` que o plano mostra
REM os ajustes de 2026-08-06 (Conformer LR 5e-5 / warmup 3000 / decay 76100;
REM RawGAT-ST dropout 0.5 / l2 3e-3 / decay 152100) antes de treinar.
REM
REM NAO usa --speaker-split por padrao: o TCC documenta particionamento
REM estratificado 70/15/15 para os 11 modelos da tabela. Split por locutor
REM muda o protocolo (teste menor/desbalanceado) e gera numero NAO comparavel
REM ao baseline -- confirmado em 2026-07-01 (RawGAT-ST: n=2250 balanceado ->
REM n=863 com 525/338). Passe "with-speaker-split" como 2o argumento para o
REM protocolo exploratorio disjunto por locutor (fora da tabela oficial).
REM
REM Pre-requisitos: dataset em data\datasets\benchmark_dataset_15k.npz,
REM ambiente com TensorFlow/PyTorch + GPU (ver docs/models/training.md).
setlocal enabledelayedexpansion
cd /d "%~dp0..\.."

REM ATUALIZADO 2026-08-06: o default apontava para
REM `benchmark_audio_raw_balanced_15k_confirmatory_v2.npz`, dataset ja APAGADO
REM (atalho de fonte de 87,6%). O retreino tem de rodar sobre a MESMA variante
REM do run diagnosticado -- variantes diferentes tem conjuntos de teste
REM diferentes e os resultados nao entram na mesma consolidacao.
set "DATASET=data/datasets/benchmark_dataset_15k.npz"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set STAMP=%%i
set "OUT=data/results/retune_ajustado_%STAMP%"
REM Sem --timeout-min o teto e DERIVADO por arquitetura a partir do custo
REM estimado (planning.EXPECTED_TRAINING_HOURS, fator 3x). O valor fixo
REM anterior (480min) MATARIA o RawGAT-ST: no clean_benchmark_15k ele levou
REM 26,9 h (1.613 min). Defina TIMEOUT_MIN=<minutos> antes de chamar para
REM forcar um teto.
set "TIMEOUT_FLAG="
if defined TIMEOUT_MIN set "TIMEOUT_FLAG=--timeout-min %TIMEOUT_MIN%"

if not exist "%DATASET%" (
  echo ERRO: dataset nao encontrado: %DATASET%
  exit /b 1
)

set MODELS="Conformer" "RawGAT-ST"
if /i "%~1"=="legacy-20260626" set MODELS="RawGAT-ST" "AASIST" "Hybrid CNN-Transformer" "MultiscaleCNN" "RandomForest" "SVM"
set "SCOPE_FLAGS="
if /i "%~1"=="extended" (
  set MODELS="Ensemble" "EfficientNet-LSTM"
  set "SCOPE_FLAGS=--scope extended --no-academic-protocol --no-optimize-hparams"
)

set "SPEAKER_SPLIT_FLAG="
if /i "%~1"=="with-speaker-split" set "SPEAKER_SPLIT_FLAG=--speaker-split --no-academic-protocol"
if /i "%~2"=="with-speaker-split" set "SPEAKER_SPLIT_FLAG=--speaker-split --no-academic-protocol"

echo == Dataset : %DATASET%
echo == Saida   : %OUT%
echo == Modelos : %MODELS%
if defined TIMEOUT_MIN (echo == Timeout/modelo : %TIMEOUT_MIN%min) else (echo == Timeout/modelo : derivado por arquitetura)

REM 5 dB e o nivel NAO VISTO no treino -- a coluna que mede generalizacao a
REM ruido. Estava faltando aqui enquanto o protocolo ja o exigia.
python scripts\benchmark\run_models_sequential.py ^
  --dataset "%DATASET%" ^
  --models %MODELS% ^
  --out "%OUT%" ^
  --epochs 100 ^
  --snr 30 20 10 5 ^
  --device-profile gpu ^
  %TIMEOUT_FLAG% ^
  %SCOPE_FLAGS% ^
  %SPEAKER_SPLIT_FLAG% ^
  --resume

echo.
echo == Retreino concluido. Resultados em %OUT%
echo == Depois:
echo ==   python scripts\reporting\consolidate_results.py %OUT%
echo ==   python scripts\reporting\validate_artifacts.py --results-dir %OUT%
echo ==   python scripts\reporting\sync_completed_benchmark_artifacts.py --summary %OUT%/run_summary.json
echo == (sincronize apenas se melhorar o baseline, atencao a robustez a 10 dB)
endlocal
