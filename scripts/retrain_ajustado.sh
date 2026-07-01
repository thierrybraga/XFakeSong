#!/usr/bin/env bash
# Retreino dos modelos AJUSTADOS após o diagnóstico do run
# clean_benchmark_full_20260626. Roda apenas os 8 modelos que receberam ajuste
# de hiperparâmetros, um por vez, com GPU, retomando os já concluídos.
#
# Uso:
#   bash scripts/retrain_ajustado.sh
#   bash scripts/retrain_ajustado.sh --neural-only      # pula SVM/RandomForest
#
# Pré-requisitos: dataset em app/datasets/benchmark_audio_raw_balanced_20k.npz,
# ambiente com TensorFlow/PyTorch + GPU (ver docs/10_TREINAMENTO.md).
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="app/datasets/benchmark_audio_raw_balanced_20k.npz"
STAMP="$(date +%Y%m%d)"
OUT="results/retune_ajustado_${STAMP}"
EPOCHS=120
SNR="30 20 10"

# Modelos ajustados (ver configs/training/retune_ajustado.yaml)
MODELS=(
  "RawGAT-ST"
  "AASIST"
  "Ensemble"
  "Hybrid CNN-Transformer"
  "EfficientNet-LSTM"
  "MultiscaleCNN"
  "RandomForest"
  "SVM"
)

if [[ "${1:-}" == "--neural-only" ]]; then
  MODELS=("RawGAT-ST" "AASIST" "Ensemble" "Hybrid CNN-Transformer" "EfficientNet-LSTM" "MultiscaleCNN")
fi

echo ">> Dataset : ${DATASET}"
echo ">> Saída   : ${OUT}"
echo ">> Modelos : ${MODELS[*]}"
echo ">> Épocas  : ${EPOCHS}   SNR: ${SNR}"

python scripts/run_models_sequential.py \
  --dataset "${DATASET}" \
  --models "${MODELS[@]}" \
  --out "${OUT}" \
  --epochs "${EPOCHS}" \
  --snr ${SNR} \
  --device-profile gpu \
  --speaker-split \
  --resume

echo
echo ">> Retreino concluído. Resultados em ${OUT}"
echo ">> Próximos passos:"
echo "   1) Comparar métricas:  python scripts/consolidate_results.py --results ${OUT}"
echo "   2) Validar artefatos:  python scripts/validate_artifacts.py --results ${OUT}"
echo "   3) Sincronizar p/ app: python scripts/sync_completed_benchmark_artifacts.py --results ${OUT}"
echo "   (sincronize apenas se as métricas melhorarem em relação ao baseline)"
