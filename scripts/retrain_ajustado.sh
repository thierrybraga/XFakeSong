#!/usr/bin/env bash
# Retreino dos modelos AJUSTADOS após o diagnóstico do run
# clean_benchmark_full_20260626. Roda apenas os modelos que receberam ajuste
# de hiperparâmetros, um por vez, com GPU, retomando os já concluídos.
#
# Uso:
#   bash scripts/retrain_ajustado.sh
#   bash scripts/retrain_ajustado.sh --neural-only    # pula SVM/RandomForest
#   bash scripts/retrain_ajustado.sh --tcc-pending    # só os 4 pendentes do TCC
#
# --tcc-pending cobre o recorte diagnosticado no consolidado de 2026-07-01:
#   RawGAT-ST e AASIST (retreino de 30/06 rodou com imagem Docker desatualizada
#   — hparams antigos e sem augmentation) + WavLM/HuBERT Original (runner SSL
#   corrigido: augmentation de ruído, early stopping e threshold calibrado).
#
# IMPORTANTE: se for rodar via Docker, reconstrua a imagem antes
# (`make build-nocache SERVICE=benchmark` ou equivalente) e confira com
# `python scripts/run_benchmark.py --plan-only` que o plano gravado mostra
# os hparams ajustados (ex.: RawGAT-ST LR 5e-5, dropout 0.35, l2 1e-3,
# use_augmentation=true) antes de treinar.
#
# NAO usa --speaker-split por padrão: o TCC (main.tex) documenta o
# particionamento estratificado 70/15/15 para os 11 modelos da tabela
# consolidada. Rodar com split por locutor mudaria o protocolo de avaliacao
# (teste menor e desbalanceado) e tornaria o resultado NAO comparavel ao
# baseline dos outros modelos — confirmado empiricamente em 2026-07-01
# (RawGAT-ST caiu de n=2250 balanceado para n=863 com 525/338, resultado
# nao comparavel). Use --with-speaker-split para o protocolo exploratorio
# disjunto por locutor (fora da tabela oficial do TCC).
#
# Pré-requisitos: dataset em app/datasets/ (ver DATASET abaixo; aceita
# override via env), ambiente com TensorFlow/PyTorch + GPU
# (ver docs/10_TREINAMENTO.md).
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-app/datasets/benchmark_audio_raw_balanced_15k.npz}"
STAMP="$(date +%Y%m%d)"
OUT="results/retune_ajustado_${STAMP}"
EPOCHS=120
SNR="30 20 10"
# 480min: AASIST/RawGAT-ST levam ~1.3min/epoca * 120 epocas = ~160min so de
# treino; sem este valor explicito, run_models_sequential.py usa o default
# de 60min e o modelo estoura o timeout antes de terminar (visto em 2026-07-01
# com AASIST: timeout aos 43 epocas/120, sem artefato salvo).
TIMEOUT_MIN=480
SPEAKER_SPLIT_FLAG=()

if [[ ! -f "${DATASET}" ]]; then
  echo "ERRO: dataset não encontrado: ${DATASET}" >&2
  echo "Defina DATASET=<caminho do .npz> ou gere o dataset (docs/12_DATASETS.md)." >&2
  exit 1
fi

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

for arg in "$@"; do
  case "$arg" in
    --neural-only)
      MODELS=("RawGAT-ST" "AASIST" "Ensemble" "Hybrid CNN-Transformer" "EfficientNet-LSTM" "MultiscaleCNN")
      ;;
    --tcc-pending)
      MODELS=("RawGAT-ST" "AASIST" "WavLM Original" "HuBERT Original")
      ;;
    --with-speaker-split)
      SPEAKER_SPLIT_FLAG=("--speaker-split")
      ;;
  esac
done

echo ">> Dataset : ${DATASET}"
echo ">> Saída   : ${OUT}"
echo ">> Modelos : ${MODELS[*]}"
echo ">> Épocas  : ${EPOCHS}   SNR: ${SNR}   Timeout/modelo: ${TIMEOUT_MIN}min"
echo ">> Speaker-split: $([ ${#SPEAKER_SPLIT_FLAG[@]} -gt 0 ] && echo 'ON (exploratorio, nao comparavel a tabela do TCC)' || echo 'OFF (protocolo do TCC)')"

python scripts/run_models_sequential.py \
  --dataset "${DATASET}" \
  --models "${MODELS[@]}" \
  --out "${OUT}" \
  --epochs "${EPOCHS}" \
  --snr ${SNR} \
  --device-profile gpu \
  --timeout-min "${TIMEOUT_MIN}" \
  "${SPEAKER_SPLIT_FLAG[@]}" \
  --resume

echo
echo ">> Retreino concluído. Resultados em ${OUT}"
echo ">> Próximos passos:"
echo "   1) Comparar métricas:  python scripts/consolidate_results.py --results ${OUT}"
echo "   2) Validar artefatos:  python scripts/validate_artifacts.py --results ${OUT}"
echo "   3) Sincronizar p/ app: python scripts/sync_completed_benchmark_artifacts.py --results ${OUT}"
echo "   (sincronize apenas se as métricas melhorarem em relação ao baseline,"
echo "    com atenção especial à robustez a 10 dB)"
