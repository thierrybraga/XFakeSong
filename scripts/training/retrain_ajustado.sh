#!/usr/bin/env bash
# Retreino dos modelos AJUSTADOS após o diagnóstico do run
# clean_benchmark_full_20260626. Roda apenas os modelos que receberam ajuste
# de hiperparâmetros, um por vez, com GPU, retomando os já concluídos.
#
# Uso:
#   bash scripts/training/retrain_ajustado.sh
#   bash scripts/training/retrain_ajustado.sh --neural-only    # pula SVM/RandomForest
#   bash scripts/training/retrain_ajustado.sh --tcc-pending    # só os 4 pendentes do TCC
#
# --tcc-pending cobre o recorte diagnosticado no consolidado de 2026-07-01:
#   RawGAT-ST e AASIST (retreino de 30/06 rodou com imagem Docker desatualizada
#   — hparams antigos e sem augmentation) + WavLM/HuBERT Original (runner SSL
#   corrigido: augmentation de ruído, early stopping e threshold calibrado).
#
# IMPORTANTE: se for rodar via Docker, reconstrua a imagem antes
# (`make build-nocache SERVICE=benchmark` ou equivalente) e confira com
# `python scripts/benchmark/run_benchmark.py --plan-only` que o plano gravado mostra
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
# Pré-requisitos: dataset em data/datasets/ (ver DATASET abaixo; aceita
# override via env), ambiente com TensorFlow/PyTorch + GPU
# (ver docs/models/training.md).
set -euo pipefail

cd "$(dirname "$0")/../.."

# ATUALIZADO 2026-08-06: o default apontava para
# `benchmark_audio_raw_balanced_15k_confirmatory_v2.npz`, dataset ja APAGADO
# (atalho de fonte de 87,6%). O retreino tem de rodar sobre a MESMA variante do
# run diagnosticado, senao o conjunto de teste muda e o resultado nao entra na
# mesma consolidacao.
DATASET="${DATASET:-data/datasets/benchmark_dataset_15k.npz}"
STAMP="$(date +%Y%m%d)"
OUT="data/results/retune_ajustado_${STAMP}"
EPOCHS=100
# 5 dB e o nivel NAO VISTO no treino — a coluna que mede generalizacao a
# ruido. Estava faltando aqui enquanto o protocolo do benchmark ja o exigia.
SNR="30 20 10 5"
# TIMEOUT_MIN vazio = derivado por arquitetura a partir do custo estimado
# (planning.EXPECTED_TRAINING_HOURS, fator 3x). O valor fixo anterior (480min)
# MATARIA o RawGAT-ST: no clean_benchmark_15k ele levou 26,9 h (1.613 min).
# Defina TIMEOUT_MIN=<minutos> no ambiente para forcar um teto.
TIMEOUT_MIN="${TIMEOUT_MIN:-}"
TIMEOUT_FLAG=()
[[ -n "${TIMEOUT_MIN}" ]] && TIMEOUT_FLAG=("--timeout-min" "${TIMEOUT_MIN}")
SPEAKER_SPLIT_FLAG=()
SCOPE_FLAGS=()

if [[ ! -f "${DATASET}" ]]; then
  echo "ERRO: dataset não encontrado: ${DATASET}" >&2
  echo "Defina DATASET=<caminho do .npz> ou gere o dataset (docs/data/public-datasets.md)." >&2
  exit 1
fi

# Modelos ajustados (ver configs/training/retune_ajustado.yaml).
# ATUALIZADO 2026-08-06 para o diagnostico do run `clean_benchmark_15k`: os
# outros nove do escopo oficial foram auditados e NAO precisam de retreino.
MODELS=(
  "Conformer"
  "RawGAT-ST"
)

for arg in "$@"; do
  case "$arg" in
    --neural-only)
      MODELS=("Conformer" "RawGAT-ST")
      ;;
    --legacy-20260626)
      # Recorte do diagnostico ANTERIOR (clean_benchmark_full_20260626),
      # mantido so para reproduzir aquele retreino.
      MODELS=("RawGAT-ST" "AASIST" "Hybrid CNN-Transformer" "MultiscaleCNN" "RandomForest" "SVM")
      ;;
    --extended)
      MODELS=("Ensemble" "EfficientNet-LSTM")
      SCOPE_FLAGS=("--scope" "extended" "--no-academic-protocol" "--no-optimize-hparams")
      ;;
    --with-speaker-split)
      SPEAKER_SPLIT_FLAG=("--speaker-split" "--no-academic-protocol")
      ;;
  esac
done

echo ">> Dataset : ${DATASET}"
echo ">> Saída   : ${OUT}"
echo ">> Modelos : ${MODELS[*]}"
echo ">> Épocas  : ${EPOCHS}   SNR: ${SNR}   Timeout/modelo: ${TIMEOUT_MIN}min"
echo ">> Speaker-split: $([ ${#SPEAKER_SPLIT_FLAG[@]} -gt 0 ] && echo 'ON (exploratorio, nao comparavel a tabela do TCC)' || echo 'OFF (protocolo do TCC)')"

python scripts/benchmark/run_models_sequential.py \
  --dataset "${DATASET}" \
  --models "${MODELS[@]}" \
  --out "${OUT}" \
  --epochs "${EPOCHS}" \
  --snr ${SNR} \
  --device-profile gpu \
  "${TIMEOUT_FLAG[@]}" \
  "${SCOPE_FLAGS[@]}" \
  "${SPEAKER_SPLIT_FLAG[@]}" \
  --resume

echo
echo ">> Retreino concluído. Resultados em ${OUT}"
echo ">> Próximos passos:"
echo "   1) Comparar métricas:  python scripts/reporting/consolidate_results.py ${OUT}"
echo "   2) Validar artefatos:  python scripts/reporting/validate_artifacts.py --results-dir ${OUT}"
echo "   3) Sincronizar p/ app: python scripts/reporting/sync_completed_benchmark_artifacts.py --summary ${OUT}/run_summary.json"
echo "   (sincronize apenas se as métricas melhorarem em relação ao baseline,"
echo "    com atenção especial à robustez a 10 dB)"
