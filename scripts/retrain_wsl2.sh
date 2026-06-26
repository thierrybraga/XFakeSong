#!/usr/bin/env bash
# =====================================================================
# XFakeSong — driver de RETREINO em WSL2 (GPU)
# =====================================================================
# Executa, sob GPU NVIDIA (WSL2), o retreino + benchmark com TODOS os
# ajustes de robustez já aplicados no código:
#   - ruído de treino calibrado por SNR (casa treino↔teste);
#   - augmentation reativada (RawNet2, SpectrogramTransformer);
#   - SpectrogramTransformer regularizado + restauração do melhor checkpoint;
#   - SVM/RF com RASTA-PLP + augmentation ruidoso;
#   - split por fonte/gerador (P0) e protocolo cross-generator (P0.4).
#
# O TF nativo no Windows NÃO usa GPU; rode este script DENTRO do WSL2.
# O dataset já existe no repo — NÃO há download.
#
# Uso (a partir da raiz do repo, dentro do WSL2):
#   bash scripts/retrain_wsl2.sh --check                  # só valida GPU/ambiente
#   bash scripts/retrain_wsl2.sh --indist                 # retreino in-distribution (14 arq.)
#   bash scripts/retrain_wsl2.sh --xgen fkvoice           # reteste cross-generator
#   bash scripts/retrain_wsl2.sh --indist --xgen fkvoice  # ambos, em sequência
#   bash scripts/retrain_wsl2.sh --model SpectrogramTransformer  # uma arquitetura
# Opções:
#   --dataset PATH   (default: app/datasets/benchmark_audio_raw_balanced_15k.npz)
#   --epochs N       (default: 100)
#   --out-prefix DIR (default: results/retrain_wsl2)
# =====================================================================
set -euo pipefail

DATASET="app/datasets/benchmark_audio_raw_balanced_15k.npz"
EPOCHS=100
OUT_PREFIX="results/retrain_wsl2"
DO_INDIST=0
DO_XGEN=""
SINGLE_MODEL=""
CHECK_ONLY=0
PY="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift;;
    --indist) DO_INDIST=1; shift;;
    --xgen) DO_XGEN="${2:?gerador requerido, ex.: fkvoice}"; shift 2;;
    --model) SINGLE_MODEL="${2:?nome da arquitetura}"; shift 2;;
    --dataset) DATASET="${2:?}"; shift 2;;
    --epochs) EPOCHS="${2:?}"; shift 2;;
    --out-prefix) OUT_PREFIX="${2:?}"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "Opção desconhecida: $1" >&2; exit 2;;
  esac
done

echo "==================================================================="
echo " XFakeSong — retreino WSL2 (GPU)"
echo "==================================================================="

# --- 1. Sanidade do ambiente -----------------------------------------
if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "AVISO: isto não parece WSL2. O TF no Windows nativo não usa GPU." >&2
fi
echo "-> nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "ERRO: nvidia-smi indisponível. Habilite GPU no WSL2." >&2; exit 1; }

echo "-> TensorFlow enxerga a GPU?"
$PY - <<'PYEOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TF", tf.__version__, "| GPUs:", gpus)
if not gpus:
    raise SystemExit(
        "ERRO: TF não vê GPU. Instale: pip install 'tensorflow[and-cuda]' "
        "(ou use o container GPU). Veja docs/22_RETREINO_WSL2.md."
    )
PYEOF

if [[ ! -f "$DATASET" ]]; then
  echo "ERRO: dataset não encontrado: $DATASET" >&2; exit 1
fi
echo "-> Dataset OK: $DATASET"

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "Ambiente validado (--check). Nada foi treinado."; exit 0
fi

# --- 2. Argumentos comuns do benchmark -------------------------------
COMMON=(--dataset "$DATASET" --epochs "$EPOCHS" --device-profile gpu --no-api)
if [[ -n "$SINGLE_MODEL" ]]; then
  COMMON+=(--model "$SINGLE_MODEL")
else
  COMMON+=(--full)
fi

run_bench () {  # $1 = sufixo do out, $2... = flags extras
  local suffix="$1"; shift
  local out="${OUT_PREFIX}_${suffix}"
  echo ">>> Benchmark [$suffix] -> $out"
  $PY scripts/run_benchmark.py "${COMMON[@]}" --out "$out" "$@"
  echo "<<< [$suffix] concluído. Artefatos + figuras (report.py) em: $out"
}

# --- 3. Execuções ----------------------------------------------------
[[ "$DO_INDIST" == "1" ]] && run_bench "indist"
[[ -n "$DO_XGEN" ]] && run_bench "xgen_${DO_XGEN}" --cross-generator "$DO_XGEN"

if [[ "$DO_INDIST" == "0" && -z "$DO_XGEN" ]]; then
  echo "Nada selecionado. Use --indist e/ou --xgen GERADOR (ou --check)." >&2
  exit 2
fi

# --- 4. Consolidação automática → figuras + tabelas do TCC -----------
# Usa o run in-distribution como fonte canônica do TCC quando existir;
# senão o cross-generator. Sobrescreve os artefatos consolidados antigos
# pelos NOVOS (resultado do treino que acabou de rodar).
CONSOL_SRC=""
[[ "$DO_INDIST" == "1" ]] && CONSOL_SRC="${OUT_PREFIX}_indist"
[[ -z "$CONSOL_SRC" && -n "$DO_XGEN" ]] && CONSOL_SRC="${OUT_PREFIX}_xgen_${DO_XGEN}"

if [[ -n "$CONSOL_SRC" && -f "$CONSOL_SRC/results.json" ]]; then
  echo ">>> Consolidando $CONSOL_SRC → results/tcc_consolidated + tcc_overleaf/figures"
  $PY scripts/consolidate_results.py "$CONSOL_SRC" \
      --out results/tcc_consolidated --copy-to tcc_overleaf/figures
  $PY scripts/update_tcc_latex.py \
      --summary results/tcc_consolidated/benchmark_summary.json \
      --output tcc_overleaf/tabelas_benchmark.tex \
      --figures-dir figures
  echo "<<< TCC atualizado: figuras substituídas + tcc_overleaf/tabelas_benchmark.tex"
  echo "    No tcc.tex, garanta:  \\input{tabelas_benchmark.tex}"
fi

echo "==================================================================="
echo " Concluído. Figuras e tabelas do TCC regeneradas a partir do treino."
echo " Recompile tcc_overleaf/tcc.tex (as figuras foram substituídas in-place)."
echo "==================================================================="
