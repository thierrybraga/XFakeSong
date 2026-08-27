#!/usr/bin/env bash
# Reexecuta APENAS WavLM Original e HuBERT Original.
#
# MOTIVO: até 2026-08-21 o runner SSL aplicava a correção de banda mas NÃO a
# registrava no bloco `config` do results.json. Os artefatos das duas entradas
# saíam sem `band_correction_hz` enquanto as nove Keras traziam 7500.0 —
# lendo só os artefatos, a conclusão natural era que os SSL rodaram sem a
# correção. Não rodaram: o log do container mostra
# "[protocolo] correção aplicada a 12162 amostras". Mas log não acompanha
# artefato, e num TCC onde proveniência é argumento isso precisa estar no
# arquivo.
#
# CUSTO: ~11 minutos somados (backbone congelado, embeddings extraídos uma vez).
# Os números NÃO devem mudar — o protocolo é o mesmo; muda o que fica gravado.
# Se mudarem além do ruído de semente, algo mais está diferente: investigue
# antes de aceitar.
#
# QUANDO RODAR: depois que a bateria terminar. Rodar antes disputa a GPU com o
# treino em curso.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SAIDA="${XFAKE_BENCHMARK_OUT:-data/results/bateria_corrigida_15k_2026-08-20}"
DATASET="${XFAKE_BENCHMARK_DATASET:-data/datasets/benchmark_dataset_15k.npz}"

if docker ps --format '{{.Names}}' | grep -q '^xfakesong_benchmark_nvidia$'; then
  echo "ERRO: a bateria ainda esta rodando. Aguarde o fim para nao disputar a GPU." >&2
  exit 1
fi

# `--resume` puliria as duas (ja concluidas com fingerprint valido). O objetivo
# aqui e justamente REGRAVAR o artefato, entao o resume fica de fora.
docker run --rm --gpus all \
  -v "$(pwd):/app" -w /app \
  -v "$(pwd)/cache/huggingface:/app/cache/huggingface" \
  -e HF_HOME=/app/cache/huggingface \
  -e TF_CPP_MIN_LOG_LEVEL=2 -e PYTHONIOENCODING=utf-8 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  --name xfake_ssl_regravar \
  xfakesong/benchmark:nvidia \
  python scripts/benchmark/run_models_sequential.py \
    --dataset "$DATASET" \
    --out "$SAIDA" \
    --models "WavLM Original" "HuBERT Original" \
    --epochs 100 --device-profile gpu --snr 30 20 10 5 \
    --band-correction-hz 7500 --checkpoint-monitor val_eer

echo
echo "Conferindo o que ficou gravado:"
python - <<'PY'
import json, glob, os
saida = os.environ.get("XFAKE_BENCHMARK_OUT",
                       "data/results/bateria_corrigida_15k_2026-08-20")
for arq in ("wavlm_original", "hubert_original"):
    for p in glob.glob(f"{saida}/{arq}/results.json"):
        cfg = json.load(open(p, encoding="utf-8")).get("config", {})
        print(f"  {arq:<18} band_correction_hz = {cfg.get('band_correction_hz')}"
              f" | bloco = {bool(cfg.get('band_correction'))}")
PY
