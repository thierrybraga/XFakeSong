# 22 — Retreino em WSL2 (GPU)

> Runbook para executar o retreino + benchmark com os ajustes de robustez
> (ver [docs/21](21_PLANO_RETREINO.md)) numa GPU NVIDIA via **WSL2**.
> O TensorFlow ≥2.11 **não usa GPU no Windows nativo** — daí o WSL2.
> Esta máquina tem **RTX 3060**, suficiente para o retreino.

## Por que WSL2

- `tf.config.list_physical_devices("GPU")` retorna **vazio** no Windows nativo
  (TF ≥2.11). Sob WSL2 + `tensorflow[and-cuda]`, a RTX 3060 é exposta.
- O **dataset já existe** no repo (`app/datasets/benchmark_audio_raw_balanced_15k.npz`,
  2.769,01 MiB). Ele deriva de 15.000 WAVs ativos em PCM linear, 16 bits,
  mono, 16 kHz, somando 2.045,61 min de áudio validado — **não há download**.
- Os ajustes de código (P0–P3) já estão aplicados; aqui só se **executa**.

## Pré-requisitos (uma vez)

```powershell
# No PowerShell (Windows), instalar WSL2 + Ubuntu:
wsl --install -d Ubuntu
# Driver NVIDIA recente no Windows já habilita a GPU no WSL2 (não instale
# driver dentro do Ubuntu).
```

## Caminho A — venv nativo no WSL2 (recomendado, mais simples)

```bash
# Dentro do Ubuntu/WSL2, na raiz do repo (ex.: /mnt/d/Github/XFakeSong):
cd /mnt/d/Github/XFakeSong
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install 'tensorflow[and-cuda]'      # habilita CUDA/cuDNN p/ a GPU

# Validar GPU + ambiente (não treina):
bash scripts/retrain_wsl2.sh --check
```

> Observação de E/S: treinar lendo o `.npz` de 2.769,01 MiB via `/mnt/d` (disco
> Windows) é mais lento que copiar para o filesystem do WSL2. Para máxima
> velocidade: `cp app/datasets/benchmark_audio_raw_balanced_15k.npz ~/ds.npz`
> e use `--dataset ~/ds.npz`.

## Caminho B — container GPU (Docker Desktop + WSL2 backend)

```bash
# Requer NVIDIA Container Toolkit / GPU habilitada no Docker Desktop.
docker compose -f docker/compose/train.nvidia.yml build
# Validar GPU dentro do container:
docker compose -f docker/compose/train.nvidia.yml run --rm tensorflow-keras \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Rodar o benchmark sequencial dentro do container:
docker compose -f docker/compose/benchmark.nvidia.yml run --rm benchmark
```

## Executar o retreino + benchmark

O driver [scripts/retrain_wsl2.sh](https://github.com/thierrybraga/XFakeSong/blob/main/scripts/retrain_wsl2.sh) encapsula tudo:

```bash
# Retreino in-distribution dos modelos do artigo + HuBERT Original SSL:
bash scripts/retrain_wsl2.sh --indist

# Reteste cross-generator (segura o XTTS/fkvoice fora do treino) — P0.4:
bash scripts/retrain_wsl2.sh --xgen fkvoice

# Ambos em sequência:
bash scripts/retrain_wsl2.sh --indist --xgen fkvoice

# Uma única arquitetura (ex.: o P1 obrigatório):
bash scripts/retrain_wsl2.sh --model SpectrogramTransformer --indist
```

Equivalente "cru" (sem o driver), via orquestrador:

```bash
python scripts/run_clean_benchmark_pipeline.py \
  --phase full \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --epochs 100 \
  --batch-size 32 \
  --device-profile gpu \
  --timeout-min 240 \
  --build
```

### HuBERT Original em WSL/Docker

O HuBERT real não deve passar pelo fallback Keras. No fluxo Docker, ele é
roteado para `scripts/run_wavlm_original_benchmark.py`, baixa
`facebook/hubert-base-ls960`, congela o backbone (`--freeze-backbone`) e treina
somente a cabeça classificadora PyTorch sobre embeddings SSL.

```bash
python scripts/run_clean_benchmark_pipeline.py \
  --models "HuBERT Original" \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --epochs 100 \
  --batch-size 32 \
  --ssl-feature-batch-size 16 \
  --device-profile gpu \
  --timeout-min 240 \
  --build
```

Para revisar sem iniciar treino nem baixar pesos:

```bash
python scripts/run_models_sequential.py \
  --models "HuBERT Original" \
  --plan-only \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz
```

### Ablação do WavLM (opcional, P2)

```bash
python scripts/ablate_wavlm_finetune.py \
  --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
  --lrs 1e-5 3e-5 1e-4 --epochs 30 --out results/ablation_wavlm
```

### Pruning do MultiscaleCNN (opcional, P3)

```bash
pip install tensorflow-model-optimization   # ausente por padrão
# from app.domain.models.training.magnitude_pruning import prune_and_finetune
```

## Saídas e tempo

- Cada execução grava em `results/retrain_wsl2_*/`: métricas por arquitetura,
  robustez AWGN, eficiência e **figuras** (`roc.png`, `robustez.png`,
  `eficiencia.png`, `convergencia.png`, `confusion_matrices.png`) via
  [benchmarks/report.py](https://github.com/thierrybraga/XFakeSong/blob/main/benchmarks/report.py).
- **Tempo**: o preset completo com os modelos do artigo + HuBERT Original em
  RTX 3060 leva **horas**; HuBERT domina parte relevante do tempo por extrair
  embeddings do backbone SSL. Rode em sessão persistente (`tmux`/`nohup`).

## Atualizar o TCC com os novos resultados (automatizado)

O `retrain_wsl2.sh` **já faz isso ao final** de cada execução: consolida o run,
regenera as figuras (substituindo as antigas in-place em `tcc_overleaf/figures/`)
e escreve o fragmento de tabelas. Manualmente, o fluxo é:

```bash
# 1) Consolidar os resultados → benchmark_summary.json + figuras nomeadas do TCC
#    (sobrescreve as antigas). Aceita um run completo OU vários runs por arquitetura.
python scripts/consolidate_results.py results/retrain_wsl2_indist \
    --out results/tcc_consolidated --copy-to tcc_overleaf/figures

# 2) Gerar as TABELAS data-driven (fragmento .tex, NÃO mexe na prosa da tese)
python scripts/update_tcc_latex.py \
    --summary results/tcc_consolidated/benchmark_summary.json \
    --output tcc_overleaf/tabelas_benchmark.tex --figures-dir figures

# 3) No main.tex, incluir uma vez:   \input{tabelas_benchmark.tex}
#    e recompilar.
```

[scripts/consolidate_results.py](https://github.com/thierrybraga/XFakeSong/blob/main/scripts/consolidate_results.py) deriva tudo
de `results/<run>/results.json` (acurácia/AUC/EER/robustez, latência/tamanho,
matrizes de confusão a partir de `scores_clean`+`y_test`, e a estabilidade a
partir do `history`). Gera exatamente os nomes que o TCC usa:
`benchmark_accuracy_auc.png`, `benchmark_eer.png`, `benchmark_robustness.png`,
`benchmark_latency.png`, `benchmark_size.png`, `training_stability.png` e
`confusion_matrices/<slug>.png`.

> **Defasagem corrigida:** o `update_tcc_latex.py` antes lia um `.tex` morto em
> `C:\Users\...\.codex\attachments` e tinha métricas/épocas hardcoded. Agora é
> data-driven (consome o summary) e, por padrão, gera só o **fragmento de
> tabelas** (seguro). A reescrita da tese inteira existe sob `--full-rewrite`
> mas é frágil (exige marcadores de seção idênticos) — **não** é o caminho
> recomendado.

> Todas as 20 figuras hoje em `tcc_overleaf/figures/` **estão em uso** pelo
> `main.tex` — o passo 1 as **substitui in-place** pelas novas; não apague nada.
