# 30 — Auditoria e consolidação do pipeline de benchmark

Consolida, em um único lugar, a **ordem de execução por custo**, a garantia de
**dataset único**, a **política de fallback** e o **levantamento de todas as
configurações de treino e inferência**. Complementa [27](27_DATASET_PIPELINE.md)
(ciclo do dataset) e [15](15_BENCHMARK.md) (métricas/execução).

---

## 1. Ordem de execução por custo (menos → mais custoso)

Fonte única: **`benchmarks/config.py::ALL_TCC_ARCHITECTURES`** — consumida por
`run_models_sequential.py` (default), pelos presets `configs/training/*.yaml` e
pelo caderno `notebooks/pipeline/04_all_architectures_full_benchmark.ipynb`. Roda
primeiro o barato (valida o pipeline cedo e falha rápido) e deixa o caro por
último.

| # | Modelo | Família | Entrada | Batch (base) | Custo |
|---:|---|---|---|---:|---|
| 1 | RandomForest | clássico | features | — | muito baixo |
| 2 | SVM | clássico (RBF) | features | — | baixo |
| 3 | MultiscaleCNN | CNN | espectrograma | 64 | baixo |
| 4 | Sonic Sleuth | CNN | espectrograma | 32 | baixo-médio |
| 5 | EfficientNet-LSTM | CNN+LSTM | espectrograma | 32 | médio |
| 6 | Conformer | Conv+Attention | espectrograma | 32 | médio-alto |
| 7 | Hybrid CNN-Transformer | CNN+Transformer | espectrograma | 32 | médio-alto |
| 8 | SpectrogramTransformer | Transformer | espectrograma | 8 | alto |
| 9 | RawNet2 | SincConv+GRU | áudio bruto | 16 | alto |
| 10 | AASIST | Sinc+GAT | áudio bruto | 16 | alto |
| 11 | RawGAT-ST | Sinc+GAT | áudio bruto | 8 | muito alto |
| 12 | WavLM | SSL (fallback CNN-1D no TF) | áudio bruto | 8 | muito alto |
| 13 | HuBERT | SSL (real/fallback) | áudio bruto | 8 | muito alto |
| 14 | Ensemble | fusão | — | 16 | máximo (depende dos demais) |

**Heurística** (não há tempos medidos versionados): família (clássico < neural),
domínio de entrada (espectrograma << áudio bruto de 16 kHz × 5 s), complexidade
(CNN < LSTM < Transformer < GAT < SSL) e cap de batch por VRAM
(`benchmarks/planning.py::_fit_to_device`). Refine com a latência/tempo de uma
corrida completa (coluna *Eficiência* do `results.json`).

---

## 2. Dataset único para todos os modelos

O benchmark **garante por construção** que todos os modelos treinam/avaliam no
mesmo `.npz`:

- `run_models_sequential.py --dataset <npz>` passa o **mesmo** arquivo a cada
  `run_benchmark.py --model <nome>`.
- `benchmarks/data.py::BenchmarkData.from_npz` **re-divide** o conjunto de forma
  estratificada e reprodutível (semente fixa, `val=0.15`/`test=0.15`) — o teste é
  idêntico para todos os modelos, independente do split original do `.npz`.
- Todos os presets `configs/training/*.yaml` apontam para
  `app/datasets/benchmark_audio_raw_balanced_20k.npz` (nome canônico).

**Atualizar o dataset = regerar esse `.npz`** (não versionado) a partir dos
splits, e todos os modelos passam a usá-lo automaticamente:

```bash
python scripts/build_dataset.py --skip-download --target <N>   # dedup→balance→splits
python scripts/export_npz_from_splits.py \
    --out app/datasets/benchmark_audio_raw_balanced_20k.npz --sample-rate 16000 --duration-sec 5.0
```

> Estado atual (auditado em sessão): após o dedup, `real=7.500` único / `fake`
> pendente de re-balance. Decisão em aberto: 7.500/7.500 (15k) agora vs. baixar
> +2.500 reais para 10k/10k (20k). Ver [27](27_DATASET_PIPELINE.md).

---

## 3. Política de fallback (explícita, evitável)

Fallbacks silenciosos comprometem o benchmark — números do fallback **não são
comparáveis** aos modelos reais. Política:

- **SSL (WavLM/HuBERT):** no caminho TensorFlow, WavLM é *sempre* fallback CNN-1D
  (não existe `TFWavLMModel`); HuBERT cai no simplificado se o backbone real
  estiver indisponível. A flag **`XFAKE_STRICT_SSL=1`** faz esses modelos
  **abortar** (em vez de degradar em silêncio) — use no preset canônico para
  garantir integridade; sem ela, o fallback é permitido mas **logado
  explicitamente** (`strict_ssl_guard` em `architectures/ssl_utils.py`).
- O ambiente (real vs fallback) é registrado no `results.json` do benchmark.
- Backbones SSL reais (PyTorch, `*_original.pt`) só na inferência/Gradio.

> Recomendação TCC: rode o benchmark TF com a ressalva documentada (WavLM/HuBERT
> = proxy CNN-1D) **ou** ative `XFAKE_STRICT_SSL` e exclua/realoque WavLM para o
> caminho PyTorch.

---

## 4. Configurações de TREINO (levantamento)

### 4.1 Globais — `app/core/config/settings.py::TrainingConfig`
`batch_size=32`, `epochs=100`, `learning_rate=1e-3`, `validation_split=0.2`,
`test_split=0.1`, `early_stopping=True` (patience 10), `reduce_lr_on_plateau=True`
(patience 5), `optimizer="adam"`, `loss="binary_crossentropy"`.
Augmentation: `use_augmentation=True`, `snr_range_db=(5,40)`.
`use_class_weighting=True`; calibração `auto_calibrate_temperature=True`
(min 50 amostras). Opt-in: `use_swa`, `use_mixup` (α=0.2), `compute_ood_threshold`
(q=0.95), `export_onnx`/`export_onnx_int8`.

### 4.2 Hiperparâmetros por modelo — **3 fontes** (cuidado com drift)
1. `architectures/registry.py::default_params` — dropout, l2, patience,
   gradient_clip, augmentation_strength (consumido pelo `training_service`).
2. `architectures/<nome>.py::create_model(...)` — LR, weight_decay, clipnorm, loss.
3. `benchmarks/planning.py::NEURAL_BENCHMARK_HPARAMS` — lr, batch, optimizer,
   scheduler, warmup, label_smoothing (usado pelo benchmark quando
   `optimize_hyperparameters=True`, default). `_fit_to_device` aplica caps de
   batch por VRAM + gating de mixed precision.

### 4.3 Presets por família — `configs/training/*.yaml`
Todos: `dataset=...balanced_20k.npz`, `epochs=100`, `snr=[30,20,10]`,
`latency_runs=30`, `optimize_hyperparameters=true`. Listas de modelos em ordem de
custo crescente.

| Preset | `device_profile` | batch | Modelos |
|---|---|---:|---|
| `classical.yaml` | cpu | 32 | RandomForest, SVM |
| `tensorflow.yaml` | gpu | 32 | MultiscaleCNN, Sonic Sleuth, EfficientNet-LSTM, SpectrogramTransformer |
| `pytorch.yaml` | gpu | — | Conformer, Hybrid CNN-Transformer, RawNet2, AASIST, RawGAT-ST |
| `ssl.yaml` | gpu | 8 | WavLM, HuBERT |
| `retune_ajustado.yaml` | gpu | — | modelos do diagnóstico de retreino |

### 4.4 Runtime/perf (env) — `app/core/performance.py`
`XFAKE_TF_INTRA/INTER_OP_THREADS`, `XFAKE_NUM_WORKERS`, `XFAKE_ENABLE_XLA`,
`XFAKE_ENABLE_ONEDNN`, `XFAKE_GPU_MEMORY_GROWTH`, `XFAKE_GPU_MEMORY_LIMIT_MB`,
`XFAKE_CUDA_MALLOC_ASYNC`, `XFAKE_ENABLE_TF32` (opt-in, Ampere+).

---

## 5. Configurações de INFERÊNCIA (levantamento)

- `configs/inference.yaml`: `models_dir=app/models`,
  `benchmark_models_dir=app/models/benchmark_final`, `registry_path=.../registry.json`,
  `results_dir=results`, Gradio `0.0.0.0:7860`, caches HF/torch/TF.
- `app/domain/services/detection_service.py`: carrega modelos de `models_dir`;
  GPU via `app.core.gpu.setup_gpu` (memory-growth, probe cuDNN com fallback p/ CPU,
  mixed precision em Tensor Cores).
- `settings.py::APIConfig`: `0.0.0.0:8000`, rate limit 60 rpm, upload 100 MB.

---

## 6. Auditoria do pipeline — achados e correções (esta sessão)

| Achado | Estado |
|---|---|
| Resume de download duplicava falantes (800 dups fkvoice) | ✅ resume por estado + dedup |
| `hf_hub_download` travava sem timeout | ✅ `hf_download_retry` (timeout+retry) |
| Balance antes do dedup → desbalanceamento | ✅ ETAPA 1.5 dedup antes do balance |
| Normalização re-rodava tudo a cada build | ✅ cache idempotente (manifesto) |
| Hparams em 3 lugares, doc citava 2 | ✅ doc + comentários de ref. cruzada |
| WavLM/HuBERT fallback não explícito | ✅ `XFAKE_STRICT_SSL` + caveat nos docs |
| Ordem de modelos não era por custo | ✅ `ALL_TCC_ARCHITECTURES` reordenada |
| Caderno com EPOCHS=20 | ✅ alinhado a 100 (canônico) |

## 7. Pendências metodológicas (fora de correções pontuais)

1. **Split disjunto por falante real** — exige consolidar `speaker_manifest.json`
   (o tier `large` promete, mas o dado atual é estratificado).
2. **Vazamento de fonte BRSpeech-DF** (real+fake do mesmo corpus) — risco de
   aprender artefatos do corpus; mitigar com protocolo cross-generator/por-falante.
3. **Backbones SSL reais no benchmark** — portar WavLM/HuBERT para o caminho
   PyTorch para resultados comparáveis à literatura.
