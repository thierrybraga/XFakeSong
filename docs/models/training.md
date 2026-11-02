# Treinamento das Arquiteturas

Este documento detalha o processo de treinamento, configurações de otimização e peculiaridades de cada arquitetura implementada no SpeechChecker.

## Ambientes consolidados

O treinamento operacional foi organizado por família computacional. Os arquivos
ficam em `infrastructure/docker/environments/`, os presets em `configs/training/`
e o entrypoint por família em `tools/training/train_by_family.py`.

| Família | Config | Modelos |
|---|---|---|
| `classical-tabular` | `classical.yaml` | RandomForest, SVM |
| `spectral-convolutional` | `spectral_convolutional.yaml` | MultiscaleCNN |
| `spectral-attention` | `tensorflow.yaml` | Conformer, CCT, AST |
| `waveform-end-to-end` | `pytorch.yaml` | RawNet2, AASIST, RawGAT-ST |
| `ssl-pretrained` | `ssl.yaml` | WavLM Original, HuBERT Original |
| `extended` | `extended.yaml` | Sonic Sleuth, EfficientNet-LSTM, Ensemble |
```bash
python tools/training/train_by_family.py --family classical-tabular --plan-only
python tools/training/train_by_family.py --family spectral-convolutional --models MultiscaleCNN --epochs 100 --device-profile gpu
python tools/training/train_by_family.py --family waveform-end-to-end --models RawNet2 AASIST --epochs 100 --device-profile gpu
python tools/training/train_by_family.py --family ssl-pretrained --models "WavLM Original" "HuBERT Original" --epochs 100 --device-profile gpu
```

Todos os entrypoints chamam `benchmarks/run_models_sequential.py`, que executa
`benchmarks/run_benchmark.py --model <nome>` para cada arquitetura, salvando logs,
modelos, métricas e figuras em pasta própria por modelo.

## 1. Pipeline de Treinamento

### 1.1 Fluxo Geral

```
TrainingService.train_model(architecture, dataset_path, config)
    │
    ├─ get_architecture_info()           ← registry.py: valida arquitetura
    ├─ np.load(dataset_path)             ← suporta .npz com X_train/y_train[/X_val/y_val]
    ├─ create_model_fn(input_shape, num_classes, **params)  ← factory.py
    ├─ ModelTrainer(TrainingConfig)
    │       ├─ SecureTrainingPipeline     ← divisão 70/15/15 sem data leakage
    │       ├─ AudioAugmenter            ← 7 técnicas de augmentation
    │       ├─ model.compile(optimizer, loss, metrics)
    │       ├─ model.fit(...)            ← com callbacks
    │       └─ save_model()               ← model.keras + scaler.pkl + config.json
    └─ ModelMetadata                     ← retorna métricas + caminho do arquivo
```

### 1.2 Formato de Dados

O `TrainingService` aceita arquivos `.npz` com as seguintes chaves:

| Chave | Tipo | Descrição |
|-------|------|-----------|
| `X_train` | ndarray | Features de treino |
| `y_train` | ndarray | Labels de treino |
| `X_val` | ndarray | Features de validação (opcional) |
| `y_val` | ndarray | Labels de validação (opcional) |

Se `X_val`/`y_val` estiverem ausentes, o `SecureTrainingPipeline` cria automaticamente os splits (70/15/15) com verificação de leakage.

O parâmetro `num_classes` pode ser passado no `config`; se omitido, é inferido automaticamente via `np.unique(y_train).size`.

### 1.2.1 Datasets de treino — tiers `small` / `medium` / `large`

O tamanho e a composição do dataset são padronizados em **tiers** definidos em
`app/domain/dataset_metadata/dataset_catalog.py` (`DATASET_TIERS`) — fonte única de verdade
compartilhada por `tools/dataset/build_dataset.py`, pela aba Datasets do Gradio, pelo
benchmark e pela documentação. Escolher um tier pré-configura tamanho, fontes e
estratégia de split. Detalhes de fontes/licenças em
[docs/data/public-datasets.md](../data/public-datasets.md).

| Tier | Por classe | Total | Split | Habilita | Uso típico no treino |
|------|-----------:|------:|-------|----------|----------------------|
| `test` | 100 | 200 | 70/15/15 estratificado | nenhum (abaixo do mínimo clássico) | smoke de ponta a ponta |
| `small` | 5.000 | 10.000 | 70/15/15 estratificado | clássicos, CNNs leves e principais neurais | iteração robusta |
| `medium` | 7.500 | 15.000 | 70/15/15 estratificado | todas as 14 arquiteturas do fluxo genérico | treino/avaliação auxiliar |
| `large` | 10.000 | 20.000 | **disjunto por falante** + cross-generator | todas as 14 + auditoria de falantes | execução estendida |

> ⚠️ **Os tiers pertencem ao fluxo legado.** Para reproduzir o estudo do TCC,
> use `data/datasets/benchmark_dataset_15k.npz` (7.500 pares), conforme o
> [artigo canônico](../evaluation/canonical-study-and-legacy-inventory.md).
> O corpus pai e a exportação integral de 40.980 amostras são documentados no
> [Protocolo de Dataset](../data/dataset-protocol.md); o NPZ integral não está
> presente neste checkout. Os comandos `build_dataset.py --tier` só reconstroem
> os artefatos antigos.

**Montar um tier** (download + balanceamento + splits + `dataset_config.json`):

```bash
python tools/dataset/build_dataset.py --tier small     # 5.000/classe, 10k total
python tools/dataset/build_dataset.py --tier medium    # 7.500/classe, tier genérico (não é o artigo)
python tools/dataset/build_dataset.py --tier large     # 10.000/classe, split por falante

# override do tamanho mantendo fontes/split do tier:
python tools/dataset/build_dataset.py --tier medium --target 7500
```

> **Contexto:** a seção de tiers abaixo descreve o fluxo legado genérico de
> construção/treino, não a configuração canônica do artigo. A reprodução do
> TCC usa o artefato `benchmark_dataset_15k.npz` e está documentada no
> [inventário canônico](../evaluation/canonical-study-and-legacy-inventory.md).

**Como o tier afeta o treino legado:**

- **Prontidão por modelo.** O treino só habilita um modelo quando o tier atinge o
  mínimo de amostras dele — `small` já cobre os principais modelos; `medium`
  cobre o conjunto do fluxo genérico; `large` aumenta a margem
  para auditorias.
- **Split.** `test/small/medium` usam 70/15/15 estratificado; `large` usa split
  **disjunto por falante** (`speaker_manifest.json`), medindo generalização a
  usuários não vistos e evitando vazamento de falante entre treino e teste.
- **Pipeline de ponta a ponta** (download → benchmark) por tier, no fluxo legado:

```bash
python benchmarks/run_tcc_pipeline.py --download --tier medium --full-benchmark --npz data/datasets/legacy_medium_15k.npz
```

Recomendação prática no fluxo legado: prototipe hiperparâmetros em `small`,
produza os números finais no `medium` e use `large` para auditoria de falantes.

**No fluxo de construção do corpus pai** não há tier a escolher; o estudo do
artigo usa o alvo explícito de 15.000 amostras durante a exportação:

```bash
python tools/dataset/export_paired_npz.py --out data/datasets/benchmark_dataset_15k.npz --target-samples 15000 --no-compress
```

### 1.3 ModelTrainer

Classe principal de treinamento (`app/domain/models/training/trainer.py`). Os
callbacks ficam em `callbacks.py`, a calibração pós-treino em `calibration.py`,
o salvamento e o `input_contract` em `artifacts.py` e o EER de validação em
`metrics.py`. Funcionalidades:

- **Compilação**: respeita o `compile()` feito pelo `create_model` de cada arquitetura (LR, loss, clipnorm); quando recompila, o otimizador vem do `OptimizerFactory`
- **SecureTrainingPipeline**: Previne data leakage — aplica `StandardScaler` apenas com dados de treino e armazena o scaler para inferência
- **Class weighting automático** (`use_class_weighting=True`): calcula `compute_class_weight('balanced')` e passa via `model.fit(class_weight=...)`. Compensa datasets desbalanceados típicos de deepfake (ASVspoof ~9:1). Suporta y categórico e one-hot.
- **Calibração automática de temperatura** (`auto_calibrate_temperature=True`): após o treino, busca o T que minimiza NLL no val set (Guo et al., ICML 2017). Salvo no `input_contract` e aplicado automaticamente pelo Predictor. Requer ≥ `calibration_min_samples` (default 50) no val set.
- **Sprint 2.3 — Stochastic Weight Averaging (SWA)** (`use_swa=True`, opt-in): mantém média móvel dos pesos nas últimas ~20% épocas e aplica ao final do treino. +0.5–1.5% accuracy típica (Izmailov et al., UAI 2018). Aplicado via `SWACallback`. Compatível com BatchNorm (recomputa stats automaticamente se `bn_update_data` for fornecido).
- **Sprint 2.4 — Mixup** (`use_mixup=True`, opt-in): interpola pares de amostras no batch (`λ ~ Beta(α, α)`). Convertido em soft labels automaticamente. **Importante**: incompatível com `class_weighting` (Keras requer int classes); o trainer detecta e desativa class weighting automaticamente.
- **Sprint 2.5 — OOD threshold** (`compute_ood_threshold=True`): calibra threshold de Out-of-Distribution no val set usando energy score / entropia (quantile=0.95). Predições com score abaixo do threshold são marcadas como `is_ood=True` pelo Predictor.
- **Sprint 3.2 — Mixed precision auto** (`use_mixed_precision=None` → auto): detecta GPU Tensor Core (CC≥7.0: V100, T4, RTX 20xx+, A100, H100) e habilita `mixed_float16` automaticamente. ~2× speedup + 50% menos VRAM.
- **Sprint 3.4 — ONNX export**: NÃO roda no treino. As flags `export_onnx`/`export_onnx_int8` do `TrainingConfig` não são lidas — o método do trainer que as usaria nunca teve chamador e saiu na divisão do módulo (2026-10-31). A exportação existe sob demanda pela API (`POST /api/v1/training/export-onnx`, `app/interfaces/web/routers/training.py`), sobre um modelo já carregado. Requer `pip install tf2onnx onnxruntime`.
- **Sprint 4.5 — EER threshold adaptativo**: calibra automaticamente o threshold de Equal Error Rate (FPR=FNR) no val set ao final do treino. Salvo em `input_contract.eer_threshold` e usado pelo Predictor como threshold de classificação (em vez de 0.5 fixo). Reportado também `eer_value` (métrica anti-spoofing padrão).
- **input_contract**: Salvo junto ao modelo — garante que inferência use o mesmo formato/features/temperatura/ood_threshold/eer_threshold do treino
- **Formato de save**: `.keras` (formato nativo Keras 3, substitui o legado `.h5`)

### 1.7 K-fold Cross Validation (Sprint 4.1)

```python
from app.domain.services.training_service import TrainingService

service = TrainingService(models_dir="data/models")
cv_result = service.cross_validate_model(
    architecture="AASIST",
    dataset_path="dataset.npz",
    config={"epochs": 50, "batch_size": 16, "learning_rate": 0.0008},
    n_folds=5,
    save_fold_models=False,  # apenas métricas, não persiste modelos
)
if cv_result.status.value == "success":
    agg = cv_result.data['aggregated']
    print(f"Accuracy: {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}")
    print(f"Best fold: {cv_result.data['best_fold']}")
```

### 1.8 Hyperparameter Tuning com Optuna (Sprint 4.2)

```python
from app.domain.models.training.hyperparameter_tuning import (
    suggest_search_space,
    tune_hyperparameters,
)

# Search space sugerido por arquitetura
search_space = suggest_search_space('AASIST')  # ou customizado

result = tune_hyperparameters(
    architecture='AASIST',
    dataset_path='dataset.npz',
    base_config={'epochs': 20, 'optimizer': 'adamw'},  # poucas épocas
    search_space=search_space,
    n_trials=30,
    metric='val_accuracy',
    direction='maximize',
    use_pruning=True,  # Hyperband — poda trials ruins cedo
)
print(f"Best: {result['best_params']} → {result['best_score']:.4f}")
```

### 1.10 Multi-Model Fusion (Sprint 4.4)

```python
result = detection_service.detect_multi_model(
    audio_data,
    model_names=['AASIST_v1', 'Conformer_v1', 'RawNet2_v1'],
    fusion='weighted_avg',         # ou 'soft_voting', 'majority_vote', 'max_conf'
    weights=[0.4, 0.4, 0.2],       # opcional; default uniforme
    use_tta=True,
)
fused = result.data
print(f"Consenso: is_fake={fused.is_fake}, conf={fused.confidence:.3f}")
print(f"Acordo entre modelos: {fused.metadata['model_agreement']:.2%}")
for r in fused.metadata['per_model']:
    print(f"  {r['model']}: {r['fake_prob']:.3f}")
```

### 1.14 MC Dropout para Uncertainty (Sprint 5.4)

Quantifica incerteza epistêmica via N forward passes com dropout ativo:

```python
# Uso direto via Predictor (em DetectionService)
result = service.predictor.predict_with_uncertainty(
    model_info, features, n_samples=20,
)
data = result.data
print(f"is_fake={data['is_deepfake']}  conf={data['confidence']:.3f}")
print(f"Epistemic uncertainty: {data['epistemic_uncertainty']:.4f}")
print(f"Predictive entropy:    {data['predictive_entropy']:.4f}")

# Decisão "abstenha-se" em casos de alta incerteza
if data['is_uncertain']:
    print("⚠️  Modelo incerto sobre esta amostra — submeter a humano")
else:
    print(f"✓  Decisão confiável: {'fake' if data['is_deepfake'] else 'real'}")
```

### 1.4 Callbacks Padrão

Definidos em `_prepare_callbacks()`:

| Callback | Função |
|----------|--------|
| `TerminateOnNaN` | Interrompe imediatamente se loss = NaN/Inf |
| `EarlyStopping` | `val_loss`, patience=`early_stopping_patience`, restaura melhores pesos |
| `ReduceLROnPlateau` | Fator 0.5, min_lr=1e-7 quando val_loss estagna |
| `ModelCheckpoint` | Salva melhor modelo por val_loss (se `checkpoint_path` passado) |
| `BackupAndRestore` | Preserva pesos/otimizador/época em `backup_dir` (se passado) — permite retomar o `fit()` de onde parou após reinício do host/Docker; `benchmarks/fit_neural.py` sempre passa `backup_dir`, então todo treino do benchmark tem esse resume automático |
| `TensorBoard` | Histogramas + gráfico (se `tensorboard_dir` passado) |
| `CSVLogger` | Histórico em CSV (se `csv_log_path` passado) |
| `EpochProgressLogger` | Log `[TRAIN]` de progresso por época/batch (intervalo via `XFAKE_TRAIN_BATCH_LOG_INTERVAL_S`, default 60s); desde 2026-07-31 também loga `rss_mb` (RSS do processo, `/proc/self/status`) — diagnóstico de memória por batch |

`.batch(batch_size)` encadeia `.prefetch(tf.data.AUTOTUNE)` (tanto no caminho
`from_tensor_slices` quanto no gerador para datasets grandes, `trainer.py`) —
sem isso a GPU ficava ociosa esperando o próximo lote (~30% de utilização
observada no Conformer) em vez de sobrepor preparo de dado (CPU) com cômputo
(GPU).

Para RawNet2/AASIST/RawGAT-ST/MultiscaleCNN, o **treino do benchmark**
(não o `ModelTrainer` isoladamente) desliga o auto-JIT do XLA no subprocesso
(`XFAKE_ENABLE_XLA=0`, `_XLA_UNFRIENDLY_TRAINING_ARCHITECTURES` em
`benchmarks/run_models_sequential.py`) — o auto-JIT desses grafos
(SincConv, GAT dinâmico, GRU) estourava RAM do host ou VRAM da GPU já no 1º
batch. É a contraparte, do lado do treino, da lista
`Predictor._XLA_UNFRIENDLY_ARCHITECTURES` já usada na inferência (ver
`docs/models/inference.md`) — os dois conjuntos de arquiteturas não são
idênticos porque a causa raiz (compile-time vs. runtime) é diferente.

### 1.5 Data Augmentation

`AudioAugmenter` (`app/domain/models/training/augmentation.py`) implementa 7 técnicas via `tf.data.Dataset.map()`:

| Técnica | Descrição |
|---------|-----------|
| `_add_noise` | Ruído gaussiano (σ = `noise_factor`) |
| `_time_shift` | Deslocamento circular temporal (`tf.roll`) |
| `_volume_change` | Fator de amplitude aleatório ±`volume_factor` |
| `_frequency_mask` | SpecAugment: zera faixa de frequência |
| `_time_mask` | SpecAugment: zera faixa temporal |
| `_rawboost` | RawBoost (Tak et al. 2022): ruído convolutivo + impulsivo + estacionário |
| `_codec_simulation` | Artefatos de codec lossy (MP3/AAC): quantização + atenuação HF |

Mixup no treino é aplicado por `apply_mixup_to_dataset()` quando `use_mixup=True` (ver §1.3).

### 1.6 Otimizadores e Losses

`OptimizerFactory` suporta: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `adadelta`, `adamax`, `nadam`.

As losses são definidas no `create_model` de cada arquitetura (ex.: AM-Softmax/OC-Softmax em `head_layers.py`). A antiga `LossFactory` nunca teve chamador e saiu em 2026-10-18.

`WarmupCosineDecaySchedule`: warmup linear seguido de decaimento cosseno — recomendado para Conformer e Transformers.

---

## 2. Hiperparâmetros por Arquitetura

`get_recommended_hyperparameters()` (`app/domain/models/training/hyperparameter_defaults.py`)
é **código morto** — o próprio docstring do módulo o marca como extraído de
um módulo depreciado sem chamador real. As 9 arquiteturas do escopo oficial
usam `benchmarks/planning.py::NEURAL_BENCHMARK_HPARAMS` (efetivas sob
`optimize_hyperparameters=True`, o default); as do escopo estendido (Sonic
Sleuth, EfficientNet-LSTM, Ensemble, WavLM/HuBERT porte Keras) e classificam
Batch/LR não capturados por `planning.py` usam
`architectures/registry.py::default_params` + o `create_model(...)` de cada
arquitetura.

| Arquitetura | Batch (GPU) | LR | Épocas | Dropout | Regularização | Observação |
|-------------|-------|----|--------|---------|-----|------------|
| **AASIST** | 24 | 3e-4 | 100 | 0.2 | L2 2e-4 | raw audio + SincConv + GAT S/T; janela 48.000 (3 s) |
| **RawGAT-ST** | 16 | 5e-5 | 100 | 0.35 | L2 1e-3 | raw audio + GAT espectral/temporal; janela 48.000 (3 s) |
| **MultiscaleCNN** | 32 (base 64, cap de VRAM) | 1e-3 | 100 | 0.5 | wd 1e-2 (AdamW) | Res2Net-50; `l2_reg_strength` do plano é config morto |
| **SpectrogramTransformer** | 8 | 1e-5 | 100 | 0.25 | wd 1e-5 | AST pré-LN, pesos AudioSet transferidos; warmup=3000 |
| **Conformer** | 32 | 1e-4 | 100 | 0.1 | wd 1e-4 | Conformer-M (16 blocos, 4 cabeças fixas — `attention_heads` do plano é config morto); warmup=1500 |
| **EfficientNet-LSTM** | 16 (extended.yaml) | ver `create_model` | 100 | 0.25 | — | tenta ImageNet; fine-tune bloco final quando disponível |
| **Hybrid CNN-T** | 32 | 3e-4 | 100 | 0.2 | L2 1e-4 | CCT + WarmupCosineDecay (warmup=1500) |
| **RawNet2** | 16 | 1e-4 | 100 | 0.3 | L2 1e-4 (dead) / wd 1e-4 | SincNet/FMS; GRU(1024) 1 camada; mixed precision desligado |
| **WavLM Original / HuBERT Original** | 128 (cabeça) | 1e-3 | 100 | — | wd 1e-4 | backbone PyTorch congelado; runner SSL dedicado |
| **WavLM / HuBERT** (porte Keras) | 16 (extended.yaml) | ver `create_model` | 100 | 0.2 / 0.3 | — | backbone congelado, soma ponderada estilo SUPERB |
| **Sonic Sleuth** | 16 (extended.yaml) | ver `create_model` | 100 | 0.3 | — | variante `sonic_sleuth_paper` usa dropout 0.1, não é o default |
| **Ensemble (adaptive)** | 16 (extended.yaml) | ver `create_model` | 100 | — | — | parte de modelos pré-treinados |
| **SVM** | Full | N/A | N/A | — | — | StandardScaler obrigatório |
| **Random Forest** | Full | N/A | N/A | — | — | n_jobs=-1 |

Batch "GPU" já reflete o cap de VRAM de `planning.py::_fit_to_device`
(RawNet2/RawGAT-ST/AST capados em 16, AASIST em 24, demais em 32) — não o
`batch_size` bruto do plano. Linhas marcadas "ver `create_model`"/"config
morto" não têm um valor único confiável para citar aqui: ou o parâmetro
nunca chega ao construtor (não é lido por `create_model`), ou o LR é o
default hardcoded na assinatura da função — confira o arquivo da arquitetura
em `app/domain/models/architectures/<nome>.py` antes de citar um número.

---

## 3. Peculiaridades por Arquitetura

### 3.1 AASIST

Descreve a variante **default** (paper-faithful — ver
[§ "Sobre as variantes 'default'"](#sobre-as-variantes-default-do-aasist-e-rawgat-st)
abaixo). As peculiaridades abaixo NÃO se aplicam a `variant="cnn_gru_simple"`
(legado, espectrograma + CNN 2D + Bi-GRU), que só existe para recarregar
checkpoints antigos.

- Entrada: **áudio bruto**, janela canônica 48.000 amostras (3 s @ 16 kHz);
  `AudioFeatureNormalization` normaliza a forma de onda antes do front-end
  `SincConvLayer` (`aasist.py`) — não há reshape para espectrograma.
- Encoder: Sinc 2D + GAT espacial/temporal + master/HS-GAL/MGO (grafo
  heterogêneo, temperatura 100 nos ramos HS-GAL); cabeça `AM-Softmax`.
- Crop aleatório no treino, multicrop (3 recortes, média dos scores) na
  avaliação para arquiteturas raw-audio.

### 3.2 RawGAT-ST

Mesma ressalva: descreve a variante default (paper-faithful).

- Entrada: **áudio bruto**, janela canônica 48.000 amostras (3 s @ 16 kHz).
- Dois encoders 2D + GAT espacial/temporal + fusão por produto + terceiro GAT
  (`GraphPoolLayer` top-k, `AASISTGraphAttentionLayer`).
- Mesmo protocolo de crop aleatório (treino) / multicrop (avaliação) do
  AASIST.

### 3.3 EfficientNet-LSTM

- Backbone `EfficientNetB0`: tenta pesos ImageNet; se o ambiente estiver offline
  ou o download falhar, cai para `weights=None`.
- Fine-tuning: com ImageNet, congela o backbone e deixa treináveis `block7` e
  `top_*`; no fallback sem pesos, treina do zero.
- Pré-processamento in-model: `MelSpectrogramFrontEnd` → `DeltaFeatureLayer`
  (3 canais) → resize 224×224×3.
- Sequência temporal via Bi-LSTM[256, 128] + `AttentionLayer`.

### 3.4 MultiscaleCNN (Res2Net-50)

- Config: `[3, 4, 6, 3]` blocos, `baseWidth=26`, `scale=4`
- Pré-processamento in-model: STFT + log-mel
- `Bottle2neck`: divisão em `s=4` grupos hierárquicos → representação multi-escala dentro do bloco

### 3.5 SpectrogramTransformer

- Processamento in-model: áudio bruto → mel spectrogram via `STFTLayer`
- `ConvolutionStemLayer` para extração de patches
- Positional encoding aprendível
- Config ViT-Base: patch_size=16, d_model=768, heads=12, layers=12

### 3.6 Conformer

- `ConformerBlock`: FeedForward×½ + SelfAttention + ConvolutionModule + FeedForward×½
- Usa `WarmupCosineDecaySchedule` — crítico para estabilidade do treinamento
- `ConvolutionModule`: Pointwise → GLU → Depthwise → Swish → Pointwise
- Subsampling 4× na entrada para reduzir sequência

### 3.7 Hybrid CNN-Transformer (CCT)

- Conv Tokenizer: 2× [Conv2D + ReLU + MaxPool(3, stride=2)] — sem patch embedding estático
- Transformer: 4 camadas, 4 heads, 256 dims, pre-norm, stochastic depth
- Sequence Pooling: atenção ponderada (sem CLS token)
- Pré-processamento in-model: mel spectrogram 128 bins

### 3.8 Ensemble (Multi-Spectrogram Fusion)

- 4 variantes: `ensemble` (feature fusion), `ensemble_score`, `ensemble_lite` (2 branches), `ensemble_adaptive` (5 branches)
- `ensemble_adaptive` (TCC Eq. 27-28): pesos adaptativos por confiança
- Branch 1: Mel (128 mels) → CNN+SE; Branch 2: LFCC (20); Branch 3: CQT (84 bins); Branch 4: MFCC (20)
- Fusão: `CrossAttentionFusionLayer` + `GatedFusionLayer` → Dense(512→256→128)
- Recomendado treinar branches individualmente e depois fazer fine-tuning do ensemble

### 3.9 RawNet2

- Primeira camada: filtros SincNet/Conv1D aprendíveis (banco passa-banda)
- Blocos residuais com `Feature Map Scaling (FMS)` — mecanismo de atenção de canal leve
- `AudioResamplingLayer` (→ 16 kHz) + `AudioNormalizationLayer` (μ=0, σ=1) in-model

### 3.10 WavLM / HuBERT

- Modo completo: carrega `microsoft/wavlm-base` via `transformers` (768-dim embeddings)
- Modo simplificado: CNN 1D compatível com Keras 3 (sem HuggingFace)
- Usa `WeightedSumLayer` para combinar hidden states (call via `list(hidden_states)`)
- Estratégia recomendada: congelar backbone, treinar apenas head por 20–50 épocas

### 3.11 Sonic Sleuth

- Pré-processamento in-model: áudio bruto → LFCC + MFCC + CQT
- Pipeline leve: 3× Conv2D(32→64→128) + MaxPool → Flatten → Dense(256→128) → Dropout(0.1) → Dense(1, sigmoid)
- Melhor resultado documentado: **98,27% accuracy / EER 0,016** (ASVspoof 2019 + In-the-Wild + FakeAVCeleb)

### 3.12 SVM e Random Forest

- Requerem features tabulares pré-extraídas `(batch, n_features)`
- Não suportam mini-batch: todo o dataset deve caber em memória
- `StandardScaler` encapsulado no pipeline sklearn
- SVM: `SVC(kernel='rbf', probability=True)` — obrigatório para `predict_proba`

---

## 4. Salvamento de Modelos

| Arquivo | Conteúdo |
|---------|----------|
| `<nome>.keras` | Modelo Keras 3 (arquitetura + pesos) |
| `<nome>_config.json` | `input_contract` + metadados de treinamento |
| `scaler.pkl` | `StandardScaler` treinado (para modelos que usam normalização) |

O `input_contract` salvo no JSON garante consistência entre treino e inferência:
```json
{
  "type": "features",
  "format": "spectrogram",
  "input_shape": [128, 80],
  "architecture": "AASIST",
  "sample_rate": 16000,
  "scaler_applied": false
}
```

---

## 5. Avaliação e Métricas

`MetricsCalculator` (`app/domain/models/training/metrics.py`) calcula:

- **Básicas**: accuracy, precision, recall, F1 (weighted + macro)
- **Probabilísticas**: ROC-AUC, PR-AUC (binário); ROC-AUC OvR/OvO (multiclasse)
- **Confusão**: TP, TN, FP, FN, specificity, sensitivity, FPR, FNR, PPV, NPV
- **EER (Equal Error Rate)**: ponto onde FPR = FNR (métrica padrão ASVspoof),
  por interpolação (`scipy.interpolate.interp1d` + `brentq`) sobre os
  thresholds finitos de `roc_curve`, com fallback ao ponto de menor
  `|FPR-FNR|` se a interpolação falhar. **Correção 2026-07-31**: sklearn
  ≥1.3 passou a devolver `thresholds[0] = inf` em `roc_curve`, o que
  entrava no eixo x do `interp1d` e produzia `NaN` — o `brentq` abortava e
  **todo** cálculo de EER do projeto caía silenciosamente no fallback (menos
  preciso) sem que a interpolação nunca de fato executasse. Os thresholds
  não-finitos agora são descartados antes da interpolação.
- **min t-DCF**: `calculate_min_tdcf()`, sobre a curva DET interna (`_det_curve`)


---

## 6. Uso via API

```python
from app.domain.services.training_service import TrainingService

service = TrainingService(models_dir="data/models")
result = service.train_model(
    architecture="AASIST",
    dataset_path="dataset.npz",
    config={
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.0008,
        "optimizer": "adamw",
        "loss_function": "binary_crossentropy",
        "metrics": ["accuracy"],
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "use_augmentation": True,
        # Sprint 1.3: class weighting automático (default True)
        "use_class_weighting": True,
        # Sprint 1.4: calibração automática de temperatura (default True)
        "auto_calibrate_temperature": True,
        "calibration_min_samples": 50,
        "num_classes": 2,
        "parameters": {
            "dropout_rate": 0.2,
            "l2_reg_strength": 0.0001,
        }
    }
)
if result.status.value == "success":
    print(f"Modelo salvo em: {result.data.file_path}")
```

### Sobre as variantes "default" do AASIST e RawGAT-ST

A partir do Sprint 1.2, a variante `"default"` em `AASIST.create_model` e
`RawGAT-ST.create_model` é **alias para a versão paper-faithful** (`"aasist"` /
`"rawgat_st"`) — antes era um CNN 2D + Bi-GRU simples. O comportamento legado
continua disponível via `variant="cnn_gru_simple"`.

```python
# Cria AASIST paper-faithful (SincConv + GAT + HS-GAL + AM-Softmax)
model = create_model_by_name("AASIST", input_shape, num_classes=2)

# Comportamento legado (CNN+Bi-GRU)
model = create_model_by_name("AASIST", input_shape, num_classes=2,
                              variant="cnn_gru_simple")
```

## Execução canônica por famílias

Use `tools/training/train_by_family.py`; `train_advanced.py` é um caminho
legado de debug, bloqueado por padrão e não produz resultados acadêmicos.
As famílias canônicas são: `classical-tabular`, `spectral-convolutional`,
`spectral-attention`, `waveform-end-to-end`, `ssl-pretrained` e `extended`.
O último escopo é exploratório e exige `--no-academic-protocol`.

Os presets usam o dataset anterior, teste selado e 100 épocas. Para
resultados finais, execute múltiplas sementes com teste congelado e reporte
média, desvio e comparações pareadas, além dos ICs bootstrap por cluster.
