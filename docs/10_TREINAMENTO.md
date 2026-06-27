# Treinamento das Arquiteturas

Este documento detalha o processo de treinamento, configurações de otimização e peculiaridades de cada arquitetura implementada no XFakeSong.

## Ambientes consolidados

O treinamento operacional foi organizado por família computacional. Os arquivos
ficam em `environments/`, os presets em `configs/training/` e os entrypoints em
`scripts/train_*.py`.

| Família | Config | Entrypoint | Modelos |
|---|---|---|---|
| `classical-ml` | `configs/training/classical.yaml` | `scripts/train_classical.py` | SVM, RandomForest |
| `tensorflow-keras` | `configs/training/tensorflow.yaml` | `scripts/train_tensorflow.py` | Sonic Sleuth, EfficientNet-LSTM, MultiscaleCNN, SpectrogramTransformer |
| `pytorch-audio` | `configs/training/pytorch.yaml` | `scripts/train_pytorch.py` | RawNet2, AASIST, RawGAT-ST, Conformer, Hybrid CNN-Transformer |
| `ssl-transformers` | `configs/training/ssl.yaml` | `scripts/train_ssl.py` | WavLM, HuBERT |

```bash
python scripts/train_classical.py --plan-only
python scripts/train_tensorflow.py --models MultiscaleCNN --epochs 100 --device-profile gpu
python scripts/train_pytorch.py --models Conformer RawNet2 --epochs 100 --device-profile gpu
python scripts/train_ssl.py --models WavLM HuBERT --epochs 100 --device-profile gpu
```

Todos os entrypoints chamam `scripts/run_models_sequential.py`, que executa
`scripts/run_benchmark.py --model <nome>` para cada arquitetura, salvando logs,
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
    │       └─ save_training_artifacts() ← model.keras + scaler.pkl + config.json
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
`app/core/dataset_catalog.py` (`DATASET_TIERS`) — fonte única de verdade
compartilhada por `scripts/build_dataset.py`, pela aba Datasets do Gradio, pelo
benchmark e pela documentação. Escolher um tier pré-configura tamanho, fontes e
estratégia de split. Detalhes de fontes/licenças em
[docs/12_DATASETS.md](12_DATASETS.md).

| Tier | Por classe | Total | Split | Habilita | Uso típico no treino |
|------|-----------:|------:|-------|----------|----------------------|
| `test` | 100 | 200 | 70/15/15 estratificado | nenhum (abaixo do mínimo clássico) | smoke de ponta a ponta |
| `small` | 1.000 | 2.000 | 70/15/15 estratificado | Clássico (SVM/RF) + CNN leve | iteração rápida de hiperparâmetros |
| `medium` | 3.000 | 6.000 | 70/15/15 estratificado | até Transformer | treino/teste mais robustos |
| `large` | 10.000 | 20.000 | **disjunto por falante** + cross-generator | **todas as 14 arquiteturas** (inclui Ensemble) | execução de referência do TCC |

> O `.npz` canônico do benchmark
> (`app/datasets/benchmark_audio_raw_balanced_15k.npz`, ~15k) corresponde ao tier
> **`large`** (10k/classe com folga de balanceamento). Para o retreino dos
> modelos ajustados use sempre `large` — só ele exercita a robustez sob falantes
> não vistos e habilita o Ensemble.

**Montar um tier** (download + balanceamento + splits + `dataset_config.json`):

```bash
python scripts/build_dataset.py --tier small     # 1.000/classe — segundos a minutos
python scripts/build_dataset.py --tier medium    # 3.000/classe
python scripts/build_dataset.py --tier large     # 10.000/classe, split por falante

# override do tamanho mantendo fontes/split do tier:
python scripts/build_dataset.py --tier medium --target 4000
```

**Como o tier afeta o treino:**

- **Prontidão por modelo.** O treino só habilita um modelo quando o tier atinge o
  mínimo de amostras dele — `small` cobre Clássico + CNN leve (RawNet2, Sonic
  Sleuth, MultiscaleCNN); `medium` chega aos Transformers; o **Ensemble** exige
  `large` (≥6.000/classe).
- **Split.** `test/small/medium` usam 70/15/15 estratificado; `large` usa split
  **disjunto por falante** (`speaker_manifest.json`), medindo generalização a
  usuários não vistos e evitando vazamento de falante entre treino e teste.
- **Pipeline de ponta a ponta** (download → benchmark) por tier:

```bash
python scripts/run_tcc_pipeline.py --download --tier large --full-benchmark --speaker-split
```

Recomendação prática: prototipe hiperparâmetros em `small`, valide arquitetura em
`medium` e produza os números finais (e o retreino ajustado) em `large`.

### 1.3 ModelTrainer

Classe principal de treinamento (`app/domain/models/training/trainer.py`). Funcionalidades:

- **Compilação**: `model.compile(optimizer, loss, metrics)` usando `OptimizerFactory` e `LossFactory`
- **SecureTrainingPipeline**: Previne data leakage — aplica `StandardScaler` apenas com dados de treino e armazena o scaler para inferência
- **TTA (Test-Time Augmentation)**: `predict_with_tta()` executa 5 versões perturbadas e faz média das predições (melhora ~1–3%)
- **Class weighting automático** (`use_class_weighting=True`): calcula `compute_class_weight('balanced')` e passa via `model.fit(class_weight=...)`. Compensa datasets desbalanceados típicos de deepfake (ASVspoof ~9:1). Suporta y categórico e one-hot.
- **Calibração automática de temperatura** (`auto_calibrate_temperature=True`): após o treino, busca o T que minimiza NLL no val set (Guo et al., ICML 2017). Salvo no `input_contract` e aplicado automaticamente pelo Predictor. Requer ≥ `calibration_min_samples` (default 50) no val set.
- **Sprint 2.3 — Stochastic Weight Averaging (SWA)** (`use_swa=True`, opt-in): mantém média móvel dos pesos nas últimas ~20% épocas e aplica ao final do treino. +0.5–1.5% accuracy típica (Izmailov et al., UAI 2018). Aplicado via `SWACallback`. Compatível com BatchNorm (recomputa stats automaticamente se `bn_update_data` for fornecido).
- **Sprint 2.4 — Mixup** (`use_mixup=True`, opt-in): interpola pares de amostras no batch (`λ ~ Beta(α, α)`). Convertido em soft labels automaticamente. **Importante**: incompatível com `class_weighting` (Keras requer int classes); o trainer detecta e desativa class weighting automaticamente.
- **Sprint 2.5 — OOD threshold** (`compute_ood_threshold=True`): calibra threshold de Out-of-Distribution no val set usando energy score / entropia (quantile=0.95). Predições com score abaixo do threshold são marcadas como `is_ood=True` pelo Predictor.
- **Sprint 3.2 — Mixed precision auto** (`use_mixed_precision=None` → auto): detecta GPU Tensor Core (CC≥7.0: V100, T4, RTX 20xx+, A100, H100) e habilita `mixed_float16` automaticamente. ~2× speedup + 50% menos VRAM.
- **Sprint 3.4 — ONNX export** (`export_onnx=True`, `export_onnx_int8=True`): exporta modelo Keras → ONNX FP32 + INT8 quantizado após o treino. Requer `pip install tf2onnx onnxruntime`. Habilita deploy CPU 2-3× mais rápido e 4× menor footprint. Falhas são silenciosas — não bloqueiam `.keras`.
- **Sprint 4.5 — EER threshold adaptativo**: calibra automaticamente o threshold de Equal Error Rate (FPR=FNR) no val set ao final do treino. Salvo em `input_contract.eer_threshold` e usado pelo Predictor como threshold de classificação (em vez de 0.5 fixo). Reportado também `eer_value` (métrica anti-spoofing padrão).
- **input_contract**: Salvo junto ao modelo — garante que inferência use o mesmo formato/features/temperatura/ood_threshold/eer_threshold do treino
- **Formato de save**: `.keras` (formato nativo Keras 3, substitui o legado `.h5`)

### 1.7 K-fold Cross Validation (Sprint 4.1)

```python
from app.domain.services.training_service import TrainingService

service = TrainingService(models_dir="app/models")
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

### 1.9 MLflow Tracking (Sprint 4.3)

```python
from app.domain.models.training.mlflow_tracking import MLflowTracker

with MLflowTracker(
    experiment_name='aasist_runs',
    tags={'architecture': 'AASIST', 'dataset': 'ASVspoof2019'},
) as tracker:
    tracker.log_config(config)
    result = trainer.train(model, train_data, validation_data)
    tracker.log_training_result(result.data)
    tracker.log_calibration_params(
        temperature=trainer._calibrated_temperature,
        ood_threshold=trainer._ood_threshold,
        eer_threshold=trainer._eer_threshold,
        eer_value=trainer._eer_value,
    )

# Visualizar: mlflow ui (http://localhost:5000)
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

### 1.11 Knowledge Distillation (Sprint 5.1)

Treine um modelo leve (Sonic Sleuth) a partir de um teacher grande (Ensemble):

```python
from app.domain.models.architectures.factory import create_model_by_name
from app.domain.models.training.knowledge_distillation import (
    DistillationConfig, distill_from_teacher,
)

# Cria student leve (Sonic Sleuth ~3M params vs Ensemble ~25M)
student = create_model_by_name('Sonic Sleuth', input_shape=(16000,), num_classes=2)

config = DistillationConfig(
    temperature=4.0,    # soft targets do teacher
    alpha=0.3,          # 30% hard labels + 70% soft targets do teacher
    epochs=50,
    learning_rate=1e-3,
)

student_trained, result = distill_from_teacher(
    teacher_model_path='app/models/Ensemble_v1.keras',
    student_model=student,
    X_train=X_train, y_train=y_train,
    validation_data=(X_val, y_val),
    config=config,
)

print(f"Compression ratio: {result['compression_ratio']:.1f}×")
print(f"Best val_loss: {result['best_val_loss']:.4f}")
```

### 1.12 Quantization-Aware Training (Sprint 5.2)

```python
from app.domain.models.training.quantization_aware import (
    apply_qat, fine_tune_qat, convert_qat_to_tflite_int8,
    create_representative_dataset,
)

# 1. Aplica QAT ao modelo treinado em FP32
qat_model = apply_qat(trained_model, quantize_all=True)

# 2. Fine-tune com LR baixo (calibra ranges de quantização)
fine_tune_qat(
    qat_model, X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10, learning_rate=1e-5,
)

# 3. Converte para TFLite INT8 (deploy mobile)
rep_data = create_representative_dataset(X_train, n_samples=100)
convert_qat_to_tflite_int8(qat_model, 'model_qat_int8.tflite',
                            representative_dataset=rep_data)
# Tipicamente: 95-99% da accuracy FP32, 4× menor, 2-3× mais rápido em CPU
```

### 1.13 Streaming Inference (Sprint 5.3)

Detecção em tempo real a partir de chunks de áudio (microfone, stream):

```python
from app.domain.models.inference.streaming_inference import (
    StreamingDetector, StreamingConfig,
)

detector = StreamingDetector(
    detection_service=service,
    model_name='AASIST_v1',
    config=StreamingConfig(
        window_seconds=3.0,    # janela analisada
        hop_seconds=1.0,       # latência de detecção
        ema_alpha=0.3,         # smoothing dos scores
    ),
)

# Loop principal (PyAudio, WebSocket, file stream...)
for chunk in audio_stream:  # chunks de ~100ms
    result = detector.push(chunk)
    if result is not None:
        print(f"t={result.timestamp:.2f}s  "
              f"is_fake={result.is_fake}  "
              f"smooth_conf={result.smoothed_confidence:.3f}")

# Decisão final após N inferências
final_score = detector.get_aggregate_score(last_n=10)
print(f"Score final: {final_score:.3f}")
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
| `TensorBoard` | Histogramas + gráfico (se `tensorboard_dir` passado) |
| `CSVLogger` | Histórico em CSV (se `csv_log_path` passado) |

O módulo `OptimizedTrainingConfig` (`optimized_training_config.py`) adiciona callbacks adicionais:
`AdvancedEarlyStopping`, `GradientClippingCallback`, `LossMonitoringCallback`, `OverfittingDetectionCallback`.

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

`apply_mixup()` e `apply_cutmix()` também disponíveis para uso manual.

### 1.6 Otimizadores e Losses

`OptimizerFactory` suporta: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `adadelta`, `adamax`, `nadam`.

`LossFactory` inclui: `focal_loss`, `weighted_bce`, `am_softmax`, `label_smoothing`, `label_smoothing_bce`.

`WarmupCosineDecaySchedule`: warmup linear seguido de decaimento cosseno — recomendado para Conformer e Transformers.

---

## 2. Hiperparâmetros por Arquitetura

Baseados no TCC (Seção 6.1, Tabela 10) e em `get_recommended_hyperparameters()`:

| Arquitetura | Batch | LR | Épocas | Dropout | L2 | Observação |
|-------------|-------|----|--------|---------|-----|------------|
| **AASIST** | 16 | 8e-4 | 100 | 0.2 | 1e-4 | attention_heads=12, hidden=512 |
| **RawGAT-ST** | 24 | 8e-4 | 100 | 0.2 | 1e-4 | — |
| **MultiscaleCNN** | 64 | 2e-3 | 100 | 0.5 | 5e-4 | hidden 128/256 |
| **SpectrogramTransformer** | 16 | 1e-4 | 100 | 0.1 | 1e-5 | WarmupCosineDecay (warmup=1000) — Sprint 2.2 |
| **Conformer** | 32 | 1e-3 | 100 | 0.3 | 1e-4 | attention_heads=8, WarmupCosineDecay (warmup=1000) |
| **EfficientNet-LSTM** | 32 | 5e-4 | 100 | 0.4 | 2e-4 | fine-tune últimas 3 camadas |
| **Hybrid CNN-T** | 32 | 1e-3 | 100 | 0.2 | 1e-4 | CCT + WarmupCosineDecay (warmup=1500) — Sprint 2.2 |
| **RawNet2** | 24 | 8e-4 | 100 | 0.3 | 1e-4 | conv_filters=[64,128,256], gru=128 |
| **WavLM / HuBERT** | 8–16 | 1e-4 | 20–50 | 0.1 | 1e-4 | backbone congelado; fine-tune head |
| **Sonic Sleuth** | 32 | 1e-3 | 100 | 0.1 | 1e-4 | — |
| **Ensemble (adaptive)** | 32 | 1e-3 | 50 | 0.3 | 1e-4 | parte de modelos pré-treinados |
| **SVM** | Full | N/A | N/A | — | — | StandardScaler obrigatório |
| **Random Forest** | Full | N/A | N/A | — | — | n_jobs=-1 |

---

## 3. Peculiaridades por Arquitetura

### 3.1 AASIST

- Entrada: espectrograma → reshape interno para `(batch, time, freq, 1)` via `AudioFeatureNormalization`
- Dropout progressivo nas densas finais: `min(dropout * 1.5, 0.9)` e `min(dropout * 2, 0.9)` (evita Dropout > 1.0)
- `bidirectional_gru` usa `layers.GRU` com `dropout=dropout_rate` — cuDNN selecionado automaticamente

### 3.2 RawGAT-ST

- Mesmas peculiaridades do AASIST (usa `GraphAttentionLayer` + `AttentionLayer`)
- Mesmos fixes de dropout e GRU

### 3.3 EfficientNet-LSTM

- Backbone `EfficientNetB0` (`weights=None` — treinado do zero)
- Fine-tuning: **últimas 3 camadas** descongeladas (`efficientnet.layers[-3:]`)
- Pré-processamento in-model: `MelSpectrogramFrontEnd` → `DeltaFeatureLayer` (3 canais) → resize 224×224×3
- Sequência temporal via Bi-LSTM[256, 128] + `AttentionLayer`

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
- **EER (Equal Error Rate)**: ponto onde FPR = FNR (métrica padrão ASVspoof)
- **Curva DET**: dados FPR/FNR para plotagem

`calculate_threshold_metrics()` avalia métricas para thresholds 0.3 a 0.7 — útil para encontrar o ponto de operação ideal por dataset.

---

## 6. Uso via API

```python
from app.domain.services.training_service import TrainingService

service = TrainingService(models_dir="app/models")
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
