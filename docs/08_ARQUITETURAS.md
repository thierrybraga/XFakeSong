# Arquiteturas Neurais

O XFakeSong implementa **14 arquiteturas** de detecção de deepfake, organizadas em três categorias de acordo com o tipo de entrada. Todos os modelos expõem a interface unificada `create_model(input_shape, num_classes, **kwargs)` via `app/domain/models/architectures/factory.py`.

## Tabela Resumo

| # | Arquitetura | Entrada | Referência | Arquivo |
|---|-------------|---------|------------|---------|
| 1 | WavLM | Áudio bruto | microsoft/wavlm-base | `app/domain/models/architectures/wavlm.py` |
| 2 | HuBERT | Áudio bruto | Hidden-Unit BERT | `app/domain/models/architectures/hubert.py` |
| 3 | RawNet2 | Áudio bruto | RawNet2 (2021) | `app/domain/models/architectures/rawnet2.py` |
| 4 | Sonic Sleuth | Espectrograma | Alshehri et al. (2024) | `app/domain/models/architectures/sonic_sleuth.py` |
| 5 | AASIST | Espectrograma | GAT spectro-temporal | `app/domain/models/architectures/aasist.py` |
| 6 | RawGAT-ST | Espectrograma | Graph Attention + GRU | `app/domain/models/architectures/rawgat_st.py` |
| 7 | Conformer | Espectrograma | Conv + Transformer | `app/domain/models/architectures/conformer.py` |
| 8 | Hybrid CNN-Transformer (CCT) | Espectrograma | Bartusiak & Delp (2022) | `app/domain/models/architectures/hybrid_cnn_transformer.py` |
| 9 | Spectrogram Transformer | Espectrograma | ViT adaptado para áudio | `app/domain/models/architectures/spectrogram_transformer.py` |
| 10 | EfficientNet-LSTM | Espectrograma | Transfer learning | `app/domain/models/architectures/efficientnet_lstm.py` |
| 11 | MultiscaleCNN (Res2Net) | Espectrograma | Gao et al. TPAMI 2021 | `app/domain/models/architectures/multiscale_cnn.py` |
| 12 | Ensemble | Espectrograma | Pham et al. (2024) | `app/domain/models/architectures/ensemble.py` |
| 13 | SVM | Features tabulares | scikit-learn SVC | `app/domain/models/architectures/svm.py` |
| 14 | Random Forest | Features tabulares | scikit-learn RF | `app/domain/models/architectures/random_forest.py` |

---

## Arquiteturas de Áudio Bruto (Raw Audio)

Estas arquiteturas operam diretamente sobre a forma de onda (waveform) — entrada: `(batch, samples, 1)`, taxa de amostragem de 16 kHz.

### 1. WavLM

Modelo SSL (Self-Supervised Learning) treinado com masked prediction e denoising. Robusto a variações de canal e ruído.

- **Modo completo**: carrega `microsoft/wavlm-base` via `transformers` (embeddings 768-dim).
- **Modo simplificado**: CNN 1D compatível com Keras 3 (para ambientes sem HuggingFace).
- **Classificador**: Backbone congelado + MLP head.

### 2. HuBERT

Aprende representações de fala prevendo "unidades ocultas" (clusters de áudio mascarado) — força o modelo a aprender características fonéticas de alto nível.

- **Implementação**: 7 blocos Conv1D com GELU e strides crescentes, simulando o feature encoder do HuBERT original (compatível Keras 3).
- **Fluxo**: Feature Encoder (CNN) → Transformer Encoder → Projection Head.

### 3. RawNet2

Aprende filtros diretamente da forma de onda, sem transformações de pré-processamento.

- **Primeira camada**: filtros SincNet/Conv1D — banco de filtros passa-banda aprendível.
- **Blocos Residuais**: Feature Map Scaling (FMS) como mecanismo de atenção de canal leve.
- **Pré-processamento in-model**: `AudioResamplingLayer` (→ 16 kHz) + `AudioNormalizationLayer` (μ=0, σ=1).

---

## Arquiteturas Baseadas em Espectrograma

Entrada: `(batch, time_steps, freq_bins)` ou `(batch, time_steps, freq_bins, 1)`.

### 4. Sonic Sleuth

Arquitetura leve baseada em LFCC, MFCC e CQT. Melhor resultado: **98,27% accuracy / EER 0,016** no ASVspoof 2019 + In-the-Wild + FakeAVCeleb.

- **Pipeline**: 3× Conv2D(32→64→128, 3×3) + MaxPool → Flatten → Dense(256) → Dense(128) → Dropout(0.1) → Dense(1, sigmoid).

### 5. AASIST

Modela o áudio como grafo — aprende relações espectro-temporais explicitamente via Graph Attention Networks.

- **GraphAttentionLayer**: nós = características, arestas = correlações entre segmentos.
- **Pipeline**: CNN Encoder → Graph Attention → GRU → Classificação binária.
- **Pré-processamento in-model**: `AudioFeatureNormalization` + reshape 4D.

### 6. RawGAT-ST

Variante do AASIST com foco em Graph Attention Networks espectrais-temporais + GRU.

- Utiliza `GraphAttentionLayer` e `AttentionLayer` customizadas.
- Blocos residuais para extração de features locais antes da modelagem global por grafo.

### 7. Conformer

Evolução do Transformer que intercala convoluções com atenção para capturar contexto local **e** global simultaneamente.

- **`ConvolutionModule`**: Pointwise Conv → GLU → Depthwise Conv → Swish → Pointwise Conv.
- **`FeedForwardModule`**: Dense com normalização e ativação Swish.
- **Bloco**: FeedForward × ½ + SelfAttention + Convolution + FeedForward × ½.

### 8. Hybrid CNN-Transformer (CCT)

Implementação do Compact Convolutional Transformer aplicado a espectrogramas de fala. Até **91,47% accuracy** no ASVspoof 2019.

- **Conv Tokenizer**: 2× [Conv2D + ReLU + MaxPool(3, stride 2)] em vez de patch embedding.
- **Transformer**: 4 camadas, 4 heads, 256 dims, pre-norm, stochastic depth.
- **Sequence Pooling**: atenção ponderada no lugar de CLS token.

### 9. Spectrogram Transformer

ViT adaptado para espectrogramas de áudio com `ConvolutionStemLayer` para extração inicial de patches.

- Processamento in-model de áudio bruto → mel spectrogram via `STFTLayer`.
- Positional encoding aprendível.

### 10. EfficientNet-LSTM

Transfer learning + modelagem temporal sequencial.

- **Backbone**: EfficientNetB0 (`weights=None`).
- **Fine-tuning**: As últimas 3 camadas do backbone são descongeladas (`efficientnet.layers[-3:]`).
- **Pré-processamento in-model**: `MelSpectrogramFrontEnd` → `DeltaFeatureLayer` (3 canais) → resize para (224, 224, 3).
- **Temporal**: Bi-LSTM[256, 128] + `AttentionLayer` sobre features extraídas pelo backbone.

### 11. MultiscaleCNN (Res2Net-50)

Implementação fiel ao paper Res2Net (TPAMI 2021) — representações multi-escala **dentro** de cada bloco residual.

- **`Bottle2neck`**: Divide features em `s` grupos processados hierarquicamente, criando representações em múltiplas escalas granulares.
- **Config**: Res2Net-50 — [3, 4, 6, 3] blocos, baseWidth=26, scale=4.

### 12. Ensemble (Multi-Spectrogram Fusion)

Combina múltiplas representações espectrais — EER 0,03 no ASVspoof 2019 (Pham et al. 2024).

- **Branch 1**: Mel spectrogram (128 mels) → CNN+SE → embedding.
- **Branch 2**: LFCC (20 coefs.) → CNN+SE → embedding.
- **Branch 3**: CQT (84 bins) → CNN+SE → embedding.
- **Branch 4**: MFCC (20 coefs.) → CNN+SE → embedding.
- **Fusão (feature-level)**: `CrossAttentionFusionLayer` + `GatedFusionLayer` → Dense(512) → Dense(256) → Dense(128).
- **Variantes**: `ensemble` (feature fusion), `ensemble_score` (fusão de scores ponderada), `ensemble_lite` (2 branches), `ensemble_adaptive` (5 branches com pesos por confiança — TCC Eq. 27-28, EER 3,6%).

---

## ML Clássico (Scikit-learn)

Encapsulados para seguir a interface do projeto, úteis como baseline e em cenários de recursos limitados.

### 13. SVM (Support Vector Machine)

- **Pipeline**: `StandardScaler` + `SVC(kernel='rbf', probability=True)`.
- **Entrada**: vetor de features tabulares `(batch, n_features)`.
- **Nota**: Requer todo o dataset em memória (sem mini-batch).

### 14. Random Forest

- **Pipeline**: `StandardScaler` + `RandomForestClassifier(n_jobs=-1)`.
- **Entrada**: vetor de features tabulares `(batch, n_features)`.
- **Vantagem**: Robusto a features irrelevantes; paralelismo em CPU multi-core.

---

## Tabela de Inputs e Pré-processamento

| Arquitetura | Entrada | Formato | Pré-proc. crítico |
|-------------|---------|---------|-------------------|
| WavLM | Áudio bruto | `(batch, samples,)` | Resampling 16 kHz |
| HuBERT | Áudio bruto | `(batch, samples,)` | Resampling 16 kHz |
| RawNet2 | Áudio bruto | `(batch, samples,)` | SincNet + pré-ênfase (interno) |
| Sonic Sleuth | Áudio bruto | `(batch, samples,)` | LFCC/MFCC/CQT extraído no modelo |
| AASIST | Espectrograma | `(batch, time, freq)` | `AudioFeatureNormalization` |
| RawGAT-ST | Espectrograma | `(batch, time, freq)` | `AudioFeatureNormalization` |
| Conformer | Espectrograma | `(batch, time, freq)` | Subsampling 4× + Positional Enc. |
| Hybrid CNN-T | Áudio bruto / Espectrograma | `(batch, samples,)` | Mel 128 bins → CCT Tokenizer |
| SpectrogramTransformer | Áudio bruto / Espectrograma | `(batch, time, freq)` | STFT → ConvStem → Patches |
| EfficientNet-LSTM | Áudio bruto / Espectrograma | `(batch, samples,)` | Mel + Delta → resize 224×224×3 |
| MultiscaleCNN | Áudio bruto / Espectrograma | `(batch, time, freq)` | STFT + log-mel (interno) |
| Ensemble | Áudio bruto | `(batch, samples,)` | Mel/LFCC/CQT/MFCC extraído no modelo |
| SVM | Features | `(batch, n_features)` | StandardScaler (interno) |
| Random Forest | Features | `(batch, n_features)` | StandardScaler (interno) |

## Considerações Gerais

- **Inferência de amostra única**: expandir dimensão com `input[np.newaxis, ...]`.
- **GPU/CPU**: arquiteturas Keras detectam automaticamente a GPU disponível.
- **Carregamento seguro**: pesos `.h5`/`.keras` passam por verificações de integridade via `safe_normalization`.
- **Factory**: use `from app.domain.models.architectures.factory import create_model` para instanciar qualquer arquitetura por nome.
