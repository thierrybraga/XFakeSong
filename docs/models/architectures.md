# Arquiteturas Neurais

O XFakeSong implementa **14 arquiteturas** de detecção de deepfake, organizadas
por contrato de entrada: áudio bruto, espectrograma/LFCC e features tabulares.
Todos os modelos expõem a interface unificada
`create_model(input_shape, num_classes, **kwargs)` via
`app/domain/models/architectures/factory.py` ou pelo registry canônico em
`app/domain/models/architectures/registry.py`.

!!! success "SSL real no caminho TensorFlow (atualizado em 2026-07-27)"
    **WavLM e HuBERT deixaram de ser fallback.** Até esta data, ambos rodavam
    como uma **CNN-1D treinada do zero** no caminho TF — os números rotulados
    "WavLM"/"HuBERT" no benchmark TF não tinham relação com os modelos dos
    artigos.

    **Causa-raiz**: o `transformers` não consegue importar **nenhum** modelo TF
    neste stack, porque exige o pacote `tf-keras` quando o Keras instalado é
    3.x (`RuntimeError: ... Keras 3 ... not yet supported in Transformers`).
    Não é falha de download nem de checkpoint.

    **Solução**: o checkpoint **PyTorch** é legível sem tocar em TensorFlow.
    [`ssl_backbone.py`](../../app/domain/models/architectures/ssl_backbone.py)
    lê o `state_dict` e reimplementa o forward do backbone com operações TF,
    carregando os pesos como variáveis **não-treináveis**:

    - **HuBERT** (`facebook/hubert-base-ls960`) — atenção padrão;
    - **WavLM** (`microsoft/wavlm-base`) — inclui o **viés posicional relativo
      com gating** (`gru_rel_pos`/`rel_attn_embed`), que é a contribuição
      arquitetural do artigo;
    - esqueleto wav2vec 2.0 completo: extrator convolucional (com GroupNorm no
      1º bloco), projeção de características, convolução posicional com
      *weight norm* e encoder Transformer.

    **Fidelidade verificada** comparando com o modelo PyTorch, em **todos os 13
    hidden states**: `max|dif| ≈ 2.6e-05` (HuBERT), `9.0e-05` (WavLM base),
    `2.6e-05` (WavLM base-plus) — correlação 1,0000000000.

    **Regime de treino** (o pedido para uso downstream): backbone
    **inteiramente congelado**; treinam apenas a cabeça e uma **soma ponderada
    aprendível dos hidden states de todas as camadas** (receita SUPERB — as
    camadas intermediárias costumam carregar mais informação de artefato que a
    última). Na prática: **1,2 %** dos parâmetros no WavLM e **0,5 %** no
    HuBERT; os 94,4 M restantes ficam congelados.

    `n_trainable_layers` é ignorado com aviso — este port não faz fine-tuning
    parcial. Se o checkpoint não estiver acessível, o extrator simplificado
    volta a ser usado e `XFAKE_STRICT_SSL=1` faz o run abortar em vez de
    aceitar o fallback silenciosamente.

    Os artefatos `*_original.pt` e o runner PyTorch
    (`scripts/benchmark/run_wavlm_original_benchmark.py`) continuam sendo o
    caminho do escopo oficial para "WavLM Original"/"HuBERT Original".

## Tabela Resumo

| # | Arquitetura | Entrada | Referência | Arquivo |
|---|-------------|---------|------------|---------|
| 1 | WavLM | Áudio bruto | microsoft/wavlm-base | `app/domain/models/architectures/wavlm.py` |
| 2 | HuBERT | Áudio bruto | Hidden-Unit BERT | `app/domain/models/architectures/hubert.py` |
| 3 | RawNet2 | Áudio bruto | RawNet2 (2021) | `app/domain/models/architectures/rawnet2.py` |
| 4 | Sonic Sleuth | Áudio bruto ou espectrograma | Alshehri et al. (2024) | `app/domain/models/architectures/sonic_sleuth.py` |
| 5 | AASIST | Áudio bruto | Jung et al., ICASSP 2022 | `app/domain/models/architectures/aasist.py` |
| 6 | RawGAT-ST | Áudio bruto | SincNet + Graph Attention espectro-temporal | `app/domain/models/architectures/rawgat_st.py` |
| 7 | Conformer | Espectrograma | Conv + Transformer | `app/domain/models/architectures/conformer.py` |
| 8 | Hybrid CNN-Transformer (CCT) | Espectrograma | Bartusiak & Delp (2022) | `app/domain/models/architectures/hybrid_cnn_transformer.py` |
| 9 | Spectrogram Transformer | Espectrograma | ViT adaptado para áudio | `app/domain/models/architectures/spectrogram_transformer.py` |
| 10 | EfficientNet-LSTM | Áudio bruto ou espectrograma | Transfer learning | `app/domain/models/architectures/efficientnet_lstm.py` |
| 11 | MultiscaleCNN (Res2Net) | Espectrograma | Gao et al. TPAMI 2021 | `app/domain/models/architectures/multiscale_cnn.py` |
| 12 | Ensemble | Áudio bruto | Pham et al. (2024) | `app/domain/models/architectures/ensemble.py` |
| 13 | SVM | Features tabulares | scikit-learn SVC | `app/domain/models/architectures/svm.py` |
| 14 | Random Forest | Features tabulares | scikit-learn RF | `app/domain/models/architectures/random_forest.py` |

---

## Visão Consolidada do Estudo Experimental

O artigo atual agrupa as 14 arquiteturas por função experimental e por tipo de
entrada. Essa organização é a referência para leitura dos resultados do
benchmark de 15.000 amostras.

| Família | Modelos | Papel no experimento |
|---|---|---|
| SSL e áudio bruto | WavLM, HuBERT, RawNet2, AASIST, RawGAT-ST, Ensemble | Comparação com representações modernas, SincNet, fusão multi-feature e forma de onda direta |
| Grafos | AASIST, RawGAT-ST | Modelagem explícita de dependências espectro-temporais sobre front-end aprendido |
| Espectrograma + atenção | Conformer, Hybrid CNN-Transformer, Spectrogram Transformer | Avaliação de convolução local + atenção global |
| CNN e fusão | Sonic Sleuth, EfficientNet-LSTM, MultiscaleCNN, Ensemble | Frentes espectrais, transferência e fusão multi-feature |
| Clássicos | SVM, Random Forest | Baselines interpretáveis e rápidos em CPU |

### Decisão operacional por arquitetura

| Modelo | Decisão no artigo | Observação |
|---|---|---|
| Conformer | Demonstração principal | Maior qualidade e robustez sob AWGN no recorte oficial |
| Sonic Sleuth | Suportado fora do recorte oficial | Artefato carregável existe na raiz `data/models/`, mas não integra os 11 finais do artigo |
| Hybrid CNN-Transformer | Recorte oficial como CCT | Melhor compromisso neural entre acurácia, tamanho e latência |
| MultiscaleCNN | Recorte oficial como Res2Net | Alta acurácia, artefato maior |
| SVM | Baseline rápido | Excelente latência; frágil sob ruído AWGN |
| Random Forest | Baseline complementar | Bom desempenho, maior custo que SVM |
| RawNet2 | Estudo raw-audio | Convergente no preset GPU |
| Ensemble | Suportado pelo registry/harness | Fora do recorte oficial sincronizado atual |
| RawGAT-ST | Comparação em grafos | Estável e relativamente robusto |
| AASIST | Comparação em grafos | Receita de treino corrigida e consistente |
| HuBERT Original | Referência SSL funcional | Backbone original viável, custo elevado |
| EfficientNet-LSTM | Suportado pela Gradio/API | Fora do recorte oficial sincronizado atual |
| WavLM Original | Referência SSL experimental | Acurácia inferior a HuBERT no benchmark atual |
| Spectrogram Transformer | Recorte oficial como AST | Estável após retreino selecionado |

Os artefatos carregáveis ficam em `data/models/bench_*`; os 11 modelos finais do
artigo ficam em `data/models/benchmark_final/<slug_do_manifesto>/`. A
rastreabilidade completa está em
[Benchmark e Resultados](../evaluation/benchmark.md) e [Estudo Experimental](../evaluation/experimental-study.md).

---

## Arquiteturas de Áudio Bruto (Raw Audio)

Estas arquiteturas operam diretamente sobre a forma de onda (waveform) —
entrada típica `(batch, samples)` ou `(batch, samples, 1)`, taxa de amostragem
de 16 kHz. Quando o artefato tem `input_contract`, ele prevalece sobre esta
classificação.

### 1. WavLM

Modelo SSL (Self-Supervised Learning) treinado com masked prediction e denoising. Robusto a variações de canal e ruído.

- **Caminho TensorFlow**: backbone **real e congelado**, portado do checkpoint
  PyTorch `microsoft/wavlm-base` (ver o bloco de destaque no topo). Inclui o
  viés posicional relativo com gating do artigo.
- **Treinável**: apenas a soma ponderada dos 13 hidden states + a cabeça MLP
  (~1,1 M de 95,5 M parâmetros = 1,2 %).
- **Fallback**: CNN-1D do zero, só quando o checkpoint não está acessível
  (`XFAKE_STRICT_SSL=1` aborta em vez de aceitar).

### 2. HuBERT

Aprende representações de fala prevendo "unidades ocultas" (clusters de áudio mascarado) — força o modelo a aprender características fonéticas de alto nível.

- **Caminho TensorFlow**: backbone **real e congelado**, portado do checkpoint
  PyTorch `facebook/hubert-base-ls960`.
- **Fluxo**: extrator convolucional (7 blocos, GroupNorm no 1º) → projeção →
  convolução posicional (*weight norm*) → 12 camadas Transformer → soma
  ponderada dos hidden states → cabeça MLP.
- **Treinável**: ~0,5 M de 94,9 M parâmetros (0,5 %).

### 3. RawNet2

Aprende filtros diretamente da forma de onda, sem transformações de pré-processamento.

- **Primeira camada**: filtros SincNet/Conv1D — banco de filtros passa-banda aprendível.
- **Blocos Residuais**: Feature Map Scaling (FMS) como mecanismo de atenção de canal leve.
- **Pré-processamento in-model**: `PreEmphasisLayer` + `AudioNormalizationLayer` (μ=0, σ=1).

!!! warning "Duas configurações diferentes chamadas 'RawNet2'"
    - `rawnet2` (**default**) = **Improved RawNet** de *verificação de locutor*
      (Jung et al., 2020): Sinc **128**, blocos `[128,128,256,256,256,256]`,
      **1×**GRU(1024) — ~7,0M parâmetros.
    - `rawnet2_antispoofing` = **baseline do ASVspoof 2021** (Tak et al.):
      Sinc **20**, blocos `[20,20,128,128,128,128]`, **3×**GRU(1024) — ~17,6M.
      **É esta** a configuração com que a literatura de anti-spoofing compara
      EER. Declare qual variante gerou cada número no relatório.

### 4. AASIST

Implementação alinhada ao AASIST: áudio bruto → SincConv → encoder residual →
grafos espectro-temporais heterogêneos → classificação.

- **Front-end**: `SincConvLayer` (70 filtros, kernel 129) → `|·|` → MaxPool2D(3,3)
  → BN → SELU, como o código de referência.
- **Encoder**: 6 blocos residuais 2D `(32,32,64,64,64,64)` com pooling `(1,3)`.
- **Nós**: espectrais por `max|·|` sobre o tempo (com *positional embedding*),
  temporais por `max|·|` sobre a frequência. Pooling **0,5 (S)** e **0,7 (T)**.
- **Atenção de grafo (§2.2–2.3, corrigida em 2026-07-27)**:
  `AASISTGraphAttentionLayer` — produto par-a-par entre nós → `tanh` → redução
  a escalar → **temperatura** (2,0 nos GATs; 100,0 nas HS-GAL) → softmax, com
  projeções *com* e *sem* atenção, BN e SELU. Antes usava-se o GAT **aditivo de
  Velickovic**, que não é a formulação dos artigos.
- **HS-GAL**: `AASISTHtrgGraphAttentionLayer` com **três conjuntos de parâmetros
  de atenção por tipo de aresta** (S–S, T–T e o cruzado S–T) — a contribuição
  que dá nome à camada. Antes era uma atenção homogênea com *type embeddings*.
- **Master node**: treinável por ramo (`MasterNodeSeed`), como os
  `nn.Parameter` `master1`/`master2` do código oficial; a média dos nós é o
  fallback da própria camada.
- **Readout**: 5 componentes (max+média temporais, max+média espectrais e o
  master), fundidos por `Maximum` (MGO) entre os dois ramos.
- **Loss/saída**: saída linear (logits) por padrão; `am_softmax` aplica a margem
  CosFace **na loss** (`AMSoftmaxCrossEntropy`).

### 4b. RawGAT-ST

Mesma família do AASIST (que deriva deste trabalho): SincConv compartilhado →
**dois encoders 2D independentes** → GAT espectral (Gs) e temporal (Gt) →
**fusão element-wise** → terceiro GAT espectro-temporal → readout.

- **Entrada default**: `raw_audio`, janela canônica 48.000 amostras (3 s @
  16 kHz); variantes legadas em espectrograma continuam disponíveis só para
  desserialização de checkpoints antigos.
- Usa a **mesma atenção de grafo do paper** (`AASISTGraphAttentionLayer`,
  temperatura 2,0).
- O alinhamento de Gs e Gt antes do produto element-wise usa **top-k pooling**
  (`GraphPoolLayer(target_nodes=12)`) — a mesma primitiva de pooling de grafo
  dos artigos. Antes era `AdaptiveGraphResize`, uma projeção densa aprendível
  sobre o eixo de nós: não existe no artigo e, por combinar nós linearmente,
  não é sequer uma operação de grafo. A camada antiga virou LEGADO (só
  desserialização).
- **Treino**: recebe as mesmas augmentations de domínio raw-audio que AASIST
  e RawNet2 (crop aleatório no treino, multicrop na avaliação).

---

## Arquiteturas Baseadas em Espectrograma

Entrada: `(batch, time_steps, freq_bins)` ou `(batch, time_steps, freq_bins, 1)`.

### 6. Sonic Sleuth

Arquitetura leve baseada em LFCC, MFCC e CQT. A referência reporta melhor
resultado com LFCC: **98,27% accuracy / EER 0,016** no ASVspoof 2019 +
In-the-Wild + FakeAVCeleb.

- **Variantes**: `sonic_sleuth` (LFCC), `sonic_sleuth_mfcc`,
  `sonic_sleuth_cqt`, `sonic_sleuth_lfcc_cqt` e **`sonic_sleuth_paper`**.
- **Front-end in-model**: LFCC/MFCC/CQT via `tf.signal`; CQT é aproximada por
  filtros log-espaçados sobre STFT para compatibilidade em grafo TensorFlow.
- **Backbone default**: versão estendida do paper com 5 blocos
  Conv2D+BatchNorm+ReLU+MaxPool, SE blocks e residuais nos blocos finais — é a
  configuração do artefato treinado/promovido.
- **`sonic_sleuth_paper`**: configuração **literal da Figura 3** do artigo —
  3 blocos (32 → 64 → 128), sem BN/SE/residual, `Flatten` →
  Dense(256) → Dense(128) → Dropout(0,1) → saída, Adam(1e-3).
- Todos esses knobs (`num_conv_blocks`, `use_residual`, `use_se_blocks`,
  `use_gap_gmp`, `use_batch_norm`, `dropout_rate`) são **parâmetros reais** do
  builder desde 2026-07-27; antes existiam no registry sem nenhum efeito.

### 7. Conformer

Evolução do Transformer que intercala convoluções com atenção para capturar contexto local **e** global simultaneamente.

- **Configuração única (Conformer-M do paper, Tabela 1)**: `d_model=256`,
  **16 blocos**, 4 cabeças, `d_ff=1024`, kernel depthwise 31, `P_drop=0,1`
  uniforme no encoder. `conformer_lite`/`conformer_m` são apenas **aliases**.
- **`ConvolutionModule`**: Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv.
- **`FeedForwardModule`**: Dense com normalização e ativação Swish.
- **Bloco**: FeedForward × ½ + SelfAttention (relativa) + Convolution + FeedForward × ½.
- **Codificação posicional relativa (corrigida em 2026-07-27)**: `R` é indexado
  por **distância** relativa, em ordem decrescente (`seq_len-1 → 0`), como em
  Dai et al. (Transformer-XL). O código devolvia a ordem **crescente** apesar
  de a docstring afirmar o contrário — o termo conteúdo↔posição, que é o
  diferencial do Conformer, ficava desalinhado.

!!! note "Consolidação 2026-07-27"
    Antes havia duas variantes com os nomes **invertidos**: `conformer`
    (rotulada "Large") tinha 8 blocos e `conformer_lite` (rotulada "Medium")
    tinha 16 — a "lite" era ~2× maior que a completa. Além disso a variante de
    8 blocos sobrescrevia o dropout por módulo, anulando o `dropout_rate` do
    plano de benchmark em todo o encoder. Ficou **uma** configuração, fiel ao
    paper, e o `dropout_rate` passou a valer de fato — **exige retreino** para
    que os números publicados continuem correspondendo ao modelo.

### 8. Hybrid CNN-Transformer (CCT)

Implementação do Compact Convolutional Transformer aplicado a espectrogramas de fala. Até **91,47% accuracy** no ASVspoof 2019.

- **Conv Tokenizer**: 2× [Conv2D + ReLU + MaxPool(3, stride 2)] em vez de patch
  embedding. **Sem Squeeze-and-Excitation** — havia um SE entre a conv e o
  pooling que não existe em Hassani et al.; removido em 2026-07-27.
- **Transformer**: 4 camadas, 4 heads, 256 dims, pre-norm, stochastic depth.
- **Sequence Pooling**: atenção ponderada no lugar de CLS token.

### 9. Spectrogram Transformer

ViT adaptado para espectrogramas de áudio, com patches extraídos direto do
espectrograma (16×16, stride 10), como no AST (Gong et al., 2021).

- **Contrato de entrada (corrigido em 2026-07-27)**: **300 quadros × 128 mel**
  — o front-end do artigo (128 bandas, hop de 10 ms) aplicado à janela canônica
  de 3 s. Resulta em **348 tokens**. O contrato anterior (100×80) produzia
  apenas **63 tokens** para um ViT-Base de 85M parâmetros, que é o regime
  documentado de colapso para chute aleatório.
- **Normalização de entrada do paper** (média 0, desvio 0,5, §2.1) aplicada por
  `ASTInputNormalization` — calculada por amostra para não vazar estatística
  entre partições.
- **Variantes**: `spectrogram_transformer` (ViT-Base do paper: 768 dims,
  12 blocos, 12 cabeças, ~85M params), **`spectrogram_transformer_small`**
  (ViT-Small: 384 dims, 12 blocos, 6 cabeças, ~21M) e
  `spectrogram_transformer_lite`.
- **`pretrained=True` transfere os pesos AudioSet de verdade.** O AST do artigo
  é inicializado com ImageNet (ViT/DeiT) e refinado em AudioSet — treinar 85M
  parâmetros do zero é a origem documentada do colapso para chute aleatório
  (EER ~51%). O `transformers` não publica AST em TensorFlow (e seus modelos TF
  nem importam com Keras 3), mas o checkpoint **PyTorch** é legível sem tocar em
  TF: [`ast_pretrained.py`](../../app/domain/models/architectures/ast_pretrained.py)
  lê o `state_dict` de `MIT/ast-finetuned-audioset-10-10-0.4593` e escreve nas
  camadas Keras.
    - O embedding posicional é **reamostrado** da grade do checkpoint (12×101)
      para a deste modelo (12×29) por interpolação bilinear — o procedimento
      que o próprio artigo prescreve ao mudar a resolução de entrada.
    - O token de **destilação** do DeiT é descartado (mantemos só o CLS) e a
      cabeça de 527 classes do AudioSet **não** é transferida.
    - Mapeamento validado contra o PyTorch **bloco a bloco**: mesma entrada →
      mesma saída, `max|dif| ≈ 1e-5` (precisão de float32).
    - A flag **nunca é no-op**: fora da configuração ViT-Base, ou se o
      checkpoint não puder ser obtido, a construção levanta
      `ASTPretrainedUnavailable`.
    - Exige rede na primeira execução (~350 MB, cacheado depois). Por isso o
      `registry` mantém `pretrained=False` como default (testes/CI offline) e
      o **benchmark liga a flag** em `benchmarks/planning.py`.
- Para o regime deliberadamente sem pesos, use `spectrogram_transformer_small`
  (ViT-Small), que é um **desvio declarado** do ViT-Base.

- Processamento in-model de áudio bruto → mel spectrogram via `STFTLayer`.
- Positional encoding aprendível.

### 10. EfficientNet-LSTM

Transfer learning + modelagem temporal sequencial.

- **Front-end**: aceita áudio bruto ou espectrograma; no raw, calcula mel
  spectrogram (`n_fft=512`, `hop=160`, `n_mels=128`) e deltas.
- **Backbone**: tenta EfficientNetB0 com pesos ImageNet; se offline ou
  indisponível, cai para `weights=None`.
- **Fine-tuning**: com ImageNet, congela o backbone e deixa treináveis `block7`
  e `top_*`; sem pesos, treina do zero.
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
| AASIST | Áudio bruto | `(batch, samples, 1)` | SincConv + grafos AASIST |
| RawGAT-ST | Áudio bruto | `(batch, samples, 1)` | SincConv + GAT espectral/temporal |
| Sonic Sleuth | Áudio bruto ou espectrograma | `(batch, samples,)` ou `(batch, time, freq)` | LFCC/MFCC/CQT extraído no modelo quando raw |
| Conformer | Espectrograma | `(batch, time, freq)` | Subsampling 4× + Positional Enc. |
| Hybrid CNN-T | Áudio bruto / Espectrograma | `(batch, samples,)` | Mel 128 bins → CCT Tokenizer |
| SpectrogramTransformer | Áudio bruto / Espectrograma | `(batch, time, freq)` | STFT → ConvStem → Patches |
| EfficientNet-LSTM | Áudio bruto / Espectrograma | `(batch, samples,)` | Mel + Delta → resize 224×224×3 → escala [0, 255] (ImageNet) |
| MultiscaleCNN | Áudio bruto / Espectrograma | `(batch, time, freq)` | STFT + log-mel (interno) |
| Ensemble | Áudio bruto | `(batch, samples,)` | Mel/LFCC/CQT/MFCC extraído no modelo |
| SVM | Features | `(batch, n_features)` | StandardScaler (interno) |
| Random Forest | Features | `(batch, n_features)` | StandardScaler (interno) |

## Considerações Gerais

- **Inferência de amostra única**: expandir dimensão com `input[np.newaxis, ...]`.
- **GPU/CPU**: o grafo é o MESMO em GPU e CPU. Nada de regularização
  condicionada ao device — o EfficientNet-LSTM tinha dropout apenas no caminho
  CPU (o argumento `dropout=` da LSTM desabilita o kernel cuDNN); hoje usa
  camadas `Dropout` externas, idênticas nos dois casos.
- **Carregamento seguro**: pesos `.h5`/`.keras` passam por verificações de integridade via `safe_normalization`.
- **Factory**: use `from app.domain.models.architectures.factory import create_model` para instanciar qualquer arquitetura por nome.
- **Nada de `layers.Lambda` com função Python** nas arquiteturas: o carregador
  do Keras 3 roda em `safe_mode` por default e recusa reconstruí-las — o modelo
  treina e salva, mas não volta. Use uma camada registrada com
  `@register_keras_serializable` (ver `layers.py`:
  `LogMelSpectrogramLayer`, `TimeResizeLayer`, `WeightedScoreFusionLayer`,
  `ExpandDimsLayer`, `AxisMaxAbsLayer`).
- **Variantes legadas** (`cnn_gru_simple`, `cnn_baseline`, `bidirectional_gru`,
  `resnet_gru`, `transformer`) são compartilhadas por AASIST e RawGAT-ST em
  `legacy_variants.py`. Não correspondem a nenhum paper: existem só para
  recarregar checkpoints antigos.
- **Hiperparâmetro novo**: precisa ser um parâmetro nomeado de algum builder do
  módulo, senão vira config morto (chave no registry sem efeito). O teste
  `tests/unit/test_architectures.py::test_default_params_are_accepted_by_builder`
  falha nesse caso.
