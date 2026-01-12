# Arquiteturas Detalhadas

Esta seção fornece uma análise profunda das arquiteturas de redes neurais implementadas no sistema XfakeSong para a detecção de deepfakes de áudio. Cada arquitetura foi escolhida por suas capacidades específicas de capturar artefatos temporais, espectrais ou espaciais.

## 1. WavLM (Waveform Language Model)

### Visão Geral
O WavLM é um modelo de ponta (SOTA) treinado em grandes volumes de áudio para aprender representações universais de fala. Ele é especialmente eficaz em capturar nuances sutis que diferenciam áudio real de sintético.

### Funcionamento
O WavLM utiliza uma abordagem de aprendizado auto-supervisionado. Ele mascara partes do áudio de entrada e tenta prever as partes faltantes, aprendendo assim a estrutura intrínseca do som. Para detecção de deepfakes, utilizamos o modelo como um extrator de características robusto.

### Implementação
No XfakeSong, a implementação do WavLM é adaptável:
*   **Modo Completo**: Utiliza a biblioteca `transformers` da Hugging Face para carregar o modelo `microsoft/wavlm-base` pré-treinado.
*   **Modo Simplificado**: Caso a biblioteca `transformers` não esteja disponível ou para ambientes com recursos limitados, uma versão simplificada baseada em CNN 1D é utilizada.

**Estrutura da Classe `WavLMFeatureExtractor`:**
*   **Entrada**: Áudio bruto (raw waveform).
*   **Processamento**: O áudio passa pelo modelo WavLM (congelado por padrão) para extrair embeddings de dimensão 768.
*   **Classificação**: Os embeddings são processados por um MLP (Multi-Layer Perceptron) para a decisão final.

**Referência de Código**: [wavlm.py](file:///app/domain/models/architectures/wavlm.py)

---

## 2. HuBERT (Hidden-Unit BERT)

### Visão Geral
Semelhante ao WavLM, o HuBERT aprende representações de fala prevendo "unidades ocultas" (clusters) de áudio mascarado. É conhecido por sua robustez a ruídos.

### Funcionamento
O modelo processa o áudio bruto e o converte em uma sequência de vetores de características. Diferente de modelos que prevêem o áudio diretamente, o HuBERT prevê categorias discretas, o que força o modelo a aprender características acústicas de alto nível (como fonemas e entonação) em vez de apenas detalhes do sinal.

### Implementação
A implementação no projeto segue uma estratégia de compatibilidade com Keras 3:
*   **Simulação**: Uma arquitetura de CNN 1D profunda é usada para simular a extração de features do HuBERT, composta por múltiplos blocos convolucionais (`Conv1D`) com ativação GELU e strides específicos para downsampling.
*   **Camadas**: 7 blocos convolucionais sequenciais que reduzem a dimensão temporal e aumentam a profundidade dos canais, mimetizando a compressão de informação do modelo original.

**Referência de Código**: [hubert.py](file:///app/domain/models/architectures/hubert.py)

---

## 3. Hybrid CNN-Transformer

### Visão Geral
Esta arquitetura combina o melhor de dois mundos: a capacidade das Redes Neurais Convolucionais (CNNs) de extrair padrões locais (texturas em espectrogramas) e a capacidade dos Transformers de modelar dependências de longo prazo.

### Funcionamento
1.  **Entrada**: Espectrogramas (imagens de tempo-frequência).
2.  **CNN (Extração Local)**: Blocos residuais com Squeeze-and-Excitation processam o espectrograma para identificar artefatos visuais de manipulação.
3.  **Transformer (Modelagem Global)**: As features extraídas pela CNN são achatadas e passadas para um codificador Transformer, que analisa a coerência temporal do áudio.

### Implementação
*   **`ResidualBlock`**: Bloco básico da CNN com conexões residuais e mecanismo de atenção de canal (Squeeze-and-Excitation).
*   **`MultiScaleAttention`**: Mecanismo de atenção que opera em diferentes escalas para capturar detalhes finos e globais.
*   **Fluxo**: `Input -> CNN Blocks -> Reshape -> Transformer Encoder -> MLP Head`.

**Referência de Código**: [hybrid_cnn_transformer.py](file:///app/domain/models/architectures/hybrid_cnn_transformer.py)

---

## 4. Conformer

### Visão Geral
O Conformer é uma evolução do Transformer que integra convoluções diretamente na arquitetura do bloco de atenção, capturando tanto interações globais quanto locais de forma muito eficiente.

### Funcionamento
A arquitetura intercala camadas de atenção (Self-Attention) com módulos de convolução. Isso resolve uma limitação dos Transformers puros, que podem perder detalhes locais finos, e das CNNs puras, que têm dificuldade com contexto global.

### Implementação
*   **`ConvolutionModule`**: Um módulo sanduíche que aplica *Pointwise Conv -> Gated Linear Unit (GLU) -> Depthwise Conv -> Swish -> Pointwise Conv*.
*   **`FeedForwardModule`**: Redes densas com normalização e ativação Swish.
*   **Bloco Conformer**: Combina `FeedForward` + `SelfAttention` + `Convolution` + `FeedForward` em uma estrutura residual única.

**Referência de Código**: [conformer.py](file:///app/domain/models/architectures/conformer.py)

---

## 5. RawNet2

### Visão Geral
O RawNet2 é projetado para operar diretamente sobre o áudio bruto (waveform), dispensando a necessidade de transformações prévias como STFT ou Mel-Spectrograms. Isso permite que a rede aprenda seus próprios filtros ideais.

### Funcionamento
A rede utiliza filtros Sinc (baseados na função sinc) na primeira camada para simular um banco de filtros passa-banda aprendível. Em seguida, aplica blocos residuais com *Feature Aggregation* para compor as características finais.

### Implementação
*   **`AudioNormalizationLayer`**: Normaliza o áudio de entrada (média zero, variância unitária).
*   **`MultiScaleConv1DBlock`**: Aplica convoluções com diferentes tamanhos de kernel (3, 5, 7) em paralelo para capturar características em múltiplas resoluções temporais.
*   **Filtros Sinc**: (Simulados via Conv1D nesta implementação) Atuam como a primeira camada de extração de features físicas.

**Referência de Código**: [rawnet2.py](file:///app/domain/models/architectures/rawnet2.py)

---

## 6. AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)

### Visão Geral
O AASIST é uma das arquiteturas mais avançadas para detecção de spoofing, modelando o áudio como um grafo onde as relações entre diferentes segmentos de tempo e frequência são aprendidas explicitamente.

### Funcionamento
Utiliza *Graph Attention Networks* (GATs) para processar representações espectro-temporais. O modelo constrói um grafo onde os nós representam características do áudio e as arestas representam as correlações entre elas, permitindo detectar inconsistências complexas geradas por vocoders neurais.

### Implementação
*   **Camadas de Grafo**: Implementa `GraphAttentionLayer` para processar as relações não-Euclidianas nos dados.
*   **Pipeline**:
    1.  Encoder CNN para extrair features iniciais.
    2.  Módulo de Grafo para refinamento de features.
    3.  Camadas GRU (Gated Recurrent Unit) para modelagem sequencial final.
    4.  Classificação binária.

**Referência de Código**: [aasist.py](file:///app/domain/models/architectures/aasist.py)
