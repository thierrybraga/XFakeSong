# Documentação de Inferência das Arquiteturas

Este documento detalha os métodos de inferência e o fluxo de dados para cada arquitetura implementada no projeto XfakeSong. Todas as arquiteturas seguem uma interface unificada para facilitar a integração no pipeline de detecção, mas diferem significativamente em seus requisitos de entrada e processamento interno.

## Interface Unificada

Todas as arquiteturas expõem uma função de fábrica `create_model` e retornam um objeto que implementa (ou emula) a interface do Keras/TensorFlow.

### Assinatura Padrão
```python
def create_model(input_shape: Tuple[int, ...], num_classes: int, architecture: str, **kwargs) -> Model
```

### Método de Inferência
A inferência é realizada através do método padrão `.predict()`:
```python
predictions = model.predict(input_data)
# Saída: Array de probabilidades (N, num_classes) ou (N, 1) para binário
```

---

## 1. Arquiteturas Baseadas em Features (Espectrogramas)

Estas arquiteturas esperam como entrada tensores representando características tempo-frequência (espectrogramas, MFCCs, etc.).

### Pré-requisitos Comuns
*   **Input Shape:** `(time_steps, feature_bins)` ou `(time_steps, feature_bins, channels)`
*   **Normalização:** Geralmente esperam dados normalizados externamente, mas muitas possuem camadas de `BatchNormalization` ou `AudioFeatureNormalization` na entrada.

### 1.1 AASIST (Anti-spoofing Audio Spoofing and Deepfake Detection)
*   **Tipo de Entrada:** Espectrograma.
*   **Fluxo de Inferência:**
    1.  **Normalização:** Camada `AudioFeatureNormalization`.
    2.  **Adaptação:** `apply_reshape_for_cnn` para garantir formato 4D `(batch, time, freq, 1)`.
    3.  **Extração de Features:** Blocos convolucionais multi-escala (SincNet-like ou CNN padrão).
    4.  **Modelagem Temporal:** Camadas GRU (Gated Recurrent Units) ou Atenção em Grafo (GAT).
    5.  **Classificação:** Camadas densas com regularização L2 e Dropout -> Softmax.
*   **Saída:** Probabilidade de ser `Real` vs `Fake`.

### 1.2 RawGAT-ST
*   **Tipo de Entrada:** Espectrograma.
*   **Fluxo de Inferência:**
    1.  Similar ao AASIST, mas foca no uso de **Graph Attention Networks (GAT)** para modelar relações espectro-temporais complexas.
    2.  Utiliza mecanismos de atenção para ponderar a importância de diferentes regiões do espectrograma.

### 1.3 EfficientNet-LSTM & MultiscaleCNN
*   **Tipo de Entrada:** Espectrograma.
*   **Fluxo de Inferência:**
    1.  **Feature Extraction:** Backbone CNN (EfficientNet ou CNN customizada multi-escala) extrai mapas de características visuais do espectrograma.
    2.  **Temporal Aggregation:** Camadas LSTM ou Global Pooling processam a sequência de features.
    3.  **Head:** Classificador denso.

### 1.4 Transformadores (SpectrogramTransformer, Conformer)
*   **Tipo de Entrada:** Espectrograma.
*   **Fluxo de Inferência:**
    1.  **Embedding:** Projeção linear das features de frequência + Positional Encoding.
    2.  **Atenção:** Múltiplos blocos de Self-Attention (Transformer) ou Convolução + Atenção (Conformer).
    3.  **Pooling:** Global Average Pooling.
    4.  **Head:** MLP Classificador.

---

## 2. Arquiteturas Baseadas em Áudio Bruto (Raw Audio)

Estas arquiteturas operam diretamente sobre a forma de onda do áudio (waveform), aprendendo filtros diretamente dos dados.

### Pré-requisitos Comuns
*   **Input Shape:** `(samples, 1)` ou `(samples,)`.
*   **Taxa de Amostragem:** Fixa (geralmente 16kHz). O modelo pode conter camadas de reamostragem, mas recomenda-se enviar na taxa correta.

### 2.1 RawNet2
*   **Tipo de Entrada:** Áudio Bruto (Waveform).
*   **Processamento Interno (In-Graph):**
    1.  **Resampling:** `AudioResamplingLayer` (garante 16kHz).
    2.  **Normalização:** `AudioNormalizationLayer` (Média 0, Std 1).
*   **Fluxo de Inferência:**
    1.  **SincNet/Conv1D:** Primeira camada aprende filtros passa-banda diretamente do sinal.
    2.  **Residual Blocks:** Blocos residuais com FMS (Feature Map Scaling).
    3.  **GRU:** Modelagem temporal das features extraídas.
    4.  **Head:** Classificador.

### 2.2 HuBERT (Hidden Unit BERT)
*   **Tipo de Entrada:** Áudio Bruto.
*   **Fluxo de Inferência:**
    1.  **Feature Encoder:** CNNs 1D que reduzem a dimensionalidade temporal (simulando tokenização).
    2.  **Transformer Encoder:** Camadas de atenção (BERT-like) para capturar contexto global.
    3.  **Projection:** Projeção final para a tarefa de classificação binária (fine-tuning).
    *Nota:* A implementação utiliza uma versão otimizada compatível com Keras 3, simulando a estrutura do HuBERT original.

### 2.3 WavLM
*   **Tipo de Entrada:** Áudio Bruto.
*   **Fluxo de Inferência:**
    1.  Similar ao HuBERT, mas treinado com tarefas de "Masked Prediction" e "Denoising", tornando-o robusto a variações de canal e ruído.
    2.  Utiliza um backbone pré-treinado (congelado ou fine-tuned) seguido de um classificador MLP.

### 2.4 Sonic Sleuth
*   **Tipo de Entrada:** Áudio Bruto.
*   **Fluxo de Inferência:**
    1.  Arquitetura leve customizada que combina convoluções 1D eficientes com mecanismos de atenção simplificados.
    2.  Focada em inferência rápida para cenários de recursos limitados.

---

## 3. Machine Learning Clássico

Modelos baseados em `scikit-learn` encapsulados para seguir a interface do projeto.

### 3.1 SVM (Support Vector Machine)
*   **Tipo de Entrada:** Features Tabulares (Vetor de características globais).
*   **Wrapper:** Classe `SVMModel`.
*   **Fluxo de Inferência:**
    1.  **Pipeline:** `StandardScaler` (padronização média/desvio) -> `SVC` (Kernel RBF padrão).
    2.  **Predict:** O método `.predict()` do wrapper delega para o pipeline do sklearn.
    3.  **Probabilidades:** Usa `predict_proba` se `probability=True`, caso contrário retorna a classe direta.

### 3.2 Random Forest
*   **Tipo de Entrada:** Features Tabulares.
*   **Wrapper:** Classe `RandomForestModel`.
*   **Fluxo de Inferência:**
    1.  **Pipeline:** Opcionalmente inclui normalização (embora árvores não exijam estritamente).
    2.  **Ensemble:** Agregação de múltiplas árvores de decisão.
    3.  **Saída:** Média das predições das árvores (voto majoritário ou média de probabilidades).

---

## Tabela Resumo de Inputs

| Arquitetura | Tipo de Entrada | Formato Esperado (Exemplo) | Pré-processamento Crítico |
| :--- | :--- | :--- | :--- |
| **AASIST** | Features | `(batch, time, freq)` | Normalização de Features |
| **RawGAT-ST** | Features | `(batch, time, freq)` | Normalização de Features |
| **RawNet2** | Áudio Bruto | `(batch, samples, 1)` | N/A (Feito no modelo) |
| **HuBERT** | Áudio Bruto | `(batch, samples, 1)` | Resampling 16kHz |
| **WavLM** | Áudio Bruto | `(batch, samples, 1)` | Resampling 16kHz |
| **Conformer** | Features | `(batch, time, freq)` | Positional Encoding (Interno) |
| **SVM** | Features (Flat) | `(batch, n_features)` | Scaling (Interno no Pipeline) |
| **RandomForest**| Features (Flat) | `(batch, n_features)` | N/A |

## Considerações de Implementação

*   **Batching:** Todos os modelos esperam a primeira dimensão como o tamanho do lote (`batch_size`). Para inferência de uma única amostra, deve-se expandir a dimensão: `input[np.newaxis, ...]`.
*   **GPU vs CPU:** As arquiteturas Deep Learning (Keras) detectam automaticamente a presença de GPU. Algumas camadas (como GRU) possuem implementações condicionais (`CuDNNGRU` vs `GRU`) para otimizar desempenho em cada hardware.
*   **Segurança:** O carregamento de pesos utiliza `safe_normalization` e verificações de integridade para evitar execução de código malicioso em arquivos `.h5` ou `.keras`.
