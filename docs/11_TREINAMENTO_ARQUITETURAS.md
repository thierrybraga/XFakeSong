# Documentação de Treinamento das Arquiteturas

Este documento detalha o processo de treinamento, configurações de otimização e peculiaridades de cada arquitetura implementada no XfakeSong.

## 1. Visão Geral do Pipeline de Treinamento

O projeto utiliza um pipeline de treinamento unificado, centralizado na classe `ModelTrainer` e configurado via `OptimizedTrainingConfig`. O objetivo principal é garantir reprodutibilidade e robustez contra overfitting, dado o tamanho limitado de alguns datasets de deepfake.

### Estratégias Globais
*   **Otimizadores:** A maioria das arquiteturas utiliza `Adam` ou `AdamW` (para melhor regularização).
*   **Loss Functions:**
    *   Binária: `BinaryCrossentropy` (ou `SparseCategoricalCrossentropy` com 2 classes).
    *   Multiclasse: `SparseCategoricalCrossentropy`.
*   **Métricas:** Acurácia, Precision, Recall e AUC-ROC são monitoradas padrão.

### Callbacks e Anti-Overfitting
Todas as arquiteturas beneficiam-se de um conjunto robusto de callbacks (definidos em `app/domain/models/training/optimized_training_config.py`):
1.  **AdvancedEarlyStopping:** Monitora `val_loss` com paciência configurável (padrão: 15 épocas) e restaura os melhores pesos.
2.  **ReduceLROnPlateau:** Reduz a taxa de aprendizado (fator 0.5) se a perda de validação estagnar por 8 épocas.
3.  **GradientClipping:** Previne explosão de gradientes, essencial para RNNs (LSTMs/GRUs) e Transformers.
4.  **OverfittingDetection:** Callback customizado que alerta ou interrompe o treino se a divergência entre treino e validação exceder um limiar.

---

## 2. Arquiteturas de Deep Learning (Raw Audio)

Estas arquiteturas processam diretamente a forma de onda e são sensíveis à taxa de amostragem e normalização.

### 2.1 AASIST (Advanced Audio Spoofing Detection)
*   **Otimizador:** `AdamW` (Adam com Weight Decay desacoplado).
*   **Learning Rate:** 0.001 inicial.
*   **Regularização:**
    *   Forte uso de **L2 Regularization** (`l2_reg_strength`) nas camadas densas finais.
    *   **Dropout Progressivo:** A taxa de dropout aumenta nas camadas mais profundas (até 2x a taxa base).
*   **Peculiaridades:**
    *   Utiliza camadas customizadas `AudioFeatureNormalization` no início do grafo.
    *   Exige tensores 4D `(batch, time, freq, 1)` simulados via reshape, mesmo operando conceitualmente sobre features espectrais ou raw.

### 2.2 RawNet2
*   **Otimizador:** `Adam` (LR=0.001).
*   **Camadas Críticas:**
    *   **SincNet / Conv1D:** A primeira camada aprende filtros passa-banda. Seus pesos são sensíveis à inicialização.
    *   **AudioResamplingLayer:** Garante que a entrada esteja em 16kHz dentro do grafo.
*   **Peculiaridades:**
    *   Utiliza `Feature Map Scaling (FMS)` nos blocos residuais, atuando como um mecanismo de atenção de canal leve.
    *   O modelo é sensível a variações de amplitude, por isso inclui `AudioNormalizationLayer` (Média 0, Desvio 1) obrigatória.

### 2.3 HuBERT & WavLM (Transformers)
*   **Otimizador:** `Adam` (LR=0.001).
*   **Estrutura de Treinamento:**
    *   Implementados para compatibilidade com Keras 3.
    *   Geralmente utilizados em modo **Fine-Tuning**: O backbone (Feature Encoder + Transformer) pode ser congelado, treinando apenas o `Projection Head`.
*   **Peculiaridades:**
    *   Consomem muita memória VRAM devido aos mecanismos de auto-atenção.
    *   Recomendado o uso de `GradientAccumulation` (simulado via batch size menor) em GPUs menores.

---

## 3. Arquiteturas Baseadas em Features (Espectrogramas)

### 3.1 EfficientNet-LSTM (Híbrido)
*   **Estratégia:** Transfer Learning + Modelagem Temporal.
*   **Backbone:** `EfficientNetB0`.
    *   **Inicialização:** `weights=None` (Treinamento do zero) ou `ImageNet` (se especificado explicitamente).
    *   **Freezing:** Por padrão, congela todas as camadas exceto as últimas 20 (`efficientnet.layers[:-20]`). Isso foca o treinamento na adaptação de features de alto nível.
*   **Pré-processamento In-Graph:**
    *   Converte áudio raw para espectrograma via `STFTLayer` -> `MagnitudeLayer`.
    *   Redimensiona para `(224, 224, 3)` para compatibilidade com a EfficientNet.
*   **Peculiaridades:**
    *   Combina CNN (espacial) com Bi-LSTM (temporal). O redimensionamento das features da CNN para a LSTM é crítico (`Reshape` layer).

### 3.2 Conformer
*   **Otimizador:** `Adam`.
*   **Arquitetura:** Combina Convoluções (local) com Transformers (global).
*   **Peculiaridades:**
    *   Implementa `ConformerBlock` customizado.
    *   Altamente dependente de **Data Augmentation** (Time Masking, Frequency Masking) para evitar overfitting rápido.

---

## 4. Machine Learning Clássico

Modelos que não utilizam gradiente descendente estocástico (SGD) da mesma forma que redes neurais profundas.

### 4.1 SVM (Support Vector Machine)
*   **Backend:** `sklearn.svm.SVC`.
*   **Pipeline:**
    1.  `StandardScaler`: Normalização (Média=0, Var=1) é obrigatória para SVMs baseados em kernel RBF.
    2.  `SVC`: Configurado com `probability=True` para permitir saída de scores de confiança (necessário para AUC).
*   **Treinamento:**
    *   Não suporta "mini-batch" nativo. O dataset inteiro deve caber na memória RAM.
    *   Utiliza `create_feature_loader` para carregar features tabulares pré-extraídas.

### 4.2 Random Forest
*   **Backend:** `sklearn.ensemble.RandomForestClassifier`.
*   **Pipeline:** Similar ao SVM, mas menos sensível à normalização (embora o `StandardScaler` seja mantido por padronização).
*   **Peculiaridades:**
    *   Treinamento paralelo (CPU multi-core) habilitado por padrão (`n_jobs=-1`).
    *   Robusto a features irrelevantes, servindo como bom baseline.

---

## 5. Tabela Resumo de Hiperparâmetros Padrão

| Arquitetura | Otimizador | LR Inicial | Batch Size (Rec.) | Épocas (Típico) | Peculiaridade Principal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AASIST** | AdamW | 1e-3 | 32 | 50-100 | Regularização L2 agressiva |
| **RawNet2** | Adam | 1e-3 | 32 | 100 | Primeira camada SincNet/Conv1D |
| **EfficientNet**| Adam | 1e-3 | 16 | 30 | Freezing parcial do backbone |
| **HuBERT** | Adam | 1e-4 | 8 | 20 | Fine-tuning de backbone pesado |
| **SVM** | N/A (SMO) | N/A | Full Batch | N/A | Exige normalização de features |
