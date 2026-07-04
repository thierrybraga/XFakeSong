# Revisão das Arquiteturas vs. Literatura — Status Atual

Esta revisão documenta a conformidade das 14 arquiteturas com suas referências
acadêmicas e com a implementação atual do projeto. Itens que antes eram
propostas já foram incorporados ao código; as lacunas abaixo são o estado
residual, não um plano antigo.

## Resumo Executivo

O projeto está alinhado com a literatura de anti-spoofing em quatro eixos
centrais:

- **Raw audio fiel às referências**: AASIST, RawGAT-ST e RawNet2 usam SincNet ou
  SincConv sobre waveform, não espectrogramas simulados.
- **Front-end espectral robusto**: LFCC é o padrão para treinos novos via
  `feature_frontend="lfcc"`; log-mel permanece como fallback para artefatos
  legados.
- **Augmentation por domínio**: RawBoost entra no treino de raw audio;
  SpecAugment entra no treino de espectrogramas.
- **SSL moderno**: WavLM/HuBERT expõem fine-tuning parcial e variantes
  SSL→AASIST, mas o caminho TensorFlow ainda depende de disponibilidade de
  backbone real; WavLM real é PyTorch-only.

As principais lacunas atuais são: inicialização AST com pesos AudioSet/ImageNet,
variante Res2Net-26 mais leve, calibração clássica ainda opt-in e custo elevado
dos backbones SSL originais.

## Status de Implementação

| Item | Status | Arquivos principais |
|---|---|---|
| RawBoost | Implementado para raw audio | `app/domain/models/training/rawboost.py`, `training_wizard.py` |
| LFCC train↔inference | Implementado, com fallback log-mel legado | `app/domain/services/detection/audio_preprocessing.py` |
| SpecAugment | Implementado para espectrogramas | `app/domain/models/training/spec_augment.py` |
| Fine-tuning SSL parcial | Implementado quando backbone TF existe | `app/domain/models/architectures/ssl_utils.py` |
| SSL→AASIST | Implementado em variantes `wavlm_aasist`/`hubert_aasist` | `wavlm.py`, `hubert.py`, `ssl_utils.py` |
| RawGAT-ST fiel ao paper | Implementado: raw audio + grafo espectral/temporal | `rawgat_st.py` |
| OC-Softmax | Implementado como camada/loss opcional | `layers.py` |
| EER/min-tDCF | Métricas implementadas no pipeline de avaliação | `training/metrics.py` |
| SVM/RF calibrados | Opt-in por `CalibratedClassifierCV` | `classical_ml_helpers.py`, `svm.py`, `random_forest.py` |

## Conformidade por Arquitetura

| # | Modelo | Referência | Conformidade | Observação atual |
|---|---|---|:---:|---|
| 1 | AASIST | Jung et al., ICASSP 2022 | Alta | Default raw audio com SincConv, 6 blocos residuais e grafos heterogêneos; variantes legadas seguem disponíveis |
| 2 | RawNet2 | Jung et al., 2020 | Alta | SincNet + FMS + GRU 1024; RawBoost fica no treino, não no grafo do modelo |
| 3 | RawGAT-ST | Tak et al., 2021 | Alta | Reescrito para raw audio, grafo espectral/temporal e fusão element-wise |
| 4 | WavLM | Chen et al., 2022; Tak et al., 2022 | Média | WavLM real é PyTorch-only; caminho TF usa fallback simplificado, artefatos `.pt` preservam backbone original |
| 5 | HuBERT | Hsu et al., 2021 | Média/Alta | Tenta `TFHubertModel`/conversão quando disponível; fallback Keras mantém compatibilidade |
| 6 | Conformer | Gulati et al., 2020 | Alta | Subsampling 4×, atenção relativa, blocos macaron FF + conv |
| 7 | Sonic Sleuth | Alshehri et al., 2024 | Alta com extensão | LFCC/MFCC/CQT in-model; backbone atual amplia a CNN do paper com SE/residuais |
| 8 | Spectrogram Transformer | Gong et al., 2021 | Média | AST/ViT treinado do zero; `pretrained=True` é metadado reservado, sem pesos AudioSet |
| 9 | EfficientNet-LSTM | CNN-LSTM e EfficientNet em espectrogramas | Boa | Tenta EfficientNetB0 ImageNet e cai para treino do zero se offline |
| 10 | MultiscaleCNN (Res2Net) | Gao et al., 2019 | Boa | Res2Net-50 com `Bottle2neck`; falta variante Res2Net-26 leve |
| 11 | Hybrid CNN-Transformer (CCT) | Hassani et al., 2021; Bartusiak & Delp, 2022 | Boa | Conv tokenizer + Transformer + sequence pooling |
| 12 | Ensemble | Pham et al., 2024 | Boa | STFT compartilhado + Mel/LFCC/CQT/MFCC; variantes feature, score, lite e adaptive |
| 13 | SVM | Baseline clássico | Boa | `StandardScaler` + SVC RBF; calibração externa é opt-in |
| 14 | Random Forest | Baseline clássico | Boa | RF paralelizado; calibração externa é opt-in |

## Pontos Transversais

### RawBoost

**Referência:** Tak et al., ICASSP 2022.

O projeto possui implementação TensorFlow-native de RawBoost para áudio bruto.
Ela substitui o placeholder antigo de ruído gaussiano fixo e deve ser aplicada a
AASIST, RawGAT-ST, RawNet2 e caminhos SSL quando o treino recebe waveform.

### LFCC/CQT

**Referências:** baselines ASVspoof e Sahidullah et al., 2015.

LFCC preserva resolução em altas frequências, onde vocoders frequentemente
deixam artefatos. O front-end unificado usa `lfcc_from_waveform` para garantir
paridade treino↔inferência. CQT permanece em Sonic Sleuth e Ensemble como
representação complementar.

### SpecAugment

**Referência:** Park et al., 2019.

SpecAugment está disponível para modelos de espectrograma via mascaramento em
tempo e frequência. Ele deve ser usado somente após o front-end espectral; os
testes protegem contra aplicação no domínio errado.

### SSL Fine-tuning

**Referências:** WavLM, HuBERT e trabalhos de SSL para ASVspoof.

`set_ssl_backbone_trainability` permite congelamento total, fine-tuning parcial
das últimas N camadas ou fine-tuning total. A recomendação prática é manter
backbones congelados para CPU/demonstração e liberar poucas camadas em GPU com
learning rate baixo.

### SSL→AASIST

As variantes `wavlm_aasist` e `hubert_aasist` conectam hidden states SSL a um
back-end de grafo inspirado no AASIST. Esse caminho reflete a tendência SOTA de
combinar representações SSL com back-ends anti-spoofing especializados.

### Métricas e Calibração

O pipeline reporta EER e min-tDCF, além de temperatura, threshold EER e OOD
quando estes campos existem no `input_contract`. SVM/RF podem usar calibração
externa com `CalibratedClassifierCV`, mas isso aumenta custo de fit e permanece
opt-in.

## Pendências Técnicas

| Pendência | Impacto | Observação |
|---|---|---|
| AST pretrained-init | Médio | Carregar pesos AudioSet/ImageNet quando houver fonte compatível |
| Res2Net-26 | Médio | Reduzir tamanho/latência do MultiscaleCNN |
| Calibração clássica default | Baixo/Médio | Avaliar custo de `CalibratedClassifierCV` antes de tornar padrão |
| WavLM TF real | Alto, mas externo | `transformers` não fornece `TFWavLMModel`; manter caminho PyTorch original |
| Documentar variantes legadas | Baixo | AASIST/RawGAT-ST ainda aceitam CNN/GRU antigas por compatibilidade |

## Referências

- Jung et al. **AASIST**, ICASSP 2022.
- Jung et al. **RawNet2 / Improved RawNet (FMS)**, 2020.
- Tak et al. **RawGAT-ST**, 2021.
- Tak et al. **RawBoost**, ICASSP 2022.
- Tak et al. **wav2vec 2.0 + data augmentation for anti-spoofing**, 2022.
- Chen et al. **WavLM**, 2022.
- Hsu et al. **HuBERT**, 2021.
- Gulati et al. **Conformer**, 2020.
- Gong et al. **AST (Audio Spectrogram Transformer)**, 2021.
- Gao et al. **Res2Net**, 2019.
- Hassani et al. **Compact Convolutional Transformers**, 2021.
- Park et al. **SpecAugment**, 2019.
- Zhang et al. **OC-Softmax / One-class spoofing**, 2021.
- Sahidullah et al. **LFCC para anti-spoofing**, 2015.
