# Revisão das Arquiteturas vs. Literatura — Melhorias Propostas

> Revisão de conformidade de cada uma das 14 arquiteturas com suas referências
> acadêmicas, focada em **otimizar a detecção de áudio fake (anti-spoofing)**.
> Cada item de melhoria cita a base na literatura. Esta é uma análise — nenhuma
> arquitetura foi alterada nesta revisão.

## Resumo executivo

A maioria das arquiteturas está **fiel ao paper** (AASIST, RawNet2, Conformer,
Sonic Sleuth, Res2Net, CCT). Os ganhos de maior impacto **não estão na topologia
das redes**, mas em três eixos transversais que a literatura de anti-spoofing
aponta como decisivos:

1. **Augmentation (RawBoost + SpecAugment)** — hoje é um *placeholder* de ruído gaussiano.
2. **Front-end (LFCC/CQT em vez de mel)** — mel-spectrograma é subótimo para spoofing.
3. **SSL fine-tuning (WavLM/HuBERT)** — os backbones estão **congelados**, perdendo o maior ganho da literatura recente.

Há também **uma divergência arquitetural real**: o **RawGAT-ST não é fiel ao paper**.

---

## Status de implementação

- ✅ **P0 implementado** — RawBoost (`training/rawboost.py`, ligado ao treino
  raw-audio + placeholders de AASIST/RawGAT-ST) e LFCC como front-end padrão
  (`audio_preprocessing.lfcc_from_waveform`, paridade train↔inference bit-a-bit,
  contrato grava `feature_frontend`; legado cai em log-mel).
- ✅ **P1 implementado** — SpecAugment (`training/spec_augment.py`, ligado ao
  treino de espectrograma) e fine-tuning parcial dos SSL (`architectures/
  ssl_utils.py`; WavLM/HuBERT com `n_trainable_layers` + LR baixo 1e-5 ao
  descongelar; default = últimas 3 camadas quando `transformers` disponível).
- ✅ **P2 implementado** — (a) **RawGAT-ST reescrito fiel ao paper**: agora opera
  sobre **áudio bruto** (`SincConvLayer` real) com grafo espectral (Gs) +
  temporal (Gt) e **fusão element-wise** (corrige a única divergência 🔴; registry
  → `raw_audio`); (b) variante **SSL→AASIST** (`wavlm_aasist`/`hubert_aasist`):
  back-end de grafo AASIST (`ssl_utils.build_ssl_aasist_backend`) sobre os
  hidden-states SSL, em vez do back-end raso (receita SOTA).
- ✅ **P3 implementado** — `min-tDCF` (métrica primária ASVspoof, em
  `training/metrics.py`, integrada a `calculate_all_metrics` junto do EER);
  `OCSoftmaxLayer` + `oc_softmax_loss` (one-class, opcional, em `layers.py`);
  AASIST com **6 blocos residuais** (paridade com o paper); RawNet2 **GRU 1024**;
  e **calibração isotônica opt-in** (`calibrate=True`, `CalibratedClassifierCV`)
  em SVM e Random Forest. (Restam só itens cosméticos: AST pretrained-init e
  Res2Net-26 — não bloqueiam treino.)

Cobertura de testes: `tests/unit/test_frontend_rawboost.py` (P0),
`tests/unit/test_p1_specaug_ssl.py` (P1),
`tests/unit/test_p2_rawgatst_sslaasist.py` (P2) e
`tests/unit/test_p3_metrics_ocsoftmax.py` (P3).

---

## Conformidade por arquitetura

| # | Modelo | Referência | Conformidade | Lacuna principal |
|---|--------|------------|:---:|------------------|
| 1 | **AASIST** | Jung et al., ICASSP 2022 | 🟢 Alta | Encoder com 4 blocos residuais (paper usa 6); SincConv 128 filtros (paper 70) |
| 2 | **RawNet2** | Jung et al., 2020 | 🟢 Alta | GRU 512 (paper 1024); sem RawBoost |
| 3 | **RawGAT-ST** | Tak et al., 2021 | 🟢 Alta ✅ | **Reescrito (P2)**: SincConv real sobre áudio bruto + grafo espectral/temporal + fusão element-wise. (Antes 🔴: era `Conv2D` falso sobre espectrograma.) |
| 4 | **WavLM** | Chen et al., 2022 / Tak et al., 2022 | 🟡 Média | Backbone **congelado**; back-end raso (conv+pooling) vs. AASIST |
| 5 | **HuBERT** | Hsu et al., 2021 | 🟡 Média | Idem WavLM (congelado, back-end simples) |
| 6 | **Conformer** | Gulati et al., 2020 | 🟢 Alta | — (subsampling + rel-pos + macaron FF corretos) |
| 7 | **Sonic Sleuth** | (LFCC/MFCC/CQT) | 🟢 Alta | LFCC interno (front-end correto p/ spoofing) |
| 8 | **Spectrogram Transformer** | AST, Gong et al., 2021 | 🟡 Média | ViT treinado **do zero** (AST usa init ImageNet/AudioSet) |
| 9 | **EfficientNet-LSTM** | CNN-LSTM (lit. híbrida) | 🟢 OK | front-end mel; EfficientNet pré-treinado em RGB natural |
| 10 | **MultiscaleCNN (Res2Net)** | Gao et al., 2019 | 🟢 OK | 49M params (pesado); front-end mel |
| 11 | **Hybrid CNN-Transformer (CCT)** | Hassani et al., 2021 | 🟢 OK | front-end mel |
| 12 | **Ensemble** | — | 🟢 OK | já usa Mel+LFCC+CQT+MFCC (bom) |
| 13 | **SVM** | clássico | 🟢 OK | sem calibração de probabilidade / tuning sistemático |
| 14 | **Random Forest** | clássico | 🟢 OK | idem |

---

## Melhorias transversais (maior impacto, ordenadas)

### 1. RawBoost — augmentation para modelos raw-audio ⭐⭐⭐
**Base:** Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
applied to Automatic Speaker Verification Anti-Spoofing" (ICASSP 2022).
Hoje `aasist.py` e `rawgat_st.py` têm apenas `simple_audio_augmenter` (ruído
gaussiano σ=0.01) — efetivamente um *placeholder*. RawBoost (ruído convolutivo
linear + impulsivo + estacionário de cor) é **a** técnica que mais melhora a
generalização para ataques não vistos em ASVspoof, sem dados extras.
**Ação:** módulo `augmentation_raw.py` com os 3 modos do RawBoost, aplicado a
AASIST/RawNet2/(RawGAT-ST)/SSL no `tf.data` pipeline.

### 2. LFCC/CQT como front-end padrão dos modelos de espectrograma ⭐⭐⭐
**Base:** baselines ASVspoof 2019/2021 (LFCC-GMM), Sahidullah et al. 2015 —
LFCC supera mel/MFCC em spoofing porque preserva resolução em **altas
frequências**, onde ficam artefatos de vocoder. Hoje o `feature_preparer`
entrega **mel-spectrograma** (`n_mels`) a MultiscaleCNN, EfficientNet-LSTM,
SpectrogramTransformer e Hybrid CNN-Transformer. Só Sonic Sleuth (LFCC interno)
e o Ensemble (branch LFCC) usam o front-end correto.
**Ação:** tornar **LFCC** (ou CQT) o `input_type=spectrogram` padrão no contrato
de treino; manter mel como opção. Já existe `LFCCLayer` em `sonic_sleuth.py` e
extrator LFCC no pipeline de features — reaproveitar.

### 3. SpecAugment para todos os modelos de espectrograma ⭐⭐
**Base:** Park et al., 2019. Mascaramento de tempo/frequência é barato e
robustece CNNs/Transformers. O treino atual usa Mixup (memória Sprint 2.4) mas
**não** SpecAugment.
**Ação:** camada/transform de time-mask + freq-mask, opt-in por modelo de spec.

### 4. Fine-tuning dos front-ends SSL (WavLM/HuBERT) ⭐⭐⭐
**Base:** Tak et al., 2022 ("…using wav2vec 2.0 and data augmentation") — o
recorde em ASVspoof 2021 (EER ~0.2%) vem de **fine-tunar** o front-end SSL, não
de congelá-lo. Hoje `freeze_weights=True` por padrão em ambos.
**Ação:** expor `freeze_wavlm=False` com **descongelamento parcial** (últimas N
camadas) + LR discriminativo (backbone 1e-5, head 1e-3). Documentar custo de VRAM.

### 5. Combinação SSL + back-end AASIST ⭐⭐
**Base:** a receita SOTA atual é *wav2vec2/WavLM → grafo AASIST*. Hoje WavLM/HuBERT
usam back-end raso (conv 1D + attention pooling).
**Ação:** variante `wavlm_aasist` que pluga os hidden-states SSL no back-end
GAT+HS-GAL já existente em `layers.py`.

### 6. RawGAT-ST fiel ao paper (correção arquitetural) ⭐⭐
**Base:** Tak et al., 2021. A implementação atual contradiz o nome: recebe
espectrograma, simula SincNet com `Conv2D` (kernels 1×15/1×9/1×5) e usa GAT
genérico.
**Ação:** reescrever usando o `SincConvLayer` real (já existe, usado pelo AASIST)
sobre raw audio + grafo **espectral** e **temporal** separados com fusão
element-wise (graph multiply), como no paper.

### 7. Função de perda one-class / OC-Softmax ⭐⭐
**Base:** Zhang et al., 2021 ("One-class Learning Towards Synthetic Voice
Spoofing Detection"). Objetivos one-class compactam *bonafide* e afastam *spoof*,
generalizando melhor a ataques não vistos que a CE binária. O AASIST já usa
AM-Softmax (margem angular, relacionada); estender a ideia.
**Ação:** oferecer `OCSoftmax`/AM-Softmax como loss opcional nos modelos raw-audio.

### 8. Métricas padronizadas: EER + min-tDCF ⭐
**Base:** protocolo ASVspoof. Já há calibração de EER threshold (Sprint 4.5);
falta expor **min-tDCF**, métrica primária da literatura.
**Ação:** adicionar min-tDCF ao relatório de avaliação/treino.

### 9. Refinamentos pontuais por modelo
- **AASIST:** aumentar encoder para 6 blocos residuais (paridade com paper).
- **RawNet2:** permitir GRU 1024 (config do paper) como variante "paper".
- **Spectrogram Transformer:** init estilo AST (pesos AudioSet) quando disponível.
- **Res2Net:** oferecer variante Res2Net-26 (mais leve que os 49M atuais).
- **SVM/RF:** `CalibratedClassifierCV` (Platt/isotonic) + GridSearch já existente.

---

## Priorização sugerida (custo × impacto)

| Prioridade | Item | Impacto | Esforço |
|:---:|------|:---:|:---:|
| P0 | RawBoost (#1) | Alto | Médio |
| P0 | LFCC default (#2) | Alto | Baixo (reuso) |
| P1 | SpecAugment (#3) | Médio-Alto | Baixo |
| P1 | SSL fine-tuning (#4) | Alto | Médio |
| P2 | RawGAT-ST rewrite (#6) | Médio | Médio-Alto |
| P2 | SSL+AASIST (#5) | Médio-Alto | Médio |
| P3 | OC-Softmax (#7), min-tDCF (#8), refinamentos (#9) | Médio | Baixo-Médio |

> Recomendação: começar por **P0 (RawBoost + LFCC default)** — são os de melhor
> retorno e reaproveitam código já existente (`LFCCLayer`, extrator LFCC, pipeline
> `tf.data`). Ambos beneficiam **todos** os modelos relevantes de uma vez.

---

## Referências

- Jung et al. **AASIST**, ICASSP 2022.
- Jung et al. **RawNet2 / Improved RawNet (FMS)**, 2020.
- Tak et al. **RawGAT-ST**, 2021.
- Tak et al. **RawBoost**, ICASSP 2022.
- Tak et al. **wav2vec 2.0 + data augmentation for anti-spoofing**, 2022.
- Chen et al. **WavLM**, 2022. · Hsu et al. **HuBERT**, 2021.
- Gulati et al. **Conformer**, 2020.
- Gong et al. **AST (Audio Spectrogram Transformer)**, 2021.
- Gao et al. **Res2Net**, 2019. · Hassani et al. **CCT**, 2021.
- Park et al. **SpecAugment**, 2019.
- Zhang et al. **OC-Softmax / One-class spoofing**, 2021.
- Sahidullah et al. **LFCC para anti-spoofing**, 2015.
