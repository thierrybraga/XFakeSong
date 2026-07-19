# Documentação de Inferência das Arquiteturas

Este documento descreve o fluxo real de inferência do XFakeSong: descoberta do
artefato, resolução do contrato de entrada, preparação do tensor e
pós-processamento da predição. A fonte de verdade para a entrada de um modelo
treinado é sempre o `input_contract` salvo no `_config.json`; quando ele não
existe, o sistema cai para o `input_requirements` do registry e, por último,
para heurísticas de shape.

## Fluxo Atual

```text
arquivo/AudioData
  -> ModelLoader: descobre .keras/.h5/.pkl/.pt
  -> ModelInfo: arquitetura, input_shape, input_contract, scaler, temperatura
  -> FeaturePreparer: raw_audio | spectrogram | tabular
  -> Predictor: TensorFlow | PyTorch SSL original | sklearn
  -> p_fake, p_real, threshold, confidence, OOD
```

`ModelLoader` procura artefatos na raiz de `app/models/` e em
`app/models/benchmark_final/<slug>/bench_*`. Modelos TensorFlow são carregados
sob demanda com `custom_objects`; modelos sklearn carregam o scaler lateral
quando existe; artefatos `.pt` de WavLM/HuBERT originais usam wrapper PyTorch
lazy.

## Resolução do Contrato de Entrada

`FeaturePreparer._resolve_input_requirements` aplica esta prioridade:

1. `input_contract` do treino: `input_type`, `format`, `input_shape`,
   `feature_frontend`, `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `n_lfcc`.
2. `ArchitectureRegistry.input_requirements`: fallback genérico para modelos
   sem contrato.
3. Inferência pelo `input_shape`: modelos legados com shape `(T, F)` são tratados
   como espectrograma; shape `(T, 1)` tende a raw audio.

Todo caminho reamostra para o `sample_rate` do contrato, por padrão `16 kHz`,
antes de calcular SincConv, STFT, LFCC ou features tabulares.

## Preparação de Features

### Raw Audio

Usado por AASIST, RawGAT-ST, RawNet2, WavLM, HuBERT, Ensemble e por variantes de
arquiteturas que aceitam forma de onda direta.

1. Downmix para mono.
2. Normalização peak.
3. Center-crop para áudios longos ou repetição (`tile`) para curtos, sem zero-padding.
4. Saída como `(T,)` ou `(T, 1)`, espelhando o artefato treinado.

### Espectrograma

Usado por Conformer, SpectrogramTransformer, Hybrid CNN-Transformer,
MultiscaleCNN, Sonic Sleuth e EfficientNet-LSTM quando o contrato pede
`input_type="spectrogram"`.

O front-end é calculado em `app/domain/services/detection/audio_preprocessing.py`
com `tf.signal`, o mesmo núcleo usado no treino:

| Campo | Padrão atual | Observação |
|---|---:|---|
| `sample_rate` | 16000 | reamostrado antes do front-end |
| `n_fft` | 512 | janela Hann no caminho unificado |
| `hop_length` | 128 | paridade treino/inferência |
| `n_mels` | 80 | log-mel legado |
| `n_lfcc` | 80 | LFCC dos treinos novos |

`feature_frontend="lfcc"` usa banco linear + log + DCT-II, preservando mais
resolução em altas frequências. Modelos legados sem `feature_frontend` continuam
em `logmel` para não quebrar paridade com o treino antigo.

### Tabular Segmentado

Usado por SVM e Random Forest. Quando o artefato não declara `feature_types`, o
fallback robusto é:

```text
spectral + cepstral + temporal + prosodic
```

O áudio é dividido em segmentos de 1 s, sem overlap, com normalização por
segmento. Cada segmento gera features e a agregação padrão é `mean`; também são
suportadas `median`, `std` e `all` (`mean + std + min + max`). O vetor final é
ajustado para a dimensão esperada pelo scaler/modelo por truncamento ou padding.

## Inferência por Backend

### TensorFlow/Keras

1. `prepare_batch_for_model` adiciona batch e ajusta shape.
2. Se existir ONNX ao lado do artefato e `onnxruntime` estiver disponível, ele é
   tentado primeiro.
3. Caso contrário, usa `tf.function` com XLA quando a arquitetura permite.
4. Arquiteturas com `tf.signal` in-graph ou graph attention dinâmico pulam XLA:
   Sonic Sleuth, Ensemble, WavLM, HuBERT, AASIST e RawGAT-ST.
5. Saídas são normalizadas para probabilidade: softmax para logits 2D, sigmoid
   para saída escalar.
6. Aplica `temperature` calibrada, `eer_threshold` quando presente e calcula
   `ood_score` baseado em entropia/energia.

### PyTorch SSL Original

Artefatos `bench_wavlm_original.pt` e `bench_hubert_original.pt` carregam
`torch`/`transformers` apenas na primeira predição. O wrapper normaliza cada
waveform para 16.000 amostras, executa o backbone original congelado
(`WavLMModel` ou `HubertModel`) e aplica o classificador salvo no checkpoint.

### Scikit-learn

SVM e Random Forest recebem vetor 2D `(batch, n_features)`. Se existir scaler
lateral (`*_scaler.pkl`), ele transforma o vetor antes de `predict_proba`.
A convenção de classe é índice `0 = real`, índice `1 = fake`.

## Tabela de Entradas por Arquitetura

| Arquitetura | Contrato principal | Front-end de inferência | Observação |
|---|---|---|---|
| AASIST | raw audio | SincConv + grafos no modelo | default alinhado ao paper; variantes legadas podem usar espectrograma |
| RawGAT-ST | raw audio | SincNet + grafos espectral/temporal | reescrito para raw audio |
| RawNet2 | raw audio | SincNet + FMS + GRU | normalização/corte no preparador |
| WavLM | raw audio | TF fallback ou PyTorch original | WavLM real é PyTorch-only |
| HuBERT | raw audio | TF HuBERT quando disponível ou fallback | `.pt` original usa `HubertModel` |
| Ensemble | raw audio | STFT compartilhado + Mel/LFCC/CQT/MFCC | variantes feature, score, lite e adaptive |
| Sonic Sleuth | raw ou espectrograma | LFCC/MFCC/CQT in-model quando raw | default LFCC |
| EfficientNet-LSTM | raw ou espectrograma | mel + delta + resize | tenta EfficientNetB0 ImageNet; fallback offline |
| MultiscaleCNN | espectrograma | log-mel/LFCC via contrato | Res2Net-style |
| Conformer | espectrograma | log-mel/LFCC via contrato | encoder Conformer com rel-pos |
| Hybrid CNN-Transformer | espectrograma | CCT tokenizer | aceita fallback raw em variantes |
| SpectrogramTransformer | espectrograma | ConvStem + patches | AST treinado do zero |
| SVM | tabular | segmented aggregated features | scaler + SVC RBF |
| Random Forest | tabular | segmented aggregated features | scaler opcional + RF |

## Saída Padronizada

A resposta de predição sempre expõe:

| Campo | Significado |
|---|---|
| `is_deepfake` | decisão binária usando `eer_threshold` quando calibrado |
| `confidence` | probabilidade da classe predita |
| `p_fake` / `p_real` | probabilidades explícitas |
| `temperature_applied` | temperatura de calibração pós-hoc |
| `classification_threshold` | threshold usado para `p_fake` |
| `ood_score` / `is_ood` | score e flag OOD quando há limiar no contrato |

## Incerteza e TTA

`Predictor.predict_with_uncertainty` usa Monte Carlo Dropout para modelos
TensorFlow com dropout ativo e retorna incerteza epistêmica, entropia preditiva
e flag `is_uncertain`. `predict_batch(..., use_tta=True)` aplica pequenas
perturbações de ruído, deslocamento temporal e volume, depois faz média das
predições.

## ONNX

Modelos TensorFlow podem ter um `.onnx` FP32 ao lado do `.keras`. Quando a sessão
ONNX falha por shape/op não suportado, o fallback para TensorFlow é automático.

```python
from app.domain.models.inference.onnx_export import OnnxInferenceSession

with OnnxInferenceSession("app/models/model.onnx") as session:
    predictions = session.predict(features)
```
