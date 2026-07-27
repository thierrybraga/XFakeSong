# 28 — Protocolo Final de ML (versão consolidada)

Este documento é a **referência canônica** da metodologia da versão final do
XFakeSong: pré-processamento, splits, treinamento, ruído, calibração,
hiperparâmetros e métricas dos **11 modelos promovidos**. Ele consolida o que
está implementado no código (caminhos citados em cada seção) e ancora cada
decisão na literatura da área. Documentos históricos (RETREINO_AJUSTES,
planos 21/25/26) permanecem como trilha de auditoria — os números válidos são
somente os da seção [Resultados finais](#8-resultados-finais-consolidados).

## 1. Pré-processamento

Implementação: `app/domain/features/benchmark_frontend.py` (fonte única
treino↔inferência) e `benchmarks/data.py::prepare_input_for_architecture`.

| Etapa | Política |
| --- | --- |
| Decodificação | mono, `float32`, 16 kHz (`soxr_hq`) |
| Janela do corpus | 5 s (80.000 amostras) por amostra |
| AGC | RMS/LUFS (`app/utils/silero_vad.apply_agc`), idêntica no corpus e na inferência |
| Normalização | z-score por amostra (frontend raw); dB-ref-max (log-Mel) |
| Janela curta | repetição (`tile`), sem zero-padding |
| Janela por família | raw 1 s (AASIST/RawGAT-ST legadas) ou 4,04 s (64.600, convenção do baseline ASVspoof 2021); log-Mel 128 bandas; vetor tabular de 63 descritores (SVM/RF) |

O contrato de entrada de cada modelo (`input_contract` no
`bench_*_config.json`) grava janela, frontend, estratégia de crop e
calibração — o `Predictor` honra o contrato na inferência, garantindo
paridade por construção.

## 2. Dataset e splits

Dataset canônico: `data/datasets/benchmark_dataset.npz` — 40.980 amostras
(20.490 reais / 20.490 falsas), CETUC pareado com clones XTTS-v2, splits
congelados 33.226/3.976/3.778 com **test-lock** (o teste nunca participa de
treino, validação, calibração ou HPO). Detalhes e auditoria:
[Protocolo de Dataset](../data/dataset-protocol.md),
[Dataset do Benchmark](../data/benchmark-dataset.md) e
[Pipeline e Auditoria](../data/pipeline-and-audit.md).

> Os resultados desta página foram obtidos sobre o **dataset anterior** — 15.000
> amostras de quatro fontes PT-BR, splits 10.500/2.250/2.250 —, já apagado do
> disco. A ressalva abaixo se refere a ele.

**Ressalva de validade (obrigatória ao citar resultados):** o conjunto é
balanceado por classe mas **confundido por fonte** (MLS/TTS-Portuguese só
contêm reais; Fake Voices só falsas; apenas BRSpeech tem as duas classes do
mesmo locutor). Todos os resultados são portanto **in-domain**. A validação
anti-atalho (teste isolado em BRSpeech, única fonte sem atalho possível:
EER dos SSL permanece < 0,5%) mitiga, mas não elimina, essa limitação.

O dataset canônico atual elimina esse confundimento por construção — as duas
classes compartilham locutor e frase, e o oráculo de fonte fica em 50%. As
garantias exigidas para reivindicar generalização (proveniência hierárquica,
oráculo de atalho, bootstrap por cluster, disjunção dupla) estão no
[Protocolo de Dataset](../data/dataset-protocol.md). **Os resultados desta
página precedem esse dataset e requerem retreino.**

## 3. Treinamento

Orquestração: `benchmarks/runner.py` + `app/domain/services/training_service.py`
(+ `SecureTrainingPipeline`). Regras:

- **Orçamento padronizado**: 100 épocas por modelo, batch conforme
  `NEURAL_BENCHMARK_HPARAMS` (`benchmarks/planning.py`), seed 42, sem early
  stopping (comparabilidade entre arquiteturas).
- **Compile-respect**: LR/otimizador/scheduler pertencem ao `create_model`
  de cada arquitetura; o benchmark não os sobrescreve.
- **Checkpoint guardado**: o "melhor por val_loss" só é restaurado se não
  degradar o val re-avaliado (`TrainingService._guarded_checkpoint_restore`)
  — proteção contra épocas instáveis/NaN observadas empiricamente.
- **Shuffle real por época** no `ModelTrainer` (o Keras ignora `shuffle`
  com `tf.data`; o embaralhamento é explícito).
- **SSL (WavLM/HuBERT originais, PyTorch)**: backbone congelado + cabeça
  com *weighted layer sum* (softmax sobre as 13 hidden states) e pooling
  mean⊕std — protocolo SUPERB (Yang et al., 2021); janela 4 s. Contrato
  gravado no `.pt` (`embedding_config`) e honrado por
  `app/domain/models/inference/ssl_head.py`.

## 4. Ruído e robustez

Protocolo AWGN **no domínio da forma de onda, antes de qualquer frontend**
(`benchmarks/runner.py::_prepare_protocol_splits`):

- **Avaliação**: SNR 30/20/10 dB com semente determinística
  `seed+20000+snr` — a mesma realização de ruído para todas as
  arquiteturas (comparabilidade).
- **Treino**: 1 cópia AWGN estática (sementes disjuntas das de avaliação).
  Os resultados promovidos de AASIST/RawGAT-ST usaram, no lugar da cópia
  estática, augmentation **dinâmico por época**
  (`architecture_specific_augmentation=True`: sorteia AWGN por SNR, RawBoost
  completo LnL+ISD+SSI, simulação de codec, RIR sintética, shift temporal e
  compressão dinâmica — a cópia estática causava overfit à realização fixa
  de ruído; composição segue o RawBoost, Tak et al., ICASSP 2022).
  **Nota**: por uniformidade comparativa, esse regime por-arquitetura é
  hoje *opt-in* e deve ser reportado como ablação — os números promovidos de
  AASIST/RawGAT-ST são declarados com esse regime explicitado.
- **Codec**: round-trip MP3 64k / Opus 24k via ffmpeg
  (`benchmarks/perturbations.py`), aplicado no mesmo ponto do protocolo.

## 5. Threshold e calibração

- **Comparação acadêmica**: accuracy, precision, recall, F1 e matrizes usam
  threshold fixo `0,5` (`accuracy_at_fixed_threshold`).
- **Operação/produção**: o threshold EER calculado exclusivamente na validação
  persiste no `input_contract`; quando avaliado, aparece separadamente como
  `accuracy_at_calibrated_threshold`.
- `accuracy_at_eer_oracle` usa o EER do próprio conjunto avaliado e é apenas teto
  diagnóstico; não é uma métrica operacional nem substitui a tabela principal.- **Temperature scaling** (Guo et al., ICML 2017): grid search de T por
  NLL na validação (`TemperatureScaler`), aplicado pelo `Predictor`.
- **Calibração Platt** adicional quando o modelo permanece miscalibrado
  (bloco `calibration` no `input_contract`, aplicado por
  `apply_posthoc_calibration` em
  `app/domain/services/detection/predictor.py`). Monotônica: não altera
  EER/ranking, só a confiança. Aplicada na versão final a AASIST
  (ECE 4,60%→1,46%), RawGAT-ST (23,07%→2,75%) e RandomForest.
- **ECE** (15 bins) reportado em toda avaliação
  (`benchmarks/evaluate.py`).

## 6. Otimização de hiperparâmetros

Três fontes, mantidas em sincronia (ver CLAUDE.md):
`registry.default_params` (dropout, L2, patience, clip, augmentation),
`create_model` de cada arquitetura (LR, weight decay, scheduler, loss) e
`NEURAL_BENCHMARK_HPARAMS` em `benchmarks/planning.py` (valores efetivos do
benchmark). O `benchmark_plan.json` de cada run registra os valores
realmente usados — é a fonte de verdade para reprodutibilidade.

## 7. Métricas e relatório

Por modelo e condição (limpo, 30/20/10 dB, MP3, Opus):
accuracy, precision, recall, F1, AUC-ROC, **EER**, **min t-DCF** (protocolo
ASVspoof 2021 — EER como métrica primária da tarefa DF, min t-DCF para o
cenário tandem), ECE, curvas ROC e **DET** (escala probit), matriz de
confusão, eficiência (parâmetros, MB, latência). **IC 95% por bootstrap**
(1.000 reamostragens, percentil) em EER/AUC/accuracy — com n=2.250 o IC do
EER é ±0,5–1 pp, sem o qual o ranking fino não é interpretável. O protocolo
O protocolo exige adicionalmente bootstrap por `cluster_ids` e métricas por
fonte/gerador (macro e worst-group).

## 8. Resultados finais consolidados

Fonte única: `data/results/final_consolidated_20260715/` (promovidos em
`data/models/benchmark_final/`, run seed 42, test-lock, protocolo acima).
**Escopo: in-domain** (ver §2).

| Modelo | EER [IC95] | Acc | Acc@10dB | ECE¹ |
| --- | --- | --- | --- | --- |
| Conformer | 0,18% [0,00–0,36] | 99,82% | 98,0% | 1,88% |
| HuBERT Original | 0,18% [0,00–0,44] | 99,87% | 96,8% | 0,32% |
| Res2Net (MultiscaleCNN) | 0,36% [0,09–0,58] | 99,69% | 97,5% | 0,37% |
| WavLM Original | 0,36% [0,09–0,58] | 99,69% | 98,8% | 0,35% |
| SVM | 0,58% [0,31–1,16] | 99,24% | 93,8% | 0,98% |
| CCT (Hybrid CNN-Transformer) | 0,71% [0,40–1,16] | 99,20% | 95,0% | 0,77% |
| AST (SpectrogramTransformer) | 0,98% [0,53–1,47] | 99,02% | 93,2% | 0,97% |
| RandomForest | 2,09% [1,47–2,71] | 97,82% | 92,4% | 11,28%¹ |
| RawNet2 | 2,71% [2,09–3,47] | 97,16% | 93,0% | 0,81% |
| AASIST | 4,89% [3,92–5,73] | 95,02% | 88,7% | 4,60→1,46%¹ |
| RawGAT-ST | 6,22% [5,20–7,20] | 93,60% | 84,0% | 23,07→2,75%¹ |

¹ ECE medido nos scores brutos; AASIST/RawGAT-ST/RandomForest recebem
calibração Platt na inferência (valores pós-calibração indicados). A
calibração não altera EER.

## 8.1 Execução por famílias

O manifesto em `benchmarks/config.py` é a fonte única dos escopos. A suíte
oficial contém 11 modelos; Sonic Sleuth, EfficientNet-LSTM e Ensemble pertencem
ao escopo `extended` e nunca são consolidados automaticamente com o artigo.

Ordem recomendada: `classical-tabular`, `spectral-convolutional`,
`spectral-attention`, `waveform-end-to-end`, `ssl-pretrained` e, em execução
separada, `extended`. Cada família deve passar por `--plan-only`, seed 42,
validação de artefatos e então seeds 42–46 para candidatos finais.

Exemplo:

```bash
python scripts/training/train_by_family.py --family classical-tabular --plan-only
python scripts/training/train_by_family.py --family ssl-pretrained --seeds 42 43 44 45 46 --resume
```

O `--resume` só reutiliza um modelo quando `run_fingerprint.json` corresponde ao
dataset/test-lock, comando, código e revisão atuais. O plano também grava
`effective_training_config.json` normalizado e hasheado.
## 9. Limitações declaradas

1. Resultados **in-domain** (confundimento fonte-classe, §2) — não
   sustentam generalização cross-domain/cross-gerador sem o protocolo atual.
2. Semente única (42); multi-sementes suportado
   (`run_models_sequential --seeds`) mas não executado no run final.
3. WavLM/HuBERT no caminho TF do benchmark usam fallback CNN-1D — os
   resultados "originais" reportados vêm dos backbones SSL reais em
   PyTorch (artefatos `*_original.pt`).
4. Avaliação cross-dataset (ASVspoof/In-the-Wild) pendente.

## Referências

- Yamagishi et al., *ASVspoof 2021: accelerating progress in spoofed and
  deepfake speech detection* — [arXiv:2109.00537](https://arxiv.org/abs/2109.00537);
  [Plano de avaliação](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)
  (EER primária p/ DF; min t-DCF p/ tandem).
- Jung et al., *AASIST: Audio Anti-Spoofing using Integrated
  Spectro-Temporal Graph Attention Networks*, ICASSP 2022.
- Tak et al., *End-to-end anti-spoofing with RawNet2*, ICASSP 2021;
  *RawGAT-ST*, ASVspoof Workshop 2021.
- Tak et al., *RawBoost: A Raw Data Boosting and Augmentation Method
  applied to Automatic Speaker Verification Anti-Spoofing*, ICASSP 2022 —
  [arXiv:2111.04433](https://arxiv.org/abs/2111.04433).
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017
  (temperature scaling); Platt, 1999 (Platt scaling).
- Yang et al., *SUPERB: Speech processing Universal PERformance
  Benchmark*, Interspeech 2021 (weighted layer sum p/ SSL).
- ASVspoof 5 (2024–) — [plano de avaliação](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf)
  (referência para avaliação externa futura).
