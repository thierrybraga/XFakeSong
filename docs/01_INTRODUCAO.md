# Visão Geral do Sistema XFakeSong

## Introdução

O **XFakeSong** é uma plataforma open source para **detecção de deepfakes de
áudio** (anti-spoofing). Ele combina processamento digital de sinais (DSP) e
aprendizado profundo para analisar um sinal de voz, extrair representações e
classificá-lo como **real** (bonafide) ou **fake** (spoof).

O foco do projeto é **acadêmico e reprodutível**: comparar arquiteturas de
detecção num fluxo local e auditável, gerando as métricas, tabelas e figuras de
um TCC. A arquitetura segue **Clean Architecture**, separando domínio, casos de
uso, interfaces e infraestrutura para facilitar manutenção, testes e adição de
novos modelos/extratores.

## O que o sistema faz

### 1. Detecção com 14 arquiteturas

Um registry unifica **14 arquiteturas** sob a interface
`create_model_by_name(architecture, input_shape, num_classes)`
(`app/domain/models/architectures/factory.py`):

| Categoria | Modelos |
| --- | --- |
| Raw-audio (forma de onda) | RawNet2, AASIST, RawGAT-ST, WavLM, HuBERT |
| Espectrograma / Transformers | Sonic Sleuth, Conformer, SpectrogramTransformer, Hybrid CNN-Transformer |
| CNN / fusão | EfficientNet-LSTM, MultiscaleCNN, Ensemble |
| Clássicos (tabular) | SVM, Random Forest |

Veja [Arquiteturas Neurais](08_ARQUITETURAS.md) e a
[Revisão das Arquiteturas](14_REVISAO_ARQUITETURAS.md).

### 2. Extração de features

O sistema extrai um conjunto rico de características acústicas (cepstral,
espectral, prosódica, perceptual, formante, complexidade, preditiva, etc.). No
**caminho de detecção**, porém, o front-end real é enxuto e ditado pelo
`input_contract` do treino: **forma de onda bruta** (raw-audio), **log-mel** ou
**LFCC** (default anti-spoofing), calculados in-graph com `tf.signal`. Detalhes
em [Features de Áudio](04_FEATURES.md).

### 3. Treinamento e inferência

- **Treino** pelo `TrainingService` (mesmo caminho da API e do benchmark), com
  class weighting, calibração de temperatura, threshold por EER e salvamento do
  modelo de inferência sem estado do otimizador. Ver [Treinamento](10_TREINAMENTO.md).
- **Inferência** pelo `ModelLoader` + `Predictor` (ONNX→TF com fallback),
  aplicando o `input_contract` (temperatura/EER/OOD). Ver
  [Inferência](09_INFERENCIA.md).

### 4. Interfaces

- **UI Gradio**: análise de áudio, análise forense/explicabilidade, assistente
  de treino, gestão de datasets e perfis de voz.
- **API FastAPI** (`/api/v1`): detecção single/multi-model, incerteza
  (MC Dropout), features, treino, datasets e perfis. Ver
  [API Reference](07_API_REFERENCE.md).

### 5. Benchmark do TCC

`scripts/run_tcc_pipeline.py` automatiza dataset → split → treino → inferência →
**métricas, tabelas LaTeX e figuras PNG** (`tcc_report.md`, `dataset.md`).
Métricas padrão da área: acurácia, AUC-ROC, **EER** e **min-tDCF**. Ver
[Benchmark e TCC](15_BENCHMARK.md).

## Tecnologias

- **Linguagem**: Python 3.11 (ambiente de referência; compatível 3.11–3.13).
- **Áudio/DSP**: librosa, NumPy, SciPy, soundfile.
- **ML**: TensorFlow + Keras 3, scikit-learn; ONNX Runtime (inferência opcional).
- **Web**: FastAPI (API), Gradio (UI), Uvicorn.
- **Deploy/Qualidade**: Docker multi-stage, pytest, ruff/black, bandit,
  GitHub Actions. Ver [CI/CD e Segurança](17_CICD_SEGURANCA.md).

## Próximos passos

Comece pela [Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md). Termos
técnicos (EER, min-tDCF, ASVspoof, LFCC, SSL…) estão no
[Glossário](18_GLOSSARIO.md).
