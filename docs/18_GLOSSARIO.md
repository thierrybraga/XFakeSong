# Glossário

Termos usados no XFakeSong e na literatura de detecção de deepfakes de áudio
(anti-spoofing). Para o pipeline em si, veja [Inferência](09_INFERENCIA.md),
[Treinamento](10_TREINAMENTO.md) e [Benchmark](15_BENCHMARK.md).

## Tarefa e dados

- **Deepfake de áudio / spoof** — voz sintética ou manipulada (TTS, conversão
  de voz, replay) que tenta se passar por uma voz humana real.
- **Anti-spoofing / detecção** — classificar um áudio como **bonafide** (real)
  ou **spoof** (fake). É a tarefa central do projeto.
- **Bonafide** — amostra de fala genuína (humana). Classe "real".
- **ASVspoof** — série de *challenges* e datasets de referência da área; define
  protocolos e a métrica **min-tDCF**.
- **SNR (Signal-to-Noise Ratio)** — razão sinal/ruído (dB); usado nos testes de
  robustez (degradação controlada do áudio).

## Métricas

- **Acurácia** — fração de classificações corretas. Sensível a desbalanceamento.
- **EER (Equal Error Rate)** — ponto em que a taxa de falsos positivos (FAR) se
  iguala à de falsos negativos (FRR). Métrica principal de anti-spoofing: **quanto
  menor, melhor**. O threshold de EER é calibrado no conjunto de validação.
- **FAR / FRR** — *False Acceptance Rate* (spoof aceito como real) e *False
  Rejection Rate* (real rejeitado como spoof).
- **min-tDCF (minimum tandem Detection Cost Function)** — custo mínimo do sistema
  combinado (ASV + detector) usado no ASVspoof; pondera os dois tipos de erro.
  **Menor é melhor.** Implementado em `app/domain/models/training/metrics.py`.
- **AUC-ROC** — área sob a curva ROC; probabilidade de ranquear um spoof acima
  de um bonafide. 1.0 = perfeito, 0.5 = aleatório.
- **Matriz de confusão / distribuição de scores** — diagnósticos visuais por
  modelo no relatório do benchmark.

## Front-end de áudio (features)

- **Forma de onda / waveform / PCM** — o sinal bruto 1D (amostras no tempo).
  Entrada dos modelos *raw-audio*.
- **STFT (Short-Time Fourier Transform)** — espectro de curto prazo; base de
  mel/LFCC. Parâmetros: `n_fft` (tamanho da janela), `hop_length` (passo).
- **Espectrograma mel / log-mel** — energia por banda na escala mel (perceptual);
  `n_mels` bandas. Front-end espectrograma clássico.
- **MFCC** — coeficientes cepstrais na escala mel; compactam o envelope espectral.
- **LFCC (Linear-Frequency Cepstral Coefficients)** — como MFCC, mas em escala
  **linear** de frequência. Costuma **superar o mel em anti-spoofing** — é o
  front-end espectrograma *default* do projeto (`feature_frontend="lfcc"`).
- **CQT (Constant-Q Transform)** — transformada com resolução log-frequencial.
- **input_contract** — metadados gravados no treino (front-end, `n_fft`/`hop`/
  `n_mels`, sample rate, temperatura, EER) que garantem **paridade
  treino↔inferência**. Lido pelo `FeaturePreparer`/`ModelLoader`.

## Arquiteturas e blocos

- **SincNet / Sinc-Convolution** — 1ª camada que aprende filtros passa-banda
  parametrizados (sinc) direto na forma de onda (RawNet2, RawGAT-ST).
- **Graph attention (espectro-temporal)** — atenção sobre um grafo de nós
  espectrais/temporais; núcleo do **AASIST** e do **RawGAT-ST**.
- **SSL (Self-Supervised Learning)** — *backbones* pré-treinados sem rótulos
  (**WavLM**, **HuBERT**) usados como extratores; fine-tuning parcial com LR baixo.
- **Ensemble** — fusão de múltiplos ramos/modelos para decisão final.
- **OC-Softmax (one-class softmax)** — *loss* que compacta a classe bonafide e
  afasta spoofs; útil contra ataques não vistos.

## Treino e calibração

- **Class weighting** — pesos por classe para compensar desbalanceamento.
- **Temperature scaling** — calibração pós-treino que divide os logits por uma
  temperatura *T* para a confiança casar com a acurácia (Guo et al., 2017).
- **OOD (Out-of-Distribution)** — entrada fora da distribuição de treino;
  sinalizada por um *energy score* calibrado.
- **MC Dropout** — múltiplos *forward passes* com dropout ativo para estimar
  **incerteza epistêmica** (Gal & Ghahramani, 2016).
- **TTA (Test-Time Augmentation)** — média de predições sobre versões aumentadas
  da entrada.
- **RawBoost / SpecAugment** — augmentations para raw-audio e para espectrograma,
  respectivamente.
- **SWA (Stochastic Weight Averaging)** — média de pesos ao longo do treino para
  melhor generalização.

## Engenharia

- **ONNX / ONNX Runtime** — formato/*runtime* de inferência portável; o
  `Predictor` usa ONNX quando disponível (mais rápido em CPU) com fallback p/ TF.
- **Clean Architecture** — separação em camadas (domínio → casos de uso →
  interfaces/infra) com dependências apontando para dentro.
- **TCC** — Trabalho de Conclusão de Curso; o benchmark gera as tabelas e
  figuras finais.
