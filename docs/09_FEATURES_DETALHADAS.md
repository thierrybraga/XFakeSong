# Funcionalidades Detalhadas: Extração de Features

Este documento detalha o processo de extração de características (features) de áudio no sistema XfakeSong. O sistema utiliza uma abordagem modular e extensível, suportando múltiplos domínios de análise para capturar artefatos sutis introduzidos por algoritmos de deepfake.

## Visão Geral do Pipeline

O pipeline de extração de features segue o fluxo:
1.  **Entrada**: Objeto `AudioData` contendo o sinal de áudio bruto e metadados.
2.  **Pré-processamento**: Normalização, reamostragem (padrão 16kHz ou 22.05kHz) e segmentação (opcional).
3.  **Extração**: Execução de extratores especializados para cada domínio (espectral, temporal, etc.).
4.  **Agregação**: Combinação de vetores de características e cálculo de estatísticas (média, desvio padrão, etc.) se operando em modo segmentado.
5.  **Saída**: Objeto `AudioFeatures` pronto para inferência ou treinamento.

---

## 1. Características Espectrais (Spectral)

Focam na distribuição de energia do sinal no domínio da frequência. Essenciais para detectar anomalias na reconstrução de frequências, comuns em vocoders neurais.

*   **Implementação**: `SpectralFeatureExtractor`
*   **Bibliotecas Base**: `librosa`, `numpy`
*   **Algoritmo Principal**: Short-Time Fourier Transform (STFT)

### Features Extraídas
*   **Spectral Centroid**: "Centro de massa" do espectro. Indica o brilho do som.
*   **Spectral Rolloff**: Frequência abaixo da qual concentra-se X% (85%, 95%) da energia.
*   **Spectral Bandwidth**: Largura de banda espectral.
*   **Spectral Flatness**: Medida de quão parecido o sinal é com ruído branco.
*   **Spectral Contrast**: Diferença de decibéis entre picos e vales no espectro.
*   **Spectral Slope/Skewness/Kurtosis/Spread**: Momentos estatísticos da distribuição espectral.
*   **Spectral Flux**: Taxa de mudança do espectro entre frames consecutivos.

---

## 2. Características Cepstrais (Cepstral)

Representam o envelope espectral de curto prazo, modelando o trato vocal humano. São o padrão ouro em reconhecimento de fala e muito sensíveis a manipulações de identidade.

*   **Implementação**: `CepstralFeatureExtractor`
*   **Bibliotecas Base**: `librosa`, `scipy`

### Features Extraídas
*   **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 a 20 coeficientes que modelam a percepção auditiva humana.
*   **Delta & Delta-Delta MFCC**: Primeira e segunda derivadas temporais dos MFCCs, capturando a dinâmica da fala.
*   **Log Mel Spectrogram**: Espectrograma em escala Mel com amplitude logarítmica.
*   **PLP (Perceptual Linear Prediction)**: Variação dos MFCCs que usa equal volume e compressão de intensidade.
*   **LPCC (Linear Prediction Cepstral Coefficients)**: Derivados de análise LPC (Linear Predictive Coding).

---

## 3. Características Temporais (Temporal)

Analisam o sinal diretamente no domínio do tempo, capturando dinâmicas de energia e estrutura temporal.

*   **Implementação**: `TemporalFeatureExtractor`

### Features Extraídas
*   **Energy (RMS, Short-time)**: Energia do sinal por frame.
*   **Zero Crossing Rate (ZCR)**: Taxa de mudança de sinal (positivo/negativo). Útil para distinguir voz de ruído.
*   **Teager Energy**: Operador de energia não-linear que estima a energia real do sistema físico.
*   **Envelope Features**: Attack time, Decay time, Sustain level (ADSR simplificado).
*   **Temporal Dynamics**: Modulação de amplitude e Tremolo rate.

---

## 4. Características Prosódicas (Prosodic)

Capturam aspectos suprassegmentais da fala como entonação, ritmo e ênfase. Deepfakes muitas vezes falham em reproduzir a micro-prosódia natural.

*   **Implementação**: `ProsodicFeatureExtractor`
*   **Algoritmo Principal**: YIN (para estimativa de F0)

### Features Extraídas
*   **F0 (Fundamental Frequency)**: Pitch da voz. Extraído contorno de F0.
*   **F0 Statistics**: Média, desvio padrão, range, slope, mediana, quartis.
*   **Voicing Probability**: Probabilidade de o frame conter voz (vs. silêncio/ruído).
*   **Jitter**: Perturbação ciclo-a-ciclo na frequência fundamental (micro-tremor).
*   **Shimmer**: Perturbação ciclo-a-ciclo na amplitude.

---

## 5. Características de Formantes (Formant)

Modelam as ressonâncias do trato vocal. Deepfakes podem gerar formantes com larguras de banda ou relações de frequência não naturais.

*   **Implementação**: `FormantFeatureExtractor`
*   **Algoritmo Principal**: Linear Predictive Coding (LPC) com algoritmo de Levinson-Durbin.

### Features Extraídas
*   **F1, F2, F3, F4**: Frequências centrais dos 4 primeiros formantes.
*   **Bandwidths**: Largura de banda de cada formante (indica dissipação de energia).
*   **Formant Amplitudes**: Amplitude relativa dos picos.
*   **Vowel Space Area**: Área do polígono formado por F1 e F2, indicando articulação.
*   **Formant Dispersion**: Média da distância entre formantes consecutivos.

---

## 6. Qualidade Vocal (Voice Quality)

Métricas detalhadas sobre a estabilidade e "textura" da fonação.

*   **Implementação**: `VoiceQualityFeatureExtractor`

### Features Extraídas
*   **HNR (Harmonics-to-Noise Ratio)**: Razão entre energia harmônica e ruído.
*   **NHR (Noise-to-Harmonics Ratio)**: Inverso do HNR.
*   **VTI (Voice Turbulence Index)**: Mede a turbulência de alta frequência (respiração, soprosidade).
*   **SPI (Soft Phonation Index)**: Mede a energia de baixa frequência relativa à alta (qualidade suave/tensa).
*   **DFA (Detrended Fluctuation Analysis)**: Mede a auto-similaridade do sinal de pitch.

---

## 7. Características Perceptuais (Perceptual)

Baseadas em modelos psicoacústicos da audição humana. Capturam artefatos que podem ser audíveis mas não óbvios em espectrogramas tradicionais.

*   **Implementação**: `PerceptualFeatureExtractor`
*   **Escalas**: Bark e ERB (Equivalent Rectangular Bandwidth)

### Features Extraídas
*   **Loudness**: Intensidade subjetiva (Modelo de Stevens/Zwicker).
*   **Sharpness**: Sensação de agudeza/brilho (energia em altas frequências ponderada).
*   **Roughness**: Rugosidade auditiva causada por modulações rápidas (15-300Hz).
*   **Fluctuation Strength**: Sensação de flutuação lenta (< 20Hz).
*   **Tonality**: Distinção entre componentes tonais e ruído.
*   **Masking**: Efeitos de mascaramento simultâneo.

---

## 8. Complexidade e Caos (Complexity)

Tratam o sinal de voz como um sistema dinâmico não-linear. Deepfakes gerados por modelos estatísticos tendem a ter complexidade diferente da voz humana natural (que é caótica determinística).

*   **Implementação**: `ComplexityFeatureExtractor`
*   **Bibliotecas Base**: Algoritmos otimizados customizados (`numba` ready).

### Features Extraídas
*   **Entropy Measures**:
    *   **Approximate Entropy (ApEn)**: Regularidade de padrões.
    *   **Sample Entropy (SampEn)**: Similar ao ApEn, mas mais robusto.
    *   **Permutation Entropy**: Complexidade baseada em ordem ordinal.
    *   **Multiscale Entropy**: Entropia calculada em múltiplas escalas de tempo.
*   **Chaos Measures**:
    *   **Correlation Dimension**: Estimativa da dimensão do atrator do sistema.
    *   **Lyapunov Exponent**: Taxa de divergência de trajetórias (sensibilidade às condições iniciais).
*   **Fractal Dimensions**:
    *   **Higuchi Fractal Dimension**: Complexidade fractal no domínio do tempo.
    *   **Hurst Exponent**: Memória de longo prazo da série temporal.
