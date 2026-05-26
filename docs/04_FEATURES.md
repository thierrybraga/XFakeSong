# Extração de Features de Áudio

O coração do XFakeSong é seu pipeline de extração de características. O sistema transforma sinais de áudio brutos em representações matemáticas que alimentam os modelos de detecção de deepfake.

## Registry de Extratores

O sistema usa o padrão **Registry** (`FeatureExtractorRegistry`) para gerenciar extratores dinamicamente. Novos extratores podem ser adicionados sem modificar o orquestrador.

### Tipos de Features (`FeatureType`)

Definido canonicamente em `app/core/interfaces/audio.py`:

| Valor | Descrição |
|-------|-----------|
| `SPECTRAL` | Distribuição de energia no domínio da frequência |
| `MEL_SPECTROGRAM` | Espectrograma em escala Mel |
| `TEMPORAL` | Dinâmica de energia e estrutura no domínio do tempo |
| `PROSODIC` | Entonação, ritmo, pitch, jitter, shimmer |
| `PERCEPTUAL` | Características psicoacústicas (Bark, ERB, loudness) |
| `ADVANCED` | Features preditivas (LPC, coeficientes de reflexão) |
| `CEPSTRAL` | Envelope espectral de curto prazo (trato vocal) |
| `FORMANT` | Ressonâncias do trato vocal (F1–F4) |
| `VOICE_QUALITY` | Estabilidade e textura da fonação |
| `COMPLEXITY` | Métricas de complexidade e caos (ApEn, SampEn, Hurst) |
| `CUSTOM` | Extratores customizados registrados pelo usuário |

> **Nota**: `app/domain/features/types.py` e `app/domain/features/interfaces.py` são re-exports
> que apontam para a fonte canônica em `app/core/interfaces/audio.py`.

---

## Arquitetura de Extração

O pipeline de extração usa três camadas:

```
ExtractorLoader  →  _extractors: Dict[FeatureType, IFeatureExtractor]
FeatureExtractorCore  →  itera apenas os tipos solicitados em ExtractionConfig
Adapters  →  envolvem extratores do domínio e expõem extract(AudioData) → ProcessingResult
```

Todos os extratores são registrados com **chaves `FeatureType` enum** — sem strings legadas.

---

## Extratores Disponíveis

Registrados automaticamente pelo `ExtractorLoader` a partir dos adaptadores em
`app/domain/features/adapters/`:

| FeatureType | Adaptador | Extrator Interno |
|-------------|-----------|------------------|
| `SPECTRAL` | `SpectralFeatureExtractor` (direto) | — |
| `MEL_SPECTROGRAM` | `MelSpectrogramExtractor` (direto) | — |
| `TEMPORAL` | `TemporalExtractorWrapper` | `TemporalFeatureExtractor` |
| `PROSODIC` | `ProsodicExtractorWrapper` | `ProsodicFeatureExtractor` |
| `CEPSTRAL` | `CepstralExtractorWrapper` | `CepstralFeatureExtractor` |
| `FORMANT` | `FormantExtractorWrapper` | `FormantFeatureExtractor` |
| `VOICE_QUALITY` | `VoiceQualityExtractorWrapper` | `VoiceQualityFeatureExtractor` |
| `PERCEPTUAL` | `PerceptualExtractorWrapper` | `PerceptualFeatureExtractor` |
| `COMPLEXITY` | `ComplexityExtractorWrapper` | complexity extractor |
| `ADVANCED` | `PredictiveExtractorWrapper` | `PredictiveFeatureExtractor` |

Extratores ausentes (ImportError) são ignorados silenciosamente — o pipeline funciona com os tipos disponíveis.

---

## Features por Categoria

### `SPECTRAL` — `SpectralFeatureExtractor`
**Taxa de amostragem padrão**: 22.050 Hz

Detecta anomalias na reconstrução de frequências — artefatos comuns em vocoders neurais.

| Feature | Descrição |
|---------|-----------|
| Spectral Centroid | "Centro de massa" do espectro |
| Spectral Rolloff | Frequência abaixo da qual se concentra 85%/95% da energia |
| Spectral Bandwidth | Largura de banda espectral |
| Spectral Flatness | Proximidade do sinal com ruído branco |
| Spectral Contrast | Diferença dB entre picos e vales no espectro |
| Spectral Flux | Taxa de mudança do espectro entre frames consecutivos |
| Spectral Decrease | Tendência de decaimento espectral |
| Crest Factor | Relação pico-RMS (detecta transientes) |
| Irregularity | Variação de amplitude entre parciais adjacentes |
| Roughness | Rugosidade causada por modulações rápidas |
| Inharmonicity | Desvio dos parciais em relação à série harmônica |

---

### `CEPSTRAL` — `CepstralExtractorWrapper`
**Taxa de amostragem padrão**: 22.050 Hz

Modela o trato vocal humano — altamente sensível a manipulações de identidade.

| Feature | Descrição |
|---------|-----------|
| MFCC (13 coefs.) | Mel-Frequency Cepstral Coefficients |
| Delta-MFCC | Primeira derivada temporal dos MFCCs |
| Delta-Delta-MFCC | Segunda derivada temporal dos MFCCs |
| Log Mel Spectrogram | Espectrograma em escala Mel com amplitude logarítmica |
| PLP | Perceptual Linear Prediction |
| LPCC | Linear Prediction Cepstral Coefficients |

---

### `PROSODIC` — `ProsodicExtractorWrapper`
**Taxa de amostragem padrão**: 22.050 Hz | **Algoritmo F0**: YIN

Deepfakes frequentemente falham em reproduzir a micro-prosódia natural da fala.

| Feature | Descrição |
|---------|-----------|
| F0 (contorno) | Pitch — frequência fundamental |
| F0 Statistics | Média, desvio padrão, range, slope, mediana, quartis |
| Voicing Probability | Probabilidade de frame voiced vs. silêncio |
| Jitter | Perturbação ciclo-a-ciclo na frequência fundamental |
| Shimmer | Perturbação ciclo-a-ciclo na amplitude |
| HNR | Harmonics-to-Noise Ratio |

---

### `FORMANT` — `FormantExtractorWrapper`
**Algoritmo**: LPC + Levinson-Durbin

Modela as ressonâncias do trato vocal. Deepfakes podem gerar formantes com larguras de banda
ou relações de frequência não naturais.

| Feature | Descrição |
|---------|-----------|
| F1–F4 | Frequências centrais dos 4 primeiros formantes |
| Bandwidths | Largura de banda de cada formante |
| Vowel Space Area | Área do polígono F1×F2 (indica articulação) |
| Formant Dispersion | Distância média entre formantes consecutivos |

---

### `VOICE_QUALITY` — `VoiceQualityExtractorWrapper`

Métricas de estabilidade e textura da fonação.

| Feature | Descrição |
|---------|-----------|
| NHR | Noise-to-Harmonics Ratio |
| VTI | Voice Turbulence Index |
| SPI | Soft Phonation Index |
| DFA | Detrended Fluctuation Analysis |

---

### Outras Categorias

| FeatureType | Módulo | Features Representativas |
|-------------|--------|--------------------------|
| `TEMPORAL` | `extractors/temporal/` | Energy RMS, ZCR, Teager Energy, ADSR |
| `PERCEPTUAL` | `extractors/perceptual/` | Loudness (Zwicker), Sharpness, Fluctuation Strength |
| `COMPLEXITY` | `extractors/complexity/` | ApEn, SampEn, Higuchi Fractal Dim., Hurst Exponent |
| `ADVANCED` | `extractors/predictive/` | LPC stability, reflection coefficients |
| `MEL_SPECTROGRAM` | `extractors/mel/` | Log Mel Spectrogram (variantes) |

---

## Adicionando um Novo Extrator

```python
from app.core.interfaces.audio import AudioData, AudioFeatures, FeatureType, IFeatureExtractor
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.domain.features.extractor_registry import (
    extractor_registry, ExtractorSpec, ExtractorComplexity
)


class MeuExtrator(IFeatureExtractor):
    def extract(self, audio_data: AudioData) -> ProcessingResult:
        # lógica de extração
        features = {"minha_feature": np.array([1.0, 2.0])}
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=AudioFeatures(
                features=features,
                feature_type=FeatureType.CUSTOM,
                extraction_params=self.get_extraction_params(),
            )
        )

    def extract_features(self, audio_data: AudioData) -> ProcessingResult:
        return self.extract(audio_data)

    def get_feature_type(self) -> FeatureType:
        return FeatureType.CUSTOM

    def get_feature_names(self):
        return ["minha_feature"]

    def get_extraction_params(self):
        return {"param1": "valor"}


# Registrar no sistema
extractor_registry.register(ExtractorSpec(
    name="meu_extrator",
    feature_type=FeatureType.CUSTOM,
    complexity=ExtractorComplexity.MEDIUM,
    description="Descrição breve do extrator",
    extractor_class=MeuExtrator,
    default_params={"param1": "valor"},
    input_requirements={"sample_rate": 22050},
))
```

---

## Configuração

| Variável de Ambiente | Descrição | Padrão |
|---------------------|-----------|--------|
| `DEEPFAKE_PARALLEL_EXTRACTION` | Extração paralela de múltiplos arquivos | `false` |

Features são normalizadas automaticamente no pipeline para estabilidade numérica no treinamento.
