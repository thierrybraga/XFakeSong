# Funcionalidades Core: Extração de Features

A extração de características (Feature Extraction) é o coração do sistema XfakeSong. O sistema transforma sinais de áudio brutos em representações matemáticas que alimentam os modelos de detecção de deepfake.

## Feature Extractor Registry

O sistema implementa um **Registry Pattern** (`FeatureExtractorRegistry`) para gerenciar dinamicamente os algoritmos de extração disponíveis. Isso permite que novos extratores sejam adicionados sem modificar o código do orquestrador.

### Tipos de Features Suportados (`FeatureType`)

O sistema categoriza as features em:

1. **SPECTRAL**: Relacionadas ao espectro de frequência (ex: Spectral Flux, Roll-off).
2. **CEPSTRAL**: Relacionadas ao domínio cepstral (ex: MFCC, LFCC).
3. **PROSODIC**: Relacionadas à prosódia e entonação (ex: Pitch, Formantes).
4. **COMPLEXITY**: Medidas de complexidade do sinal (ex: Entropia, Dimensão Fractal).
5. **TEMPORAL**: Características no domínio do tempo (ex: Zero Crossing Rate).

## Extratores Implementados

### 1. Cepstral Features
Focadas na representação do timbre e características vocais.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Padrão ouro em processamento de fala.
- **LFCC (Linear Frequency Cepstral Coefficients)**: Eficaz para capturar artefatos de alta frequência comuns em deepfakes.

### 2. Complexity Features
Analisa a "naturalidade" do sinal através de teoria do caos e fractais. Vozes sintéticas tendem a ter padrões de complexidade diferentes de vozes humanas naturais.
- **Chaos**: Expoentes de Lyapunov.
- **Entropy**: Entropia de Shannon, Entropia Aproximada.
- **Fractal**: Dimensão Fractal de Higuchi.

### 3. Prosodic Features
Analisa a melodia e ritmo da fala.
- **Pitch**: Frequência fundamental (F0).
- **Jitter**: Variação micro da frequência fundamental.
- **Shimmer**: Variação micro da amplitude.

## Como Adicionar um Novo Extrator

Para adicionar um novo extrator, você deve:

1. Criar uma classe que implemente a interface `IFeatureExtractor`.
2. Definir a especificação do extrator (`ExtractorSpec`).
3. Registrar a classe no `FeatureExtractorRegistry`.

Exemplo de estrutura:

```python
class MeuNovoExtrator(IFeatureExtractor):
    def extract(self, audio_data):
        # Lógica de extração
        pass

# Registro
registry.register(ExtractorSpec(
    name="meu_extrator",
    feature_type=FeatureType.CUSTOM,
    ...
))
```

## Configuração de Extração

A extração pode ser configurada via `.env` ou `PipelineConfig`:
- **Paralelismo**: `DEEPFAKE_PARALLEL_EXTRACTION=true` permite extrair features de múltiplos arquivos simultaneamente.
- **Normalização**: As features são normalizadas para garantir estabilidade numérica no treinamento.
