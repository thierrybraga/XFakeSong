# XFakeSong — Guia para Codex

## Visão Geral

XFakeSong é uma plataforma open source de detecção de áudio deepfake. Combina análise espectral clássica com modelos Transformer modernos (WavLM, HuBERT), com foco em Explainable AI (XAI) e processamento 100% local.

**Interface**: Gradio (`python main.py --gradio`, porta 7860)
**Python**: 3.11+ | **Licença**: MIT

---

## Comandos Essenciais

```bash
# Iniciar interface Gradio
python main.py --gradio

# Criar estrutura de pastas (bootstrap)
python main.py --bootstrap-dirs

# Scripts interativos
start.bat          # Windows
./start.sh         # Linux/macOS

# Testes
pytest tests/                   # todos os testes
pytest tests/unit/              # somente unitários
pytest --cov=app tests/         # com relatório de cobertura

# Formatação e lint
black app/ tests/
isort app/ tests/
flake8 app/ tests/
bandit -r app/

# Docker (produção)
docker-compose up --build -d
docker-compose logs -f
docker-compose down
```

---

## Arquitetura

O projeto segue **Clean Architecture** com cinco camadas:

```
app/
├── domain/          # Regras de negócio puras — sem dependência de frameworks
│   ├── features/    # Extração de features (extractors/, adapters/, registry)
│   ├── models/      # Arquiteturas neurais (architectures/, training/, inference/)
│   └── services/    # DetectionService, TrainingService, UploadService, etc.
├── application/     # Casos de uso, Pipeline (stages/, workflows/, dto/)
├── core/            # Infraestrutura transversal (config, utils, interfaces, training)
├── interfaces/      # Adaptadores de entrada
│   ├── gradio/      # Interface web com 8 abas
│   └── cli/         # CLI com menus interativos
├── routers/         # Endpoints FastAPI (detection, training, features, etc.)
└── schemas/         # Modelos Pydantic (request/response)
```

**Fluxo de produção (direto):**
```
HTTP/Gradio → routers/ ou interfaces/ → domain/services/ → domain/models/ → resultado
```

**Fluxo via pipeline orquestrado (customizável):**
```
Input → UploadStage → FeatureExtractionStage → DetectionStage → PipelineResult
```

**Regra de ouro**: código em `app/domain/` nunca importa de `app/interfaces/`, `app/routers/` ou frameworks externos diretamente. Novas bibliotecas externas entram via wrapper em `app/core/` ou `app/domain/features/adapters/`.

---

## Stack Tecnológico

| Categoria | Tecnologia |
|-----------|------------|
| ML / DL | Keras 3, TensorFlow, scikit-learn |
| Áudio | librosa, soundfile, scipy, numpy |
| Interface | Gradio 4.x |
| API | FastAPI (em desenvolvimento) |
| Deploy | Docker, Hugging Face Spaces |
| Qualidade | black, isort, flake8, bandit, pytest, pytest-cov |

---

## Modelos Implementados (14 arquiteturas)

| Modelo | Entrada | Particularidade |
|--------|---------|-----------------|
| WavLM | Áudio bruto | SSL — `microsoft/wavlm-base` ou CNN 1D fallback |
| HuBERT | Áudio bruto | 7 blocos Conv1D simulando feature encoder |
| RawNet2 | Áudio bruto | Filtros SincNet + FMS + GRU |
| Sonic Sleuth | Espectrograma | LFCC/MFCC/CQT — 98,27% accuracy |
| AASIST | Espectrograma | Graph Attention Networks spectro-temporais |
| RawGAT-ST | Espectrograma | GAT + GRU |
| Conformer | Espectrograma | Conv local + Self-Attention global |
| Hybrid CNN-Transformer | Espectrograma | CCT — Conv tokenizer + Transformer |
| Spectrogram Transformer | Espectrograma | ViT adaptado com ConvStem |
| EfficientNet-LSTM | Espectrograma | Transfer learning + Bi-LSTM |
| MultiscaleCNN (Res2Net) | Espectrograma | Multi-escala hierárquica dentro do bloco |
| Ensemble | Espectrograma | 4 branches (Mel+LFCC+CQT+MFCC) + fusão |
| SVM | Features tabulares | StandardScaler + SVC(rbf) |
| Random Forest | Features tabulares | n_jobs=-1, paralelismo CPU |

Use `from app.domain.models.architectures.factory import create_model` para instanciar por nome.

---

## Variáveis de Ambiente (`.env`)

```env
DEEPFAKE_ENV=development           # development | production
GRADIO_SERVER_PORT=7860
DEEPFAKE_MODELS_DIR=./app/models
DEEPFAKE_LOG_LEVEL=INFO            # DEBUG | INFO | WARNING | ERROR
DEEPFAKE_PARALLEL_EXTRACTION=false
```

Copie `.env.example` para `.env` antes de executar.

---

## Padrões de Código

- **Formatação**: `black` (linha máxima 88 chars), `isort`
- **Tipos**: type hints obrigatórios em funções públicas
- **Logging**: `logging.getLogger(__name__)` — nunca `print()`
- **Novas regras de negócio**: adicionar em `app/domain/` sem dependência de frameworks
- **Novas features**: implementar `IFeatureExtractor`, registrar em `FeatureExtractorRegistry`
- **Testes**: espelham estrutura de `app/` nas pastas `tests/unit/`, `tests/integration/`, `tests/api/`

---

## Datasets

Coloque áudios em:
```
app/datasets/
├── real/    # Áudios genuínos
└── fake/    # Áudios sintéticos/deepfake
```

Para datasets públicos (ASVspoof, WaveFake, In-the-Wild, etc.), consulte [`docs/12_DATASETS.md`](docs/12_DATASETS.md).

---

## Documentação

Toda a documentação técnica está em `docs/`, gerada via MkDocs Material (`mkdocs.yml`).

| Arquivo | Conteúdo |
|---------|----------|
| `01_INTRODUCAO.md` | Visão geral e funcionalidades |
| `02_INSTALACAO_CONFIGURACAO.md` | Setup local, Docker e HF Spaces |
| `03_ARQUITETURA.md` | Clean Architecture e estrutura de pastas |
| `04_FEATURES.md` | Todos os tipos de features e como adicionar novos |
| `05_GUIA_DEV.md` | Padrões de código, logging, convenções |
| `06_TESTES.md` | Estratégia de testes e CI/CD |
| `07_API_REFERENCE.md` | Endpoints REST da API |
| `08_ARQUITETURAS.md` | Arquiteturas neurais detalhadas |
| `09_INFERENCIA.md` | Fluxo de inferência por arquitetura |
| `10_TREINAMENTO.md` | Configuração de treinamento e hiperparâmetros |
| `11_DEPLOY_HUGGINGFACE.md` | Deploy no HF Spaces |
| `12_DATASETS.md` | Datasets públicos com links e licenças |
| `13_COLAB_GUIDE.md` | Execução via Google Colab |
| `14_REVISAO_ARQUITETURAS.md` | Auditoria técnica das arquiteturas versus literatura |
| `15_BENCHMARK.md` | Benchmark, métricas e geração de resultados para TCC |
