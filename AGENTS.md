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

# Docker (produção) — perfis segmentados em docker/compose/*.yml
make build && make up      # ou: docker compose -f docker/compose/inference.cpu.yml up --build -d
make logs
make down
```

---

## Arquitetura

O projeto segue **Clean Architecture** com quatro camadas:

```
app/
├── domain/          # Regras de negócio puras — sem dependência de frameworks
│   ├── features/    # Extração de features (extractors/, adapters/, registry)
│   ├── models/      # Arquiteturas neurais (architectures/, training/ com
│   │                #   secure_training_pipeline.py, inference/)
│   ├── services/    # DetectionService, TrainingService, UploadService, etc.
│   ├── dataset_metadata/  # Catálogo de datasets e manifesto de falantes
│   └── xai/         # Explicabilidade dos modelos (Grad-CAM, SHAP, contrato tabular)
├── core/            # Infraestrutura transversal (config, db, auth, contracts)
│   └── contracts/   # Contratos abstratos SOLID (IFeatureExtractor, IDetectionService, ...)
└── interfaces/      # As 3 interfaces de entrada do projeto
    ├── gradio/      # app.py (montagem unificada), schema_patch.py, 5 seções role-based
    │                # (Painel, Detectar, Investigar, Treinar, Gerenciar) em tabs/, utils/
    ├── cli/         # CLI com menus interativos (context.py, menus/)
    └── web/         # FastAPI: main_fastapi.py, routers/ (endpoints), schemas/
                      # (Pydantic request/response), static/, templates/ (Jinja2)
```

Fora dessas camadas, `app/` também tem: `utils/` (utilitários centrais de
áudio/arquivos/sistema/VAD + `colab.py`, helper isolado só para execução via
Google Colab), e `dependencies.py`/`deploy_hf.py` (compartilhados entre
interfaces, por isso ficam em `app/` e não dentro de uma interface específica).

**Fluxo de produção:**
```
HTTP/Gradio → interfaces/web/routers/ ou interfaces/gradio/ → domain/services/ → domain/models/ → resultado
```

**Regra de ouro**: código em `app/domain/` nunca importa de `app/interfaces/` (gradio, cli ou web) nem de frameworks externos diretamente. Novas bibliotecas externas entram via wrapper em `app/core/` ou `app/domain/features/adapters/`. `app/core/contracts/` (contratos abstratos) e `app/interfaces/` (adaptadores de entrada: Gradio/CLI/Web) têm nomes deliberadamente distintos para não confundir as duas camadas.

---

## Stack Tecnológico

| Categoria | Tecnologia |
|-----------|------------|
| ML / DL | Keras 3, TensorFlow, scikit-learn |
| Áudio | librosa, soundfile, scipy, numpy |
| Interface | Gradio 4.x |
| API | FastAPI (implementada: routers, middleware de segurança, rate limiting) |
| Deploy | Docker, Hugging Face Spaces |
| Qualidade | black, isort, flake8, bandit, pytest, pytest-cov |

---

## Modelos Implementados (14 arquiteturas)

| Modelo | Entrada | Particularidade |
|--------|---------|-----------------|
| WavLM | Áudio bruto | SSL real — backbone `microsoft/wavlm-base` (state_dict PyTorch portado p/ Keras, congelado); CNN-1D do zero só como fallback quando o checkpoint não está acessível |
| HuBERT | Áudio bruto | SSL real — backbone `facebook/hubert-base-ls960`, mesmo esquema de fallback do WavLM |
| RawNet2 | Áudio bruto | Filtros SincNet + FMS + GRU |
| Sonic Sleuth | Espectrograma | LFCC/MFCC/CQT — 98,27% accuracy reportado no paper original (Alshehri et al., 2024) |
| AASIST | Áudio bruto | Graph Attention Networks spectro-temporais |
| RawGAT-ST | Áudio bruto | GAT + GRU |
| Conformer | Espectrograma | Conv local + Self-Attention global |
| Hybrid CNN-Transformer | Espectrograma | CCT — Conv tokenizer + Transformer |
| Spectrogram Transformer | Espectrograma | ViT adaptado com ConvStem, pré-treinado AudioSet |
| EfficientNet-LSTM | Espectrograma | Transfer learning + Bi-LSTM |
| MultiscaleCNN (Res2Net) | Espectrograma | Multi-escala hierárquica dentro do bloco |
| Ensemble | Multi-representação | 4 branches (Mel+LFCC+CQT+MFCC) + fusão MLP |
| SVM | Features tabulares (63) | StandardScaler + SVC(rbf) |
| Random Forest | Features tabulares (63) | n_jobs=-1, paralelismo CPU |

Cobertas em dois escopos de benchmark (`benchmarks/config.py`) — oficial (11: os 9
acima exceto Sonic Sleuth/EfficientNet-LSTM/Ensemble, mais WavLM Original e
HuBERT Original via runner PyTorch dedicado) e estendido (5: Sonic Sleuth,
EfficientNet-LSTM, Ensemble, WavLM e HuBERT em porte Keras). Detalhes em
[`docs/models/architectures.md`](docs/models/architectures.md).

Use `from app.domain.models.architectures.factory import create_model` para instanciar por nome.

---

## Variáveis de Ambiente (`.env`)

```env
DEEPFAKE_ENV=development           # development | production
GRADIO_SERVER_PORT=7860
DEEPFAKE_MODELS_DIR=./data/models
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

Coloque áudios em (raiz canônica consolidada em 2026-07-14 — antes havia
fragmentação com `app/datasets/`):
```
data/datasets/
├── real/    # Áudios genuínos
└── fake/    # Áudios sintéticos/deepfake
```

Para datasets públicos (ASVspoof, WaveFake, In-the-Wild, etc.), consulte [`docs/data/public-datasets.md`](docs/data/public-datasets.md).

---

## Documentação

Toda a documentação técnica está em `docs/`, gerada via MkDocs Material
(`mkdocs.yml`). [`docs/index.md`](docs/index.md) é o índice canônico e
completo — consulte-o em vez de duplicar a lista aqui. Os
mais usados no dia a dia de desenvolvimento (reestruturados em 2026-07-19 de
arquivos numerados na raiz de `docs/` para pastas por tema — os nomes antigos
como `03_ARQUITETURA.md` não existem mais):

| Arquivo | Conteúdo |
|---------|----------|
| [`docs/architecture/overview.md`](docs/architecture/overview.md) | Clean Architecture e estrutura de pastas |
| [`docs/architecture/audio-features.md`](docs/architecture/audio-features.md) | Todos os tipos de features e como adicionar novos |
| [`docs/development/developer-guide.md`](docs/development/developer-guide.md) | Padrões de código, logging, convenções |
| [`docs/development/quality-and-testing.md`](docs/development/quality-and-testing.md) | Estratégia de testes e CI/CD |
| [`docs/models/architectures.md`](docs/models/architectures.md) | Arquiteturas neurais detalhadas |
| [`docs/models/training.md`](docs/models/training.md) | Configuração de treinamento e hiperparâmetros |
| [`docs/evaluation/benchmark.md`](docs/evaluation/benchmark.md) | Benchmark, métricas e geração de resultados para TCC |
| [`docs/evaluation/retraining-adjustments.md`](docs/evaluation/retraining-adjustments.md) | Ajustes de hiperparâmetros pós-diagnóstico e retreinos aplicados |
| [`docs/data/dataset-protocol.md`](docs/data/dataset-protocol.md) | Protocolo de dataset (split sem vazamento, locutor×sentença) |
