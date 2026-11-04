# Estrutura de pastas

## Estrutura atual recomendada

```text
SpeechChecker/
├── app/                         # pacote Python da aplicacao
│   ├── core/                    # configuracao, contratos, DB e infraestrutura
│   │   ├── contracts/
│   │   ├── interfaces/          # compatibilidade temporaria
│   │   ├── config/
│   │   └── db/
│   ├── domain/                  # regras de negocio
│   │   ├── dataset_metadata/
│   │   ├── features/
│   │   ├── models/              # arquiteturas, treinamento e inferência
│   │   ├── services/
│   │   └── xai/
│   ├── interfaces/              # adaptadores CLI, Gradio e API
│   ├── utils/
│   ├── dependencies.py
│   ├── deploy_hf.py
│   └── file_storage.py
├── benchmarks/                  # harness experimental importável
├── configs/                     # configurações de dataset, inferência e treino
├── data/                        # dados, artefatos e estado local; sem código-fonte
│   ├── cache/
│   ├── datasets/
│   ├── logs/
│   ├── models/
│   ├── results/                 # inclui results/paper/
│   ├── temp/
│   ├── uploads/
│   ├── voice_profiles/
│   └── app.db                   # banco SQLite local
├── docs/                        # fonte editorial MkDocs
├── infrastructure/
│   ├── docker/                  # compose/ e environments/
│   └── site/                    # site gerado, ignorado pelo Git
├── tests/                       # unit/, integration/, api/, functional/ e smoke/
├── tools/                       # CLIs dataset/, training/, reporting/ e ops/
├── app.py                       # entrypoint Hugging Face Spaces
├── main.py                      # entrypoint local
└── requirements*.txt, Makefile, mkdocs.yml
```

### Regras de organizacao

- O código da aplicação fica em `app/`; ferramentas operacionais ficam em `tools/` e o motor experimental em `benchmarks/`.
- `data/` guarda datasets, cache, modelos, resultados e outros artefatos locais; código-fonte não deve ser colocado ali.
- `data/voice_profiles/` é o armazenamento canônico dos dados de perfis de voz; o serviço resolve seu caminho pelo diretório de dados configurado.
- `data/cache/` é o repositório canônico local para cache compartilhado. Os perfis de inferência montam `data/` em `/app/data`; os de treino vinculam a raiz do projeto em `/app`. Em ambos, o cache mantém o caminho único `/app/data/cache/`.
- `data/results/paper/` contém fontes e artefatos acadêmicos; os demais resultados de execução são regeneráveis.
- `infrastructure/site/` é a saída do MkDocs: não edite manualmente.
- `infrastructure/docker/compose/` e `infrastructure/docker/environments/` são os caminhos canônicos dos perfis Compose e Dockerfiles de desenvolvimento.
- O `Dockerfile` permanece na raiz para o deploy Docker do Hugging Face Spaces.
- `app/` permanece como pacote Python principal: imports `app.*` são usados pelo
  entrypoint, pelas interfaces, ferramentas, testes e builds Docker. A migração
  para `src/`/`speechchecker/` seria uma mudança pública de imports sem benefício
  funcional identificado nesta auditoria.
- `benchmarks/`, `configs/`, `data/`, `docs/`, `infrastructure/`, `tests/` e
  `tools/` têm escopos distintos e nomes convencionais; não foram renomeados.

## Auditoria da árvore

A avaliação por área, as colisões verificadas e os limites da auditoria estão em
[Auditoria e Organização](project-structure.md). A árvore desta página é a fonte
única para os caminhos canônicos do repositório.

## Arquivos movidos na reorganização observada

| Origem antiga | Destino atual | Motivo |
| --- | --- | --- |
| `app/core/interfaces/*` | `app/core/contracts/*` | nome mais preciso para contratos SOLID |
| `app/core/utils/*` | `app/utils/*` | utilitários fora de infraestrutura core |
| `app/core/xai/*` | `app/domain/xai/*` | XAI pertence ao dominio de detecção |
| `app/core/dataset_catalog.py` | `app/domain/dataset_metadata/dataset_catalog.py` | catálogo e conhecimento de dataset sao dominio |
| `app/core/speaker_manifest.py` | `app/domain/dataset_metadata/speaker_manifest.py` | manifesto de falantes e conhecimento de dataset |
| `app/routers/*` | `app/interfaces/web/routers/*` | routers sao adaptadores HTTP |
| `app/schemas/*` | `app/interfaces/web/schemas/*` | schemas sao contratos HTTP |
| Tema visual | `app/interfaces/gradio/static/css/theme.css` | tema fica junto da única interface visual; a antiga página Jinja foi removida |
| `app/main_fastapi.py` | `app/interfaces/web/main_fastapi.py` | app FastAPI e adaptador de entrada |
| `app/gradio_schema_patch.py` | `app/interfaces/gradio/schema_patch.py` | patch especifico do adaptador Gradio |

## Estrutura futura opcional

Nao criar `modules/` agora. Quando os contextos estiverem prontos para extracao,
criar pacotes por contexto pode ser feito incrementalmente:

```text
app/domain/services/detection/    -> app/modules/detection/
app/domain/features/              -> app/modules/features/
app/domain/models/training/       -> app/modules/training/
```

Essa migracao deve ser precedida por aliases de compatibilidade e testes de
contrato para evitar quebrar notebooks, scripts e artefatos.
