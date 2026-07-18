# Estrutura de pastas

## Estrutura atual recomendada

```text
app/
├── core/
│   ├── auth/
│   ├── config/
│   ├── contracts/
│   ├── db/
│   └── interfaces/        # compatibilidade temporaria
├── domain/
│   ├── dataset_metadata/
│   ├── features/
│   ├── models/
│   ├── services/
│   └── xai/
├── interfaces/
│   ├── cli/
│   ├── gradio/
│   └── web/
├── models/                # artefatos carregaveis pela aplicacao
└── utils/
```

## Arquivos movidos na reorganizacao observada

| Origem antiga | Destino atual | Motivo |
| --- | --- | --- |
| `app/core/interfaces/*` | `app/core/contracts/*` | nome mais preciso para contratos SOLID |
| `app/core/utils/*` | `app/utils/*` | utilitarios fora de infraestrutura core |
| `app/core/xai/*` | `app/domain/xai/*` | XAI pertence ao dominio de deteccao |
| `app/core/dataset_catalog.py` | `app/domain/dataset_metadata/dataset_catalog.py` | catalogo e conhecimento de dataset sao dominio |
| `app/core/speaker_manifest.py` | `app/domain/dataset_metadata/speaker_manifest.py` | manifesto de falantes e conhecimento de dataset |
| `app/routers/*` | `app/interfaces/web/routers/*` | routers sao adaptadores HTTP |
| `app/schemas/*` | `app/interfaces/web/schemas/*` | schemas sao contratos HTTP |
| `app/static/*` | `app/interfaces/web/static/*` | assets pertencem ao adaptador web |
| `app/templates/*` | `app/interfaces/web/templates/*` | templates pertencem ao adaptador web |
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
