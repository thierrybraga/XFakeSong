# 23 — Interface Gradio, Abas e Fluxos de Análise

A interface Gradio é a superfície principal de demonstração do XFakeSong. Ela
fica disponível junto com a API FastAPI quando o app é iniciado por:

```bash
python main.py --gradio
```

ou via Docker:

```bash
docker compose up --build -d
```

Endereços padrão:

| Recurso | URL |
| --- | --- |
| Interface Gradio | `http://localhost:7860/gradio/` |
| Página inicial | `http://localhost:7860/` |
| Healthcheck | `http://localhost:7860/api/v1/system/health` |
| OpenAPI | `http://localhost:7860/api/docs` |

## Configurações Necessárias

Antes de abrir a interface, confirme:

| Item | Configuração recomendada |
| --- | --- |
| Modelos default | `app/models/bench_*` e `app/models/benchmark_final_manifest.json` |
| Diretório de modelos | `DEEPFAKE_MODELS_DIR=app/models` |
| Porta | `GRADIO_SERVER_PORT=7860` |
| Host Docker/HF | `GRADIO_SERVER_NAME=0.0.0.0` |
| Ambiente demo | `DEEPFAKE_ENV=production` |
| Treino em demo pública | `ENABLE_TRAINING=false` |
| Sincronização HF | `MODEL_REPO_ID` + `XFAKE_SYNC_MODELS_ON_BOOT=true` |
| GPU | WSL2/Linux/Docker GPU ou Hugging Face GPU Space |
| Persistência | `XFAKE_STORAGE_DIR=/data` em Hugging Face Storage |

O healthcheck deve retornar `status=healthy` e `models_loaded=14` quando os
modelos consolidados estão disponíveis.

## Barra Superior

A barra superior concentra feedback operacional:

- estado online/offline;
- indicação de GPU;
- número de modelos carregados;
- contagem de perfis;
- notificações pendentes;
- botões de tema e idioma.

As notificações resumem eventos do backend, como inicialização de GPU,
descoberta de modelos, erros de carregamento e avisos de inferência. Use-as
como primeiro ponto de diagnóstico antes de abrir logs do container.

## Aba Painel

Objetivo: visão rápida do estado do sistema.

| Bloco | Uso |
| --- | --- |
| KPIs | análises recentes, modelos carregados, perfis e datasets |
| Últimas análises | histórico resumido das predições |
| Status do sistema | ambiente, GPU, banco e armazenamento |
| Modelos disponíveis | cards resumindo arquiteturas carregáveis |

Use esta aba para confirmar se o app subiu corretamente e se os modelos default
foram encontrados antes de executar uma análise.

## Aba Detectar

Objetivo: executar inferência em áudio individual ou em lote.

### Análise de Áudio

Fluxo recomendado:

1. Abra **Detectar → Análise de Áudio**.
2. Envie um arquivo `.wav`, `.mp3`, `.flac` ou grave pelo microfone.
3. Opcionalmente abra **Configurações Avançadas**.
4. Escolha uma arquitetura/modelo quando quiser forçar um artefato específico.
5. Use **Inferência Segmentada** para áudios longos.
6. Clique em **Analisar Áudio**.

Saídas:

| Saída | Descrição |
| --- | --- |
| Classificação | classe predita (`REAL`, `DEEPFAKE` ou erro operacional) |
| Confiança | probabilidade calibrada da classe predita |
| Forma de onda | visualização temporal do sinal |
| Espectrograma Mel | visualização tempo-frequência |
| Prosódia | energia RMS e pitch quando aplicável |
| JSON técnico | modelo usado, probabilidades, metadados e features |

A inferência usa `DetectionService`, `ModelLoader` e `Predictor`. Os modelos são
carregados sob demanda a partir de `app/models/`, evitando carregar todos os
pesos no startup.

### Análise em Lote

Fluxo:

1. Abra **Detectar → Análise em Lote**.
2. Envie múltiplos arquivos de áudio.
3. Execute a análise.
4. Revise gráfico de distribuição, tabela por arquivo e relatório exportável.

Use lote para demonstrações com conjuntos pequenos e triagem inicial. Para
benchmark científico, use `scripts/run_tcc_pipeline.py` ou
`scripts/run_benchmark.py`.

## Aba Investigar

Objetivo: análise forense e explicabilidade além da classificação binária.

| Análise | Finalidade |
| --- | --- |
| Forma de onda | inspeção visual de amplitude, cortes e silêncio |
| Espectrograma | padrões espectrais e artefatos de síntese |
| Features acústicas | MFCC, LFCC, CQT, RMS, ZCR e métricas espectrais |
| Prosódia | energia, pitch e variações temporais |
| Qualidade vocal | jitter, shimmer, HNR e estabilidade |
| Metadados | duração, sample rate e informações técnicas |
| Explicabilidade | regiões/atributos que influenciam a decisão quando disponível |

Use esta aba para explicar por que uma amostra foi classificada como suspeita e
para gerar material visual de apoio à apresentação.

## Aba Treinar

Objetivo: criar ou atualizar modelos a partir de datasets organizados.

| Etapa | Descrição |
| --- | --- |
| Dataset | seleção de dados reais/fake ou `.npz` consolidado |
| Modelo | arquitetura, variante e hiperparâmetros |
| Validação | split, balanceamento e checagens de compatibilidade |
| Execução | treino, progresso, métricas e salvamento |

Saídas esperadas:

- modelo salvo em `app/models/bench_<modelo>.*`;
- config em `app/models/bench_<modelo>_config.json`;
- logs e métricas de treino;
- gráficos de convergência quando disponíveis;
- compatibilidade imediata com a aba **Detectar** e API.

Em deploy público ou apresentação, mantenha `ENABLE_TRAINING=false` para evitar
treinos acidentais. Para treinamento real, prefira WSL2/Linux com GPU, Docker
GPU ou Hugging Face GPU Space com Storage.

## Aba Gerenciar

Objetivo: administrar datasets, histórico, modelos e artefatos.

| Área | Uso |
| --- | --- |
| Datasets | baixar, validar, balancear e preparar dados |
| Modelos | listar artefatos disponíveis e configs |
| Histórico | consultar análises e exportar registros |
| Perfis de voz | gerenciar amostras de referência quando habilitado |
| Sistema | ações de refresh, diagnóstico e limpeza controlada |

Para o benchmark oficial, a preparação robusta do dataset deve ser feita via
scripts, mas a aba Gerenciar ajuda a verificar se os dados e modelos estão
visíveis para a aplicação.

## Relação com Benchmark e Notebooks

| Objetivo | Interface | Notebook/script equivalente |
| --- | --- | --- |
| Testar um áudio | Detectar | `notebooks/pipeline/03_inference.ipynb` |
| Estudar features | Investigar | `notebooks/features/01_feature_extraction_study.ipynb` |
| Treinar um modelo | Treinar | `notebooks/pipeline/02_training_model.ipynb` |
| Rodar benchmark completo | Não recomendado pela UI | `scripts/run_tcc_pipeline.py` |
| Auditar todos os modelos | Painel/Gerenciar | `notebooks/pipeline/04_all_architectures_full_benchmark.ipynb` |
| Gerar resultados do TCC | Scripts | `docs/15_BENCHMARK.md` |

## Checklist de Validação da UI

Antes de uma apresentação:

- `http://localhost:7860/gradio/` abre sem erro.
- `/api/v1/system/health` retorna `healthy`.
- A barra superior mostra GPU quando aplicável.
- A aba **Detectar** lista modelos treinados.
- Um áudio curto gera classificação, confiança e gráficos.
- A aba **Investigar** renderiza forma de onda e espectrograma.
- A aba **Treinar** está desativada em demo pública ou habilitada apenas em
  ambiente controlado.
- A aba **Gerenciar** mostra datasets/modelos esperados.
- Logs do container não exibem stack trace após startup.

## Problemas Comuns

| Sintoma | Causa provável | Correção |
| --- | --- | --- |
| Nenhum modelo aparece | `app/models` vazio ou `DEEPFAKE_MODELS_DIR` errado | sincronizar Model Hub ou restaurar `app/models/bench_*` |
| Upload falha silenciosamente | `allowed_paths`/temp dir incorreto | usar build atual e `GRADIO_TEMP_DIR=/tmp/gradio` |
| Treino bloqueado | `ENABLE_TRAINING=false` | habilitar somente em ambiente de treino |
| GPU não aparece | Windows nativo/sem passthrough | usar WSL2, Docker GPU ou GPU Space |
| Space perde arquivos | disco efêmero | montar Storage em `/data` |
| Notificações acumulam | eventos do backend não lidos | abrir acordeão de notificações e marcar como lidas |
