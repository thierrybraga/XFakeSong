# Relatorio de refatoracao arquitetural

## Escopo desta auditoria

Data: 2026-07-08.

Objetivo: auditar `docs/` e `app/`, consolidar a arquitetura alvo, adicionar
documentacao local nas pastas fonte e preservar compatibilidade de imports
legados durante a migracao.

## Estudo realizado

Foram lidos os documentos principais em `docs/`, incluindo arquitetura,
features, API, treinamento, inferencia, benchmark, dataset, frontend, seguranca,
retreino, ambientes e auditoria de pipeline.

Conclusao: a documentacao ja define uma Clean Architecture consistente. A
melhor evolucao e incremental, orientada por contextos do dominio.

## Auditoria inicial

| Categoria | Achado |
| --- | --- |
| Arquivos Python em `app` | 261 |
| Linhas Python em `app` | ~59.937 |
| Maiores funcoes | callbacks Gradio e `FeaturePreparer.prepare_input` |
| Maiores classes | `VoiceProfileService`, `ModelTrainer`, `ArchitectureRegistry` |
| Violacao critica inicial | `benchmarks/` estava ausente apesar de imports ativos |
| Compatibilidade | aliases mantidos para materiais externos ainda usando `app.core.interfaces` |
| Artefatos misturados | `data/models/benchmark_final` contem pesos, metricas e figuras |
| Acoplamento | `core/db/setup.py` carregava modelos de dominio no import do modulo |

## Alteracoes aplicadas nesta etapa

| Alteracao | Motivo | Beneficio | Impacto |
| --- | --- | --- | --- |
| `app/core/interfaces/*` reexporta `app/core/contracts/*` | preservar imports antigos | notebooks e scripts legados continuam importando | sem mudanca de regra de negocio |
| `benchmarks/` restaurado do estado versionado | corrigir quebra de imports publicos | testes, scripts e notebooks voltam a encontrar o harness | recupera comportamento existente |
| imports de modelos em `core/db/setup.py` tornados tardios | reduzir acoplamento em import-time | `app.core.db.setup` deixa de carregar dominio antes do bootstrap | sem mudanca de schema |
| notebooks em `docs/notebooks/pipeline` atualizados para `app.core.contracts` | remover uso documentado de caminho legado | exemplos seguem a arquitetura atual | documental |
| `training_wizard_presenter.py` extraido | separar apresentacao visual do wizard | reduz tamanho/acoplamento de `training_wizard.py` | sem mudanca de callback |
| READMEs em pastas principais de `app` | documentacao obrigatoria local | onboarding e navegabilidade | documental |
| Novos docs arquiteturais em `docs/` | centralizar decisoes e roadmap | rastreabilidade | documental |

## Arquivos criados

- `app/README.md`
- `app/core/**/README.md`
- `app/core/interfaces/*.py`
- `benchmarks/README.md`
- `app/domain/**/README.md`
- `app/interfaces/**/README.md`
- `app/utils/README.md`
- `docs/architecture/modular-architecture.md`
- `docs/architecture/modules.md`
- `docs/architecture/folder-structure.md`
- `docs/architecture/patterns.md`
- `docs/architecture/dependency-graph.md`
- `docs/development/coding-standards.md`
- `docs/development/conventions.md`
- `docs/archive/refactoring-report.md`

## Arquivos movidos, renomeados e removidos

Nenhum arquivo foi movido, renomeado ou removido nesta etapa. A arvore ja estava
com muitos moves/deletes antes desta auditoria; eles estao listados em
`git status`.

## Pendencias P0

1. Separar resultados de benchmark de pesos carregaveis em `data/models`.
2. Continuar quebrando callbacks Gradio gigantes em presenters/callback modules.
3. Consolidar o destino arquitetural de `benchmarks/` como modulo de avaliacao
   ou pacote externo de experimentos.
4. Mover o bootstrap persistente para uma composicao de infraestrutura dedicada
   quando houver repositorios explicitos.

## Roadmap

1. Fechar compatibilidade de imports e testes rapidos.
2. Consolidar `benchmarks` como harness de avaliacao com fronteira documentada.
3. Extrair presenters da UI Gradio.
4. Separar `DetectionService` em casos de uso menores.
5. Isolar persistencia e repositorios.
6. Introduzir contratos por bounded context.
7. Avaliar extracao para microsservicos por contexto.
