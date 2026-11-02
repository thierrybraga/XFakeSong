# Auditoria e organização do projeto

Auditoria da árvore em 2026-10-18. A árvore canônica proposta, com os caminhos
completos e regras de organização, está em [Estrutura de pastas](folder-structure.md).
Diretórios locais persistidos e arquivos gerados podem existir a mais em uma
máquina de desenvolvimento.

## Avaliação por área

| Área | Resultado da auditoria | Decisão |
| --- | --- | --- |
| Raiz | Contém entrypoints, manifests de dependências, configuração de ferramentas, governança e build. `app.py` tem o mesmo basename do pacote `app/`; a verificação de import confirmou que Python resolve `import app` para `app/__init__.py`, enquanto `python app.py` continua sendo o entrypoint do Space. | Manter entrypoints e manifests na raiz; não criar outra pasta genérica `src/` que obrigaria mudar centenas de imports `app.*`, ferramentas, testes e builds Docker. Preservar `app.py` enquanto o deploy Hugging Face depender desse nome. |
| `app/` | `core/`, `domain/`, `interfaces/` e `utils/` correspondem às camadas descritas. `app/domain/models/` guarda implementação de modelos; `data/models/` guarda artefatos treinados. | Manter `app/` como pacote importável e nome já utilizado pelo projeto. Remover armazenamento de voz do pacote de código. |
| `benchmarks/` | Módulos Python relacionados são importados diretamente por `benchmarks.*` em testes, ferramentas, serviços e notebooks. | Manter os scripts juntos ao harness. Não dividir runners/métricas em subpacotes sem uma migração coordenada de imports e interfaces de CLI. |
| `configs/` | Tem configurações de dataset, inferência e presets de treinamento. `app/core/config/` é código Python de configuração; `configs/` é configuração declarativa do repositório. | Manter ambos: seus nomes próximos refletem responsabilidades distintas, não duplicação. |
| `data/` | É o armazenamento local para cache, datasets, logs, modelos, resultados, temporários, uploads, banco SQLite e perfis de voz. | Manter o conteúdo de runtime fora do Git/código. Cache compartilhado permanece somente em `data/cache/`. |
| `docs/` | É a fonte editorial do MkDocs, agrupada por tema. `archive/` guarda histórico; `assets/`, `notebooks/`, `javascripts/` e `stylesheets/` têm funções distintas. | Manter documentação histórica e assets. Não editar manualmente `infrastructure/site/`, que é gerado a partir daqui. |
| `infrastructure/` | Guarda `docker/` e o diretório de saída do site MkDocs. Dockerfiles por ambiente e perfis Compose ficam juntos. | Manter o caminho Docker consolidado. `site/` é derivado, deve continuar ignorado e ser regenerado pelo MkDocs. |
| `tests/` | `unit/`, `integration/`, `api/`, `functional/` e `smoke/` definem escopos de execução diferentes; fixtures comuns ficam em `conftest.py`. | Manter categorias. Novos testes acompanham a camada/componente testado. |
| `tools/` | CLI operacional está separada em `dataset/`, `training/`, `reporting/` e `ops/`; a lógica central fica em `app/` ou `benchmarks/`. | Manter a divisão por finalidade e nomes de scripts `verbo_objeto.py`. |

## Ajuste aplicado

`VoiceProfileService` gravava dados em `app/voice_profiles/`, misturando estado
mutável com código. O destino canônico é `data/voice_profiles/`, obtido de
`OperationalPaths` para respeitar o diretório de armazenamento configurado. O
script `tools/ops/migrate_voice_profiles_storage.py`
tem dry-run por padrão e migra diretórios e caminhos persistidos na tabela
SQLite `voice_profiles` somente com `--apply`; ele não imprime valores de caminho
ou dados pessoais. A migração do banco local do repositório foi aplicada junto
com a mudança.

## Convenções e limites

- Pastas de primeiro nível usam nomes descritivos e estabelecidos em projetos
  Python. Nenhuma renomeação geral é indicada; `app`, `benchmarks`, `configs`,
  `data`, `docs`, `infrastructure`, `tests` e `tools` têm funções distintas.
- Módulos Python usam `snake_case`; CLIs operacionais começam com um verbo; testes
  seguem `test_*.py`; configurações declaram formato/família no nome.
- Não foram encontrados grupos de arquivos de código/configuração idênticos
  (arquivos com mais de 512 bytes) na varredura de conteúdo. Essa checagem não
  compara datasets, modelos, cache, banco local ou saída HTML gerada.
- Conteúdo local volumoso em `data/` não é inspecionado arquivo por arquivo nem
  removido automaticamente. Pastas históricas e material de pesquisa também não
  são órfãos apenas por não serem importados pelo código de produção.
- `pyproject.toml` concentra configuração de qualidade/testes. Se o projeto
  passar a distribuir um pacote instalável via PyPI, definir metadados PEP 621 e
  uma política de dependências de distribuição será uma tarefa separada.

## Verificações da auditoria

O inventário cobriu nomes e subpastas das áreas de primeiro nível. As verificações
executadas foram:

- AST de 451 arquivos Python com gramática-alvo 3.11: zero erros de sintaxe.
- Hash exato de 539 arquivos de texto/código com mais de 512 bytes: zero grupos
  duplicados. Dados locais e site gerado foram excluídos.
- Links locais nas fontes Markdown ativas: zero links quebrados; arquivos sob
  `docs/archive/` foram excluídos desta validação de links históricos.
- 49 destinos Markdown locais declarados no `mkdocs.yml`: todos existem.
- Os cinco arquivos Compose passaram em `docker compose config --quiet`, usando
  valores descartáveis para as variáveis obrigatórias de autenticação.
- `data/app.db` passou em `PRAGMA quick_check`; nenhum campo de caminho do perfil
  contém o prefixo legado e há um registro com o prefixo canônico.
- Não há diretórios vazios nas áreas de código/documentação, excluindo caches de
  Python e saída gerada. O diretório legado de perfis não existe mais.
- `git diff --check` não encontrou erros de whitespace.

O ambiente de auditoria tem somente Python 3.14, sem Python 3.11/3.12, `pytest`,
NumPy ou MkDocs instalados. Por isso não foi possível executar a suíte de testes,
importar o serviço de voz com as dependências da aplicação ou reconstruir o site
com `mkdocs build --strict`. A validação Python foi sintática, não substitui os
testes de runtime. Nenhum dataset, modelo, cache ou conteúdo pessoal do banco foi
removido ou exibido.
