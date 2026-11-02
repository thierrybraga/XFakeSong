# Plano de auditoria e adequação para publicação e TCC

## Objetivo

Preparar o SpeechChecker para uma publicação pública reproduzível e para uma
apresentação de TCC verificável. O plano cobre código, instalação, dados,
modelos, testes, segurança, documentação, containers e rastreabilidade dos
resultados científicos.

Este documento registra a revisão estrutural inicial e define os gates de
auditoria. Ele não declara que todas as verificações abaixo já foram executadas.

## Revisão estrutural inicial

- O projeto separa a aplicação (`app/`), scripts operacionais (`tools/`),
  harness científico (`benchmarks/`), configurações (`configs/`), documentação
  (`docs/`), containers (`infrastructure/docker/`) e testes (`tests/`).
- Os comandos de benchmark estavam separados do harness em `scripts/benchmark/`.
  Como os arquivos tinham responsabilidades distintas e não eram cópias, eles
  foram reunidos em `benchmarks/`; imports, comandos documentados e caminhos de
  execução foram atualizados. `tools/README.md` passa a explicar essa divisão.
- O projeto possui suítes separadas em `unit/`, `api/`, `functional/`,
  `integration/` e `smoke/`; o `pyproject.toml` exclui smoke da execução padrão.
- O CI normal executa testes, Bandit, documentação e Docker; Ruff e pip-audit
  continuam consultivos nesse fluxo. Foi definido um workflow separado e
  estrito para tags de release, cobrindo lint, dependências, testes, docs e
  container. Ver `docs/development/release-gates.md`.
- A documentação de desenvolvimento e avaliação já é extensa. A auditoria
  completa deve priorizar divergências entre documentação, configuração ativa e
  arquivos distribuídos, em vez de duplicar explicações.
- `docs/data/dataset-protocol.md` estava excluído pelo padrão amplo `data/` do
  `.gitignore`, apesar de ser referenciado pela documentação pública. A regra
  foi refinada para permitir esse documento; confirmar que ele entra no próximo
  commit/release (sem incluir artefatos de dados ignorados).
- Os dados e modelos ficam sob `data/`, que é ignorada pelo Git em grande parte.
  A disponibilidade pública de checkpoints e do dataset usado no TCC precisa
  ser confirmada por releases ou repositório de artefatos, com licença e hashes.
- O artigo `data/results/paper/main.tex` foi escolhido como referência
  metodológica canônica do TCC. O NPZ avaliado (`benchmark_dataset_15k.npz`) e
  seu test lock existem localmente; o SHA-256 confere com o artigo. A execução
  local de 2026-08-20 mantém resumo geral `running`, caminhos de saída ausentes
  e métricas diferentes das tabelas consolidadas. Os números continuam não
  reconciliados e não devem ser tratados como confirmatórios até vincular run,
  código, configuração e artefatos.
- As 36 referências gráficas encontradas nos três arquivos LaTeX do artigo
  resolvem para arquivos presentes no checkout (22 arquivos de figura). A
  proveniência precisa ser reconciliada e o PDF atual, anterior à fonte LaTeX,
  deve ser recompilado depois de regenerar tabelas e figuras.
- O NPZ avaliado no artigo tem 15.000 amostras, não é versionado e existe no
  checkout local; o corpus pai contém 40.980 amostras. Revisões upstream e
  comandos de reconstrução estão documentados, mas isso não confirma a
  redistribuição do corpus derivado. Os cards upstream declaram MIT, enquanto
  a licença CPML do XTTS restringe o uso de suas saídas. A compatibilidade e a
  base de consentimento/identificação dos locutores precisam de revisão antes
  de publicar áudio/NPZ.
- O guia do dataset ainda tinha comandos no caminho antigo; foram atualizados.
  Os comandos de clone também foram corrigidos para refletir o diretório
  `XFakeSong` criado pelo Git.
- A configuração de exemplo ativava `DEEPFAKE_PARALLEL_EXTRACTION`, embora a
  documentação declare que a extração paralela não tem leitor no código. A
  variável foi retirada do exemplo ativo para não prometer um recurso inexistente.
- Os grupos de dependências agora separam runtime, CPU, desenvolvimento/testes,
  documentação, notebooks, XAI e experimentos. As instalações limpas em Python
  3.11/3.12 são gates do workflow de release; ainda precisam passar na primeira
  execução antes de publicar.
- O inventário mostrou caminhos antigos em dois notebooks JSON; eles foram
  atualizados junto à consolidação. A validação de links, comandos e execução
  dos notebooks continua pendente.

## Plano por fases

### Fase 0 — Congelar escopo e estado de referência

- Registrar commit-base, branch, `git status` e alterações locais antes de
  qualquer limpeza ou reestruturação adicional.
- Definir a versão de Python, sistemas operacionais suportados e o recorte de
  publicação: aplicação, API, demo, código de treino ou pacote completo.
- Separar resultados oficiais do TCC de experimentos exploratórios e indicar
  qual conjunto de dados/modelos representa cada resultado.

**Aceite:** escopo e artefatos oficiais identificados; nenhuma alteração local
preexistente foi descartada.

### Fase 1 — Estrutura, imports e arquitetura

- Verificar imports cruzados conforme a regra de Clean Architecture e remover
  dependências de interface do domínio.
- Conferir que cada entrypoint funciona a partir da raiz, que todos os paths
  relativos são calculados a partir de uma raiz explícita e que Docker/WSL
  montam os arquivos nos novos locais.
- Procurar duplicação real de lógica, comandos obsoletos e scripts sem
  chamadores; não remover rotinas experimentais sem registrar o motivo e o
  destino dos resultados.
- Validar índice de documentação, navegação MkDocs, links locais, nomes de
  pastas e consistência dos nomes `benchmark`/`benchmarks`.

**Aceite:** imports públicos, CLIs documentadas e builds dos perfis de container
resolvem todos os caminhos; referências a `scripts/benchmark/` foram corrigidas
ou classificadas como históricas.

### Fase 2 — Instalação e distribuição pública

- Reconciliar `requirements*.txt`, `constraints.txt`, configurações de
  `pyproject.toml`, imagens Docker e workflows de CI; documentar quais arquivos
  são fontes de dependências e quais são constraints de compatibilidade.
- Manter os grupos em `requirements*.txt` conforme
  `docs/development/release-gates.md`; instalar só runtime na aplicação e os
  perfis opcionais apenas quando o recurso correspondente for usado.
- Fazer instalação limpa em ambiente sem cache e verificar inicialização da
  CLI, Gradio e FastAPI nos perfis suportados.
- Revisar `.env.example`, defaults, diretórios criados no primeiro uso, erros
  de configuração e comportamento sem GPU, sem internet e sem pesos baixados.
- Confirmar licenças de datasets, checkpoints, dependências, fontes e figuras;
  publicar manifesto de artefatos com origem, licença, SHA-256, tamanho e
  instrução de obtenção. Não adicionar dados privados ou pesos grandes ao Git.
- Conferir `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  tags/releases e instruções de reporte de vulnerabilidade.

**Aceite:** uma pessoa externa consegue instalar uma variante documentada,
executar uma verificação local e localizar a licença/proveniência de cada
artefato público.

### Fase 3 — Testes e qualidade

Executar em ordem crescente de custo e guardar os relatórios da CI:

1. `pytest tests/unit/` e validação de cobertura para regras críticas.
2. `pytest tests/api/ tests/functional/` para contratos e fluxos de interface.
3. `pytest tests/integration/` em ambiente controlado com dados sintéticos e
   diretórios de modelos temporários.
4. `pytest -m smoke tests/smoke/` em uma matriz definida, com dependências e
   recursos explicitados.
5. Ruff/Black/isort, análise de tipos se configurada, Bandit e auditoria de
   dependências/licenças.
6. Testes de instalação e build dos perfis Docker CPU e GPU em runners
   compatíveis; não presumir que a CI sem GPU cobre o perfil CUDA.

**Aceite:** suítes críticas passam em ambiente limpo; exceções conhecidas são
registradas com issue, impacto, responsável e critério de remoção.

### Fase 4 — Dados, protocolo e reprodutibilidade científica

- Verificar origem, permissão de uso, composição, balanceamento, splits,
  locutores, frases e ausência de vazamento do dataset de avaliação.
- Confirmar que o test lock corresponde exatamente aos arrays de teste usados
  pelos resultados publicados e que nenhum ajuste foi feito após observar o
  teste final.
- Fixar seeds, versões de bibliotecas, hardware, comandos, configurações e
  hashes de dataset, checkpoints e código de cada run apresentado.
- Reproduzir um benchmark reduzido de ponta a ponta e uma repetição do resultado
  oficial; comparar métricas e explicar variações esperadas por hardware.
- Conferir que tabelas, figuras, valores do artigo e README são derivados dos
  mesmos resultados versionados ou disponibilizados com manifesto.

**Aceite:** uma execução documentada reconstrói os artefatos citados no TCC, ou
qualquer diferença é descrita e quantificada.

**Bloqueio já identificado:** o artigo e o run local usam o mesmo hash do NPZ,
mas as métricas, os identificadores do split de teste e os caminhos de origem
não estão reconciliados. A apresentação não deve afirmar que as tabelas foram
reproduzidas até ligar cada métrica à configuração, logs, checkpoints e
predições do run correto. Uma nova campanha só deve ser iniciada após preservar
e avaliar os artefatos existentes.

### Fase 5 — Segurança, privacidade e operação

- Revisar autenticação/autorização, upload de áudio, limites de tamanho/duração,
  tipos de arquivo, nomes temporários, limpeza e tratamento de erros.
- Validar rate limiting, CORS, cabeçalhos, logs sem dados sensíveis, caminhos de
  arquivos e modo de produção da API/Gradio.
- Verificar secrets em histórico e workflows, permissões mínimas de tokens,
  imagens base, dependências vulneráveis e superfícies de rede dos containers.
- Documentar retenção e remoção dos áudios enviados e confirmar que o
  processamento local descrito corresponde ao comportamento real.

**Aceite:** nenhum segredo é exigido no checkout público; os fluxos de upload e
serving têm limites e política de dados documentados e cobertos por verificações.

### Fase 6 — Documentação e apresentação do TCC

- Revisar README para promessa do produto, escopo atual, demo, instalação curta,
  limites, links canônicos e comandos que funcionam literalmente.
- Gerar documentação MkDocs em ambiente limpo e corrigir links quebrados,
  navegação duplicada e conteúdo arquivado apresentado como instrução atual.
- Criar um roteiro de apresentação com problema, hipótese, protocolo,
  arquiteturas, controles contra vazamento, resultados, limitações e
  demonstração offline.
- Regenerar tabelas/figuras do artigo a partir do run confirmatório e compilar
  `main.tex`; guardar a versão do PDF e o manifesto dos artefatos usados na
  apresentação.
- Preparar backup da demo, vídeo/capturas sem dados pessoais, comandos de
  recuperação e slides com referências dos resultados.
- Fazer uma revisão externa de clareza: alguém que não conhece o repositório
  deve conseguir seguir o guia de instalação e identificar o que cada métrica
  significa.

**Aceite:** a documentação pública e a apresentação usam a mesma versão,
protocolo e resultados, e a demo tem um caminho de contingência sem internet.

## Riscos iniciais a confirmar

| Prioridade | Questão | Evidência para fechar |
| --- | --- | --- |
| P0 | Os pesos e o dataset citados são realmente acessíveis e redistribuíveis? | URLs públicas, licenças, hashes e teste de download limpo. |
| P0 | O protocolo do teste final está congelado e rastreável nos resultados apresentados? | Test lock, hash do NPZ, manifesto de split e metadados do run. |
| P1 | Os caminhos e comandos funcionam após centralizar os entrypoints em `benchmarks/`? | Varredura de referências, `--help`, testes do pipeline e build Docker. |
| P1 | As instruções de instalação correspondem às dependências mantidas? | Instalações limpas para cada perfil documentado. |
| P1 | Ruff e CVEs bloqueiam a publicação da release? | Sim, pelo workflow `release-gate.yml`; o CI normal mantém avisos para desenvolvimento. |
| P1 | Gradio/API aplicam os limites e controles descritos? | Testes de upload, autenticação, rate limit e configuração de produção. |
| P2 | O índice e as páginas arquivadas diferenciam claramente conteúdo atual? | Build MkDocs, checagem de links e revisão da navegação. |

## Registro de execução

| Verificação | Estado | Evidência |
| --- | --- | --- |
| Inventário inicial das pastas e referências de benchmark | Feito | `benchmarks/` e `scripts/benchmark/` comparados; não havia nomes de arquivos duplicados. |
| Consolidação dos entrypoints e atualização de referências textuais | Feito; validação funcional pendente | Scripts movidos para `benchmarks/`; READMEs e notebooks atualizados. |
| Consolidação da referência canônica e distinção dos fluxos por tier | Feito; revisão funcional pendente | Defaults, catálogo Gradio, docs, navegação MkDocs e notebooks agora apontam para o recorte do artigo e identificam tiers como fluxo genérico histórico. |
| Inventário prévio de legado e proposta de limpeza | Feito; exclusões pendentes | `docs/evaluation/canonical-study-and-legacy-inventory.md`; resultados e fontes históricas preservados. |
| Testes unitários, integração, API, smoke e segurança | Pendente | Executar conforme Fase 3. |
| Compilação sintática `app/ benchmarks/ tools/ tests/` | Feito | `python -m compileall -q app benchmarks tools tests`. |
| Configuração Compose CPU e GPU | Feito | Quatro perfis `config` passaram com credenciais descartáveis definidas só no processo. |
| Suíte unitária | Bloqueada no ambiente atual | `pytest` não instalado; Python do host é 3.14, fora da faixa documentada; instalar ambiente suportado 3.11/3.12. |
| Build MkDocs limpo + verificação de artefato | Concluído localmente em 2026-10-17 | `mkdocs build --strict --clean` (MkDocs 1.6.1, Python 3.14) e `python tools/ops/check_docs_site.py infrastructure/site` passaram; o checker confirmou páginas Markdown, destinos e fragmentos. A CI mantém a validação em Python 3.11. |
| Integridade das referências de figuras no LaTeX | Feito parcialmente | 36 referências resolvem para 22 arquivos locais; proveniência ainda precisa ser reconciliada. |
| Compilação do artigo LaTeX | Pendente | PDF local é anterior à fonte atual; recompilar após reconciliar resultados e regenerar tabelas/figuras. |
| Disponibilidade/reconstrução das fontes do dataset | Feito parcialmente | Upstream e revisões estão documentados; NPZ 15k/test lock existem localmente, não no Git. Licença/consentimento seguem pendentes. |
| Reconciliação dos resultados do artigo | Pendente crítico | Hash do dataset coincide, mas métricas e caminhos de origem divergem; ver inventário canônico do estudo. |
| Licença e autorização para redistribuir corpus derivado | Pendente crítico | Cards upstream declaram MIT; XTTS CPML limita saídas a uso não comercial. Confirmar compatibilidade e consentimento antes de hospedar artefatos. |
| Perfis de dependência e gates de release | Definidos; primeira execução pendente | `requirements*.txt`, `.github/workflows/release-gate.yml` e `docs/development/release-gates.md`. |
| Instalações limpas e builds CPU/GPU | Pendente | Executar conforme Fase 2 e Fase 3. |
| Auditoria de dados/licenças e reprodução do TCC | Pendente | Executar conforme Fase 4. |
| Auditoria de segurança de produção | Pendente | Executar conforme Fase 5. |
| Revisão final de documentação e ensaio de defesa | Pendente | Executar conforme Fase 6. |
