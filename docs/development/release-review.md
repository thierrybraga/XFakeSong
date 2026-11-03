# Registro da revisão manual de release

Registro dos itens de
[revisão manual obrigatória](release-gates.md#revisao-manual-obrigatoria). Cada
revisão é datada e descreve o commit candidato daquele momento. Uma revisão não
vale para commits posteriores que mudem dados, modelos, métricas ou o artigo.

## Revisão de 2026-11-02

**Candidato:** branch `refactor/fases-1-a-4`, antes de qualquer tag.

**Veredito: a publicação acadêmica está bloqueada.** Há pendência P0 de licença
(item 1) e de proveniência dos resultados (item 2). Os dois exigem decisão humana;
nenhum foi resolvido nesta revisão.

| Item | Situação | Quem decide |
| --- | --- | --- |
| 1. Licença, consentimento e origem | **P0 aberto** | Orientação ou instituição |
| 2. Métricas do artigo x artefatos | **P0 aberto** | Autor do artigo |
| 3. Versão, changelog e instalação | Pendente | Mantenedor |
| 4. PDF e apresentação | Pendente, depende do item 2 | Autor do artigo |

### 1. Licença, consentimento e origem

- O corpus combina `falabrasil/cetuc` e `unfake/fake_voices`, cujos cards declaram
  MIT. Isso não define a licença do corpus composto: as vozes sintéticas vêm do
  XTTS-v2, sob CPML, que limita o modelo e as saídas a uso não comercial e exige que
  os termos acompanhem cópias das saídas
  ([protocolo do dataset, §8](../data/dataset-protocol.md)).
- O protocolo também registra que falta uma base declarada de consentimento para
  sintetizar e publicar vozes identificáveis.
- **Enquanto isso não for resolvido, o corpus e os modelos treinados nele não devem
  ser redistribuídos** nem descritos como MIT. A licença MIT do repositório cobre o
  código.

### 2. Métricas do artigo x artefatos

O artigo descreve uma geração de resultados diferente da que está no checkout.
`data/results/paper/tabelas_benchmark.tex` e os números do texto de `main.tex`
coincidem com a geração anterior, guardada em
`data/results/archive/pre_bateria_2026-08-20/models_benchmark_final/`, enquanto
`main.tex` cita a impressão digital de teste `66ffed25…` do run atual. Por isso uma
tabela do artigo não pode ser apresentada como reprodução do run atual.

Em 2026-11-03, a cópia que ficava em `data/models/benchmark/` foi para uma quarentena
local, fora do repositório e com manifesto. Os 33 arquivos de artefatos dela têm o
mesmo SHA-256 na pasta acima; só o `README.md` não tem cópia lá.

Medidas no conjunto de teste limpo, em porcentagem:

| Modelo | Acurácia atual | EER atual | Acurácia anterior | EER anterior |
| --- | ---: | ---: | ---: | ---: |
| AASIST | 94,50 | 3,18 | 94,72 | 2,60 |
| Conformer | 98,70 | 1,45 | 98,91 | 0,29 |
| HuBERT Original | 93,42 | 6,80 | 93,92 | 5,93 |
| Hybrid CNN-Transformer | 98,12 | 1,16 | 99,57 | 0,43 |
| MultiscaleCNN | 95,80 | 4,05 | 97,76 | 2,17 |
| RandomForest | 86,11 | 8,25 | 91,53 | 6,95 |
| RawGAT-ST | 84,44 | 3,91 | 87,55 | 11,87 |
| RawNet2 | 91,17 | 8,10 | 95,88 | 3,18 |
| SVM | 91,75 | 7,09 | 93,20 | 6,95 |
| SpectrogramTransformer | 98,55 | 0,29 | 99,71 | 0,14 |
| WavLM Original | 96,16 | 3,62 | 96,09 | 3,62 |

"Atual" vem de `data/results/benchmark/run_summary.json`; "anterior", dos
`artifact_manifest.json` em
`data/results/archive/pre_bateria_2026-08-20/models_benchmark_final/`. A geração
atual é a que tem correção de banda, e a queda de desempenho em relação à anterior é
coerente com a remoção do atalho de reamostragem descrito em `CLAUDE.md`, sem que esta
revisão tenha isolado essa causa modelo a modelo.

Dois avisos para a reconciliação:

- O run atual guarda o RawNet2 válido em `_superseded/rawnet2_d9f23c0d`. O retreino
  abortado na época 5 saiu de `data/results/benchmark/rawnet2/` em 2026-11-03 e está
  na quarentena local. Os números da tabela acima são do run válido.
- Promover a geração atual para o app não se resume a restaurar
  `data/results/archive/pre_bateria_2026-08-20/models_benchmark_final/`, que é a
  geração anterior. Hoje WavLM e HuBERT da geração atual não chegam à inferência.

### 3. Versão, changelog e instalação

- O projeto não tinha `CHANGELOG.md`: o arquivo foi criado nesta revisão.
- Não há tags, e `pyproject.toml` não declara versão. `app/__init__.py` declara
  `1.1.0`. A escolha do número e a criação da tag ficam para o mantenedor.
- As instruções de instalação foram atualizadas para Gradio 6.17 e para o uso de
  `constraints.txt`.

### 4. PDF e apresentação

`data/results/paper/main.pdf` é de 2026-08-27 e `main.tex` é de 2026-10-18, então o
PDF não reflete a fonte. Ele precisa ser gerado de novo depois do item 2.

### Gates automáticos nesta revisão

Executados localmente, em Windows com Python 3.11, salvo indicação em contrário. Os
detalhes estão no [registro de execução do plano](refactoring-action-plan.md#registro-de-execucao).

| Gate | Resultado |
| --- | --- |
| Ruff, Bandit `-lll` e MkDocs estrito | Passam |
| Suíte rápida | 1.049 aprovados, 0 falhas, 10 pulados. Os 10 pulados também passam com `nbformat`, `nbclient` e `shap` instalados |
| Smoke | 10 aprovados |
| Instalação em 3.11 e 3.12 | **Só resolvida para Linux, sem instalar.** O Python 3.12 não está nesta máquina. A instalação real é validada na CI |
| `tools/ops/check_pip_audit.py` | Aprova em Windows e na lista de pacotes de Linux. Tolera 84 IDs em 3 pacotes, registrados e justificados em `configs/security/pip-audit-exceptions.json` |
| Validação dos perfis Compose | Os 5 perfis são válidos, com o env fictício do workflow |
| Build Docker | **Não executado.** Roda na primeira execução do workflow no GitHub |
| Workflow `release-gate.yml` | **Ainda não executado no GitHub.** Nada foi enviado ao repositório remoto |
