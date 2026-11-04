# Dependências e gates de release

Este documento define os perfis de instalação e as verificações obrigatórias
antes de publicar uma versão. Os arquivos de requirements descrevem grupos
funcionais; os Dockerfiles por família continuam sendo a fonte dos ambientes
de treino especializados.

## Perfis de dependências

| Perfil | Arquivo | Conteúdo e uso |
| --- | --- | --- |
| Runtime padrão | `requirements.txt` | Aplicação completa com TensorFlow, PyTorch e Transformers. |
| Runtime CPU | `requirements-cpu.txt` | Base compartilhada e TensorFlow CPU, sem as bibliotecas CUDA do TensorFlow. |
| Desenvolvimento e testes | `requirements-dev.txt` | Runtime padrão, pytest, Ruff, Black/isort, Bandit e pip-audit. |
| Documentação | `requirements-docs.txt` | MkDocs Material, sem instalar a stack de ML. |
| Notebooks | `requirements-notebooks.txt` | Desenvolvimento/testes mais Jupyter, IPython e nbconvert. |
| XAI | `requirements-xai.txt` | SHAP; instalar junto ao runtime. |
| Experimentos | `requirements-experiments.txt` | ONNX, Optuna, MLflow e quantização; instalar junto ao runtime. |

Instalações típicas:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-docs.txt
python -m pip install -r requirements-notebooks.txt
python -m pip install -r requirements.txt -r requirements-xai.txt
python -m pip install -r requirements.txt -r requirements-experiments.txt
```

`requirements-dev.txt` já inclui o runtime, portanto é suficiente para
contribuição e execução dos testes. Os perfis opcionais são suplementares e
não devem ser adicionados ao runtime padrão sem necessidade funcional.

Os requirements declaram faixas. Para reproduzir o ambiente das imagens e do gate
de release, acrescente `-c constraints.txt` a qualquer um dos comandos acima.

## Pilha web e seus tetos

A pilha web subiu de Gradio 4.44 para 6.17, mantendo `torch`, `transformers`,
TensorFlow, Keras, NumPy e scikit-learn nas versões anteriores. Os tetos formam uma
cadeia: mudar um deles exige reavaliar os outros.

| Pino | Motivo |
| --- | --- |
| `torch==2.5.1`, `transformers==4.45.2` | Deliberados: versões mais novas causaram segfault no container GPU, e a reprodução do SSL depende delas. |
| `huggingface_hub<1.0` | Exigido pelo `transformers` fixado. |
| `gradio>=6.17.3,<6.18` | O Gradio 6.18 pede `huggingface_hub>=1.2`. |
| `pillow>=12.3,<13` | O Gradio 6.17 exige `pillow<13`. Os avisos que restam no Pillow só têm correção na 13. |

O conjunto foi testado com a suíte rápida, o smoke, os fluxos da interface no
navegador e a comparação de predições dos modelos existentes
([registro no plano](refactoring-action-plan.md#registro-de-execucao)).

## Gates automáticos

`.github/workflows/release-gate.yml` roda ao enviar uma tag `v*` ou por
execução manual. Rode-o no commit candidato antes de criar a tag; o push da
tag repete a verificação. **Só crie a release no GitHub depois que todos os
gates passarem.** O workflow não publica artefatos por conta própria.

| Gate | Critério de aprovação |
| --- | --- |
| Instalação e compatibilidade | `requirements.txt` e `requirements-cpu.txt` instalam com `constraints.txt` e passam `pip check` em Python 3.11 e 3.12. |
| Lint | Ruff termina sem achados em `app/`, `tests/`, `benchmarks/` e `tools/`. |
| Testes | Suíte rápida termina sem falhas e o relatório de cobertura é gerado. Ainda não há piso percentual global configurado; defini-lo exige medir a baseline antes de torná-lo bloqueante. Smoke com TensorFlow/GPU permanece opt-in por não haver runner GPU garantido. |
| SAST | Bandit não encontra achados HIGH. |
| Dependências | `tools/ops/check_pip_audit.py` audita todos os perfis instalados e reprova qualquer aviso fora das [exceções registradas](#excecoes-do-pip-audit). |
| Documentação | `mkdocs build --strict --clean` e `python tools/ops/check_docs_site.py infrastructure/site` passam; o segundo comando valida geração de páginas Markdown, destinos locais e fragmentos HTML. |
| Container | Build CPU de publicação e validação de todos os perfis Compose passam. Isso não certifica execução GPU/CUDA. |

O workflow normal de CI continua rápido: Ruff e auditoria de dependências
podem publicar avisos enquanto o desenvolvimento corrige a base. A checagem
estrita no workflow de release é o gate final e não deve ser substituída por
um resultado verde do CI normal.

## Exceções do pip-audit

Um aviso só é tolerado se estiver em `configs/security/pip-audit-exceptions.json`,
com o pacote, a versão exata, os IDs, a justificativa, o motivo de não atualizar
(`blocked_by`) e a data de revisão (`review_by`). O gate reprova quando:

- surge um aviso que a lista não cobre, mesmo em um pacote que já tem exceção;
- a data de revisão passou;
- a versão instalada difere da registrada;
- uma exceção aponta para um pacote ausente do ambiente;
- um pacote não pôde ser auditado, ou o `pip-audit` não terminou.

Exceções em vigor, com revisão até 2027-02-01:

| Pacote | IDs | Por que não é explorável aqui | Risco residual aceito |
| --- | ---: | --- | --- |
| `pillow` 12.3.0 | 39 | O projeto não abre imagem, fonte ou PDF de terceiros, e os uploads só aceitam áudio. | Nenhum caminho conhecido. |
| `torch` 2.5.1 | 19 | Os artefatos carregados são locais, e nenhuma rota recebe arquivo de modelo. O `weights_only=True` **não** protege nesta versão. | Quem grava em `data/models`, publica no repositório de sincronização ou tem a chave da API pode executar código. |
| `transformers` 4.45.2 | 26 | O Gradio e a detecção só usam checkpoints fixos ou diretórios locais. Não há `trust_remote_code`, `Trainer` nem tokenizadores. | A rota de treino da API aceita `parameters` livres que escolhem o repositório, e os checkpoints não fixam `revision`. |

Trate a chave da API como acesso administrativo enquanto o risco residual da rota de
treino estiver aberto.

Ao reprovar por um aviso novo, leia o aviso, confira se ele é alcançável a partir do
código e só então atualize o pino ou registre o ID com a justificativa. Trocar a data
de revisão sem reavaliar anula o propósito do gate.

## Revisão manual obrigatória

As revisões feitas ficam no [registro da revisão manual](release-review.md).

Os gates automatizados não validam autorização para publicar dados nem a
validade científica do artigo. Antes de criar a release, revisar e registrar:

1. licença, consentimento, origem, hashes e instruções de obtenção do dataset,
   modelos, checkpoints, figuras e fontes de terceiros;
2. correspondência entre métricas do artigo, configuração, código, split,
   predições e artefatos do run confirmatório;
3. versão/tag, changelog, instruções de instalação e limitações conhecidas;
4. PDF do artigo e materiais de apresentação gerados a partir dos artefatos
   reconciliados.

Qualquer pendência P0 de licença/dados ou de proveniência dos resultados bloqueia
a publicação acadêmica e a apresentação dos números como reproduzidos. A CI não
tem GPU; aprovação dos perfis NVIDIA significa apenas que a configuração Compose
é válida.
