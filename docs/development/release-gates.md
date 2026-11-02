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

## Gates automáticos

`.github/workflows/release-gate.yml` roda ao enviar uma tag `v*` ou por
execução manual. Rode-o no commit candidato antes de criar a tag; o push da
tag repete a verificação. **Só crie a release no GitHub depois que todos os
gates passarem.** O workflow não publica artefatos por conta própria.

| Gate | Critério de aprovação |
| --- | --- |
| Instalação e compatibilidade | `requirements.txt` e `requirements-cpu.txt` instalam e passam `pip check` em Python 3.11 e 3.12. |
| Lint | Ruff termina sem achados em `app/`, `tests/`, `benchmarks/` e `tools/`. |
| Testes | Suíte rápida termina sem falhas e o relatório de cobertura é gerado. Ainda não há piso percentual global configurado; defini-lo exige medir a baseline antes de torná-lo bloqueante. Smoke com TensorFlow/GPU permanece opt-in por não haver runner GPU garantido. |
| SAST | Bandit não encontra achados HIGH. |
| Dependências | `pip-audit` não encontra vulnerabilidades nos perfis instalados. Uma exceção só vale com ID, justificativa e prazo registrados, além de uma supressão explícita no workflow; sem isso, continua bloqueante. |
| Documentação | `mkdocs build --strict --clean` e `python tools/ops/check_docs_site.py infrastructure/site` passam; o segundo comando valida geração de páginas Markdown, destinos locais e fragmentos HTML. |
| Container | Build CPU de publicação e validação de todos os perfis Compose passam. Isso não certifica execução GPU/CUDA. |

O workflow normal de CI continua rápido: Ruff e auditoria de dependências
podem publicar avisos enquanto o desenvolvimento corrige a base. A checagem
estrita no workflow de release é o gate final e não deve ser substituída por
um resultado verde do CI normal.

## Revisão manual obrigatória

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
