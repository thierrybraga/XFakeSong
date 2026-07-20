# Padrão da documentação

Esta pasta é simultaneamente a fonte do site MkDocs e um vault do Obsidian.

## Estrutura

- `getting-started/`: introdução, instalação e primeiros passos.
- `architecture/`: arquitetura, módulos, padrões e features.
- `models/`: arquiteturas, treinamento e inferência.
- `evaluation/`: benchmark, protocolos, estudos e retreinos ativos.
- `interfaces/`: Gradio e API REST.
- `development/`: contribuição, qualidade, CI/CD e ambientes.
- `data/`: datasets, protocolos e auditorias.
- `deployment/`: publicação e operação.
- `reference/`: glossário e FAQ.
- `archive/`: planos concluídos e relatórios históricos.
- `notebooks/`: notebooks executáveis e seu índice próprio.

## Convenções

1. Use nomes ASCII em `kebab-case.md`.
2. Mantenha um único título H1 no início de cada página.
3. Use links Markdown relativos com extensão `.md`.
4. Não use caminhos absolutos locais.
5. Guarde imagens reutilizáveis em `assets/attachments/`.
6. Não edite runs em `data/results/` como documentação. A única exceção versionada
   é `data/results/paper/`, que contém as fontes e figuras finais do artigo.
7. Registre páginas novas no `nav` de `mkdocs.yml`.

Links Markdown relativos são escolhidos porque funcionam no GitHub, MkDocs e
Obsidian. Wikilinks podem ser usados localmente, mas não devem ser salvos nos
documentos versionados.
