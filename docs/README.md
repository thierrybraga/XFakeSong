# Visão geral da documentação

Esta pasta reúne a documentação técnica e operacional do XFakeSong e é a
**única fonte editorial** do website. O diretório `site/` é gerado por MkDocs,
não deve ser editado nem versionado e pode ser removido a qualquer momento.

## Como navegar

- Comece pela página inicial em [index.md](index.md) para um mapa rápido do projeto.
- Use os guias por tema para instalação, arquitetura, benchmark, deploy e contribuição.
- Os notebooks ativos agora ficam em [notebooks/README.md](notebooks/README.md) e foram integrados à documentação do sistema.

## Estrutura principal

- [index.md](index.md): ponto de entrada da documentação.
- [getting-started/concepts.md](getting-started/concepts.md) a [evaluation/pipeline-audit.md](evaluation/pipeline-audit.md): guias técnicos e operacionais.
- [notebooks/](notebooks/): notebooks organizados por estudo, pipeline e modelos.
- [evaluation/retraining-adjustments.md](evaluation/retraining-adjustments.md): registro de ajustes aplicados após diagnóstico.

## Publicação do website

- Desenvolvimento local: `mkdocs serve`.
- Validação equivalente ao CI: `mkdocs build --strict`.
- Publicação: o workflow `.github/workflows/static.yml` gera `site/` a partir
  de `docs/` e publica o artefato no GitHub Pages.

Alterações devem ser feitas somente em `docs/`, `mkdocs.yml`, estilos ou scripts
da documentação. Nunca edite HTML diretamente em `site/`.

## Recomendação de leitura

1. Instalação e configuração.
2. Arquitetura e features.
3. Treinamento e benchmark.
4. Interface, deploy e contribuição.
