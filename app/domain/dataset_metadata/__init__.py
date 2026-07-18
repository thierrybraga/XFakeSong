"""Metadados de dataset do domínio XFakeSong.

Consolida conhecimento específico do domínio de construção de datasets de
áudio (catálogo de fontes/licenças e manifesto de falantes) — movido de
`app/core/` (infraestrutura genérica) para `app/domain/` (regras de negócio
específicas do projeto), seguindo a Clean Architecture do projeto.

- :mod:`app.domain.dataset_metadata.dataset_catalog` — catálogo de datasets
  (nome, fonte, licença, prefixos de arquivo).
- :mod:`app.domain.dataset_metadata.speaker_manifest` — manifesto sidecar
  arquivo→falante, usado para splits disjuntos por falante.
"""
