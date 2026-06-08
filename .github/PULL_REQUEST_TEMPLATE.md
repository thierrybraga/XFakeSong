# Descrição

<!-- O que este PR muda e por quê. Linke issues relacionadas (#123). -->

## Tipo de mudança

- [ ] Correção de bug
- [ ] Nova funcionalidade
- [ ] Refatoração / manutenção
- [ ] Documentação
- [ ] Segurança

## Checklist

- [ ] `./scripts/run_tests.sh fast` verde localmente
- [ ] Cobertura mantida ou aumentada para o código novo
- [ ] `bandit -r app benchmarks scripts -lll` sem achados HIGH
- [ ] `mkdocs build --strict` ok quando há mudança em `docs/`
- [ ] Sem segredos, tokens ou credenciais no diff

## Notas de segurança

<!-- Impacto em autenticação, uploads, extração de arquivos, deps,
     execução de comandos, etc. Escreva "nenhum" se não se aplica. -->
