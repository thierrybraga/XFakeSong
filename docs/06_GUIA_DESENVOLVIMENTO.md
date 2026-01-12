# Guia de Desenvolvimento e Contribuição

Este guia é destinado a desenvolvedores que desejam manter ou expandir o sistema XfakeSong.

## Padrões de Código

O projeto segue estritamente a **Clean Architecture**. Ao adicionar novas funcionalidades:

1. **Novas Regras de Negócio**: Adicione em `app/domain`. Não dependa de frameworks aqui.
2. **Novos Fluxos**: Adicione em `app/application`. Use o padrão Pipeline se for um processo sequencial.
3. **Novas Bibliotecas**: Se precisar de uma lib externa (ex: librosa, pandas), crie um adaptador ou wrapper em `app/core` ou `app/domain/adapters` para não acoplar o domínio diretamente à biblioteca.

## Logging e Debugging

O sistema possui um módulo de logging robusto configurado em `app/core/config/settings.py`.

- **Logs em Arquivo**: Salvos em `logs/deepfake_system.log`.
- **Nível de Log**: Configurável via `.env` (`DEEPFAKE_LOG_LEVEL=DEBUG`).

Use o logger padrão em seus módulos:
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Iniciando processamento...")
logger.error("Falha ao carregar arquivo", exc_info=True)
```

## Onde encontrar Resultados

O sistema gera artefatos durante a execução:
- **Modelos Treinados**: Salvos em `app/models/` (ou caminho configurado em `DEEPFAKE_MODELS_DIR`).
- **Gráficos e Métricas**: Salvos em `app/results/`.
- **Arquivos Temporários**: `temp/` (limpos automaticamente se configurado).

## Adicionando Dependências

Se instalar um novo pacote pip, lembre-se de atualizar o `requirements.txt`:

```bash
pip freeze > requirements.txt
```

## Testes (Recomendação)

Embora o sistema atual foque em implementação funcional, recomenda-se criar testes unitários para novos extratores em `tests/` (a ser criado), espelhando a estrutura de `app/`.
