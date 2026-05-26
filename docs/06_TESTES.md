# Guia de Testes e Qualidade

Este guia apresenta a estratégia de testes do projeto XfakeSong, a organização da suíte e como executar, medir cobertura e manter a qualidade contínua em CI/CD.

## Estrutura da Suíte

Os testes residem em `tests/` com camadas claras para facilitar manutenção e velocidade:

```
tests/
├── unit/                 # Testes unitários (componentes isolados)
│   ├── test_core_utils.py
│   ├── test_trainer.py
│   ├── test_upload_service.py
│   ├── test_architectures.py
│   └── test_audio_processor.py
├── integration/          # Testes de integração (fluxos entre módulos)
│   ├── test_training_integration.py
│   └── test_detection_integration.py
├── api/                  # Testes de API REST
│   ├── test_datasets.py
│   ├── test_detection.py
│   ├── test_training.py
│   └── test_features.py
└── functional/           # Testes funcionais e de frontend
    ├── test_frontend_routes.py
    └── test_detection_flow.py
```

### Objetivos por camada
- Unit: validar contratos e lógica interna sem dependências externas reais.
- Integration: garantir a cooperação correta entre serviços e pipelines.
- API: assegurar contratos HTTP estáveis (status, payloads, autenticação).
- Functional: cobrir fluxos de usuário e disponibilidade de rotas/UI.

## Execução dos Testes

Com ambiente ativo e dependências instaladas:

### Todos os testes
```bash
pytest tests
```

### Verbose
```bash
pytest -v tests
```

### Categoria específica
```bash
pytest tests/unit
pytest tests/integration
pytest tests/api
pytest tests/functional
```

### Arquivo específico
```bash
pytest tests/integration/test_training_integration.py
```

## Cobertura de Código

Com `pytest-cov` instalado:
```bash
pytest --cov=app tests
```

Relatórios adicionais podem ser habilitados via parâmetros `--cov-report`.

## Observações Importantes

1. Mocks de TensorFlow: testes que dependem de carga pesada são mockados ou possuem skip condicional para manter leveza e confiabilidade.
2. Arquivos Temporários: uso de `tmp_path` para não tocar no ambiente real.
3. Autenticação: endpoints sensíveis exigem cabeçalho de API Key nos testes.
4. Estabilidade: mantenha fixtures e interfaces sempre alinhadas ao core.

## Integração Contínua (CI)

- A suíte roda automaticamente no pipeline CI.
- Ao adicionar novas funcionalidades:
  - Inclua teste unitário correspondente.
  - Se houver interação entre serviços, adicione teste de integração.
  - Atualize mocks e fixtures quando interfaces mudarem.

