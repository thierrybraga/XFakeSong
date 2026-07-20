# Padroes de codigo

## Nomeacao

- Arquivos Python: `snake_case.py`.
- Classes: `PascalCase`.
- Funcoes e variaveis: `snake_case`.
- Constantes: `UPPER_SNAKE_CASE`.
- DTOs HTTP: nomes explicitos em `interfaces/web/schemas`.

## Imports

- Codigo novo deve usar `app.core.contracts`, nao `app.core.interfaces`.
- `domain` nao importa `interfaces`.
- `interfaces` pode importar `domain`, `core` e `dependencies`.
- Imports pesados opcionais devem ser lazy quando afetam startup.

## Tamanho recomendado

- Funcoes publicas: idealmente abaixo de 80 linhas.
- Classes de servico: dividir ao passar de uma responsabilidade clara.
- Arquivos Gradio: extrair callbacks grandes para presenters/helpers.

## Logging

Use:

```python
import logging

logger = logging.getLogger(__name__)
```

Evite `print()` fora de CLI, scripts interativos ou blocos `if __name__ == "__main__"`.
