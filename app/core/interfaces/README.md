# interfaces

Aliases de compatibilidade para contratos antigos.

## Responsabilidade

Mantem funcionando imports legados como `app.core.interfaces.base`.

## Quando usar

Nao use em codigo novo. Prefira `app.core.contracts`. Esta pasta existe para
preservar notebooks, scripts e documentacao gerada durante a migracao.

## Arquivos

- `base.py`: reexporta `app.core.contracts.base`.
- `audio.py`: reexporta `app.core.contracts.audio`.
- `services.py`: reexporta `app.core.contracts.services`.
