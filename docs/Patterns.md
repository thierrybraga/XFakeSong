# Padroes arquiteturais

## Padroes usados

| Padrao | Onde | Uso |
| --- | --- | --- |
| Clean Architecture | `core`, `domain`, `interfaces` | dependencias apontam para dentro |
| Registry | features e arquiteturas | descoberta dinamica |
| Factory | `architectures/factory.py` | criacao por nome |
| Strategy | extratores e backends | algoritmos intercambiaveis |
| Adapter | `interfaces/*`, `features/adapters` | adaptar frameworks/implementacoes |
| Facade | servicos de dominio | simplificar orquestracao para UI/API |

## Regras SOLID

- SRP: separar UI, servicos, extratores, modelos e infraestrutura.
- OCP: novos extratores/modelos entram por registry.
- LSP: implementacoes devem respeitar os contratos em `core/contracts`.
- ISP: contratos devem permanecer pequenos e especificos.
- DIP: interfaces/adapters dependem de contratos, nao de detalhes de UI.

## Padroes de erro

- APIs HTTP retornam Problem Details.
- Dominio retorna `ProcessingResult` quando aplicavel.
- Logs usam `logging.getLogger(__name__)`.
- Excecoes de dominio/infra ficam em `app/core/exceptions.py`.
