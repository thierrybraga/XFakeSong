# XFakeSong — Guia para Claude Code

Este projeto usa [AGENTS.md](AGENTS.md) como guia canonico para agentes de
codigo. Claude Code deve seguir as mesmas regras de arquitetura, comandos,
padroes de teste e convencoes documentadas nesse arquivo.

## Atalhos Operacionais

```bash
python main.py --gradio
python main.py --bootstrap-dirs
pytest tests/
docker compose up --build -d
```

Para detalhes tecnicos, consulte:

| Tema | Fonte canonica |
| --- | --- |
| Arquitetura e limites de camadas | [AGENTS.md](AGENTS.md) e [docs/03_ARQUITETURA.md](docs/03_ARQUITETURA.md) |
| Desenvolvimento | [docs/05_GUIA_DEV.md](docs/05_GUIA_DEV.md) |
| Testes | [docs/06_TESTES.md](docs/06_TESTES.md) |
| Modelos e treinamento | [docs/08_ARQUITETURAS.md](docs/08_ARQUITETURAS.md) e [docs/10_TREINAMENTO.md](docs/10_TREINAMENTO.md) |
