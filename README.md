---
title: XfakeSong
emoji: 🛡️
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
license: mit
---

# XfakeSong

XfakeSong e uma plataforma open source para deteccao de deepfakes de audio.
Ela combina features espectrais, modelos neurais modernos e processamento local
para apoiar pesquisa, auditoria e experimentacao forense.

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active_development-orange?style=for-the-badge)
[![CI](https://github.com/thierrybraga/XFakeSong/actions/workflows/ci.yml/badge.svg)](https://github.com/thierrybraga/XFakeSong/actions/workflows/ci.yml)

## Inicio Rapido

```bash
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --bootstrap-dirs
python main.py --gradio
```

A interface Gradio fica disponivel em `http://localhost:7860`.

No Windows, o fluxo interativo tambem pode ser iniciado com:

```bash
start.bat
```

## Documentacao

A documentacao completa esta em `docs/` e e publicada via MkDocs:
[thierrybraga.github.io/XFakeSong](https://thierrybraga.github.io/XFakeSong/).

| Objetivo | Documento |
| --- | --- |
| Entender o projeto | [Introducao](docs/01_INTRODUCAO.md) |
| Instalar e configurar | [Instalacao e Configuracao](docs/02_INSTALACAO_CONFIGURACAO.md) |
| Entender a arquitetura | [Arquitetura](docs/03_ARQUITETURA.md) |
| Adicionar features | [Features de Audio](docs/04_FEATURES.md) |
| Contribuir com codigo | [Guia do Desenvolvedor](docs/05_GUIA_DEV.md) |
| Rodar testes | [Testes e Qualidade](docs/06_TESTES.md) |
| Usar a API | [API Reference](docs/07_API_REFERENCE.md) |
| Revisar modelos | [Arquiteturas Neurais](docs/08_ARQUITETURAS.md) |
| Rodar inferencia | [Inferencia](docs/09_INFERENCIA.md) |
| Treinar modelos | [Treinamento](docs/10_TREINAMENTO.md) |
| Publicar no Hugging Face | [Deploy Hugging Face](docs/11_DEPLOY_HUGGINGFACE.md) |
| Preparar datasets | [Datasets Publicos](docs/12_DATASETS.md) |
| Executar no Colab | [Guia Google Colab](docs/13_COLAB_GUIDE.md) |
| Validar aderencia tecnica | [Revisao das Arquiteturas](docs/14_REVISAO_ARQUITETURAS.md) |
| Medir resultados | [Benchmark e TCC](docs/15_BENCHMARK.md) |

## Comandos Essenciais

```bash
python main.py --gradio
python main.py --bootstrap-dirs
pytest tests/
pytest --cov=app tests/
docker compose up --build -d
docker compose logs -f
docker compose down
```

## Contribuicao e Seguranca

Leia [CONTRIBUTING.md](CONTRIBUTING.md) antes de abrir pull requests e
[SECURITY.md](SECURITY.md) para reportar vulnerabilidades. O projeto segue a
licenca MIT; consulte [LICENSE](LICENSE).
