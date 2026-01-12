---
title: XfakeSong
emoji: 🛡️
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# 🛡️ XfakeSong — Enterprise Deepfake Audio Detection

> **Plataforma Open Source de Inteligência Artificial para Detecção de Áudio Sintético e Forense Digital.**

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active_development-orange?style=for-the-badge)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)
[![CI](https://github.com/thierrybraga/XFakeSong/actions/workflows/ci.yml/badge.svg)](https://github.com/thierrybraga/XFakeSong/actions/workflows/ci.yml)

## 🎯 Visão Geral

O **XfakeSong** é uma solução robusta e modular projetada para combater a crescente ameaça de deepfakes de áudio. Utilizando arquiteturas de ponta (WavLM, HuBERT) e princípios de **Explainable AI (XAI)**, o sistema oferece ferramentas precisas para pesquisadores, analistas de segurança e desenvolvedores.

### 🌟 Diferenciais
*   **Arquitetura Híbrida**: Combina análise espectral clássica com Transformers modernos.
*   **Interface Unificada**: GUI baseada em Gradio para inferência, treinamento e análise.
*   **Pipeline Modular**: Facilidade para plugar novos modelos e extratores de features.
*   **Foco em Privacidade**: Processamento local ou containerizado, sem dependência de APIs externas inseguras.
*   **Open Source Ready**: Documentação extensiva e configurações de CI/CD prontas para uso.

---

## 📚 Documentação Oficial

A documentação completa para desenvolvedores e usuários está disponível em nossa [Wiki / GitHub Pages](https://thierrybraga.github.io/XFakeSong/).

### 🔹 Guias Iniciais
- [**01 - Introdução e Visão Geral**](docs/01_INTRODUCAO.md): Propósito e capacidades.
- [**02 - Instalação e Configuração**](docs/02_INSTALACAO_CONFIGURACAO.md): Setup do ambiente.
- [**06 - Guia de Desenvolvimento**](docs/06_GUIA_DESENVOLVIMENTO.md): Padrões para contribuição.
- [**07 - Testes e Qualidade**](docs/07_TESTES.md): Estrutura, execução e cobertura.

### 🔹 Guias Técnicos Avançados
- [**03 - Arquitetura do Sistema**](docs/03_ARQUITETURA.md): Clean Architecture e Design Patterns.
- [**04 - Funcionalidades Core**](docs/04_FUNCIONALIDADES_CORE.md): Algoritmos e Features.
- [**08 - Arquiteturas Detalhadas**](docs/08_ARQUITETURAS_DETALHADAS.md): Visão profunda dos modelos.
- [**10 - Inferência de Arquiteturas**](docs/10_INFERENCIA_ARQUITETURAS.md): Fluxos de dados.
- [**11 - Treinamento de Arquiteturas**](docs/11_TREINAMENTO_ARQUITURAS.md): Otimização e treinamento.
- [**13 - Datasets Públicos de Áudio**](docs/13_DATASETS_PUBLICOS.md): Links, licenças e downloads.

### 🔹 Deploy e Operação
- [**14 - Guia Google Colab**](docs/14_GOOGLE_COLAB_GUIDE.md): Execução em nuvem com GPUs.
- [**12 - Deploy Hugging Face**](docs/12_DEPLOY_HUGGINGFACE.md): Guia de deploy em nuvem.
- [**07 - API Reference**](docs/07_API_REFERENCE.md): Endpoints REST.

---

## 🚀 Quick Start (Comece Agora)

### Pré-requisitos
- Python 3.11+
- Docker (Opcional, recomendado para produção)

### Instalação Local

```bash
# 1. Clone o repositório
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong

# 2. Setup Automático (Windows)
start.bat
# Escolha a opção [1] para Teste Local ou [3] para Instalar Dependências

# 3. Setup Manual (Linux/Mac)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --bootstrap-dirs
python main.py --gradio
```

Acesse a interface em: `http://localhost:7860`

---

## 🤝 Como Contribuir

Este é um projeto **Open Source** e adoramos receber contribuições da comunidade!

1.  Leia nosso [Guia de Contribuição](CONTRIBUTING.md).
2.  Consulte o [Código de Conduta](CODE_OF_CONDUCT.md).
3.  Veja as [Issues Abertas](https://github.com/thierrybraga/XFakeSong/issues) para encontrar onde ajudar.

### Roadmap 🗺️
- [x] Detecção Baseada em Features Espectrais
- [x] Integração com WavLM e HuBERT
- [x] Documentação de Treinamento e Inferência
- [x] CI/CD Pipeline (Linting & Tests)
- [ ] API REST Completa com FastAPI (v2)
- [ ] Suporte a Multi-GPU para Treinamento Distribuído
- [ ] Dashboard de Monitoramento em Tempo Real

---

## 🛡️ Segurança

Levamos a segurança a sério. Consulte nossa [Política de Segurança](SECURITY.md) para saber como reportar vulnerabilidades de forma responsável.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
