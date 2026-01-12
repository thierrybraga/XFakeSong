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

# XfakeSong — Deepfake Audio Detection System

> Sistema avançado para upload, extração de features, treinamento e inferência de detecção de deepfake de áudio com interface Gradio.

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-development-orange)

## 📚 Documentação Completa

A documentação detalhada do projeto foi organizada em módulos para facilitar o entendimento:

- [**01 - Introdução e Visão Geral**](docs/01_INTRODUCAO.md): Entenda o propósito e as capacidades do sistema.
- [**02 - Instalação e Configuração**](docs/02_INSTALACAO_CONFIGURACAO.md): Guia passo a passo para configurar o ambiente e variáveis `.env`.
- [**03 - Arquitetura do Sistema**](docs/03_ARQUITETURA.md): Detalhes sobre a Clean Architecture e o padrão Pipeline utilizado.
- [**04 - Funcionalidades Core**](docs/04_FUNCIONALIDADES_CORE.md): Explicação profunda sobre os algoritmos de extração de features (Cepstral, Complexity, etc).
- [**05 - Estrutura do Projeto**](docs/05_ESTRUTURA_PROJETO.md): Mapa completo de arquivos e pastas.
- [**06 - Guia de Desenvolvimento**](docs/06_GUIA_DESENVOLVIMENTO.md): Padrões de código e dicas para contribuidores.
- [**07 - API Reference**](docs/07_API_REFERENCE.md): Documentação completa dos endpoints da API REST.

---

## 🚀 Quick Start

### Pré-requisitos
- Python 3.11+
- Pip atualizado

### Instalação Rápida

```bash
# 1. Clone e entre no diretório
git clone <URL_REPO>
cd TCC

# 2. Crie e ative o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
# Windows: copy .env.example .env
# Linux: cp .env.example .env
```

### Inicializando o Sistema

Você pode iniciar o sistema via script (menu Opção 1) ou manualmente:

Antes da primeira execução, crie a estrutura de pastas:
```bash
python main.py --bootstrap-dirs
```

Inicie a interface web (Gradio):
```bash
python main.py --gradio --gradio-port 7860
```
Acesse: `http://127.0.0.1:7860/`

---

## 🛠️ Dicas de Uso

### Interface Web (Gradio)
1. **Aba "Análise Única"**: Faça upload de um áudio e verifique se é Real ou Fake com o modelo carregado.
2. **Aba "Treino/Modelos"**: Configure hiperparâmetros e inicie o treinamento de novos modelos usando os datasets em `datasets/`.
3. **Aba "Resultados & Gráficos"**: Visualize métricas de performance e histórico de execuções.

### Diretórios Importantes
- `app/models/`: Onde os modelos treinados (.pth, .h5) são salvos.
- `app/results/`: Onde gráficos e métricas JSON são armazenados.
- `logs/`: Logs de execução para debugging.

---

## 🔧 Solução de Problemas

| Problema | Solução |
|----------|---------|
| `net::ERR_ABORTED` | Evite cliques múltiplos rápidos na UI local. Em modo `--gradio-share`, aguarde a fila. |
| Erro de Importação | Execute sempre da raiz (`TCC/`) usando `python main.py ...`. |
| Porta Ocupada | Use `--gradio-port 7861` (ou outra porta livre). |

Para validação rápida de sintaxe em todo o projeto:
```bash
python -m compileall -q app main.py
```

---

*Para mais detalhes técnicos, consulte a [Documentação de Arquitetura](docs/03_ARQUITETURA.md).*
