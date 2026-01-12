# Visão Geral do Sistema XfakeSong

## Introdução
O **XfakeSong** é um sistema avançado desenvolvido para a detecção de deepfakes de áudio. O projeto combina técnicas de processamento digital de sinais (DSP) e aprendizado de máquina (Deep Learning) para analisar, extrair características e classificar áudios como "Real" ou "Fake".

O sistema foi projetado com uma arquitetura modular baseada em princípios de **Clean Architecture**, facilitando a manutenção, escalabilidade e testes de novos algoritmos de extração de características.

## Funcionalidades Principais

### 1. Interface de Usuário (Gradio)
O sistema conta com uma interface web interativa construída com Gradio, permitindo:
- Upload de arquivos de áudio para análise.
- Visualização de resultados em tempo real.
- Gerenciamento de treinamento de modelos.
- Visualização de gráficos e métricas.

### 2. Extração de Features (Características)
O núcleo do sistema reside na sua capacidade robusta de extração de características de áudio, organizadas em:
- **Cepstral**: MFCCs, LFCCs, PLP, etc.
- **Complexity**: Dimensão Fractal, Entropia, Caos.
- **Spectral**: Centróide espectral, Roll-off, Fluxo.
- **Prosodic**: Pitch, Jitter, Shimmer.

### 3. Pipeline de Processamento
Utiliza um orquestrador de pipelines para gerenciar o fluxo de dados, desde o carregamento do áudio bruto até a inferência final, garantindo consistência e tratamento de erros.

### 4. Treinamento Seguro
Módulo dedicado ao treinamento de modelos de Deep Learning, com validação cruzada temporal e gestão de datasets (Fake vs Real).

## Tecnologias Utilizadas
- **Linguagem**: Python 3.11+
- **Interface**: Gradio
- **Processamento de Áudio**: Librosa, NumPy, SciPy
- **Machine Learning**: PyTorch / TensorFlow (a verificar na implementação final)
- **Arquitetura**: Clean Architecture, Pipeline Pattern

## Próximos Passos
Para começar a utilizar o sistema, consulte o guia de [Instalação e Configuração](./02_INSTALACAO_CONFIGURACAO.md).
