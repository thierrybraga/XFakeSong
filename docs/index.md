---
hide:
  - navigation
  - toc
---

# XfakeSong: Detecção de Deepfake de Áudio

<div class="md-banner" style="padding: 4rem 2rem; text-align: center; margin: -2rem -2rem 2rem -2rem; border-radius: 0 0 2rem 2rem;">
  <h1 style="color: white; margin-bottom: 1rem; font-weight: 900; font-size: 3rem;">XfakeSong</h1>
  <p style="color: rgba(255,255,255,0.9); font-size: 1.4rem; max-width: 800px; margin: 0 auto 2rem;">
    Plataforma Enterprise Open Source para detecção forense de áudio sintético e segurança digital.
  </p>
  <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
    <a href="02_INSTALACAO_CONFIGURACAO/" class="md-button md-button--primary" style="background-color: white; color: var(--md-primary-fg-color); border: none; font-size: 1.1rem; padding: 0.8rem 2rem;">
      Começar Agora
    </a>
    <a href="03_ARQUITETURA/" class="md-button" style="border: 1px solid rgba(255,255,255,0.5); color: white; font-size: 1.1rem; padding: 0.8rem 2rem;">
      Ver Arquitetura
    </a>
  </div>
  <div style="margin-top: 3rem; display: flex; gap: 1rem; justify-content: center;">
    <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
    <img src="https://img.shields.io/badge/Framework-Gradio-orange?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio">
  </div>
</div>

## Por que XfakeSong?

O **XfakeSong** não é apenas mais um classificador. É um ecossistema completo de forense digital projetado para pesquisadores, analistas de segurança e desenvolvedores que precisam de transparência e robustez.

<div class="grid cards" markdown>

-   :material-fingerprint: **Precisão Forense**

    ---

    Utiliza uma combinação de **features espectrais** (MFCC, CQT, Chroma) e embeddings profundos (WavLM, HuBERT) para detectar até as manipulações mais sutis.

-   :material-eye-outline: **Explainable AI (XAI)**

    ---

    Não confie apenas no resultado. Visualize **mapas de calor**, espectrogramas e análises fractais que justificam cada decisão do modelo.

-   :material-shield-lock: **Privacidade em Primeiro Lugar**

    ---

    Projetado para rodar 100% **offline** ou em containers isolados. Seus dados sensíveis nunca deixam sua infraestrutura.

-   :material-cube-scan: **Modularidade Extrema**

    ---

    Arquitetura plug-and-play. Adicione novos modelos, extratores de features ou pipelines de pré-processamento sem reescrever o núcleo.

</div>

## Funcionalidades em Destaque

=== "🔍 Inferência Multimodal"

    Carregue arquivos de áudio e receba análises detalhadas de múltiplos modelos simultaneamente (Ensemble Learning).

    ```python
    # Exemplo de uso da API
    from app.services import AudioProcessor

    processor = AudioProcessor()
    result = processor.analyze("suspect_audio.wav")
    print(f"Probabilidade de Fake: {result.fake_score:.2%}")
    ```

=== "📊 Dashboard Interativo"

    Interface gráfica poderosa construída com **Gradio**, permitindo:
    
    *   Upload drag-and-drop
    *   Visualização de formas de onda em tempo real
    *   Ajuste fino de parâmetros de detecção
    *   Exportação de relatórios em PDF/JSON

=== "🐳 Deploy Simplificado"

    Leve para produção em minutos com suporte nativo a Docker e Hugging Face Spaces.

    ```bash
    docker-compose up -d --build
    ```

## Comece sua Jornada

<div class="grid cards" markdown>

-   [**Guia de Instalação**](02_INSTALACAO_CONFIGURACAO.md)
    {: .card-link }
    Configure seu ambiente local ou Docker em minutos.

-   [**Arquitetura do Sistema**](03_ARQUITETURA.md)
    {: .card-link }
    Entenda os princípios de Clean Architecture que guiam o projeto.

-   [**Testes e Qualidade**](07_TESTES.md)
    {: .card-link }
    Rode a suíte de testes e acompanhe cobertura de código.

-   [**Treinando Modelos**](11_TREINAMENTO_ARQUITETURAS.md)
    {: .card-link }
    Aprenda a criar e refinar seus próprios detectores de deepfake.

-   [**Contribua**](06_GUIA_DESENVOLVIMENTO.md)
    {: .card-link }
    Junte-se à nossa comunidade e ajude a combater a desinformação.

</div>

<hr>

<p align="center" style="font-size: 0.9rem; color: var(--md-default-fg-color--light);">
  Desenvolvido com ❤️ pela equipe XfakeSong. Licenciado sob MIT.
</p>
