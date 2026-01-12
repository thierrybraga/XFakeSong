# Bem-vindo ao XfakeSong

<p align="center">
  <img src="https://img.shields.io/badge/Deepfake-Detection-red?style=for-the-badge&logo=security" alt="Deepfake Detection">
  <img src="https://img.shields.io/badge/Audio-Analysis-blue?style=for-the-badge&logo=waveform" alt="Audio Analysis">
  <img src="https://img.shields.io/badge/Powered%20By-Gradio-orange?style=for-the-badge&logo=gradio" alt="Gradio">
</p>

O **XfakeSong** é uma plataforma de ponta projetada para combater a desinformação detectando áudios gerados por Inteligência Artificial (Deepfakes).

---

## 🚀 Funcionalidades Principais

<div class="grid cards" markdown>

-   :material-upload: **Upload e Análise**
    
    Carregue arquivos de áudio (.wav, .mp3) e receba uma classificação instantânea (Real vs Fake) com pontuação de confiança.

-   :material-chart-line: **Extração de Features**
    
    Visualize características profundas do áudio, incluindo espectrogramas, MFCCs, e análises de complexidade fractal.

-   :material-brain: **Treinamento de Modelos**
    
    Treine seus próprios modelos de Deep Learning diretamente pela interface, customizando hiperparâmetros e datasets.

-   :material-security: **Segurança e Privacidade**
    
    Processamento local seguro ou via container Docker, garantindo que seus dados não saiam do seu controle.

</div>

## 🔍 Como Funciona?

O sistema utiliza uma arquitetura de pipeline robusta baseada em **Clean Architecture**:

1.  **Ingestão**: O áudio é carregado e normalizado.
2.  **Processamento**: Algoritmos matemáticos extraem assinaturas digitais do som (Features).
3.  **Inferência**: Modelos de IA analisam essas assinaturas em busca de artefatos sintéticos.
4.  **Resultado**: Um veredito é apresentado com métricas visuais.

[Entenda a Arquitetura em Detalhes](03_ARQUITETURA.md){ .md-button .md-button--primary }

## 🛠️ Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/TCC.git

# Execute o script de inicialização
./start.sh  # Linux/Mac
# ou
.\start.bat # Windows
```

[Guia Completo de Instalação](02_INSTALACAO_CONFIGURACAO.md){ .md-button }
