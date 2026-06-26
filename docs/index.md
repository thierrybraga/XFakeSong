# Documentação do XFakeSong

O **XFakeSong** é uma plataforma open source para detecção de deepfakes de
áudio com execução local: interface Gradio, API FastAPI e pipelines modulares
de extração de features, treinamento, inferência e benchmark reprodutível para
o TCC.

Esta página é o mapa da documentação. As páginas detalhadas abaixo são as
fontes canônicas de cada assunto.

## Leitura por objetivo

| Se você quer… | Leia |
| --- | --- |
| Entender a área (anti-spoofing, ameaças, métricas) | [Conceitos e Fundamentos](00_CONCEITOS.md) |
| Entender o escopo do projeto | [Introdução](01_INTRODUCAO.md) |
| Instalar e executar a aplicação | [Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md) |
| Navegar pela Clean Architecture | [Arquitetura](03_ARQUITETURA.md) |
| Trabalhar com extração de features | [Features de Áudio](04_FEATURES.md) |
| Contribuir com código | [Guia do Desenvolvedor](05_GUIA_DEV.md) |
| Validar qualidade e testes | [Testes e Qualidade](06_TESTES.md) |
| Integrar via HTTP | [API Reference](07_API_REFERENCE.md) |
| Comparar as arquiteturas neurais | [Arquiteturas Neurais](08_ARQUITETURAS.md) |
| Rodar predição com modelos treinados | [Inferência](09_INFERENCIA.md) |
| Treinar modelos | [Treinamento](10_TREINAMENTO.md) |
| Usar a interface Gradio e suas abas | [Interface Gradio](23_INTERFACE_GRADIO.md) |
| Publicar no Hugging Face Spaces | [Deploy Hugging Face](11_DEPLOY_HUGGINGFACE.md) |
| Publicar documentação e demo | [GitHub Pages e Hugging Face](24_PUBLICACAO_GITHUB_HF.md) |
| Preparar datasets | [Datasets Públicos](12_DATASETS.md) |
| Executar no Google Colab | [Guia Google Colab](13_COLAB_GUIDE.md) |
| Auditar a aderência das arquiteturas | [Revisão das Arquiteturas](14_REVISAO_ARQUITETURAS.md) |
| Rodar o benchmark consolidado | [Benchmark e Resultados](15_BENCHMARK.md) |
| Ler a fundamentação e análise experimental no GitHub Pages | [Estudo Experimental](20_ESTUDO_EXPERIMENTAL.md) |
| Estudar com os notebooks | [Guia de Notebooks](16_NOTEBOOKS.md) |
| Entender CI/CD e segurança | [CI/CD e Segurança](17_CICD_SEGURANCA.md) |
| Consultar termos técnicos | [Glossário](18_GLOSSARIO.md) |
| Tirar dúvidas rápidas | [Perguntas Frequentes (FAQ)](19_FAQ.md) |

## Artefatos consolidados

| Item | Local |
| --- | --- |
| Artigo para Overleaf | `tcc_overleaf/main.tex` |
| Pacote Overleaf | `tcc_overleaf.zip` |
| Dataset do benchmark atual | `app/datasets/benchmark_audio_raw_balanced_15k.npz` |
| Modelos default da aplicação | `app/models/bench_*` |
| Modelos completos por arquitetura | `app/models/benchmark_final/` |
| Métricas, gráficos e relatórios | `results/` |

## Estudo experimental no GitHub Pages

A documentação agora incorpora a fundamentação técnica e a análise experimental
consolidadas a partir do trabalho:

- equações de síntese, pré-processamento, VAD e extração de características em
  [Features de Áudio](04_FEATURES.md) e [Estudo Experimental](20_ESTUDO_EXPERIMENTAL.md);
- fluxograma de predição e treinamento em [Arquitetura](03_ARQUITETURA.md);
- descrição das 14 arquiteturas e decisão operacional em
  [Arquiteturas Neurais](08_ARQUITETURAS.md);
- resultados, artefatos, modelos treinados e rastreabilidade em
  [Benchmark e Resultados](15_BENCHMARK.md);
- uso da interface, abas, notificações e fluxos de análise em
  [Interface Gradio](23_INTERFACE_GRADIO.md);
- publicação coordenada de documentação e demonstração em
  [GitHub Pages e Hugging Face](24_PUBLICACAO_GITHUB_HF.md);
- versão navegável do estudo em
  [Estudo Experimental](20_ESTUDO_EXPERIMENTAL.md).

## Visão de uma página

- **14 arquiteturas** de detecção: 12 neurais (AASIST, RawGAT-ST, RawNet2,
  WavLM, HuBERT, Conformer, SpectrogramTransformer, Hybrid CNN-Transformer,
  EfficientNet-LSTM, MultiscaleCNN, Ensemble, Sonic Sleuth) + 2 clássicas
  (SVM, RandomForest).
- **Front-end real** por modelo: forma de onda bruta (raw-audio), log-mel ou
  **LFCC** (espectrograma), via `tf.signal` in-graph — paridade treino↔inferência
  garantida pelo `input_contract`.
- **Métricas** padrão da área: acurácia, AUC-ROC, **EER** e **min-tDCF**
  (ASVspoof), além de latência e tamanho do modelo.
- **Plataforma**: Python 3.11, TensorFlow/Keras 3, FastAPI + Gradio, Docker
  multi-stage, CI com testes/segurança/docs.

## Fluxos principais

```mermaid
flowchart LR
    A["Áudio bruto"] --> B["Upload e validação"]
    B --> C["Extração de features"]
    C --> D["Modelo treinado"]
    D --> E["Predição"]
    E --> F["Relatório e explicabilidade"]
```

```mermaid
flowchart LR
    A["Dataset real/fake"] --> B["Pré-processamento"]
    B --> C["Pipeline de features"]
    C --> D["Treinamento"]
    D --> E["Métricas"]
    E --> F["Modelo exportado"]
```

```mermaid
flowchart LR
    A["Benchmark TCC"] --> B["dataset.md"]
    A --> C["tcc_report.md"]
    A --> D["figures/*.png"]
    A --> E["architectures/&lt;modelo&gt;/"]
    E --> F["app/models/benchmark_final"]
```

## Comandos rápidos

```bash
python main.py --bootstrap-dirs                 # cria a estrutura de diretórios
python main.py --gradio                         # sobe a UI Gradio + API em :7860
./scripts/run_tests.sh fast                     # suíte rápida (sem smoke)
docker compose up --build -d                    # produção (Docker)
python scripts/run_tcc_pipeline.py --smoke --epochs 1 --batch-size 4
```

Para detalhes de ambiente, dependências e variáveis `.env`, veja
[Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md).
