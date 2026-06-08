# Guia de Integração com Google Colab

Este guia detalha como executar a plataforma XFakeSong de detecção de deepfakes de áudio no Google Colab.

## 1. Configuração Inicial

### 1.1. Preparar o Google Drive
1. Crie uma pasta chamada `XFakeSong` na raiz do seu Google Drive.
2. Faça upload do conteúdo do projeto para essa pasta.
   - Opção recomendada: use `git clone` diretamente no Colab.
   - Alternativa: faça upload do arquivo zip e extraia.

### 1.2. Estrutura de Diretórios Esperada

```
/content/drive/MyDrive/XFakeSong/
├── app/
├── data/
│   ├── raw/
│   │   ├── real/
│   │   └── fake/
│   └── processed/
├── notebooks/
├── requirements.txt
└── ...
```

---

## 2. Notebooks Disponíveis

Os notebooks ficam organizados por finalidade:

| Caminho | Objetivo | Runtime |
|---|---|---|
| `notebooks/00_index.ipynb` | Índice executável e mapa dos notebooks | CPU |
| `notebooks/features/01_feature_extraction_study.ipynb` | Estudo do front-end real LFCC/log-mel/raw e features clássicas MFCC, centroid, bandwidth, ZCR e RMS | CPU |
| `notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb` | Download, processamento, split, treino, inferência e relatório do TCC | GPU recomendada |
| `notebooks/pipeline/02_training_model.ipynb` | Treino prático de um modelo pelo benchmark | CPU para SVM/RF, GPU para neurais |
| `notebooks/pipeline/03_inference.ipynb` | Leitura de predições e fluxo de inferência | CPU ou GPU |
| `notebooks/models/*.ipynb` | Um notebook de estudo para cada arquitetura | CPU para inspeção, GPU para treino |
| `notebooks/legacy/` | Notebooks antigos preservados | Conforme notebook |

Para o experimento completo do TCC, comece por
`notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb`. Ele documenta o
preset `--tcc-full-dataset`, com alvo de `10.000` amostras reais + `10.000`
amostras fake.

---

## 3. Problemas Comuns e Soluções

### 3.1. Erros de Caminho
- Execute sempre a célula de setup que monta o Drive e define `PROJECT_PATH`.
- Se ocorrer `ModuleNotFoundError`, certifique-se de que a linha `sys.path.append(PROJECT_PATH)` foi executada antes.

### 3.2. GPU Não Disponível
- Execute `!nvidia-smi` para verificar se uma GPU está disponível na sessão atual.
- Algumas arquiteturas avançadas (como AASIST) podem exigir versões específicas de CUDA — o ambiente padrão do Colab normalmente é suficiente.

### 3.3. Falta de Memória (OOM)
- Se a sessão travar durante a extração de features, reduza o `batch_size` ou processe os arquivos em lotes menores.
- Para treinamento de Transformers pesados (WavLM, HuBERT), use acumulação de gradiente com batch size menor.

---

## 4. Executando a Inferência ou Demo

1. Abra `notebooks/pipeline/03_inference.ipynb` para estudar predições,
   scores e artefatos do benchmark.
2. Para interface visual, rode `python main.py --gradio` em uma célula do
   Colab ou no terminal local.
3. Se usar Gradio Live, clique no link público exibido na saída
   (ex: `Rodando em URL pública: https://...`).

> **Dica**: Links do Gradio Live expiram após ~72 horas de inatividade. Para uso contínuo, considere o deploy no Hugging Face Spaces (ver [`11_DEPLOY_HUGGINGFACE.md`](11_DEPLOY_HUGGINGFACE.md)).
