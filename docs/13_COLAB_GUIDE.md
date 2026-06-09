# Guia de Integração com Google Colab

Este guia detalha como executar a plataforma XFakeSong de detecção de deepfakes de áudio no Google Colab.

## 1. Configuração Inicial

Os notebooks são **auto-suficientes no Colab**: a primeira célula (bootstrap)
detecta o ambiente, clona o repositório e instala as dependências de áudio.
**Não é preciso** montar o Google Drive nem subir arquivos manualmente.

### 1.1. Abrir um notebook no Colab

Abra qualquer notebook pela URL `colab.research.google.com/github/…`:

```
https://colab.research.google.com/github/thierrybraga/XFakeSong/blob/main/notebooks/<caminho>
```

Exemplos:

- Treinar um modelo: `notebooks/pipeline/02_training_model.ipynb`
- Estudar uma arquitetura: `notebooks/models/05_aasist.ipynb`
- Índice de tudo: `notebooks/00_index.ipynb`

### 1.2. Selecionar GPU e rodar o setup

1. **Ambiente de execução → Alterar o tipo de hardware → GPU** (T4 grátis).
2. Rode a **primeira célula**: ela clona o `XFakeSong`, entra na pasta e instala
   `librosa` / `soundfile` / `PyWavelets` (TensorFlow, NumPy e scikit-learn já
   vêm no Colab — por isso o setup é rápido, ~1–2 min na 1ª execução).

!!! tip "Persistir resultados entre sessões"
    O armazenamento do Colab é efêmero. Para guardar modelos/resultados, monte o
    Drive (`from google.colab import drive; drive.mount('/content/drive')`) e
    aponte o `output_dir`/`models_dir` dos notebooks para uma pasta nele.

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

Para o experimento completo do TCC, comece por
`notebooks/pipeline/01_benchmark_tcc_full_pipeline.ipynb`. Ele documenta o
preset `--tcc-full-dataset`, com alvo de `10.000` amostras reais + `10.000`
amostras fake.

---

## 3. Problemas Comuns e Soluções

### 3.1. Erros de import / caminho
- Rode a **célula de bootstrap** (a primeira) antes de qualquer outra — ela
  clona o projeto, ajusta o `sys.path` e instala as dependências.
- `ModuleNotFoundError: No module named 'app'` → o clone não rodou; reexecute a
  primeira célula.
- `ModuleNotFoundError: No module named 'transformers'` → ocorre só em
  WavLM/HuBERT; esses notebooks têm uma célula dedicada que o instala (rode-a).

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
