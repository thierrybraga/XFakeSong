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

O projeto inclui 3 notebooks para as diferentes etapas do pipeline:

### 2.1. `01_Setup_and_Extraction.ipynb`
- **Objetivo**: Instala dependências e extrai características dos arquivos de áudio brutos.
- **Entrada**: Arquivos de áudio em `data/raw/real/` e `data/raw/fake/`.
- **Saída**: `real_features.joblib` e `fake_features.joblib` em `data/processed/`.
- **Runtime**: CPU padrão (GPU recomendada para extração mais rápida).

### 2.2. `02_Training.ipynb`
- **Objetivo**: Treina o modelo de detecção de deepfakes.
- **Entrada**: Features extraídas pelo Notebook 01.
- **Saída**: Modelo treinado salvo em `app/models/`.
- **Runtime**: **GPU obrigatória.** Ative em `Ambiente de Execução > Alterar o tipo de ambiente de execução > T4 GPU`.

### 2.3. `03_Inference_and_Demo.ipynb`
- **Objetivo**: Executa a interface Gradio para testes e demonstração.
- **Funcionalidades**:
    - Gravação em tempo real ou upload de arquivo.
    - Score de probabilidade de deepfake.
    - Visualização de features e espectrogramas.
- **Runtime**: CPU ou GPU.
- **Acesso**: Gera um link público compartilhável (ex: `https://xxxx.gradio.live`).

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

## 4. Executando a Demo

1. Abra o notebook `03_Inference_and_Demo.ipynb`.
2. Execute todas as células em sequência.
3. Clique no link público exibido na saída da última célula (ex: `Rodando em URL pública: https://...`).

> **Dica**: Links do Gradio Live expiram após ~72 horas de inatividade. Para uso contínuo, considere o deploy no Hugging Face Spaces (ver [`11_DEPLOY_HUGGINGFACE.md`](11_DEPLOY_HUGGINGFACE.md)).
