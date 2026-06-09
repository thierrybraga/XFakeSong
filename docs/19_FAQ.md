# Perguntas Frequentes (FAQ)

Respostas curtas para as dúvidas mais comuns, com links para a fonte canônica.
Para o passo a passo completo de instalação e os erros detalhados, veja
[Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md).

## Ambiente e instalação

??? question "Qual versão de Python devo usar?"
    **Python 3.11** é o ambiente de referência (testado com TensorFlow 2.21);
    a faixa suportada é **3.11–3.13**. No 3.13 é preciso TensorFlow ≥ 2.20
    (versões 2.16–2.19 não têm *wheels* para 3.13).

??? question "`pip install` falha no TensorFlow"
    Quase sempre é incompatibilidade Python↔TF. Confirme a versão do Python e use
    a variante certa de requirements: `requirements.txt` (padrão),
    `requirements-cpu.txt` (CPU, ~250 MB menor) ou GPU via
    `tensorflow[and-cuda]`. As três compartilham `requirements-base.txt`.
    Rode `python scripts/doctor.py` para um diagnóstico completo.

??? question "Preciso de GPU?"
    Não. SVM, Random Forest e a inspeção dos modelos rodam em CPU. Para treinar
    redes neurais a GPU acelera bastante. No Windows, a GPU NVIDIA é usada via
    **WSL2** — veja a seção de GPU em
    [Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md).

## Execução

??? question "Erro: porta 7860 já está em uso"
    Há uma instância antiga rodando. Finalize-a ou use outra porta:
    `python main.py --gradio --gradio-port 7861`. No Docker, `docker compose down`
    antes de subir de novo.

??? question "A UI demora a abrir / o healthcheck falha no boot"
    O TensorFlow tem *boot* lento. O healthcheck usa `/api/v1/system/health`
    (não depende do Gradio renderizado) e o `start_period` é de 180 s. Aguarde o
    primeiro carregamento; predições seguintes são rápidas (há *warm-up*).

??? question "`TypeError: unhashable type: 'dict'` ao abrir a página"
    Incompatibilidade entre `gradio` e `starlette`. Use as versões pinadas
    (`gradio>=4.31,<5.0`, `starlette>=0.36,<0.42`); o `app/core/version_check.py`
    detecta e loga isso no *startup*. Detalhes em
    [Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md).

??? question "Como rodo os notebooks no Google Colab?"
    Eles são auto-suficientes: abra por
    `colab.research.google.com/github/thierrybraga/XFakeSong/blob/main/notebooks/<caminho>`,
    selecione **GPU** (Ambiente de execução → Alterar o tipo de hardware) e rode
    a **primeira célula** — ela clona o repo e instala as dependências. Os
    notebooks de modelo treinam por padrão. Guia completo:
    [Google Colab](13_COLAB_GUIDE.md).

## Treinamento

??? question "Que formato de dataset o treino espera?"
    Um `.npz` com `X_train`/`y_train` (ou `X`/`y`), ou as pastas
    `app/datasets/real/` e `app/datasets/fake/`. Veja
    [Treinamento](10_TREINAMENTO.md) e [Datasets Públicos](12_DATASETS.md).

??? question "O treino diverge (loss vira NaN)"
    O pipeline já força `float32`, *gradient clipping* e normalização segura.
    Se ainda ocorrer, reduza o *learning rate* e confira se o áudio não tem
    trechos inválidos (o pré-processamento sanitiza NaN/Inf). Use o
    **Assistente de Treino** da UI, que aplica os padrões corretos por modelo.

??? question "Qual modelo escolher para começar?"
    `MultiscaleCNN` (espectrograma) ou `RawNet2` (raw-audio) são bons pontos de
    partida leves. SVM/Random Forest treinam em segundos como baseline. Para o
    comparativo completo, rode o [Benchmark](15_BENCHMARK.md).

## Inferência

??? question "O que é o `input_contract` e por que ele importa?"
    São os metadados gravados no treino (front-end, parâmetros de STFT, *sample
    rate*, temperatura, limiar de EER) que garantem **paridade
    treino↔inferência**. O `ModelLoader` os lê e o `Predictor` os aplica
    automaticamente. Veja [Inferência](09_INFERENCIA.md).

??? question "Como a inferência usa ONNX?"
    Se houver um `.onnx` ao lado do modelo e o `onnxruntime` estiver instalado,
    o `Predictor` usa ONNX (mais rápido em CPU), com *fallback* automático para
    o TensorFlow. Nenhuma ação manual é necessária.

## Deploy

??? question "Como publicar no Hugging Face Spaces?"
    O `app.py` é o *entry point* do Space. Siga o
    [Deploy Hugging Face](11_DEPLOY_HUGGINGFACE.md) — inclui as variáveis de
    ambiente e os cuidados de boot em ambiente sem internet de saída.

??? question "Docker ou execução local?"
    Docker (`docker compose up --build -d`) isola dependências e é o modo de
    produção recomendado. Execução local (`.venv`) é melhor para desenvolvimento.
    Ambos cobertos em [Instalação e Configuração](02_INSTALACAO_CONFIGURACAO.md).

## Projeto

??? question "Como contribuir?"
    Veja o [`CONTRIBUTING.md`](https://github.com/thierrybraga/XFakeSong/blob/main/CONTRIBUTING.md)
    e o [Guia do Desenvolvedor](05_GUIA_DEV.md). PRs passam pelos gates de
    [CI/CD](17_CICD_SEGURANCA.md) (testes, docs, segurança).

??? question "Encontrei uma vulnerabilidade. O que faço?"
    **Não** abra issue pública. Use o *GitHub Private Vulnerability Reporting* —
    detalhes em [CI/CD e Segurança](17_CICD_SEGURANCA.md) e no
    [`SECURITY.md`](https://github.com/thierrybraga/XFakeSong/blob/main/SECURITY.md).
