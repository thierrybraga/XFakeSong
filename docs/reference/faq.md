# Perguntas Frequentes (FAQ)

Respostas curtas para as dúvidas mais comuns, com links para a fonte canônica.
Para o passo a passo completo de instalação e os erros detalhados, veja
[Instalação e Configuração](../getting-started/installation.md).

## Ambiente e instalação

??? question "Qual versão de Python devo usar?"
    **Python 3.11** é o ambiente de referência (testado com TensorFlow 2.21);
    a faixa suportada é **3.11–3.12**. O Python 3.13 ainda não é suportado
    porque o pino `torch==2.5.1` não tem *wheels* cp313 (suporte a 3.13 só a
    partir do torch 2.6). Para usar 3.13, suba torch/torchaudio para >=2.6 e
    revalide o import na GPU.

??? question "`pip install` falha no TensorFlow"
    Quase sempre é incompatibilidade Python↔TF. Confirme a versão do Python e use
    a variante certa de requirements: `requirements.txt` (padrão),
    `requirements-cpu.txt` (CPU, ~250 MB menor) ou GPU via
    `tensorflow[and-cuda]`. As três compartilham `requirements-base.txt`.
    Rode `python tools/ops/doctor.py` para um diagnóstico completo.

??? question "Preciso de GPU?"
    Não. SVM, Random Forest e a inspeção dos modelos rodam em CPU. Para treinar
    redes neurais a GPU acelera bastante. No Windows, a GPU NVIDIA é usada via
    **WSL2** — veja a seção de GPU em
    [Instalação e Configuração](../getting-started/installation.md).

## Execução

??? question "Erro: a porta já está em uso"
    Cada instância tem a sua porta: 7860 no Docker NVIDIA, 7861 no Docker CPU e
    7862 na aplicação local. Veja o
    [mapa de portas](../development/docker-environments.md#portas). Se outro
    programa ocupa a porta, use `python main.py --gradio --gradio-port 7863` ou
    defina `GRADIO_PORT_NVIDIA` ou `GRADIO_PORT_CPU` no `.env`. Se for uma instância
    antiga, use `make down` (ou `start.bat stop`) antes de subir de novo.

??? question "A aplicação recusa subir porque outra instância usa o mesmo `data/`"
    Todas as instâncias usam o mesmo `data/app.db`, e duas ao mesmo tempo podem
    danificar o banco. Pare a outra com o `docker stop` indicado na mensagem. Para
    rodar a aplicação local ao lado do Docker, aponte `DATABASE_URL` para outro
    arquivo. Detalhes no
    [guia de portas](../development/docker-environments.md#uma-instancia-por-vez-sobre-o-mesmo-data).

??? question "A UI demora a abrir / o healthcheck falha no boot"
    O TensorFlow tem *boot* lento. O healthcheck usa `/api/v1/system/health`
    (não depende do Gradio renderizado) e o `start_period` é de 180 s. Aguarde o
    primeiro carregamento; predições seguintes são rápidas (há *warm-up*).

??? question "`TypeError: unhashable type: 'dict'` ao abrir a página"
    Incompatibilidade entre `gradio` e `starlette` em um ambiente desatualizado.
    Reinstale com `pip install -r requirements.txt -c constraints.txt`; o
    `app/core/version_check.py` detecta e loga isso no *startup*. Detalhes em
    [Instalação e Configuração](../getting-started/installation.md).

??? question "Como rodo os notebooks no Google Colab?"
    Eles são auto-suficientes: abra por
    `colab.research.google.com/github/thierrybraga/SpeechChecker/blob/main/docs/notebooks/<caminho>`,
    selecione **GPU** (Ambiente de execução → Alterar o tipo de hardware) e rode
    a **primeira célula** — ela clona o repo e instala as dependências. Os
    notebooks de modelo treinam por padrão. Guia completo:
    [Google Colab](../getting-started/google-colab.md).

## Treinamento

??? question "Que formato de dataset o treino espera?"
    Um `.npz` com `X_train`/`y_train` (ou `X`/`y`), ou as pastas
    `data/datasets/real/` e `data/datasets/fake/`. Veja
    [Treinamento](../models/training.md) e [Datasets Públicos](../data/public-datasets.md).

??? question "O treino diverge (loss vira NaN)"
    O pipeline já força `float32`, *gradient clipping* e normalização segura.
    Se ainda ocorrer, reduza o *learning rate* e confira se o áudio não tem
    trechos inválidos (o pré-processamento sanitiza NaN/Inf). Use o
    **Assistente de Treino** da UI, que aplica os padrões corretos por modelo.

??? question "Qual modelo escolher para começar?"
    `MultiscaleCNN` (espectrograma) ou `RawNet2` (raw-audio) são bons pontos de
    partida leves. SVM/Random Forest treinam em segundos como baseline. Para o
    comparativo completo, rode o [Benchmark](../evaluation/benchmark.md).

## Inferência

??? question "O que é o `input_contract` e por que ele importa?"
    São os metadados gravados no treino (front-end, parâmetros de STFT, *sample
    rate*, temperatura, limiar de EER) que garantem **paridade
    treino↔inferência**. O `ModelLoader` os lê e o `Predictor` os aplica
    automaticamente. Veja [Inferência](../models/inference.md).

??? question "Como a inferência usa ONNX?"
    Se houver um `.onnx` ao lado do modelo e o `onnxruntime` estiver instalado,
    o `Predictor` usa ONNX (mais rápido em CPU), com *fallback* automático para
    o TensorFlow. Nenhuma ação manual é necessária.

## Deploy

??? question "Como publicar no Hugging Face Spaces?"
    O fluxo recomendado é o **Docker Space**: `Dockerfile` (raiz) →
    `docker-entrypoint.sh` → `python main.py --gradio`. `app.py` existe (entry
    point do SDK Gradio simples), mas todos os guias de deploy o desaconselham
    para este projeto — use o Docker Space. Siga o
    [Deploy Hugging Face](../deployment/hugging-face-spaces.md) — inclui as variáveis de
    ambiente e os cuidados de boot em ambiente sem internet de saída.

??? question "Docker ou execução local?"
    Docker (`make up` ou `start.bat prod`) isola dependências e é o modo de
    produção recomendado. Execução local (`.venv`) é melhor para desenvolvimento.
    Ambos cobertos em [Instalação e Configuração](../getting-started/installation.md).

## Projeto

??? question "Como contribuir?"
    Veja o [`CONTRIBUTING.md`](https://github.com/thierrybraga/XFakeSong/blob/main/CONTRIBUTING.md)
    e o [Guia do Desenvolvedor](../development/developer-guide.md). PRs passam pelos gates de
    [CI/CD](../development/ci-cd-and-security.md) (testes, docs, segurança).

??? question "Encontrei uma vulnerabilidade. O que faço?"
    **Não** abra issue pública. Use o *GitHub Private Vulnerability Reporting* —
    detalhes em [CI/CD e Segurança](../development/ci-cd-and-security.md) e no
    [`SECURITY.md`](https://github.com/thierrybraga/XFakeSong/blob/main/SECURITY.md).
