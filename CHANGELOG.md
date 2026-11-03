# Changelog

Mudanças relevantes para quem instala, opera ou reproduz o SpeechChecker. O formato
segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).

O projeto não tinha changelog nem tags antes desta entrada. O histórico anterior está
no Git, e `app/__init__.py` declara a versão `1.1.0`.

## [Não lançado]

### Deploy em Docker

- **Segredos fora da imagem CPU:** `Dockerfile.cpu.dockerignore` substituía o `.dockerignore`
  da raiz e não excluía `.env`, `data/app.db` nem os áudios de `data/temp`. O arquivo saiu, e
  as duas imagens de inferência usam agora o mesmo conjunto de regras. **Se você já publicou
  uma imagem CPU, troque as credenciais do `.env` que estava no contexto de build.**
- **`.env` lido de verdade:** `make`, `start.bat` e `start.sh` passam `--env-file .env`. Antes,
  a inferência recusava subir e o treino e o benchmark ignoravam o arquivo. **Em treino e
  benchmark, valores do `.env` como `XFAKE_BENCHMARK_OUT` e os limites de memória passam a
  valer.** Os scripts validam a configuração antes do build.
- **Modo produção:** os perfis de inferência rodam sempre com `DEEPFAKE_ENV=production`, mesmo
  que o `.env` traga `development`.
- **Porta só local por padrão:** a inferência publica em `127.0.0.1`. Quem acessa de outra
  máquina precisa definir `GRADIO_BIND_HOST`.
- **Hora certa no Histórico:** o container rodava em UTC, e a tela mostrava 01:30 às 22:30. Os
  perfis de inferência definem `TZ`, com padrão `America/Sao_Paulo`. O banco continua em UTC.
- **Modelos do Hub no container:** `MODEL_REPO_ID`, `HF_TOKEN` e `XFAKE_SYNC_MODELS_ON_BOOT` agora
  chegam à inferência. **Um `HF_TOKEN` definido no `.env` passa a entrar no ambiente do container.**
  `sync_hf_models.py` não sobrescreve mais arquivo local sem `--force`, porque em Docker
  `data/models` é a pasta do host. Com armazenamento persistente em Spaces, atualizar um modelo já
  baixado também passa a exigir `--force`.
- **Versões fixadas:** as imagens de inferência instalam as versões exatas da imagem validada, em
  `constraints-<perfil>.txt`. Antes, `nvidia-cudnn-cu12` e `tensorflow` seguiam faixas, e o CuDNN
  subiu de 9.24 para 9.27 entre dois builds. O benchmark usa a mesma imagem e herda isso.
- **Imagens menores:** o cache do pip ocupava 6,4 GB na NVIDIA e 3,5 GB na CPU, sem uso em execução.
  Sem ele, o tamanho informado pelo Docker caiu de 30,3 para 16,7 GB e de 22,7 para 15,4 GB.
- **Caminho de volta:** `tools/ops/docker_release.py` guarda a imagem e o banco antes de atualizar e
  volta com `rollback`, que espera o health check. Também em `make snapshot` e `make rollback`.
- **`make clean` restrito ao projeto:** `make clean`, `start.bat clean` e `start.sh clean` removiam
  imagens órfãs e todo o cache de build da máquina, inclusive de outros projetos. Agora só removem
  imagens órfãs da inferência. O cache de build fica, e quem quiser liberá-lo roda
  `docker builder prune` sabendo o alcance.
- **Imagens de treino:** o cache do pip também saiu delas. Em `ssl-transformers:nvidia`, o TensorFlow
  rodava em CPU sem nenhum erro, porque o Dockerfile não publicava as bibliotecas CUDA. Corrigido.
- **CI das imagens de inferência:** o workflow `docker-inference.yml` constrói e confere as duas
  imagens. Antes a CI só construía o `Dockerfile` da raiz.
- **Docker parado:** `start.bat` e `start.sh` tentam iniciar o Docker Desktop antes de desistir.
- **`GET /api/v1/system/health`:** responde 503 quando o banco falha, o diretório de modelos não
  existe ou não há modelo disponível. Antes respondia 200. O `HEALTHCHECK` das imagens de
  inferência não tenta mais `/` depois da falha, então o container fica `unhealthy` nesses casos.
- **AASIST na GPU:** `POST /api/v1/detection/analyze` falhava com `Input 'y' of 'Mul' Op has type
  float32`. Com GPU, `setup_gpu` liga `mixed_float16` na thread principal, e as sub-camadas herdavam
  essa política ao carregar o modelo. A carga agora usa uma política neutra, e o resultado na GPU
  bate com a CPU, onde a temperatura e o limiar foram medidos.

### Testes

- **Suíte isolada do banco real:** `tests/conftest.py` não definia `DATABASE_URL`, e cada execução
  gravava em `data/app.db` (runs, métricas, artefatos, snapshots e jobs de treino). Agora usa um
  banco temporário. As linhas já gravadas continuam lá.

### Calibração e fusão

- **Temperatura sobre o logit real:** a busca era uma grade de 0,5 a 4,9 sobre a
  probabilidade cortada em 1e-7. Conformer parou no piso, RawGAT-ST no teto, e o corte
  criava um teto de P(fake): 0,97891 no AASIST, atingido por 84% da validação. Treinos novos
  usam busca contínua e declaram `temperature_on_logits` no contrato. Artefatos sem o campo
  continuam como estão e reproduzem os números do seu run.
- **`tools/reporting/recalibrate_temperature.py`:** migra os modelos promovidos sem treinar.
  Transporta o limiar, então nenhum veredito da validação muda, nem EER, AUC ou t-DCF. Só
  grava com `--apply`, guardando o sidecar anterior.
- **Fusão multi-modelo:** `detect_multi_model` e `POST /multi-model` combinam `fake_score`,
  não P(fake), e decidem em 0,5. Antes, dois modelos que diziam "real" podiam fundir em
  "fake". **Muda vereditos da fusão**, e `probabilities` da resposta passa a trazer a
  pontuação fundida. `majority_vote` não contradiz mais o voto, e um modelo que falha não
  desloca mais os pesos dos seguintes.
- **`tools/reporting/select_fusion.py`:** mede modelos e conjuntos na validação, em locutor
  não visto, limpo e sob ruído, sem abrir o teste.
- **Relatórios:** tabelas e resumos ganham `Acur.@cal`, a acurácia no limiar de validação,
  que é a decisão do sistema implantado. O runner SSL passa a derivar esse limiar da
  validação limpa, em runs novos.

### Alterado

- **Pilha web:** Gradio 4.44 para 6.17, FastAPI 0.116 para 0.162, Starlette 0.47 para
  1.10, Pillow 12.3 e datasets 5. `torch`, `transformers`, TensorFlow, Keras, NumPy e
  scikit-learn não mudaram, e as predições dos modelos existentes foram conferidas
  contra a versão anterior.
- **Aparência do Gradio:** tema, CSS e cabeçalho saíram de `gr.Blocks`, porque o
  Gradio 6 os recebe em `launch()` e `mount_gradio_app()`. Quem montar a interface por
  conta própria deve passar `APPEARANCE`, de `app/interfaces/gradio/app.py`.
- **Download de datasets:** repositórios do Hugging Face baseados em script deixam de
  carregar, porque o datasets 5 removeu `trust_remote_code`.
- **Estrutura:** scripts foram para `tools/` e Docker foi para `infrastructure/`.
  Módulos grandes foram decompostos mantendo as APIs públicas: assistente de treino,
  perfis de voz, análise forense, `trainer.py`, runner do benchmark e `layers.py`. Os
  modelos `.keras` já salvos continuam carregando.

### Alterado na detecção

- **`confidence` agora é a confiança no veredito, de 0,5 a 1, em todos os modelos.** Antes
  era a probabilidade da classe decidida: com limiar calibrado de 0,94, um veredito "real"
  de P(fake) 0,80 saía com confiança 0,20, e a tela dizia "provável voz real" com "80% de
  probabilidade de ser sintética". O novo campo `fake_score` reescala P(fake) para que 50%
  seja sempre o ponto de decisão do modelo. **Nenhum veredito muda**, e `probabilities`
  segue com os valores brutos, a comparar com `classification_threshold`. Clientes da API
  que liam `confidence` como probabilidade devem usar `probabilities`. Análises já gravadas
  no histórico mantêm o valor antigo.

### Corrigido

- **Investigar:** a linha do tempo, o mapa de anomalias e o histograma do lote mostravam
  a confiança do veredito como se fosse a probabilidade de fake, com limiar fixo de 0,5.
  Agora usam P(fake) e o limiar do modelo.
- **Modelos clássicos:** um run novo agora grava a correção de banda no contrato do
  sidecar, e a inferência a reproduz. Os artefatos já promovidos tinham esse carimbo e
  não mudam. Como `fit_classical.py` entra no fingerprint, o `--resume` retreina os
  clássicos.
- **Benchmark:** o artefato preservado do run agora é uma cópia real. Com hardlink,
  qualquer gravação em `data/models/bench_*` alterava em silêncio o artefato registrado.
- **CI:** a validação dos perfis de inferência do Compose não exige mais um `.env`.

### Segurança

- O gate de release agora usa `tools/ops/check_pip_audit.py`: bloqueia todo aviso fora
  de `configs/security/pip-audit-exceptions.json`, versão diferente da registrada e
  exceção vencida. A migração reduziu os avisos de 8 para 3 pacotes.
- **Exceções abertas, com revisão até 2027-02-01:** `pillow`, `torch` e
  `transformers`, 84 IDs ao todo. A justificativa de cada uma e o motivo de não
  atualizar estão no arquivo de exceções.

### Limitações conhecidas

- **Publicação acadêmica bloqueada:** há pendências de licença do corpus e de
  proveniência das métricas do artigo. Ver
  [o registro da revisão manual](docs/development/release-review.md).
- **API de treino:** quem tem a chave da API pode escolher, em `parameters`, o
  repositório do Hugging Face carregado pelos modelos SSL. Com os pinos atuais de
  `torch` e `transformers`, isso equivale a execução de código no servidor. Trate a
  chave como acesso administrativo.
- **Gradio limitado a 6.17:** a linha 6.18 exige `huggingface_hub>=1.2`, e o
  `transformers` fixado exige `<1.0`.
- **GPU e Linux:** a CI não tem GPU. A aprovação dos perfis NVIDIA só significa que a
  configuração é válida.
