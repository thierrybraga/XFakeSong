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
- **Uma porta para cada uso:** Docker NVIDIA na 7860, Docker CPU na 7861 e aplicação local na 7862.
  Antes, as três usavam a 7860. **`GRADIO_PORT` deixa de ser lida: use `GRADIO_PORT_NVIDIA` e
  `GRADIO_PORT_CPU`.** `DEEPFAKE_API_PORT` e `DEEPFAKE_DB_PORT` também saíram. O mapa fica em
  `app/core/config/ports.py`.
- **Uma instância por vez sobre o mesmo `data/`:** `main.py`, `make up`, `start.bat` e `start.sh`
  recusam subir com outro perfil em execução. `up` direto do Compose e o botão Start do Docker
  Desktop continuam sem essa verificação.
- **Credenciais:** o `.env` da raiz é a única fonte de login e chave da API nos dois perfis, sem
  senha de reserva. O login sobrevive a um rebuild.
- **Cache do Gradio na pasta temporária:** em `data/`, ele ficava em outro disco, e o upload só
  terminava de copiar depois da resposta. O cache agora some ao recriar o container.
- **`build_smoke_test.sh` removido:** o script não passava da partida, e nem a CI nem o Makefile o
  chamavam. A porta 7868 foi liberada. Para as imagens de inferência, use
  `tools/ops/docker_release.py check`.

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

### Interface

- **Navegação reorganizada** em seis seções por objetivo: Painel, Detectar (um áudio, em
  lote, histórico), Investigar (forense, características), Perfis de voz, Treinar e
  Datasets. Antes, perfis de voz ficavam dentro de Detectar, e o histórico, em Gerenciar.
- **Detectar:** o modelo agora é uma escolha visível, só entre os instalados. Antes, a
  escolha era ignorada sem marcar uma caixa escondida. O resultado virou um cartão com o
  veredito em texto, P(fake) comparado ao limiar do modelo e avisos de pouca margem e de
  áudio fora do padrão de treino. Enviar arquivo e gravar são abas separadas, e o
  histórico guarda o nome real do arquivo.
- **Opções sem efeito removidas:** hiperparâmetros JSON na detecção; SWA, mixup,
  calibração de temperatura e limiar EER no assistente de treino. "Sprints 1-5" saiu da
  tela.
- **Layout:** largura estável entre as abas (variava de 726 a 1.179 px), sem bordas
  brancas nos cartões, figuras sem o rótulo por cima do título, tema claro completo e
  navegação fixa ao rolar. Sem rolagem horizontal da página em celular e tablet, com
  alvos de toque de 44 px.
- **Estados vazios:** gráficos, relatório e salvamento só aparecem quando há resultado.
- **Confirmação** antes de apagar todo o histórico e antes de excluir um perfil.
- **Notificações:** o sino mostra avisos e erros para a pessoa usuária. Logs de
  requisição, como `GET /favicon.ico -> 404`, saíram do painel.
- **Gravação pelo microfone:** a análise agora usa a gravação inteira, entregue ao clicar
  em Parar. O modo streaming falhava com o aviso de `process_stream` e, mesmo sem erro,
  analisava só o último pedaço (1,75 s de uma gravação de cerca de 14 s). **Durante a
  gravação, a tela mostra só a forma de onda:** saíram o veredito e os gráficos ao vivo.
- **Aviso em gravações de microfone:** o cartão diz, com qualquer veredito, que os modelos
  não foram validados com esse tipo de áudio.

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

- **Build das imagens de treino Keras:** o PR 48 do Dependabot fixou `transformers==5.22.1` em
  `tensorflow-keras/requirements.txt`, mas a imagem instala com `constraints.txt`, que fixa
  4.45.2. O pip recusaria o próximo build de `tensorflow-keras:cpu` e `:nvidia`. O pino voltou
  para 4.45.2, um teste confere cada imagem contra o próprio constraint, e o Dependabot não
  propõe mais a versão 5. As imagens já construídas não foram afetadas.
- **Detectar e API:** a análise por arquivo perdia o limiar calibrado e o indicador de
  fora do padrão, e a tela afirmava "limiar fixo 0,5" para um modelo com corte de 0,94.
- **Painel e Histórico:** tempos negativos (`-9545s atrás`) e datas em UTC mostradas como
  se fossem locais.
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
- **Gravações de microfone:** os modelos podem errar para os dois lados. Uma voz real
  dividiu os 9 modelos, e o AASIST, modelo padrão, a chamou de sintética. As medições estão
  em [o guia da interface](docs/interfaces/gradio.md).
