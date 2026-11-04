# Docker, ambientes e execução

Este guia consolida a visão operacional do projeto para Docker, ambientes de treino/inferência e execução local.

## Pontos de entrada

- `main.py`: ponto de entrada principal do CLI e da interface Gradio.
- `app/interfaces/gradio/app.py`: montagem da interface unificada.
- `app.py`: entrada para Hugging Face Spaces.
- `infrastructure/docker/environments/inference-api/docker-entrypoint.sh`: bootstrap do container.

## Estrutura recomendada

- [`infrastructure/docker/compose`](https://github.com/thierrybraga/XFakeSong/tree/main/infrastructure/docker/compose): perfis de execução.
- [`infrastructure/docker/environments`](https://github.com/thierrybraga/XFakeSong/tree/main/infrastructure/docker/environments): Dockerfiles por família de ambiente.
- [`data/results`](https://github.com/thierrybraga/XFakeSong/tree/main/data/results): artefatos regeneráveis do benchmark (não há `results/` na raiz).
- [`data/models`](https://github.com/thierrybraga/XFakeSong/tree/main/data/models): modelos inferidos e exportados.

## Fluxos principais

- Inferência local: `python main.py --gradio`, em `http://localhost:7862`
- Inferência Docker: `docker compose --env-file .env -f infrastructure/docker/compose/inference.cpu.yml up --build inference-api`, em `http://localhost:7861`
- Treino Docker: `docker compose --env-file .env -f infrastructure/docker/compose/train.nvidia.yml run --rm tensorflow-keras`
- Benchmark Docker: `docker compose --env-file .env -f infrastructure/docker/compose/benchmark.nvidia.yml run --rm benchmark`

## Portas

Cada uso tem a sua porta no host, e nenhuma se repete. Os valores em Python ficam em
`app/core/config/ports.py`, e `tests/unit/test_ports.py` confere se compose, scripts e este guia
continuam iguais a eles.

| Porta no host | Quem usa | Como mudar |
| --- | --- | --- |
| 7860 | Inferência Docker NVIDIA (`inference.nvidia.yml`) | `GRADIO_PORT_NVIDIA` no `.env` |
| 7861 | Inferência Docker CPU (`inference.cpu.yml`) | `GRADIO_PORT_CPU` no `.env` |
| 7862 | Aplicação local, sem Docker: `python main.py --gradio`, `make dev` e `start.* test` | `--port` ou `PORT` |
| 7863 em diante | Livres, para uma instância manual | Escolha ao subir |
| 8000 | `mkdocs serve` | `mkdocs serve -a 127.0.0.1:<porta>` |
| 17860 | `docker_release.py check nvidia` | `--porta` |
| 17861 | `docker_release.py check cpu` | `--porta` |

Dentro de qualquer container, a aplicação escuta sempre na 7860. É essa porta que aparece no
`CMD` das imagens, no health check, no `proxy_pass` de um proxy na mesma rede Docker e no
`app_port` do Hugging Face Spaces. `GRADIO_SERVER_PORT` também é essa porta interna: mudá-la não
muda a porta publicada. Treino e benchmark não publicam nenhuma porta.

`make up`, `start.bat` e `start.sh` mostram o endereço que o Compose publicou. Para consultá-lo:

```bash
docker compose --env-file .env -f infrastructure/docker/compose/inference.cpu.yml port inference-api 7860
```

`python tools/ops/doctor.py` mostra se a porta local está livre e quem está nas portas dos perfis
Docker. Antes desse mapa, os perfis usavam `GRADIO_PORT`, que deixou de ser lida. Se o seu `.env`
define um valor diferente de 7860, passe-o para a variável do perfil.

### Uma instância por vez sobre o mesmo `data/`

Portas diferentes permitem subir mais de uma instância, mas todas usam o mesmo `data/app.db`, e
essa combinação não é segura. O modo WAL do SQLite exige processos na mesma máquina, e um processo
no Windows e um container na VM do WSL estão em máquinas diferentes. Dois containers sobre o mesmo
banco também não foram validados.

| Ao subir | O que acontece com outra instância no ar |
| --- | --- |
| `make up`, `start.bat` ou `start.sh`, em um perfil Docker | Recusa se o outro perfil estiver rodando |
| `python main.py --gradio`, `make dev` ou `start.* test` | Recusa se um perfil Docker responder na sua porta e `DATABASE_URL` for o `data/app.db` |
| `up` direto do Compose ou **Start** no Docker Desktop | Não verifica |
| Um perfil Docker com a aplicação local já aberta | Não verifica |

Para usar outra instância ao mesmo tempo, aponte `DATABASE_URL` da aplicação local para outro
arquivo, ou troque o volume `data/` de um perfil Docker com um override do Compose.

## Credenciais

Há uma única fonte: o `.env` da raiz, que não vai para o Git nem para as imagens.

| Variável | Para quê | Quem lê |
| --- | --- | --- |
| `XFAKE_GRADIO_USERNAME` e `XFAKE_GRADIO_PASSWORD` | Login da interface | Docker NVIDIA, Docker CPU e aplicação local |
| `SPEECHCHECKER_API_KEY` | Header `X-API-Key` da API REST | Docker NVIDIA, Docker CPU e aplicação local |

O Compose entrega os valores ao criar o container, e o banco não guarda contas de usuário. Por isso
reconstruir a imagem, recriar o container ou voltar com `rollback` mantém o mesmo login. Sem esses
valores, os perfis de inferência recusam subir: não existe senha de reserva. Treino e benchmark não
têm login.

Para trocar, edite o `.env` e recrie o container, porque `restart` não relê o arquivo:

```bash
docker compose --env-file .env -f infrastructure/docker/compose/inference.nvidia.yml up -d --no-build --force-recreate inference-api
```

Guarde uma cópia em um gerenciador de senhas: o `.env` só existe nesta máquina, e `git clean -x` o
apagaria. As únicas outras credenciais são descartáveis: `docker_release.py check` gera valores
aleatórios a cada execução, para um container sem dados, e a CI usa valores fictícios.

## Tamanho e tempo de build

Valores medidos em 2026-11-03 no Docker Desktop com WSL2, um build por vez:

| Imagem | Tamanho | Build completo |
| --- | --- | --- |
| `speechchecker/inference-api:nvidia` | 16,7 GB | 13 min |
| `speechchecker/inference-api:cpu` | 15,4 GB | 9 min |

| `speechchecker/pytorch-audio:nvidia` | 16,2 GB | 13 min |
| `speechchecker/ssl-transformers:nvidia` | 16,3 GB | 14 min |
| `speechchecker/tensorflow-keras:nvidia` | 13,7 GB | 9 min |
| `speechchecker/pytorch-audio:cpu` | 13,6 GB | 29 min |
| `speechchecker/ssl-transformers:cpu` | 13,7 GB | 16 min |
| `speechchecker/tensorflow-keras:cpu` | 5,4 GB | 4 min |
| `speechchecker/classical-ml:cpu` | 1,8 GB | 1 min |

`speechchecker/benchmark:nvidia` sai do mesmo Dockerfile da inferência NVIDIA. O tempo depende da
conexão: em 2026-11-02, a 2,5 MB/s, a imagem NVIDIA levou 68 min. A retirada do cache do pip reduziu
os tamanhos de inferência de 30,3 e 22,7 GB. Com o armazenamento containerd do Docker Desktop, o
tamanho informado por `docker images` é bem maior que os arquivos da imagem, porque inclui também as
camadas comprimidas.

Reserve o espaço no disco do `docker_data.vhdx`, que fica em `C:` por padrão. O arquivo cresce
durante os builds, mesmo com espaço livre dentro dele, e não encolhe sozinho quando imagens são
removidas. Em 2026-11-03 ele chegou a 160 GB com 125 GB em uso. Para manter todas as imagens, deixe
esse arquivo em um disco com folga, em **Settings > Resources > Advanced > Disk image location**.

**Mudar só o código pode refazer a instalação.** Depois de construir a NVIDIA e a CPU nessa ordem,
só a camada do `pip install` da CPU continuou no cache de build. A causa provável é o limite de
20 GB de `builder.gc.defaultKeepStorage`, padrão do Docker Desktop, que comporta a camada de uma
imagem, mas não a das duas. Quem alterna entre as imagens pode aumentar o limite em **Settings >
Docker Engine**, por exemplo para `60GB`. O efeito desse ajuste ainda não foi medido.

## Versões fixadas

Cada imagem de inferência instala as versões exatas de
`infrastructure/docker/environments/inference-api/constraints-<perfil>.txt`, geradas da imagem
validada. Os pacotes `nvidia-*-cu12` ficam fora desses arquivos, porque `tensorflow[and-cuda]` e
`torch` pedem versões diferentes. O `torch` fixa quase todos, e o `Dockerfile.nvidia` fixa
`nvidia-cudnn-cu12` e `nvidia-cuda-nvcc-cu12`. A imagem base e os pacotes do `apt` não são fixados.

Para atualizar uma dependência:

1. Mude a versão no arquivo `constraints-<perfil>.txt` e, para `tensorflow` ou `nvidia-*`, no Dockerfile.
2. Reconstrua a imagem e valide os modelos.
3. Regrave o arquivo a partir da imagem validada:

```bash
python tools/ops/docker_release.py lock nvidia
```

## Atualizar e voltar atrás

Reconstruir sobrescreve a tag da imagem, e a versão anterior deixa de existir. Guarde-a antes de
atualizar. Com o container parado, `snapshot` também copia o banco para `data/backups/`:

```bash
docker compose --env-file .env -f infrastructure/docker/compose/inference.nvidia.yml down
```

```bash
python tools/ops/docker_release.py snapshot nvidia
```

```bash
docker compose --env-file .env -f infrastructure/docker/compose/inference.nvidia.yml up --build -d inference-api
```

Para confirmar que a nova imagem responde como deveria, sem tocar em `data/`:

```bash
python tools/ops/docker_release.py check nvidia
```

Se a nova versão falhar, volte para a imagem guardada. O comando recria o container e só termina
com sucesso depois que o health check passa:

```bash
python tools/ops/docker_release.py rollback nvidia
```

Pelo Make, em Git Bash ou WSL: `make down snapshot build up GPU=1` e `make rollback GPU=1`.

O que a volta não faz:

- **Banco:** continua como está, porque a aplicação só cria tabelas novas e nunca altera as
  existentes. Restaure só se a versão nova tiver danificado dados: pare o container e copie os
  arquivos `app.db*` de `data/backups/<pasta>/` para `data/`.
- **Modelos:** `data/models` é a pasta do host e não muda com a imagem. Um modelo novo incompatível
  com a imagem antiga precisa ser tratado à parte.

Só existe uma imagem guardada por perfil, em `<perfil>-previous`, e a que estava em uso ao voltar
fica em `<perfil>-reverted`. Cada uma ocupa disco até ser removida com `docker image rm`.

## Docker Desktop ao ligar a máquina

O Docker Desktop só inicia sozinho com a opção **Settings > General > Start Docker Desktop when you
sign in** ligada, e só depois do login no Windows. A opção vem da sua instalação e não é alterada
por este projeto. Quando o Docker sobe, `restart: unless-stopped` traz a aplicação de volta, a menos
que você a tenha parado.

Com o Docker parado, `start.bat` e `start.sh` tentam iniciá-lo com `docker desktop start`. Em Linux
sem Docker Desktop, esse comando não existe e o script só orienta a iniciar o serviço.
