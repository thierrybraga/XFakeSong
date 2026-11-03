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

- Inferência local: `python main.py --gradio`
- Inferência Docker: `docker compose --env-file .env -f infrastructure/docker/compose/inference.cpu.yml up --build inference-api`
- Treino Docker: `docker compose --env-file .env -f infrastructure/docker/compose/train.nvidia.yml run --rm tensorflow-keras`
- Benchmark Docker: `docker compose --env-file .env -f infrastructure/docker/compose/benchmark.nvidia.yml run --rm benchmark`

## Tamanho e tempo de build

Valores medidos em 2026-11-03 no Docker Desktop com WSL2, um build por vez:

| Imagem | Tamanho | Build completo |
| --- | --- | --- |
| `speechchecker/inference-api:nvidia` | 16,7 GB | 13 min |
| `speechchecker/inference-api:cpu` | 15,4 GB | 9 min |

O tempo depende da conexão: em 2026-11-02, a 2,5 MB/s, a imagem NVIDIA levou 68 min. A retirada do
cache do pip reduziu os tamanhos de 30,3 e 22,7 GB. Com o armazenamento containerd do Docker
Desktop, o tamanho informado por `docker images` é bem maior que os arquivos da imagem, porque
inclui também as camadas comprimidas.

Reserve o espaço no disco do `docker_data.vhdx`, que fica em `C:` por padrão. O arquivo cresce
durante os builds e não encolhe sozinho quando imagens são removidas.

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
