# Instalação e Configuração

Guia completo para executar o XFakeSong em **Local (Python)**, **Docker** e **Hugging Face Spaces**, com seções específicas para Windows, Linux e GPU.

---

## 📋 Pré-requisitos

| Componente | Versão mínima | Notas |
|------------|---------------|-------|
| **Python** | 3.11 | Referência (testado); compatível 3.11–3.13 — 3.13 requer TF 2.20+ |
| **pip** | 23+ | `python -m pip install --upgrade pip` |
| **Docker Desktop** | 4.x (Compose v2) | Para modo produção |
| **NVIDIA Driver** | 535+ | Apenas para GPU (Windows: via WSL2) |
| **Git LFS** | qualquer | Para clonar modelos grandes |
| **Make** (opcional) | 3.81+ | Habilita `make <target>` |

---

## 🚀 Quick start — Windows

### Cenário 1: Docker (recomendado, isola dependências)

```cmd
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
start.bat
```

No menu, escolha **[2] Modo PRODUCAO**. Aguarde o healthcheck (~60–90s na primeira vez — TensorFlow carregando). Acesse http://localhost:7860.

### Cenário 2: Python local (mais rápido para iterar)

```cmd
start.bat install
start.bat test
```

### Cenário 3: GPU NVIDIA (Windows 11 + WSL2)

```cmd
start.bat gpu
```

Pré-requisitos: WSL2 GPU Support habilitado no Docker Desktop, NVIDIA driver 535+ no Windows.

### Cenário 4: GPU NVIDIA — execução local via WSL2 Ubuntu (sem Docker)

Recomendado para **treinos longos** com a GPU diretamente acessível pelo
Python, sem overhead de Docker. TF ≥ 2.11 **não suporta GPU em Windows nativo**
— WSL2 é o único caminho prático.

```cmd
:: 1. No Windows (PowerShell admin) — instala WSL2 + Ubuntu
wsl --install -d Ubuntu

:: 2. Atualize o driver NVIDIA no WINDOWS host >= 535
::    https://www.nvidia.com/drivers
::    (NÃO instale driver dentro do WSL — o WSL2 herda a GPU via passthrough.
::     TF 2.16+ embute CUDA 12.x, que exige driver >= 535)
```

Após reiniciar, abra o Ubuntu pelo menu Iniciar e:

```bash
# 3. Dentro do Ubuntu WSL2
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
chmod +x start.sh

# 4. Instala TensorFlow com CUDA embutido (tensorflow[and-cuda]) + deps:
./start.sh cuda

# 5. (opcional) Valida que a GPU está visível ao TensorFlow:
./start.sh gpu-config

# 6. Inicia a app usando a GPU:
./start.sh gpu-local
```

O `./start.sh cuda` fará:
1. Verifica `nvidia-smi` (driver Windows via passthrough WSL2)
2. Cria `.venv` e instala `tensorflow[and-cuda]` (CUDA/cuDNN embutidos no wheel)
3. Instala `requirements.txt`

E `./start.sh gpu-local` valida o CUDA e inicia o app em `http://localhost:7860`
(usando a GPU automaticamente).

Performance esperada (RTX 3060 12 GB):
- Treino AASIST (10 epochs, 4600 amostras): **~3 min** (vs ~30 min CPU)
- Inferência por áudio: **~40 ms** (vs ~200 ms CPU)

---

## 🚀 Quick start — Linux / macOS

```bash
git clone https://github.com/thierrybraga/XFakeSong.git
cd XFakeSong
chmod +x start.sh
./start.sh prod         # Docker
./start.sh test         # Python local (usa GPU se disponível)
./start.sh gpu          # Docker + GPU
./start.sh install-gpu  # Instala TF + CUDA local (não-Docker)
./start.sh gpu-test     # Diagnóstico GPU standalone
```

Ou via `make`:
```bash
make install            # Cria .venv e instala
make up                 # docker compose up -d
make logs               # tail logs
make down               # para containers
```

---

## 📦 Comandos disponíveis (start.bat / start.sh)

| Comando | Descrição |
|---------|-----------|
| `test` | Roda Python local em `.venv` — usa GPU se disponível |
| `prod` | Sobe Docker em modo produção |
| `gpu` | Sobe Docker + GPU (NVIDIA) |
| `cuda` / `install-gpu` | Instala `tensorflow[and-cuda]` + deps em `.venv` (Linux/WSL2) |
| `gpu-local` (`wsl-gpu`) | Roda local usando a GPU (instala TF CUDA se faltar) |
| `gpu-config` | Reconhece a placa e valida se o TF enxerga a GPU |
| `nvidia-driver` | Instala driver NVIDIA no Linux (no WSL2, instale no Windows host) |
| `stop` | Para containers |
| `logs` | Tail dos logs do container (`-f`) |
| `rebuild` | Force rebuild sem cache |
| `clean` | Limpeza profunda (volumes + prune) |
| `status` | `compose ps` + `docker stats` |
| `install` | Instala dependências em `.venv` (CPU) |
| `bootstrap` | Cria estrutura de diretórios padrão |
| `deploy` | Push para Hugging Face Spaces |
| `help` | Mostra ajuda |

Sem argumento, ambos abrem **menu interativo**.

### GPU em Windows nativo — limitações

A partir do **TensorFlow 2.11**, **GPU não é mais suportado em Windows nativo**.
Wheels Windows são CPU-only. Caminhos viáveis para GPU em Windows:

| Caminho | Status | Performance |
|---------|--------|-------------|
| **WSL2 + Ubuntu + tensorflow[and-cuda]** | ✅ Recomendado | 100% CUDA, full speed |
| **Docker Desktop + GPU passthrough** | ✅ Suportado (`start.bat gpu`) | 100% CUDA |
| **tensorflow-directml-plugin** | ⚠ Requer Python ≤3.10 + TF 2.10 | ~60% velocidade CUDA |
| **TF Windows nativo (sem CUDA)** | ✗ CPU only | Funcional, mas lento |

O **Dashboard → 🎮 Diagnóstico de GPU** mostra qual situação você está e dá as instruções acionáveis.

---

## 🐳 Docker — Detalhes

### Imagem multi-stage

O `Dockerfile` usa **multi-stage build**:
1. **Stage builder**: instala `gcc`, `build-essential`, compila wheels Python.
2. **Stage runtime**: apenas runtime libs (`ffmpeg`, `libsndfile1`, `tini`, `curl`) + venv pronto do builder.

Resultado: imagem final ~60% menor que single-stage. Sem `gcc` em produção.

### docker-compose

```yaml
# Comando base
docker compose up -d

# Com GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Forçar rebuild
docker compose build --no-cache --pull

# Logs
docker compose logs -f app

# Stop e cleanup
docker compose down -v --remove-orphans
```

**Importante**: usar `docker compose` (v2, espaço) e não `docker-compose` (v1, hífen). Os scripts `start.bat`/`start.sh` detectam ambos automaticamente.

### Limites de recursos

Editáveis via `.env`:
```env
DOCKER_MEMORY_LIMIT=8G       # Default 8GB
DOCKER_CPU_LIMIT=4.0         # Default 4 CPUs
GRADIO_PORT=7860             # Porta exposta
```

Para datasets grandes ou treinamento, aumente `DOCKER_MEMORY_LIMIT` (ex: 16G).

---

## 🪟 Troubleshooting Windows

### "Docker daemon não está rodando"
Abra **Docker Desktop**. Aguarde o ícone ficar verde na bandeja. Tente novamente.

### Volumes não persistem ou são read-only
1. Docker Desktop → **Settings → Resources → File Sharing**
2. Adicione o drive onde o projeto está clonado (ex: `D:\`)
3. **Apply & Restart**

### `ERROR: No matching distribution found for tensorflow<2.20,>=2.16`

**Sintoma**: ao rodar `pip install -r requirements.txt` ou `start.bat install`:
```
ERROR: Could not find a version that satisfies the requirement tensorflow<2.20,>=2.16
       (from versions: 2.20.0rc0, 2.20.0, 2.21.0rc0, 2.21.0rc1, 2.21.0)
ERROR: No matching distribution found for tensorflow<2.20,>=2.16
```

**Causa**: você está em **Python 3.13**. TensorFlow só ganhou wheels para Python 3.13 a partir da versão **2.20**. Versões 2.16-2.19 só têm wheels para Python 3.9-3.12.

**Fix**: já corrigido no `requirements.txt` (upper bound subido para `<2.22`). Se sua cópia ainda tem `<2.20`:

```bash
# Opção A: atualizar requirements (recomendado)
git pull   # se sincronizado com upstream
pip install -r requirements.txt

# Opção B: instalar manualmente
pip install 'tensorflow>=2.16,<2.22' 'numba>=0.60'
pip install -r requirements.txt
```

Se preferir Python 3.11/3.12 (TF mais estável):
```bash
# Windows:
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### `TypeError: unhashable type: 'dict'` ao acessar `/` no browser

**Sintoma**: API funciona (`/api/v1/system/health` retorna 200) mas o Gradio (`/`) retorna HTTP 500 com stack trace incluindo:
```
File ".../gradio/routes.py", line 432, in main
    return templates.TemplateResponse(...)
...
File ".../jinja2/utils.py", line 515, in __getitem__
    rv = self._mapping[key]
TypeError: unhashable type: 'dict'
```

**Causa**: incompatibilidade entre versões.
- Gradio < 4.31 chama `templates.TemplateResponse(context_dict, name, ...)` (API antiga)
- Starlette ≥ 0.36 (puxado por `fastapi>=0.110`) mudou para `(request, name, context, ...)` (API nova)
- O dict é interpretado como `template_name` → Jinja2 falha no cache key (dict não é hashable).

**Fix imediato**:
```bash
# Atualizar requirements (já feito a partir do commit BUG.Render.1):
pip install --upgrade 'gradio>=4.31,<5.0'

# OU em Docker, força rebuild:
docker compose build --no-cache --pull
docker compose up -d
```

**Verificação**: ao subir o app, o log inicial deve mostrar uma linha:
```
INFO ... Version check: gradio=4.36.1, starlette=0.40.0, fastapi=0.115.0, jinja2=3.1.5
```

Se aparecer `ERROR ... INCOMPATIBILIDADE CRÍTICA: gradio==X.Y.Z ...`, **siga o fix acima** — o servidor sobe mas falhará na primeira request à raiz.

---

### `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`

**Sintoma**: o app **nem inicia** — falha logo no `import gradio` com:
```
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
```

**Causa**: o `huggingface_hub` **1.0** removeu a classe `HfFolder`, que o **gradio 4.x**
ainda importa. Um `pip install` sem teto de versão (`huggingface_hub>=0.20`) puxa a
linha 1.x e quebra a UI inteira — afeta qualquer instalação nova, **inclusive Linux**.

**Fix**:
```bash
pip install 'huggingface_hub>=0.25,<1.0'
```
O `requirements.txt` já fixa `huggingface_hub>=0.25,<1.0` (teto firme `<1.0`; piso
`0.25` é a exigência do `datasets`, então a faixa satisfaz **gradio E datasets**).

**Verificação**: `python scripts/doctor.py` mostra na seção *Compatibilidade de versões*:
```
OK  gradio=4.44.1, huggingface_hub=0.36.2 (compatíveis)
```
Se aparecer `IMPORT QUEBRADO` em `gradio` na seção *Dependências* ou
`INCOMPATIBILIDADE CRÍTICA: ... HfFolder ...` no log de startup, aplique o fix acima.

---

### Diagnóstico rápido do ambiente (incl. GPU)

Antes de abrir um problema, rode o doctor — ele cobre Python, dependências (distinguindo
*ausente* de *import quebrado*), compatibilidade de versões, **GPU/CUDA** (hardware NVIDIA,
visibilidade pelo TensorFlow e dicas acionáveis por SO) e porta:
```bash
python scripts/doctor.py          # diagnóstico completo
python scripts/doctor.py --fix    # tenta reinstalar deps faltantes
```
No Linux/WSL o `./start.sh doctor` faz o equivalente em shell (e checa Docker + toolkit).

---

### "Error response from daemon: cannot create container"
Provavelmente conflito de porta. Verifique se algo está usando 7860:
```cmd
netstat -ano | findstr :7860
```
Para mudar a porta:
```cmd
set GRADIO_PORT=8080
start.bat prod
```

### Build extremamente lento (10+ min)
- Confira que o `.dockerignore` existe (criado neste projeto)
- Docker Desktop → **Settings → Resources** — aumente CPUs/RAM
- Use WSL2 backend (mais rápido que Hyper-V)

### `start.bat` mostra caracteres estranhos no console
O script força `chcp 65001` (UTF-8) no topo. Se ainda assim aparecer, use **Windows Terminal** (recomendado) em vez de cmd.exe legado.

### Line endings CRLF causando erros no entrypoint
Se editar `docker-entrypoint.sh` no Notepad/VSCode no Windows e gerar CRLF:
```cmd
git config --global core.autocrlf input
git rm --cached docker-entrypoint.sh
git checkout -- docker-entrypoint.sh
```

Ou adicione `.gitattributes`:
```
*.sh text eol=lf
```

### GPU não detectada (NVIDIA)
1. Verifique driver no host:
   ```cmd
   nvidia-smi
   ```
2. Habilite GPU no WSL2 (Docker Desktop → Settings → Resources → WSL Integration)
3. Teste:
   ```cmd
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```
4. Use `start.bat gpu` (carrega `docker-compose.gpu.yml`)

---

## 🐧 Troubleshooting Linux

### Permissões em volumes montados
O container roda como `appuser` (UID 1000). Se seus diretórios estiverem com outro owner:
```bash
sudo chown -R 1000:1000 ./app/models ./app/results ./logs ./data
```

### GPU NVIDIA — Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 🛠️ Instalação manual (sem scripts)

### Python local
```bash
python -m venv .venv
source .venv/bin/activate           # Linux/Mac
.venv\Scripts\activate              # Windows

pip install --upgrade pip
pip install -r requirements.txt              # produção
pip install -r requirements-dev.txt          # + dev tools

python main.py --bootstrap-dirs              # cria diretórios
python main.py --gradio --gradio-port 7860   # inicia
```

### Docker manual
```bash
docker build -t xfakesong:latest .
docker run -d \
    --name xfakesong_app \
    -p 7860:7860 \
    -v $(pwd)/app/models:/app/app/models \
    -v $(pwd)/app/results:/app/app/results \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    xfakesong:latest
```

---

## 🤗 Deploy Hugging Face Spaces

O deploy recomendado é **Docker Space**, com modelos treinados em um
repositório separado do tipo **Model** no Hugging Face Hub. O SDK Gradio simples
não é indicado para o app completo, porque não oferece o mesmo controle sobre
TensorFlow, PyTorch, FFmpeg, GPU, healthcheck e Storage.

### Resumo operacional

1. Crie um Space com **SDK: Docker**.
2. Use `sdk: docker` e `app_port: 7860` no `README.md`.
3. Envie os modelos de `app/models/` para um Model Repo:

```bash
python scripts/upload_models_to_hf.py \
  --repo-id SEU_USUARIO/xfakesong-models \
  --dry-run

python scripts/upload_models_to_hf.py \
  --repo-id SEU_USUARIO/xfakesong-models \
  --private
```

4. Configure no Space:

```env
MODEL_REPO_ID=SEU_USUARIO/xfakesong-models
ENABLE_TRAINING=false
XFAKE_SYNC_MODELS_ON_BOOT=true
DEEPFAKE_MODELS_DIR=app/models
DEEPFAKE_ENV=production
```

5. Se o Model Repo for privado, adicione `HF_TOKEN` como **Secret**.
6. Para demo com todos os modelos, prefira GPU **L4**; para demo leve, CPU
   Upgrade/T4 pode bastar.
7. Para persistir uploads, datasets, logs ou novos treinamentos, anexe Storage
   Bucket em `/data` e defina:

```env
XFAKE_STORAGE_DIR=/data
```

Guia completo: [Deploy Hugging Face](11_DEPLOY_HUGGINGFACE.md).

---

## 🧪 Testes

```bash
make test               # suíte rápida, exclui smoke
make test-cov           # cobertura app + benchmarks
# ou diretamente
./scripts/run_tests.sh fast
./scripts/run_tests.sh cov
```

---

## ⚙️ Variáveis de Ambiente (`.env`)

Copie `.env.example` para `.env`:
```bash
cp .env.example .env    # Linux
copy .env.example .env  # Windows
# ou
make env                # via Makefile
```

**Variáveis críticas:**

| Variável | Default | Descrição |
|----------|---------|-----------|
| `DEEPFAKE_ENV` | `production` | `development` ou `production` |
| `DEEPFAKE_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `GRADIO_PORT` | `7860` | Porta exposta (compose) |
| `GRADIO_SERVER_PORT` | `7860` | Porta interna do Gradio (container) |
| `DOCKER_MEMORY_LIMIT` | `8G` | Limite RAM do container |
| `DOCKER_CPU_LIMIT` | `4.0` | Limite CPUs do container |
| `NVIDIA_GPU_COUNT` | `all` | Apenas com `docker-compose.gpu.yml` |

---

## 📁 Estrutura de diretórios (auto-criada)

```
XFakeSong/
├── app/
│   ├── models/         # Modelos treinados (.keras, .onnx, scaler.pkl) [VOLUME]
│   ├── results/        # Resultados de inferência batch [VOLUME]
│   └── ...
├── data/
│   ├── real/           # Áudios genuínos [VOLUME]
│   └── fake/           # Áudios sintéticos [VOLUME]
├── logs/               # Logs da aplicação [VOLUME]
├── .venv/              # Virtualenv local (ignorado no Docker)
├── Dockerfile          # Multi-stage build
├── docker-compose.yml          # Base
├── docker-compose.gpu.yml      # Override GPU
├── docker-entrypoint.sh        # Entrypoint do container
├── Makefile                    # Comandos uniformes
├── start.bat / start.sh        # Launchers
├── requirements.txt            # Deps runtime
├── requirements-dev.txt        # Deps dev (testes, optuna, mlflow, onnx)
└── .env                        # Suas configurações (criar a partir do .env.example)
```

---

## 🔄 Atualizando

```bash
git pull
make rebuild            # Linux/Mac/WSL com Make
# ou
start.bat rebuild       # Windows
./start.sh rebuild      # Linux/Mac sem Make
```

---

## 💡 Dicas de produção

- **Memory**: ML models pesados (Ensemble, WavLM) requerem 8GB+. Ajuste `DOCKER_MEMORY_LIMIT`.
- **Healthcheck**: configurado com `start_period: 180s` (TF + 14 modelos demoram em CPU) e `interval: 30s`. Usa `/api/v1/system/health` (não depende do Gradio renderizado).
- **Logs**: rotação automática (10MB × 5 arquivos) via `logging.options` no compose.
- **Signal handling**: `tini` como PID 1 garante shutdown limpo do Gradio em `docker stop`.
- **Não-root**: container roda como `appuser` (UID 1000), sem privilégios elevados.
- **Multi-stage**: imagem final tem só runtime libs (sem `gcc`/`build-essential`), reduzindo superfície de ataque.
- **Build CPU-only**: para ~450MB menor, use `docker build --build-arg TF_VARIANT=cpu .` (usa `tensorflow-cpu` em vez do default com CUDA libs).
- **Gradio analytics**: desabilitada via `GRADIO_ANALYTICS_ENABLED=False` — evita timeout de boot em ambientes sem internet de saída.
- **Reverse proxy + WebSocket**: Gradio queue usa WS. Em nginx/Caddy/Traefik, garanta upgrade headers:
  ```nginx
  location / {
      proxy_pass http://xfakesong:7860;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_set_header Host $host;
      proxy_read_timeout 86400;  # WS long-poll fallback
  }
  ```
- **TrustedHost**: se setar `ALLOWED_HOSTS` para domínios específicos, o middleware **automaticamente** inclui `127.0.0.1`, `localhost`, `::1` para o healthcheck do Docker continuar funcionando.
