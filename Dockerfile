# syntax=docker/dockerfile:1.7
# =====================================================================
# XFakeSong — Multi-stage Dockerfile
# =====================================================================
# Stage 1 (builder): instala dependências de compilação e gera wheels
# Stage 2 (runtime): imagem slim com apenas runtime libs + wheels
# Resultado: ~60% menor que single-stage, sem gcc/dev libs em produção
# =====================================================================

ARG PYTHON_VERSION=3.11
# PROD.7: TF_VARIANT controla qual requirements usar.
#   - "" (default): requirements.txt — TensorFlow padrão
#   - "cpu": requirements-cpu.txt — tensorflow-cpu, ~450MB menor.
#   - "gpu": requirements.txt + tensorflow[and-cuda] — CUDA wheels para Linux NVIDIA.
# Build:  docker build --build-arg TF_VARIANT=cpu .
ARG TF_VARIANT=""

# ---------- Stage 1: BUILDER ----------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TF_VARIANT

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# Dependências de COMPILAÇÃO (só no builder, não na imagem final)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libsndfile1-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Aproveita cache de layer: requirements raramente muda comparado ao código.
# PROD.7: copia ambos os requirements; seleciona qual instalar via TF_VARIANT.
COPY requirements.txt requirements-base.txt requirements-cpu.txt* ./

# Cria virtualenv isolado e instala dependências
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    if [ "${TF_VARIANT}" = "cpu" ] && [ -f requirements-cpu.txt ]; then \
        echo "[Dockerfile] Usando requirements-cpu.txt (tensorflow-cpu)"; \
        pip install -r requirements-cpu.txt; \
    elif [ "${TF_VARIANT}" = "gpu" ]; then \
        echo "[Dockerfile] Usando requirements.txt + tensorflow[and-cuda]"; \
        pip install -r requirements.txt; \
        pip install 'tensorflow[and-cuda]>=2.16,<2.22'; \
        # ---- Conflito de cuDNN (corrigido) ---------------------------------
        # torch fixa `nvidia-cudnn-cu12==9.1.0.70`, mas TF 2.21 foi compilado
        # com cuDNN 9.3 e EXIGE >=9.3.0.75. Sem o upgrade abaixo, a GPU falha
        # com: "Loaded runtime CuDNN library: 9.1.0 but source was compiled
        # with: 9.3.0 ... No DNN in stream executor". O cuDNN 9.x é
        # retrocompatível com o torch (validado: torch conv na GPU OK com
        # 9.23). `--no-deps` evita que o pip rebaixe outras libs por causa do
        # pino do torch; o aviso de incompatibilidade do torch é cosmético.
        pip install --upgrade --no-deps 'nvidia-cudnn-cu12>=9.3.0.75,<10'; \
        # ptxas/nvlink p/ XLA: sem o nvcc, o TF GPU falha com "No PTX
        # compilation provider is available" ao compilar kernels XLA no fit().
        # O PATH/XLA_FLAGS (runtime) já apontam p/ nvidia/cuda_nvcc.
        pip install --no-deps 'nvidia-cuda-nvcc-cu12'; \
        # Sanidade: cuDNN >=9.3 E ptxas presente (senão TF GPU falha).
        python -c "import importlib.metadata as m; v=m.version('nvidia-cudnn-cu12'); j,n=(int(x) for x in v.split('.')[:2]); assert (j,n)>=(9,3), 'cuDNN '+v+' < 9.3'; print('[Dockerfile] cuDNN OK p/ TF GPU:', v)"; \
        ls /opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas; \
    else \
        echo "[Dockerfile] Usando requirements.txt (tensorflow padrão)"; \
        pip install -r requirements.txt; \
    fi

# =====================================================================
# ---------- Stage 2: RUNTIME ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Metadata OCI — útil para registries (GHCR, Docker Hub, etc.)
LABEL org.opencontainers.image.title="XFakeSong" \
      org.opencontainers.image.description="Deepfake audio detection platform" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/thierrybraga/XFakeSong" \
      org.opencontainers.image.documentation="https://github.com/thierrybraga/XFakeSong/tree/main/docs"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # TF: evita logs verbose, usa só CPU por default (override em runtime)
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    # Gradio (PROD.1): desabilita telemetria — evita timeout em ambientes
    # sem internet de saída + acelera boot.
    GRADIO_ANALYTICS_ENABLED=False \
    # PROD.4: temp dir explícito (default /tmp/gradio pode estar em tmpfs cheio)
    GRADIO_TEMP_DIR=/tmp/gradio \
    # Escuta em todas as interfaces
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    # Numba/librosa cache em diretório writable
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    # Backend matplotlib não-interativo (Docker headless)
    MPLBACKEND=Agg \
    # TensorFlow GPU via tensorflow[and-cuda] instala CUDA/cuDNN no venv.
    # O runtime slim precisa desses caminhos explícitos para carregar as libs.
    NVIDIA_PYPI_LIB_DIR=/opt/venv/nvidia \
    LD_LIBRARY_PATH="/opt/venv/nvidia/cuda_nvrtc/lib:/opt/venv/nvidia/cuda_cupti/lib:/opt/venv/nvidia/cudnn/lib:/opt/venv/nvidia/nccl/lib:/opt/venv/nvidia/cusolver/lib:/opt/venv/nvidia/cuda_runtime/lib:/opt/venv/nvidia/cublas/lib:/opt/venv/nvidia/nvjitlink/lib:/opt/venv/nvidia/curand/lib:/opt/venv/nvidia/cufft/lib:/opt/venv/nvidia/cusparse/lib" \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/venv/nvidia/cuda_nvcc \
    PATH="/opt/venv/bin:/opt/venv/nvidia/cuda_nvcc/bin:$PATH"

# Apenas RUNTIME libs (sem gcc/dev) — imagem final menor
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        curl \
        tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Cria usuário não-root ANTES de criar diretórios
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --system --gid ${APP_GID} appuser && \
    useradd --system --uid ${APP_UID} --gid appuser --shell /bin/bash --create-home appuser

# Copia venv pronto do builder (sem precisar reinstalar)
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Caminho estável para os pacotes NVIDIA do tensorflow[and-cuda].
RUN set -eux; \
    nvidia_dir="$(find /opt/venv/lib -path '*/site-packages/nvidia' -type d | head -n 1)"; \
    if [ -n "${nvidia_dir}" ]; then \
        ln -s "${nvidia_dir}" /opt/venv/nvidia; \
    fi

WORKDIR /app

# Cria estrutura de diretórios com permissão correta UMA VEZ
# PROD.4: /tmp/gradio também aqui (Gradio cria sob demanda mas pode falhar
# em ambientes com /tmp restritivo).
RUN mkdir -p \
        /app/logs \
        /app/app/models \
        /app/app/results \
        /app/data/fake \
        /app/data/real \
        /tmp/numba_cache \
        /tmp/matplotlib \
        /tmp/huggingface \
        /tmp/gradio \
    && chown -R appuser:appuser \
        /app \
        /tmp/numba_cache /tmp/matplotlib /tmp/huggingface /tmp/gradio

# Entrypoint
COPY --chown=appuser:appuser docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Código da aplicação (em layer separada do venv para cache eficiente)
COPY --chown=appuser:appuser . .

# Drop para usuário não-privilegiado (sem gosu)
USER appuser

EXPOSE 7860

# Healthcheck no Dockerfile (PROD.2 + PROD.9):
# - Usa /api/v1/system/health (responde rápido SEM precisar do Gradio carregado)
# - Fallback para raiz / (Gradio HTML)
# - start_period 180s permite TF + Keras + load de modelos terminar
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsSL -o /dev/null --max-time 5 \
            http://127.0.0.1:7860/api/v1/system/health \
        || curl -fsSL -o /dev/null --max-time 5 \
            http://127.0.0.1:7860/ \
        || exit 1

# tini como PID 1 — handle correto de signals (SIGTERM, SIGINT)
ENTRYPOINT ["tini", "--", "docker-entrypoint.sh"]

CMD ["python", "main.py", "--gradio", "--gradio-port", "7860"]
