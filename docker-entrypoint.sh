#!/usr/bin/env bash
# =====================================================================
# XFakeSong — Docker container entrypoint
# Executado como appuser (UID/GID 1000). PID 1 é tini.
# =====================================================================
set -euo pipefail

# Diretórios mínimos necessários — idempotente, evita falha em re-run
for dir in \
    /app/logs \
    /app/app/models \
    /app/app/results \
    /app/data/fake \
    /app/data/real \
    /tmp/numba_cache \
    /tmp/matplotlib \
    /tmp/huggingface
do
    mkdir -p "${dir}" 2>/dev/null || true
done

# Verifica writability dos volumes montados (problema comum no Windows WSL2)
for dir in /app/logs /app/app/models /app/app/results /app/data; do
    if [ ! -w "${dir}" ]; then
        echo "[entrypoint] WARN: ${dir} is not writable. " \
             "On Windows with WSL2, ensure file sharing is enabled and " \
             "the directory has the correct permissions." >&2
    fi
done

# Log de início (útil para debug)
echo "[entrypoint] Starting XFakeSong as $(id -un) ($(id -u):$(id -g))"
echo "[entrypoint] DEEPFAKE_ENV=${DEEPFAKE_ENV:-development}"
echo "[entrypoint] GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-7860}"

# Hugging Face Spaces/demo: sincroniza modelos treinados do Hub quando
# MODEL_REPO_ID/XFAKE_MODEL_REPO_ID estiver definido. Sem essa variável, é no-op
# e usa os artefatos já empacotados/montados em app/models.
if [ "${XFAKE_SYNC_MODELS_ON_BOOT:-true}" != "false" ]; then
    python scripts/sync_hf_models.py || {
        echo "[entrypoint] WARN: model sync failed; continuing with local files." >&2
    }
fi

# Hook opcional: comando customizado no env BOOT_HOOK (ex: "python migrate.py")
# BUG FIX: eval "${BOOT_HOOK}" permite injeção de código arbitrário quando
# BOOT_HOOK vem de variável de ambiente (ex: "rm -rf /"). Usar bash -c é mais
# seguro pois restringe a um único subshell sem acesso às funções do entrypoint.
if [ -n "${BOOT_HOOK:-}" ]; then
    echo "[entrypoint] Running BOOT_HOOK: ${BOOT_HOOK}"
    bash -c "${BOOT_HOOK}"
fi

# exec garante que o comando substitui o shell (PID 1 funciona corretamente)
exec "$@"
