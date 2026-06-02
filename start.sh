#!/usr/bin/env bash
# ==============================================================================
# XFakeSong Launcher v4.0
# Plataforma operacional unificada
# Compatível: Ubuntu/Debian • Linux • macOS • WSL2
# ==============================================================================
#
# Uso: ./start.sh [comando]
#
#   EXECUÇÃO
#   test        Executa localmente (Python + venv)
#   gpu-local   Linux/WSL2 + GPU: local com TF CUDA nativo
#   prod        Docker (CPU)
#   gpu         Docker + GPU (nvidia-container-toolkit)
#
#   SETUP UBUNTU (privilegiado — usa apt/sudo)
#   setup           Setup COMPLETO Ubuntu + roda local (one-shot otimizado)
#   setup-system    Instala dependências de sistema (apt: ffmpeg, libsndfile…)
#   setup-python    Instala/configura Python 3.11+ e cria o .venv
#   nvidia-driver   Instala drivers NVIDIA (ubuntu-drivers autoinstall)
#   cuda            Instala CUDA/TensorFlow-GPU no .venv (tensorflow[and-cuda])
#   gpu-config      Configura e reconhece a placa de vídeo (driver + TF CUDA)
#
#   INSTALAÇÃO PYTHON
#   install         Instala dependências Python no .venv
#   bootstrap       Cria a estrutura de diretórios do projeto
#
#   DOCKER
#   logs / stop / rebuild / status
#
#   MANUTENÇÃO
#   doctor      Diagnóstico completo do ambiente
#   clean       Deep clean Docker (volumes + cache)
#   help        Exibe esta ajuda
#   (nenhum)    Menu interativo
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ==============================================================================
# METADATA
# ==============================================================================

readonly VERSION="4.0.0"
readonly APP_NAME="XFakeSong"
readonly DEFAULT_PORT="7860"
readonly HEALTH_TIMEOUT="300"
readonly HEALTH_INTERVAL="5"
readonly CONTAINER_NAME="xfakesong_app"
readonly PY_MIN_MINOR="10"   # Python 3.10+ aceitável; 3.11+ recomendado

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"
readonly LOG_DIR="${PROJECT_ROOT}/.logs"
readonly LOG_FILE="${LOG_DIR}/launcher.log"

# ==============================================================================
# COLORS
# ==============================================================================

if [[ -t 1 ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[0;34m'
    readonly CYAN='\033[0;36m'
    readonly BOLD='\033[1m'
    readonly NC='\033[0m'
else
    readonly RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

# ==============================================================================
# LOGGING
# ==============================================================================

mkdir -p "$LOG_DIR"

log() {
    local level="$1"; shift
    local ts; ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] [$level] $*" >> "$LOG_FILE"
}

info()    { log INFO    "$@"; echo -e "${BLUE}[INFO]${NC} $*"; }
success() { log SUCCESS "$@"; echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { log WARN    "$@"; echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { log ERROR   "$@"; echo -e "${RED}[ERROR]${NC} $*" >&2; }
fatal()   { error "$@"; exit 1; }

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

on_error() {
    local exit_code="$?"
    local line="${1:-?}"
    error "Falha na linha ${line} (exit code=${exit_code})"
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

# ==============================================================================
# CORE HELPERS
# ==============================================================================

require_bash() {
    [[ -n "${BASH_VERSION:-}" ]] || { echo "Este script requer Bash"; exit 1; }
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fatal "Dependência ausente: ${1}"
}

clear_screen() {
    command -v clear >/dev/null 2>&1 && clear || true
}

pause() {
    echo
    read -rp "Pressione ENTER para continuar..." _
}

is_interactive() { [[ -t 0 ]]; }

confirm() {
    # confirm "pergunta" → 0 se sim, 1 se não. Em não-interativo: assume sim.
    local prompt="${1:-Continuar?}"
    is_interactive || return 0
    local response
    read -rp "${prompt} [y/N] " response
    [[ "$response" =~ ^[Yy]$ ]]
}

is_wsl() {
    [[ -n "${WSL_DISTRO_NAME:-}" ]] && return 0
    [[ -f /proc/version ]] || return 1
    grep -qiE '(microsoft|wsl)' /proc/version
}

# ==============================================================================
# OS DETECTION + APT/SUDO (Ubuntu/Debian)
# ==============================================================================

OS_ID=""
OS_LIKE=""
OS_VERSION=""
OS_PRETTY=""

detect_os() {
    [[ -n "$OS_ID" ]] && return 0
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_LIKE="${ID_LIKE:-}"
        OS_VERSION="${VERSION_ID:-}"
        OS_PRETTY="${PRETTY_NAME:-$OS_ID}"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        OS_ID="macos"; OS_PRETTY="macOS"
    else
        OS_ID="unknown"; OS_PRETTY="$(uname -s)"
    fi
}

is_debian_like() {
    detect_os
    [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" || "$OS_LIKE" == *debian* ]]
}

require_apt() {
    is_debian_like || fatal "Este comando é específico para Ubuntu/Debian (apt). SO detectado: ${OS_PRETTY}"
    command -v apt-get >/dev/null 2>&1 || fatal "apt-get não encontrado."
}

# SUDO: vazio se já root, "sudo" caso contrário.
SUDO=""
ensure_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        SUDO=""
        return 0
    fi
    command -v sudo >/dev/null 2>&1 || \
        fatal "Operação requer privilégios de root. Instale 'sudo' ou rode como root."
    SUDO="sudo"
    # Aquece o cache de credenciais cedo (evita prompt no meio do apt)
    $SUDO -v || fatal "Falha na autenticação sudo"
}

APT_UPDATED=0
apt_update_once() {
    [[ "$APT_UPDATED" -eq 1 ]] && return 0
    info "Atualizando índice de pacotes (apt-get update)..."
    $SUDO apt-get update -y
    APT_UPDATED=1
}

apt_install() {
    ensure_sudo
    apt_update_once
    info "Instalando: $*"
    # 'env' garante DEBIAN_FRONTEND mesmo com sudo limpando o ambiente
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

# ==============================================================================
# ENVIRONMENT DETECTION (Python / venv)
# ==============================================================================

PYTHON_BIN=""

check_python() {
    [[ -n "$PYTHON_BIN" ]] && return 0  # idempotente

    # Prefere python3.11 / 3.12 explícitos, depois python3, depois python
    local cand
    for cand in python3.12 python3.11 python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            PYTHON_BIN="$cand"
            return 0
        fi
    done
    return 1
}

python_minor() {
    # Ecoa o minor da versão (ex.: 11 para 3.11). Vazio se indetectável.
    check_python || return 1
    "$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])' 2>/dev/null || true
}

python_version_ok() {
    local minor; minor="$(python_minor)"
    [[ -n "$minor" ]] && [[ "$minor" -ge "$PY_MIN_MINOR" ]]
}

ensure_python() {
    check_python || fatal "Python não encontrado. Rode: ./start.sh setup-python"
}

ensure_venv() {
    ensure_python
    if [[ ! -d ".venv" ]]; then
        info "Criando virtualenv em .venv/ (com ${PYTHON_BIN})..."
        # python3-venv pode faltar no Ubuntu mínimo
        if ! "$PYTHON_BIN" -m venv .venv 2>/dev/null; then
            if is_debian_like; then
                warn "Módulo venv ausente — instalando python3-venv..."
                apt_install python3-venv python3-pip
                "$PYTHON_BIN" -m venv .venv
            else
                fatal "Falha ao criar venv. Instale o pacote venv do seu Python."
            fi
        fi
    fi
}

activate_venv() {
    [[ -d ".venv" ]] || fatal ".venv não encontrada — execute: ./start.sh install"
    # shellcheck disable=SC1091
    source .venv/bin/activate
}

# ==============================================================================
# GPU
# ==============================================================================

check_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1
}

# Detecta GPU NVIDIA via PCI mesmo SEM driver instalado (para sugerir instalação)
check_nvidia_pci() {
    command -v lspci >/dev/null 2>&1 || return 1
    lspci 2>/dev/null | grep -qi 'nvidia'
}

check_cuda_tensorflow() {
    [[ -d ".venv" ]] || return 1
    activate_venv
    python - <<'PYEOF' >/dev/null 2>&1
import tensorflow as tf
exit(0 if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU') else 1)
PYEOF
}

get_gpu_status() {
    check_nvidia_gpu && echo "NVIDIA" || echo "CPU"
}

show_gpu_status() {
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " STATUS DA GPU"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if check_nvidia_gpu; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
            || nvidia-smi -L
        echo
        if [[ -d ".venv" ]] && check_cuda_tensorflow; then
            success "TensorFlow enxerga a GPU (CUDA ativo)"
        else
            warn "TensorFlow sem GPU/CUDA ativo"
            warn "  → Execute: ./start.sh cuda   (instala tensorflow[and-cuda])"
        fi
    else
        warn "Driver NVIDIA não detectado (nvidia-smi indisponível)"
        if check_nvidia_pci; then
            warn "  Mas uma GPU NVIDIA foi vista no barramento PCI."
            warn "  → Instale o driver: ./start.sh nvidia-driver"
        fi
        if is_wsl; then
            warn "WSL2: o driver é instalado no Windows host, não no WSL."
            warn "  → https://docs.nvidia.com/cuda/wsl-user-guide/"
        fi
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ==============================================================================
# DOCKER
# ==============================================================================

COMPOSE_CMD=()

get_docker_status() {
    docker info >/dev/null 2>&1 && echo "ONLINE" || echo "OFFLINE"
}

ensure_docker() {
    require_command docker
    docker info >/dev/null 2>&1 || fatal "Docker daemon não está rodando. Inicie o Docker."
}

get_compose() {
    [[ "${#COMPOSE_CMD[@]}" -gt 0 ]] && return 0

    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
        return 0
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(docker-compose)
        return 0
    fi
    fatal "Docker Compose não encontrado. Instale Docker >= 20.10 ou docker-compose."
}

wait_for_healthy() {
    local elapsed=0
    info "Aguardando container healthy (timeout ${HEALTH_TIMEOUT}s)..."

    while [[ "$elapsed" -lt "$HEALTH_TIMEOUT" ]]; do
        local status
        status="$(docker inspect \
            -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' \
            "$CONTAINER_NAME" 2>/dev/null || true)"

        printf "\r  Status: %-20s  Tempo: %3ds  " "$status" "$elapsed"

        case "$status" in
            healthy)
                echo; success "Container healthy após ${elapsed}s"; return 0 ;;
            unhealthy)
                echo; error "Container ficou unhealthy"
                "${COMPOSE_CMD[@]}" logs --tail=50 app || true; return 1 ;;
            no-healthcheck)
                echo; warn "Container sem HEALTHCHECK configurado"; return 0 ;;
            "")
                echo; error "Container não encontrado ou parou"; return 1 ;;
        esac

        sleep "$HEALTH_INTERVAL"
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done

    echo
    fatal "Timeout (${HEALTH_TIMEOUT}s) aguardando container healthy"
}

# ==============================================================================
# NETWORKING
# ==============================================================================

check_port() {
    local port="${1:-$DEFAULT_PORT}"
    local pid=""

    if command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -ti ":${port}" 2>/dev/null | head -1 || true)"
    elif command -v ss >/dev/null 2>&1; then
        pid="$(ss -tlnp 2>/dev/null | \
               awk -v p=":${port}" '$4 ~ p {match($0,/pid=([0-9]+)/,a); print a[1]}' | \
               head -1 || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pid="$(fuser "${port}/tcp" 2>/dev/null | awk '{print $1}' || true)"
    fi

    [[ -z "$pid" ]] && return 0

    warn "Porta ${port} em uso (PID ${pid})"

    if is_interactive; then
        if confirm "  Encerrar processo automaticamente?"; then
            kill "$pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                warn "SIGTERM ignorado, aplicando SIGKILL..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            success "Processo ${pid} encerrado"
        else
            fatal "Porta ${port} ocupada. Libere manualmente e tente novamente."
        fi
    else
        warn "Modo não-interativo: ignorando conflito na porta ${port}"
    fi
}

# ==============================================================================
# UBUNTU SETUP (privilegiado)
# ==============================================================================

# Pacotes de sistema necessários ao XFakeSong no Ubuntu/Debian:
#  - build-essential/python3-dev: compilar wheels nativos (numba, soundfile…)
#  - ffmpeg: decodificação de áudio (librosa/datasets)
#  - libsndfile1: backend do soundfile
#  - portaudio19-dev/libportaudio2: captura de áudio (opcional, mas comum)
#  - libgomp1: OpenMP (TensorFlow/scikit-learn)
readonly UBUNTU_SYSTEM_PKGS=(
    build-essential pkg-config git curl wget ca-certificates
    python3 python3-venv python3-dev python3-pip
    ffmpeg libsndfile1 libsndfile1-dev libportaudio2 portaudio19-dev
    libgomp1 software-properties-common
)

setup_system_deps() {
    show_header
    require_apt
    info "Instalando dependências de sistema (Ubuntu/Debian)..."
    apt_install "${UBUNTU_SYSTEM_PKGS[@]}"
    success "Dependências de sistema instaladas"
}

setup_python() {
    show_header
    require_apt

    if python_version_ok; then
        success "Python OK ($("$PYTHON_BIN" --version 2>&1))"
    else
        local cur; cur="$(python_minor || echo '?')"
        warn "Python 3.${cur} insuficiente/ausente (mín. 3.${PY_MIN_MINOR}). Instalando..."
        # Tenta o python3 padrão da distro primeiro
        apt_install python3 python3-venv python3-dev python3-pip
        PYTHON_BIN=""  # força redetecção
        if ! python_version_ok; then
            warn "A versão padrão ainda é antiga. Tentando deadsnakes (Python 3.11)..."
            ensure_sudo
            $SUDO add-apt-repository -y ppa:deadsnakes/ppa || \
                warn "Falha ao adicionar PPA deadsnakes — seguindo com o python3 da distro"
            APT_UPDATED=0
            apt_install python3.11 python3.11-venv python3.11-dev || true
            PYTHON_BIN=""
            check_python || true
        fi
    fi

    info "Criando/validando o ambiente virtual..."
    ensure_venv
    activate_venv
    pip install --upgrade pip --quiet
    success "Python e .venv prontos ($("$PYTHON_BIN" --version 2>&1))"
}

setup_nvidia_driver() {
    show_header
    require_apt

    if check_nvidia_gpu; then
        success "Driver NVIDIA já ativo:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
        if ! confirm "Reinstalar/atualizar mesmo assim?"; then
            return 0
        fi
    fi

    if is_wsl; then
        warn "Detectado WSL2: NÃO instale driver NVIDIA dentro do WSL."
        warn "Instale o driver no WINDOWS host. O WSL herda a GPU automaticamente."
        warn "  → https://docs.nvidia.com/cuda/wsl-user-guide/"
        return 0
    fi

    echo
    warn "Vou instalar o driver NVIDIA recomendado via 'ubuntu-drivers autoinstall'."
    warn "Isso requer sudo e provavelmente exigirá REINICIAR o sistema."
    confirm "Continuar com a instalação do driver?" || { info "Cancelado"; return 0; }

    ensure_sudo
    apt_install ubuntu-drivers-common

    info "Drivers recomendados detectados:"
    $SUDO ubuntu-drivers devices 2>/dev/null || true
    echo

    info "Instalando o driver recomendado (ubuntu-drivers autoinstall)..."
    $SUDO ubuntu-drivers autoinstall

    success "Driver NVIDIA instalado."
    warn "════════════════════════════════════════════════════════════"
    warn " REINICIE o sistema para carregar o driver:  sudo reboot"
    warn " Após reiniciar, valide com:  ./start.sh gpu-config"
    warn "════════════════════════════════════════════════════════════"
}

setup_cuda() {
    # No XFakeSong, o caminho SUPORTADO de GPU é o pacote pip
    # tensorflow[and-cuda], que já EMBUTE as libs CUDA/cuDNN. No host só é
    # preciso o DRIVER NVIDIA — não é necessário instalar o CUDA Toolkit
    # de sistema. Esta função instala o TF-GPU no .venv.
    show_header
    info "Configurando CUDA via tensorflow[and-cuda] (CUDA embutido no pacote pip)."

    if ! check_nvidia_gpu; then
        warn "Driver NVIDIA não detectado."
        if is_wsl; then
            warn "WSL2: instale o driver no Windows host."
        elif check_nvidia_pci; then
            warn "GPU NVIDIA presente no PCI mas sem driver."
            if confirm "Instalar o driver NVIDIA agora?"; then
                setup_nvidia_driver
                warn "Reinicie e rode './start.sh cuda' novamente após o reboot."
                return 0
            fi
        fi
        warn "Sem driver, o TensorFlow rodará em CPU. Continuando assim mesmo."
    fi

    install_gpu_dependencies
}

gpu_config() {
    # Configura e RECONHECE a placa de vídeo: mostra driver, valida TF CUDA,
    # e orienta os próximos passos conforme o que faltar.
    show_header
    info "Configuração e reconhecimento de GPU"
    show_gpu_status

    detect_os
    if check_nvidia_gpu; then
        if [[ -d ".venv" ]] && check_cuda_tensorflow; then
            success "GPU pronta para uso pelo XFakeSong."
        else
            warn "Driver OK, mas o TensorFlow ainda não usa a GPU."
            info "Próximo passo: ./start.sh cuda"
        fi
    elif is_wsl; then
        info "WSL2: garanta o driver NVIDIA no Windows host (>=525)."
    elif check_nvidia_pci && is_debian_like; then
        info "Próximo passo: ./start.sh nvidia-driver  (depois reinicie)"
    else
        info "Nenhuma GPU NVIDIA encontrada — o XFakeSong usará CPU (funcional, mais lento)."
    fi
}

# Setup COMPLETO Ubuntu + roda local (one-shot otimizado)
full_setup() {
    show_header
    detect_os
    info "Setup completo do XFakeSong para ${OS_PRETTY}"
    echo

    if ! is_debian_like; then
        warn "SO não-Ubuntu/Debian (${OS_PRETTY}): pulando 'apt' e indo direto ao Python/venv."
    else
        info "[1/4] Dependências de sistema (apt)"
        setup_system_deps
    fi

    info "[2/4] Python + ambiente virtual"
    ensure_venv
    activate_venv
    pip install --upgrade pip --quiet

    info "[3/4] Dependências Python (requirements.txt)"
    pip install -r requirements.txt

    info "[4/4] GPU / CUDA"
    if check_nvidia_gpu; then
        if ! check_cuda_tensorflow; then
            if confirm "GPU NVIDIA detectada. Instalar TensorFlow CUDA (tensorflow[and-cuda])?"; then
                install_gpu_dependencies
            fi
        else
            success "TensorFlow CUDA já ativo"
        fi
    elif check_nvidia_pci && is_debian_like && ! is_wsl; then
        warn "GPU NVIDIA no PCI sem driver."
        if confirm "Instalar o driver NVIDIA agora? (exige reboot depois)"; then
            setup_nvidia_driver
            success "Setup parcial concluído. Reinicie e rode: ./start.sh setup"
            return 0
        fi
    else
        info "Sem GPU NVIDIA ativa — seguindo em modo CPU."
    fi

    success "Setup completo!"
    echo
    if confirm "Iniciar a aplicação agora (local, http://localhost:${DEFAULT_PORT})?"; then
        run_local
    else
        info "Para iniciar depois: ./start.sh test"
    fi
}

bootstrap_dirs() {
    show_header
    ensure_venv
    activate_venv
    info "Criando estrutura de diretórios do projeto..."
    "$PYTHON_BIN" main.py --bootstrap-dirs
    success "Estrutura de diretórios criada"
}

# ==============================================================================
# INSTALLATION (Python)
# ==============================================================================

install_dependencies() {
    show_header
    ensure_venv
    activate_venv

    info "Atualizando pip..."
    pip install --upgrade pip --quiet

    info "Instalando dependências de requirements.txt..."
    pip install -r requirements.txt

    success "Dependências instaladas com sucesso"
}

install_gpu_dependencies() {
    show_header
    check_nvidia_gpu || warn "GPU NVIDIA não detectada (nvidia-smi) — instalando assim mesmo"

    ensure_venv
    activate_venv

    info "Removendo builds TensorFlow CPU anteriores..."
    pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel 2>/dev/null || true

    info "Instalando TensorFlow com suporte CUDA nativo (CUDA embutido)..."
    pip install 'tensorflow[and-cuda]>=2.16,<2.22'

    info "Sincronizando requirements restantes..."
    pip install -r requirements.txt

    success "TensorFlow CUDA instalado"
    show_gpu_status
}

# ==============================================================================
# APPLICATION RUNTIME
# ==============================================================================

# Otimizações de runtime para Linux/CPU: usa todos os núcleos físicos para
# TensorFlow e BLAS, mantém oneDNN ligado (default), evita logs verbosos.
apply_runtime_optimizations() {
    local cores
    cores="$(nproc 2>/dev/null || echo 0)"
    if [[ "$cores" -gt 0 ]]; then
        export OMP_NUM_THREADS="$cores"
        export TF_NUM_INTRAOP_THREADS="$cores"
        export TF_NUM_INTEROP_THREADS="2"
    fi
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
    export PYTHONUNBUFFERED=1
    info "Otimizações de runtime: ${cores} núcleos (OMP/TF intra-op)"
}

run_local() {
    show_header
    info "Modo local (Python direto)..."

    ensure_venv
    activate_venv

    if ! python -c "import gradio" >/dev/null 2>&1; then
        warn "Dependências não instaladas. Executando install..."
        pip install -r requirements.txt
    fi

    apply_runtime_optimizations
    check_port "$DEFAULT_PORT"
    show_gpu_status

    exec "$PYTHON_BIN" main.py --gradio --gradio-port "$DEFAULT_PORT"
}

run_gpu_local() {
    # Linux/WSL2 com CUDA nativo — tensorflow[and-cuda] em venv local, sem Docker.
    show_header
    info "Modo local + GPU (TF CUDA nativo)..."

    check_nvidia_gpu || \
        fatal "nvidia-smi não encontrado.\n  Linux: ./start.sh nvidia-driver (depois reboot)\n  WSL2: instale o driver no Windows host."

    ensure_venv
    activate_venv

    if ! check_cuda_tensorflow; then
        info "TensorFlow CUDA não detectado — instalando..."
        pip uninstall -y tensorflow tensorflow-cpu 2>/dev/null || true
        pip install 'tensorflow[and-cuda]>=2.16,<2.22'
    else
        success "TensorFlow CUDA já instalado"
    fi

    apply_runtime_optimizations
    check_port "$DEFAULT_PORT"
    show_gpu_status

    exec "$PYTHON_BIN" main.py --gradio --gradio-port "$DEFAULT_PORT"
}

run_docker() {
    local gpu_mode="${1:-0}"

    show_header
    ensure_docker
    get_compose

    local compose_args=(-f docker-compose.yml)

    if [[ "$gpu_mode" == "1" ]]; then
        [[ -f docker-compose.gpu.yml ]] || fatal "docker-compose.gpu.yml não encontrado"
        compose_args+=(-f docker-compose.gpu.yml)
        info "Modo GPU ativo (nvidia-container-toolkit necessário)"
    fi

    info "Buildando e iniciando containers..."
    "${COMPOSE_CMD[@]}" "${compose_args[@]}" up --build -d

    wait_for_healthy
    success "Aplicação disponível em http://localhost:${DEFAULT_PORT}"
}

stop_containers() {
    show_header
    ensure_docker
    get_compose
    "${COMPOSE_CMD[@]}" down
    success "Containers parados e removidos"
}

show_logs() {
    ensure_docker
    get_compose
    "${COMPOSE_CMD[@]}" logs -f --tail=100 app
}

rebuild() {
    show_header
    ensure_docker
    get_compose

    info "Rebuild completo sem cache..."
    "${COMPOSE_CMD[@]}" down
    "${COMPOSE_CMD[@]}" build --no-cache --pull
    "${COMPOSE_CMD[@]}" up -d
    wait_for_healthy
    success "Rebuild concluído"
}

show_status() {
    show_header
    ensure_docker
    get_compose

    echo
    "${COMPOSE_CMD[@]}" ps
    echo

    timeout 5 docker stats \
        --no-stream \
        --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}' \
        2>/dev/null || true
}

# ==============================================================================
# CLEANUP
# ==============================================================================

clean_environment() {
    show_header
    ensure_docker
    get_compose

    echo
    warn 'ATENÇÃO: esta operação remove volumes Docker e cache de build'
    echo

    confirm 'Continuar?' || { info "Operação cancelada"; return 0; }

    "${COMPOSE_CMD[@]}" down -v --remove-orphans
    docker image prune -f
    docker builder prune -f

    success 'Limpeza concluída'
}

# ==============================================================================
# DOCTOR
# ==============================================================================

run_doctor() {
    show_header
    local failures=0

    check_item() {
        local label="$1" cmd="$2"
        printf '  %-44s' "$label"
        if eval "$cmd" >/dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAIL${NC}"
            failures=$((failures + 1))
        fi
    }

    detect_os
    echo
    echo "  SO: ${OS_PRETTY}"
    echo

    check_item 'Python 3.10+'               'python_version_ok'
    check_item 'pip'                        'command -v pip3 || command -v pip'
    check_item '.venv'                      '[ -d .venv ]'
    check_item 'requirements.txt'           '[ -f requirements.txt ]'
    check_item 'main.py'                    '[ -f main.py ]'
    check_item ".env"                       '[ -f .env ]'
    check_item 'curl'                       'command -v curl'
    check_item "Porta ${DEFAULT_PORT} livre" \
        "! ( lsof -ti :${DEFAULT_PORT} 2>/dev/null || \
             ss -tlnp 2>/dev/null | grep -q ':${DEFAULT_PORT} ' ) >/dev/null 2>&1"

    if is_debian_like; then
        check_item 'ffmpeg'                 'command -v ffmpeg'
        check_item 'libsndfile (soundfile)' \
            '[ -x .venv/bin/python ] && .venv/bin/python -c "import soundfile"'
    fi

    check_item 'Docker daemon'              'docker info'
    check_item 'Docker Compose'             'docker compose version || docker-compose version'
    check_item 'docker-compose.yml'         '[ -f docker-compose.yml ]'

    if is_wsl; then
        check_item 'WSL2'                   'is_wsl'
    fi

    if check_nvidia_gpu; then
        check_item 'nvidia-smi'             'nvidia-smi -L'
        check_item 'TensorFlow CUDA'        'check_cuda_tensorflow'
    else
        check_item 'GPU NVIDIA (driver)'    'check_nvidia_gpu'
    fi

    echo
    if [[ "$failures" -eq 0 ]]; then
        success "Diagnóstico concluído — nenhuma falha encontrada"
    else
        warn "${failures} item(ns) com FAIL — veja as sugestões de setup acima"
        info "Log completo em: ${LOG_FILE}"
        return 0   # doctor é informativo: não aborta
    fi
}

# ==============================================================================
# HEADER / UI
# ==============================================================================

show_header() {
    clear_screen

    detect_os
    local docker_status gpu_status
    docker_status="$(get_docker_status 2>/dev/null || echo "N/A")"
    gpu_status="$(get_gpu_status 2>/dev/null || echo "N/A")"

    echo -e "
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BOLD}${APP_NAME} Launcher v${VERSION}${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

 Ambiente:
 ──────────────────────────────────────────────────────────────────
 SO      : ${OS_PRETTY}
 Docker  : ${docker_status}
 GPU     : ${gpu_status}
 Projeto : $(basename "$PROJECT_ROOT")
 Logs    : ${LOG_FILE}
"
}

show_help() {
    cat <<'EOF'

Uso: ./start.sh [comando]

  EXECUÇÃO
  ─────────────────────────────────────────────────────────────
  test              Executa localmente (Python + venv)
  gpu-local         Linux/WSL2 + GPU: local com TF CUDA nativo
  prod              Docker (CPU)
  gpu               Docker + GPU (nvidia-container-toolkit)

  SETUP UBUNTU (usa apt/sudo)
  ─────────────────────────────────────────────────────────────
  setup             Setup COMPLETO Ubuntu + roda local (one-shot)
  setup-system      Dependências de sistema (ffmpeg, libsndfile, build…)
  setup-python      Instala/configura Python 3.11+ e cria o .venv
  nvidia-driver     Instala o driver NVIDIA (ubuntu-drivers autoinstall)
  cuda              Instala CUDA/TensorFlow-GPU no .venv
  gpu-config        Configura e reconhece a placa de vídeo

  INSTALAÇÃO PYTHON
  ─────────────────────────────────────────────────────────────
  install           Instala dependências Python no .venv
  bootstrap         Cria a estrutura de diretórios do projeto

  DOCKER
  ─────────────────────────────────────────────────────────────
  logs / stop / rebuild / status

  MANUTENÇÃO
  ─────────────────────────────────────────────────────────────
  doctor            Diagnóstico completo do ambiente
  clean             Deep clean Docker (volumes + cache de build)
  help / -h         Exibe esta ajuda

  Sem argumentos → menu interativo

EOF
}

show_menu() {
    show_header
    cat <<'EOF'

 ══════════════════════════════════════════════════════
                        EXECUÇÃO
 ══════════════════════════════════════════════════════
 [1] Executar localmente (Python)
 [2] Local + GPU (TF CUDA nativo)
 [3] Docker — CPU
 [4] Docker — GPU (nvidia-container-toolkit)

 ══════════════════════════════════════════════════════
              SETUP UBUNTU (apt/sudo)
 ══════════════════════════════════════════════════════
 [5] ⚡ Setup COMPLETO + rodar (one-shot)
 [6] Dependências de sistema (ffmpeg, libsndfile…)
 [7] Python 3.11+ e ambiente virtual
 [8] Instalar driver NVIDIA
 [9] Instalar CUDA / TensorFlow-GPU
 [10] Configurar e reconhecer GPU

 ══════════════════════════════════════════════════════
                    PYTHON / DOCKER
 ══════════════════════════════════════════════════════
 [11] Instalar dependências Python (.venv)
 [12] Bootstrap diretórios
 [13] Docker: status     [14] Docker: logs
 [15] Docker: rebuild    [16] Docker: stop

 ══════════════════════════════════════════════════════
                      MANUTENÇÃO
 ══════════════════════════════════════════════════════
 [17] Diagnóstico do ambiente (doctor)
 [18] Deep clean Docker/cache

 ══════════════════════════════════════════════════════
 [h] Ajuda    [q] Sair
 ══════════════════════════════════════════════════════

EOF
}

# ==============================================================================
# INTERACTIVE MENU
# ==============================================================================

interactive_menu() {
    is_interactive || fatal 'Menu interativo requer TTY (stdin não é terminal)'

    while true; do
        show_menu
        read -rp 'Selecione uma opção: ' opt

        case "$opt" in
            1)   run_local ;;            # exec — substitui o shell
            2)   run_gpu_local ;;        # exec — idem
            3)   run_docker 0; pause ;;
            4)   run_docker 1; pause ;;
            5)   full_setup ;;           # pode exec (run_local) ao final
            6)   setup_system_deps; pause ;;
            7)   setup_python; pause ;;
            8)   setup_nvidia_driver; pause ;;
            9)   setup_cuda; pause ;;
            10)  gpu_config; pause ;;
            11)  install_dependencies; pause ;;
            12)  bootstrap_dirs; pause ;;
            13)  show_status; pause ;;
            14)  show_logs ;;
            15)  rebuild; pause ;;
            16)  stop_containers; pause ;;
            17)  run_doctor; pause ;;
            18)  clean_environment; pause ;;
            h|H) show_help; pause ;;
            q|Q|0) exit 0 ;;
            *)   warn "Opção inválida: '${opt}'"; sleep 1 ;;
        esac
    done
}

# ==============================================================================
# COMMAND DISPATCHER
# ==============================================================================

main() {
    require_bash
    cd "$PROJECT_ROOT"

    local cmd="${1:-menu}"

    case "$cmd" in
        # Execução
        test)            run_local ;;
        gpu-local|wsl-gpu) run_gpu_local ;;   # wsl-gpu mantido p/ compat
        prod)            run_docker 0 ;;
        gpu)             run_docker 1 ;;
        # Setup Ubuntu
        setup)           full_setup ;;
        setup-system)    setup_system_deps ;;
        setup-python)    setup_python ;;
        nvidia-driver|nvidia-drivers) setup_nvidia_driver ;;
        cuda)            setup_cuda ;;
        gpu-config)      gpu_config ;;
        # Instalação Python
        install)         install_dependencies ;;
        install-gpu)     install_gpu_dependencies ;;   # compat
        bootstrap)       bootstrap_dirs ;;
        # Docker
        logs)            show_logs ;;
        stop)            stop_containers ;;
        rebuild)         rebuild ;;
        status)          show_status ;;
        clean)           clean_environment ;;
        # Manutenção
        doctor)          run_doctor ;;
        help|-h|--help)  show_help ;;
        menu)            interactive_menu ;;
        *)               fatal "Comando desconhecido: '${cmd}'\nUso: ./start.sh help" ;;
    esac
}

main "$@"
