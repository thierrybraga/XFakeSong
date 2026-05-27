#!/usr/bin/env bash
# ==============================================================================
# XFakeSong Launcher v3.1
# Plataforma operacional unificada
# Compatível: Linux • macOS • WSL2
# ==============================================================================
#
# Uso: ./start.sh [comando]
#
# Comandos disponíveis:
#   test        Executa localmente (Python + venv)
#   wsl-gpu     WSL2 + GPU: local com TF CUDA nativo
#   prod        Docker (CPU)
#   gpu         Docker + GPU (nvidia-container-toolkit)
#   install     Instala dependências Python no .venv
#   install-gpu Instala TensorFlow CUDA no .venv
#   logs        Exibe logs em tempo real
#   stop        Para e remove containers
#   rebuild     Rebuild sem cache
#   status      Status e métricas dos containers
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

readonly VERSION="3.1.0"
readonly APP_NAME="XFakeSong"
readonly DEFAULT_PORT="7860"
readonly HEALTH_TIMEOUT="300"
readonly HEALTH_INTERVAL="5"
readonly CONTAINER_NAME="xfakesong_app"

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

is_wsl() {
    [[ -n "${WSL_DISTRO_NAME:-}" ]] && return 0
    [[ -f /proc/version ]] || return 1
    grep -qiE '(microsoft|wsl)' /proc/version
}

# ==============================================================================
# ENVIRONMENT DETECTION
# ==============================================================================

# BUG FIX: variáveis com guard, não readonly dentro de função condicional.
# readonly dentro de função chamada múltiplas vezes causa "readonly variable" fatal.
PYTHON_BIN=""

check_python() {
    [[ -n "$PYTHON_BIN" ]] && return 0  # já detectado — idempotente

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
        return 0
    fi

    return 1
}

ensure_python() {
    check_python || fatal "Python não encontrado. Instale Python 3.11+."
}

ensure_venv() {
    ensure_python
    if [[ ! -d ".venv" ]]; then
        info "Criando virtualenv em .venv/ ..."
        "$PYTHON_BIN" -m venv .venv
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

check_cuda_tensorflow() {
    [[ -d ".venv" ]] || return 1
    activate_venv
    # Heredoc com aspas simples evita expansão prematura no script host
    python - <<'PYEOF' >/dev/null 2>&1
import tensorflow as tf
exit(0 if tf.test.is_built_with_cuda() else 1)
PYEOF
}

get_gpu_status() {
    check_nvidia_gpu && echo "NVIDIA" || echo "CPU"
}

show_gpu_status() {
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " GPU STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if check_nvidia_gpu; then
        nvidia-smi -L
        echo
        if check_cuda_tensorflow; then
            success "TensorFlow CUDA ativo"
        else
            warn "TensorFlow sem suporte CUDA"
            warn "  → Execute: ./start.sh install-gpu"
        fi
    else
        warn "GPU NVIDIA não detectada"
        if is_wsl; then
            warn "WSL2: verifique driver NVIDIA no Windows host"
            warn "  → https://docs.nvidia.com/cuda/wsl-user-guide/"
        fi
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ==============================================================================
# DOCKER
# ==============================================================================

# BUG FIX: array em vez de string escalar.
# Com IFS=$'\n\t' (espaço excluído), a variável "docker compose" NÃO faz
# word-split — tentaria executar o binário "docker compose" (com espaço).
# Usando array e "${COMPOSE_CMD[@]}", a expansão é sempre correta.
COMPOSE_CMD=()

get_docker_status() {
    # Não falha se Docker não estiver instalado
    docker info >/dev/null 2>&1 && echo "ONLINE" || echo "OFFLINE"
}

ensure_docker() {
    require_command docker
    docker info >/dev/null 2>&1 || fatal "Docker daemon não está rodando. Inicie o Docker."
}

get_compose() {
    # BUG FIX: idempotente — detecta uma única vez e reutiliza.
    # A versão anterior usava "readonly COMPOSE_CMD" dentro da função,
    # o que causava erro fatal "readonly variable" na segunda chamada.
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
                echo
                success "Container healthy após ${elapsed}s"
                return 0
                ;;
            unhealthy)
                echo
                error "Container ficou unhealthy"
                "${COMPOSE_CMD[@]}" logs --tail=50 app || true
                return 1
                ;;
            no-healthcheck)
                echo
                warn "Container sem HEALTHCHECK configurado"
                return 0
                ;;
            "")
                echo
                error "Container não encontrado ou parou"
                return 1
                ;;
            # "starting" e outros estados intermediários: continua aguardando
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

    # BUG FIX: lsof não está disponível em todos os sistemas Linux (Alpine, Debian
    # mínimo, etc.). Cascata: lsof → ss → fuser → skip.
    if command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -ti ":${port}" 2>/dev/null | head -1 || true)"
    elif command -v ss >/dev/null 2>&1; then
        # ss: extrai PID do campo "users:(("proc",pid=NNN,...))
        pid="$(ss -tlnp 2>/dev/null | \
               awk -v p=":${port}" '$4 ~ p {match($0,/pid=([0-9]+)/,a); print a[1]}' | \
               head -1 || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pid="$(fuser "${port}/tcp" 2>/dev/null | awk '{print $1}' || true)"
    fi

    [[ -z "$pid" ]] && return 0  # porta livre

    warn "Porta ${port} em uso (PID ${pid})"

    if is_interactive; then
        read -rp "  Encerrar processo automaticamente? [y/N] " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
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
# INSTALLATION
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

    check_nvidia_gpu || fatal "GPU NVIDIA não detectada (nvidia-smi não responde)"

    ensure_venv
    activate_venv

    info "Removendo builds TensorFlow CPU anteriores..."
    pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel 2>/dev/null || true

    # BUG FIX: tensorflow[and-cuda]==2.20.* não existe. Versão máxima atual: 2.19.
    # Usar constraint aberto para pegar a versão mais recente compatível.
    info "Instalando TensorFlow com suporte CUDA nativo..."
    pip install 'tensorflow[and-cuda]>=2.16,<3.0'

    info "Sincronizando requirements restantes..."
    pip install -r requirements.txt

    success "TensorFlow CUDA instalado"
    show_gpu_status
}

# ==============================================================================
# APPLICATION RUNTIME
# ==============================================================================

run_local() {
    show_header
    info "Modo local (Python direto)..."

    ensure_venv
    activate_venv

    # BUG FIX: versão anterior chamava install_dependencies() (pip install completo)
    # em todo launch — lento e desnecessário. Apenas verifica se venv existe.
    if ! python -c "import gradio" >/dev/null 2>&1; then
        warn "Dependências não instaladas. Executando install..."
        pip install -r requirements.txt
    fi

    check_port "$DEFAULT_PORT"
    show_gpu_status

    exec "$PYTHON_BIN" main.py --gradio --gradio-port "$DEFAULT_PORT"
}

run_wsl_gpu() {
    # WSL2 com CUDA nativo — tensorflow[and-cuda] em venv local, sem Docker.
    show_header
    info "Modo WSL2 + GPU (TF CUDA nativo)..."

    is_wsl || warn "Não detectado como WSL2 — continuando mesmo assim"

    check_nvidia_gpu || \
        fatal "nvidia-smi não encontrado.\nNo Windows host: instale driver NVIDIA >= 525.\nNo WSL2: não precisa instalar CUDA separado."

    ensure_venv
    activate_venv

    if ! check_cuda_tensorflow; then
        info "TensorFlow CUDA não detectado — instalando..."
        pip uninstall -y tensorflow tensorflow-cpu 2>/dev/null || true
        pip install 'tensorflow[and-cuda]>=2.16,<3.0'
    else
        success "TensorFlow CUDA já instalado"
    fi

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
        [[ -f docker-compose.gpu.yml ]] || \
            fatal "docker-compose.gpu.yml não encontrado"
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

    read -rp 'Continuar? [y/N] ' response
    [[ "$response" =~ ^[Yy]$ ]] || { info "Operação cancelada"; return 0; }

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
        local label="$1"
        local cmd="$2"
        printf '  %-44s' "$label"
        if eval "$cmd" >/dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAIL${NC}"
            failures=$((failures + 1))
        fi
    }

    echo

    check_item 'Docker daemon'              'docker info'
    check_item 'Docker Compose'             'docker compose version || docker-compose version'
    check_item 'Python 3'                   'command -v python3'
    check_item 'pip'                        'command -v pip3 || command -v pip'
    check_item '.venv'                      '[ -d .venv ]'
    check_item 'requirements.txt'           '[ -f requirements.txt ]'
    check_item 'docker-compose.yml'         '[ -f docker-compose.yml ]'
    check_item ".env"                       '[ -f .env ]'
    check_item 'main.py'                    '[ -f main.py ]'
    check_item "Porta ${DEFAULT_PORT} livre" \
        "! ( lsof -ti :${DEFAULT_PORT} 2>/dev/null || \
             ss -tlnp 2>/dev/null | grep -q ':${DEFAULT_PORT} ' ) >/dev/null 2>&1"
    check_item 'curl'                       'command -v curl'

    if is_wsl; then
        check_item 'WSL2'                   'is_wsl'
    fi

    if check_nvidia_gpu; then
        check_item 'nvidia-smi'             'nvidia-smi -L'
        check_item 'TensorFlow CUDA'        'check_cuda_tensorflow'
    else
        check_item 'GPU NVIDIA'             'check_nvidia_gpu'
    fi

    echo

    if [[ "$failures" -eq 0 ]]; then
        success "Diagnóstico concluído — nenhuma falha encontrada"
    else
        error "${failures} falha(s) encontrada(s)"
        info "Log completo em: ${LOG_FILE}"
        return 1
    fi
}

# ==============================================================================
# HEADER / UI
# ==============================================================================

show_header() {
    clear_screen

    # BUG FIX: 2>/dev/null garante que show_header nunca falha mesmo se Docker
    # não estiver instalado (ex: ao rodar ./start.sh help num ambiente limpo).
    local docker_status gpu_status
    docker_status="$(get_docker_status 2>/dev/null || echo "N/A")"
    gpu_status="$(get_gpu_status 2>/dev/null || echo "N/A")"

    echo -e "
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BOLD}${APP_NAME} Launcher v${VERSION}${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

 Ambiente:
 ──────────────────────────────────────────────────────────────────
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
  wsl-gpu           WSL2 + GPU: local com TF CUDA nativo
  prod              Docker (CPU)
  gpu               Docker + GPU (nvidia-container-toolkit)

  INSTALAÇÃO
  ─────────────────────────────────────────────────────────────
  install           Instala dependências Python no .venv
  install-gpu       Instala TensorFlow CUDA no .venv

  DOCKER
  ─────────────────────────────────────────────────────────────
  logs              Exibe logs em tempo real
  stop              Para e remove containers
  rebuild           Rebuild sem cache + restart
  status            Status e métricas dos containers

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
 [2] WSL2 + GPU (TF CUDA nativo)
 [3] Docker — CPU
 [4] Docker — GPU (nvidia-container-toolkit)

 ══════════════════════════════════════════════════════
                       GPU / CUDA
 ══════════════════════════════════════════════════════
 [5] Instalar TensorFlow CUDA no .venv
 [6] Diagnóstico GPU

 ══════════════════════════════════════════════════════
                        DOCKER
 ══════════════════════════════════════════════════════
 [7]  Status containers
 [8]  Logs realtime
 [9]  Rebuild completo
 [10] Stop containers

 ══════════════════════════════════════════════════════
                      MANUTENÇÃO
 ══════════════════════════════════════════════════════
 [11] Instalar dependências Python
 [12] Diagnóstico ambiente (doctor)
 [13] Deep clean Docker/cache

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
            1)   run_local ;;           # exec — substitui shell, sem retorno
            2)   run_wsl_gpu ;;         # exec — idem
            3)   run_docker 0 ;;
            4)   run_docker 1 ;;
            5)   install_gpu_dependencies; pause ;;
            6)   show_gpu_status; pause ;;
            7)   show_status; pause ;;
            8)   show_logs ;;
            9)   rebuild; pause ;;
            10)  stop_containers; pause ;;
            11)  install_dependencies; pause ;;
            12)  run_doctor; pause ;;
            13)  clean_environment; pause ;;
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

    cd "$PROJECT_ROOT"  # garante que paths relativos (.venv, requirements.txt) funcionam

    local cmd="${1:-menu}"

    case "$cmd" in
        test)           run_local ;;
        wsl-gpu)        run_wsl_gpu ;;
        prod)           run_docker 0 ;;
        gpu)            run_docker 1 ;;
        install)        install_dependencies ;;
        install-gpu)    install_gpu_dependencies ;;
        logs)           show_logs ;;
        stop)           stop_containers ;;
        rebuild)        rebuild ;;
        status)         show_status ;;
        clean)          clean_environment ;;
        doctor)         run_doctor ;;
        help|-h|--help) show_help ;;
        menu)           interactive_menu ;;
        *)              fatal "Comando desconhecido: '${cmd}'\nUso: ./start.sh help" ;;
    esac
}

main "$@"
