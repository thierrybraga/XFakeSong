# XFakeSong Launcher v3

```bash
#!/usr/bin/env bash
# ==============================================================================
# XFakeSong Launcher v3
# Plataforma operacional unificada
# Compatível: Linux • macOS • WSL2
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ==============================================================================
# METADATA
# ==============================================================================

readonly VERSION="3.0.0"
readonly APP_NAME="XFakeSong"
readonly DEFAULT_PORT="7860"
readonly HEALTH_TIMEOUT="300"
readonly HEALTH_INTERVAL="5"
readonly CONTAINER_NAME="xfakesong_app"

readonly SCRIPT_NAME="$(basename "$0")"
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
    readonly MAGENTA='\033[0;35m'
    readonly BOLD='\033[1m'
    readonly NC='\033[0m'
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly BLUE=''
    readonly CYAN=''
    readonly MAGENTA=''
    readonly BOLD=''
    readonly NC=''
fi

# ==============================================================================
# LOGGING
# ==============================================================================

mkdir -p "$LOG_DIR"

log() {
    local level="$1"
    shift

    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "[$ts] [$level] $*" >> "$LOG_FILE"
}

info() {
    log INFO "$@"
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    log SUCCESS "$@"
    echo -e "${GREEN}[OK]${NC} $*"
}

warn() {
    log WARN "$@"
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    log ERROR "$@"
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

fatal() {
    error "$@"
    exit 1
}

# ==============================================================================
# ERROR HANDLING
# ==============================================================================

on_error() {
    local exit_code="$?"
    local line="$1"

    error "Falha na linha ${line} (exit code=${exit_code})"
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

# ==============================================================================
# CORE HELPERS
# ==============================================================================

require_bash() {
    [[ -n "${BASH_VERSION:-}" ]] || {
        echo "Este script requer Bash"
        exit 1
    }
}

require_command() {
    local cmd="$1"

    command -v "$cmd" >/dev/null 2>&1 || \
        fatal "Dependência ausente: ${cmd}"
}

clear_screen() {
    command -v clear >/dev/null 2>&1 && clear || true
}

pause() {
    echo
    read -rp "Pressione ENTER para continuar..." _
}

is_interactive() {
    [[ -t 0 ]]
}

is_wsl() {
    [[ -n "${WSL_DISTRO_NAME:-}" ]] && return 0

    [[ -f /proc/version ]] || return 1

    grep -qiE '(microsoft|wsl)' /proc/version
}

# ==============================================================================
# ENVIRONMENT DETECTION
# ==============================================================================

check_python() {

    if command -v python3 >/dev/null 2>&1; then
        readonly PYTHON_BIN="python3"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        readonly PYTHON_BIN="python"
        return 0
    fi

    return 1
}

ensure_python() {
    check_python || fatal "Python não encontrado"
}

ensure_venv() {

    ensure_python

    if [[ ! -d ".venv" ]]; then
        info "Criando virtualenv..."
        "$PYTHON_BIN" -m venv .venv
    fi
}

activate_venv() {

    [[ -d ".venv" ]] || fatal ".venv não encontrada"

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

    python - <<EOF >/dev/null 2>&1
import tensorflow as tf
exit(0 if tf.test.is_built_with_cuda() else 1)
EOF
}

get_gpu_status() {

    if check_nvidia_gpu; then
        echo "NVIDIA"
    else
        echo "CPU"
    fi
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
        fi

    else

        warn "GPU NVIDIA não detectada"

        if is_wsl; then
            warn "Verifique driver NVIDIA no Windows"
        fi
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ==============================================================================
# DOCKER
# ==============================================================================

get_docker_status() {

    if docker info >/dev/null 2>&1; then
        echo "ONLINE"
    else
        echo "OFFLINE"
    fi
}

ensure_docker() {

    require_command docker

    docker info >/dev/null 2>&1 || \
        fatal "Docker daemon não está rodando"
}

get_compose() {

    if docker compose version >/dev/null 2>&1; then
        readonly COMPOSE_CMD="docker compose"
        return
    fi

    if command -v docker-compose >/dev/null 2>&1; then
        readonly COMPOSE_CMD="docker-compose"
        return
    fi

    fatal "Docker Compose não encontrado"
}

wait_for_healthy() {

    local elapsed=0

    info "Aguardando container healthy..."

    while [[ "$elapsed" -lt "$HEALTH_TIMEOUT" ]]; do

        local status

        status="$(docker inspect \
            -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' \
            "$CONTAINER_NAME" \
            2>/dev/null || true)"

        printf "\rStatus: %-20s Tempo: %ss" "$status" "$elapsed"

        case "$status" in

            healthy)
                echo
                success "Container healthy"
                return 0
                ;;

            unhealthy)
                echo
                error "Container unhealthy"
                $COMPOSE_CMD logs --tail=50 app
                return 1
                ;;

            no-healthcheck)
                echo
                warn "Container sem HEALTHCHECK"
                return 0
                ;;

            "")
                echo
                error "Container não encontrado"
                return 1
                ;;

        esac

        sleep "$HEALTH_INTERVAL"

        elapsed=$((elapsed + HEALTH_INTERVAL))

    done

    echo

    fatal "Timeout aguardando container healthy"
}

# ==============================================================================
# NETWORKING
# ==============================================================================

check_port() {

    local port="${1:-$DEFAULT_PORT}"

    if lsof -i ":${port}" >/dev/null 2>&1; then

        local pid

        pid="$(lsof -ti ":${port}" | head -1)"

        warn "Porta ${port} em uso (PID ${pid})"

        read -rp "Encerrar automaticamente? [y/N] " response

        if [[ "$response" =~ ^[Yy]$ ]]; then

            kill "$pid" || true

            sleep 5

            if kill -0 "$pid" >/dev/null 2>&1; then
                warn "Aplicando SIGKILL..."
                kill -9 "$pid"
            fi

            success "Processo encerrado"

        else
            return 1
        fi
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
    pip install --upgrade pip

    info "Instalando dependências..."
    pip install -r requirements.txt

    success "Dependências instaladas"
}

install_gpu_dependencies() {

    show_header

    ensure_venv

    activate_venv

    check_nvidia_gpu || fatal "GPU NVIDIA não detectada"

    info "Removendo TensorFlow CPU..."

    pip uninstall -y \
        tensorflow \
        tensorflow-cpu \
        tensorflow-intel \
        >/dev/null 2>&1 || true

    info "Instalando TensorFlow CUDA..."

    pip install 'tensorflow[and-cuda]==2.20.*'

    info "Instalando requirements..."

    pip install -r requirements.txt

    success "TensorFlow CUDA instalado"
}

# ==============================================================================
# APPLICATION RUNTIME
# ==============================================================================

run_local() {

    show_header

    info "Inicializando modo local..."

    install_dependencies

    check_port "$DEFAULT_PORT"

    show_gpu_status

    activate_venv

    exec "$PYTHON_BIN" \
        main.py \
        --gradio \
        --gradio-port "$DEFAULT_PORT"
}

run_docker() {

    local gpu_mode="${1:-0}"

    show_header

    ensure_docker

    get_compose

    local compose_args=(
        -f docker-compose.yml
    )

    if [[ "$gpu_mode" == "1" ]]; then
        compose_args+=( -f docker-compose.gpu.yml )
    fi

    info "Buildando containers..."

    $COMPOSE_CMD \
        "${compose_args[@]}" \
        up \
        --build \
        -d

    wait_for_healthy

    success "Aplicação disponível em http://localhost:${DEFAULT_PORT}"
}

stop_containers() {

    show_header

    ensure_docker

    get_compose

    $COMPOSE_CMD down

    success "Containers parados"
}

show_logs() {

    ensure_docker

    get_compose

    $COMPOSE_CMD logs -f --tail=100 app
}

rebuild() {

    show_header

    ensure_docker

    get_compose

    info "Rebuild sem cache..."

    $COMPOSE_CMD down

    $COMPOSE_CMD build --no-cache --pull

    $COMPOSE_CMD up -d

    wait_for_healthy
}

show_status() {

    show_header

    ensure_docker

    get_compose

    $COMPOSE_CMD ps

    echo

    timeout 5 docker stats \
        --no-stream \
        --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' \
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
    warn 'ATENÇÃO: esta operação remove volumes e cache Docker'
    echo

    read -rp 'Continuar? [y/N] ' response

    [[ "$response" =~ ^[Yy]$ ]] || return 0

    $COMPOSE_CMD down -v --remove-orphans

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

        printf '%-40s' "$label"

        if eval "$cmd" >/dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAIL${NC}"
            failures=$((failures + 1))
        fi
    }

    echo

    check_item 'Docker daemon' 'docker info'
    check_item 'Docker Compose' 'docker compose version'
    check_item 'Python' 'command -v python3'
    check_item 'Pip' 'command -v pip'
    check_item '.venv' '[ -d .venv ]'
    check_item 'requirements.txt' '[ -f requirements.txt ]'
    check_item 'docker-compose.yml' '[ -f docker-compose.yml ]'
    check_item 'Porta 7860 livre' '! lsof -i :7860'

    if check_nvidia_gpu; then
        check_item 'TensorFlow CUDA' 'check_cuda_tensorflow'
    fi

    echo

    if [[ "$failures" -eq 0 ]]; then
        success 'Diagnóstico concluído sem falhas'
    else
        error "${failures} falhas encontradas"
        return 1
    fi
}

# ==============================================================================
# HEADER / UI
# ==============================================================================

show_header() {

    clear_screen

    local docker_status
    local gpu_status

    docker_status="$(get_docker_status)"
    gpu_status="$(get_gpu_status)"

    cat <<EOF

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${BOLD}${APP_NAME} Launcher v${VERSION}${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

 Ambiente:
 ──────────────────────────────────────────────────────────────────
 Docker:        ${docker_status}
 GPU:           ${gpu_status}
 Projeto:       $(basename "$PROJECT_ROOT")
 Logs:          ${LOG_FILE}

EOF
}

show_help() {

cat <<EOF

Comandos disponíveis:

  test              Executa localmente
  prod              Executa via Docker
  gpu               Docker + GPU

  install           Instala dependências
  install-gpu       Instala TensorFlow CUDA

  logs              Exibe logs
  stop              Para containers
  rebuild           Rebuild sem cache
  status            Status runtime

  clean             Limpeza Docker/cache
  doctor            Diagnóstico ambiente

  help              Exibe ajuda

EOF
}

show_menu() {

    show_header

    cat <<EOF

 =====================================================
                    EXECUÇÃO
 =====================================================

 [1] Executar localmente (Python)
 [2] Executar via Docker
 [3] Executar Docker + GPU

 =====================================================
                    GPU / CUDA
 =====================================================

 [4] Instalar TensorFlow CUDA
 [5] Diagnóstico GPU

 =====================================================
                    DOCKER
 =====================================================

 [6] Status containers
 [7] Logs realtime
 [8] Rebuild completo
 [9] Stop containers

 =====================================================
                    MANUTENÇÃO
 =====================================================

 [10] Instalar dependências
 [11] Diagnóstico ambiente (doctor)

 =====================================================
                    LIMPEZA
 =====================================================

 [12] Deep clean Docker/cache

 =====================================================
                    SISTEMA
 =====================================================

 [h] Help
 [q] Sair

EOF
}

# ==============================================================================
# INTERACTIVE MENU
# ==============================================================================

interactive_menu() {

    is_interactive || fatal 'Menu interativo indisponível'

    while true; do

        show_menu

        read -rp 'Selecione uma opção: ' opt

        case "$opt" in

            1)
                run_local
                ;;

            2)
                run_docker 0
                ;;

            3)
                run_docker 1
                ;;

            4)
                install_gpu_dependencies
                pause
                ;;

            5)
                show_gpu_status
                pause
                ;;

            6)
                show_status
                pause
                ;;

            7)
                show_logs
                ;;

            8)
                rebuild
                pause
                ;;

            9)
                stop_containers
                pause
                ;;

            10)
                install_dependencies
                pause
                ;;

            11)
                run_doctor
                pause
                ;;

            12)
                clean_environment
                pause
                ;;

            h|H)
                show_help
                pause
                ;;

            q|Q|0)
                exit 0
                ;;

            *)
                warn 'Opção inválida'
                sleep 1
                ;;

        esac
    done
}

# ==============================================================================
# COMMAND DISPATCHER
# ==============================================================================

main() {

    require_bash

    local cmd="${1:-menu}"

    case "$cmd" in

        test)
            run_local
            ;;

        prod)
            run_docker 0
            ;;

        gpu)
            run_docker 1
            ;;

        install)
            install_dependencies
            ;;

        install-gpu)
            install_gpu_dependencies
            ;;

        logs)
            show_logs
            ;;

        stop)
            stop_containers
            ;;

        rebuild)
            rebuild
            ;;

        status)
            show_status
            ;;

        clean)
            clean_environment
            ;;

        doctor)
            run_doctor
            ;;

        help|-h|--help)
            show_help
            ;;

        menu)
            interactive_menu
            ;;

        *)
            fatal "Comando inválido: ${cmd}"
            ;;

    esac
}

main "$@"
```

## Melhorias aplicadas

* Arquitetura modular por domínio
* Logging persistente
* Error trapping global
* Melhor UX operacional
* Menu contextual organizado
* Healthcheck robusto
* Compatibilidade Linux/macOS/WSL2
* Melhor tratamento Docker Compose
* SIGTERM antes de SIGKILL
* Melhor portabilidade Bash
* Status operacional no topo
* Timeout em docker stats
* Estrutura pronta para CI/CD
* Fluxo operacional mais previsível
* Redução de duplicação
* Melhor separação runtime/setup
* Melhor observabilidade
