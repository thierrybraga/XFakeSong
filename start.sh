#!/usr/bin/env bash
# =====================================================================
# XFakeSong Launcher (Linux / macOS / WSL2)
# Wrapper para builds locais + docker compose
# Uso: ./start.sh [test|prod|gpu|stop|logs|rebuild|clean|install|install-gpu|wsl-setup|gpu-test|bootstrap]
#
# Modos GPU:
#   install-gpu  → Instala TF com suporte CUDA (tensorflow[and-cuda]) em Linux/WSL2
#   wsl-setup    → Mostra passos completos de setup CUDA dentro do WSL2
#   gpu-test     → Roda diagnóstico standalone (sem iniciar app)
#   test         → Modo TESTE local (Python) — usa GPU se disponível
# =====================================================================
set -uo pipefail

# Cores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# -------------------- helpers --------------------

show_header() {
    clear
    echo -e "${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}                              XFAKESONG LAUNCHER                               ${NC}"
    echo -e "${BLUE}===============================================================================${NC}"
    echo ""
}

# Detecta variante do compose: 'docker compose' (v2) ou 'docker-compose' (v1)
detect_compose() {
    if docker compose version &>/dev/null; then
        echo "docker compose"
        return 0
    fi
    if command -v docker-compose &>/dev/null; then
        echo "docker-compose"
        return 0
    fi
    return 1
}

check_docker_running() {
    if ! docker info &>/dev/null; then
        echo -e "${RED}Docker daemon não está rodando.${NC}"
        echo -e "${YELLOW}Inicie o Docker Desktop ou systemctl start docker${NC}"
        return 1
    fi
    return 0
}

check_python() {
    if command -v python3 &>/dev/null; then
        PY=python3
    elif command -v python &>/dev/null; then
        PY=python
    else
        echo -e "${RED}Python 3 não encontrado!${NC}"
        return 1
    fi
    export PY
    return 0
}

# Detecta se estamos em WSL (Linux dentro do Windows)
is_wsl() {
    if [ -n "${WSL_DISTRO_NAME:-}" ]; then return 0; fi
    if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then return 0; fi
    return 1
}

# Detecta GPU NVIDIA via nvidia-smi (funciona em Linux nativo E WSL2)
check_nvidia_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    nvidia-smi -L >/dev/null 2>&1
}

# Diz se o pacote TF atual está compilado com CUDA
tf_has_cuda() {
    [ -d ".venv" ] || return 1
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -c "import tensorflow as tf; import sys; sys.exit(0 if tf.test.is_built_with_cuda() else 1)" 2>/dev/null
}

# Resolve comando docker compose (v1 ou v2) UMA VEZ
get_compose() {
    if [ -z "${DC:-}" ]; then
        DC=$(detect_compose) || {
            echo -e "${RED}Docker Compose não encontrado.${NC}"
            echo -e "${YELLOW}Instale Docker Desktop (inclui Compose v2)${NC}"
            return 1
        }
        export DC
    fi
    echo "$DC"
}

wait_for_healthy() {
    # Aguarda container ficar healthy com timeout de 5 min
    local container="xfakesong_app"
    local max_wait=300  # 5 min
    local elapsed=0
    local interval=5

    echo -e "${YELLOW}Aguardando container ficar healthy (max ${max_wait}s)...${NC}"
    while [ $elapsed -lt $max_wait ]; do
        local status
        status=$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo "")
        if [ -z "$status" ]; then
            echo -e "${RED}Container '$container' não encontrado.${NC}"
            return 1
        fi

        printf "\r  Status: %-15s (${elapsed}s)" "$status"

        case "$status" in
            healthy)
                echo ""
                echo -e "${GREEN}✓ Container healthy${NC}"
                return 0
                ;;
            unhealthy)
                echo ""
                echo -e "${RED}✗ Container unhealthy. Veja logs:${NC}"
                $DC logs --tail=50 app
                return 1
                ;;
            "")
                echo ""
                echo -e "${RED}✗ Container parou inesperadamente.${NC}"
                return 1
                ;;
        esac

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo ""
    echo -e "${RED}✗ Timeout aguardando healthy (${max_wait}s)${NC}"
    return 1
}

# -------------------- ações --------------------

check_port_local() {
    # BUG.Startup.3: pré-check porta 7860 antes de iniciar localmente
    local port="${1:-7860}"
    if lsof -i ":${port}" >/dev/null 2>&1 || \
       (command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":${port} "); then
        local old_pid
        old_pid=$(lsof -ti ":${port}" 2>/dev/null | head -1)
        echo ""
        echo -e "${RED}===============================================================${NC}"
        echo -e "${RED} Porta ${port} já em uso${NC}${old_pid:+ pelo PID $old_pid}${NC}"
        echo -e "${RED}===============================================================${NC}"
        echo "Encerre a instância anterior:"
        [ -n "$old_pid" ] && echo "  kill -9 ${old_pid}"
        echo "Ou rode em outra porta:"
        echo "  python main.py --gradio --gradio-port 7861"
        echo ""
        if [ -n "$old_pid" ]; then
            read -rp "Encerrar PID ${old_pid} automaticamente? [y/N] " kill_yn
            if [[ "$kill_yn" =~ ^[yY] ]]; then
                kill -9 "$old_pid" 2>/dev/null && \
                    { echo "PID encerrado. Aguardando porta liberar..."; sleep 3; } || \
                    return 1
            else
                return 1
            fi
        else
            return 1
        fi
    fi
    return 0
}

mode_test() {
    show_header
    echo -e "${GREEN}Iniciando Modo TESTE (Local Python)...${NC}"
    check_python || return 1

    if [ ! -d ".venv" ]; then
        echo "Criando ambiente virtual..."
        $PY -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip >/dev/null
    pip install -r requirements.txt

    # BUG.Startup.3: verifica porta livre antes de iniciar
    check_port_local 7860 || return 1

    # GPU.8: Diagnóstico rápido de GPU antes de iniciar — informa ao usuário
    # se está rodando com aceleração ou em CPU.
    echo ""
    echo -e "${BLUE}--- Status de GPU ---${NC}"
    if check_nvidia_gpu; then
        nvidia-smi -L | sed 's/^/  /'
        if tf_has_cuda; then
            echo -e "${GREEN}  ✓ TensorFlow tem suporte CUDA. GPU será usada para treino/inferência.${NC}"
        else
            echo -e "${YELLOW}  ⚠ GPU NVIDIA detectada mas TF está sem CUDA.${NC}"
            echo -e "${YELLOW}    Rode: ./start.sh install-gpu  (para instalar tensorflow[and-cuda])${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠ nvidia-smi não encontrado — modo CPU.${NC}"
        if is_wsl; then
            echo "    (Você está em WSL — atualize o driver NVIDIA do Windows ≥ 525.x)"
        fi
    fi
    echo ""

    echo "Iniciando aplicação em http://localhost:7860 ..."
    $PY main.py --gradio --gradio-port 7860
}

# GPU.8: Instala TF com suporte CUDA (Linux/WSL2)
mode_install_gpu() {
    show_header
    echo -e "${GREEN}Instalando TensorFlow com suporte CUDA (GPU)...${NC}"
    echo ""

    # 1. Pré-requisito: nvidia-smi precisa estar disponível
    if ! check_nvidia_gpu; then
        echo -e "${RED}✗ nvidia-smi não encontrado ou GPU não detectada.${NC}"
        echo ""
        if is_wsl; then
            echo -e "${YELLOW}Você está em WSL2 mas a GPU não está visível.${NC}"
            echo "Soluções:"
            echo "  1. Driver NVIDIA Windows ≥ 525.x:  https://www.nvidia.com/drivers"
            echo "  2. WSL2 atualizado:  wsl --update  (em PowerShell admin)"
            echo "  3. Reinicie o WSL:   wsl --shutdown  e abra novamente"
        else
            echo -e "${YELLOW}Instale drivers NVIDIA primeiro (Linux):${NC}"
            echo "  Ubuntu/Debian:  sudo apt install nvidia-driver-535"
            echo "  Reinicie o sistema após instalar."
        fi
        return 1
    fi
    nvidia-smi -L | sed 's/^/  /'

    check_python || return 1
    [ -d ".venv" ] || $PY -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip

    echo ""
    echo -e "${BLUE}Removendo build CPU anterior...${NC}"
    pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel 2>/dev/null || true

    echo ""
    echo -e "${BLUE}Instalando tensorflow[and-cuda]...${NC}"
    echo "(Download ~ 1 GB — leva alguns minutos)"
    pip install --upgrade 'tensorflow[and-cuda]>=2.16,<2.22' || {
        echo -e "${RED}Falha ao instalar tensorflow[and-cuda].${NC}"
        echo "Possíveis causas:"
        echo "  - Python > 3.13: TF 2.20+ ainda não tem wheels CUDA para 3.13 oficiais"
        echo "  - Sem espaço em disco (~3 GB necessário)"
        echo "  - Pacote pip outdated: pip install --upgrade pip"
        return 1
    }

    echo ""
    echo -e "${BLUE}Instalando resto das dependências...${NC}"
    pip install -r requirements.txt

    echo ""
    echo -e "${BLUE}Validando setup GPU...${NC}"
    python - <<'PYEOF'
import tensorflow as tf
print(f"TF:                  {tf.__version__}")
print(f"built_with_cuda:     {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs visíveis:       {len(gpus)}")
for i, g in enumerate(gpus):
    d = tf.config.experimental.get_device_details(g)
    print(f"  #{i}: {d.get('device_name', g)} CC={d.get('compute_capability', '?')}")
if not gpus:
    print()
    print("⚠ TF compilado com CUDA mas não está expondo a GPU.")
    print("  Possíveis causas:")
    print("  - Driver NVIDIA Windows < 525.x (atualize)")
    print("  - cuDNN incompatível — pip install nvidia-cudnn-cu12")
    print("  - LD_LIBRARY_PATH precisa apontar para libs CUDA")
PYEOF

    echo ""
    echo -e "${GREEN}✓ Instalação concluída. Inicie com: ./start.sh test${NC}"
}

# GPU.8: Setup completo de WSL2 + CUDA (interativo, informativo)
mode_wsl_setup() {
    show_header
    echo -e "${GREEN}Setup WSL2 + GPU CUDA — Passo a Passo${NC}"
    echo ""

    if ! is_wsl; then
        echo -e "${YELLOW}Você NÃO está em WSL.${NC}"
        echo "Este modo é específico para WSL2 (Linux dentro do Windows)."
        echo ""
        echo "Para entrar no WSL primeiro:"
        echo "  1. No PowerShell (admin):  wsl --install -d Ubuntu"
        echo "  2. Reinicie o computador"
        echo "  3. Abra o app 'Ubuntu' no menu Iniciar"
        echo "  4. Clone o repo:  git clone <url> e cd no diretório"
        echo "  5. Rode novamente:  ./start.sh wsl-setup"
        return 1
    fi

    echo -e "${BLUE}Você está em WSL: ${WSL_DISTRO_NAME:-?}${NC}"
    echo ""

    # 1. nvidia-smi
    echo -e "${BLUE}[1/4] Verificando driver GPU passthrough...${NC}"
    if check_nvidia_gpu; then
        nvidia-smi -L | sed 's/^/  /'
        echo -e "${GREEN}  ✓ GPU visível via passthrough WSL2.${NC}"
    else
        echo -e "${RED}  ✗ GPU não visível no WSL.${NC}"
        echo ""
        echo "Solução (no Windows host):"
        echo "  1. Atualize driver NVIDIA Windows ≥ 525.x: https://www.nvidia.com/drivers"
        echo "  2. PowerShell admin:  wsl --update"
        echo "  3. Feche todas as janelas WSL:  wsl --shutdown"
        echo "  4. Reabra o WSL e rode este script novamente."
        return 1
    fi
    echo ""

    # 2. Python
    echo -e "${BLUE}[2/4] Verificando Python...${NC}"
    if ! check_python; then
        echo "Instalando Python 3 + venv..."
        sudo apt update && sudo apt install -y python3 python3-pip python3-venv
        check_python || return 1
    fi
    $PY --version
    echo ""

    # 3. TF com CUDA
    echo -e "${BLUE}[3/4] Instalando TF com suporte CUDA...${NC}"
    mode_install_gpu || return 1
    echo ""

    # 4. Done
    echo -e "${BLUE}[4/4] Pronto!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Setup WSL2 + GPU concluído.${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Iniciar app:        ./start.sh test"
    echo "Testar GPU:         ./start.sh gpu-test"
}

# GPU.8: Roda app/core/gpu de forma standalone (sem subir Gradio)
mode_gpu_test() {
    show_header
    echo -e "${GREEN}Diagnóstico de GPU standalone...${NC}"
    echo ""
    check_python || return 1
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Sem .venv. Rode primeiro: ./start.sh install${NC}"
        return 1
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python - <<'PYEOF'
from app.core.gpu import setup_gpu, describe_gpu_setup, get_setup_result
import json
r = setup_gpu()
print()
print("STATUS:", describe_gpu_setup())
print()
print("DETALHES:")
print(json.dumps({k: v for k, v in r.items() if k != "errors"}, indent=2, default=str))
if r.get("errors"):
    print("\nWARNINGS:")
    for e in r["errors"]:
        print(f"  - {e}")
PYEOF
}

mode_prod() {
    local gpu="${1:-0}"
    show_header
    echo -e "${GREEN}Iniciando Modo PRODUÇÃO (Docker)...${NC}"

    check_docker_running || return 1
    DC=$(get_compose) || return 1

    local compose_args="-f docker-compose.yml"
    if [ "$gpu" = "1" ]; then
        echo -e "${BLUE}GPU mode habilitado (NVIDIA)${NC}"
        compose_args="$compose_args -f docker-compose.gpu.yml"
    fi

    echo "Build + Up..."
    # shellcheck disable=SC2086
    $DC $compose_args up --build -d

    wait_for_healthy && {
        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  Aplicação rodando em http://localhost:7860${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
        echo ""
        echo "Comandos úteis:"
        echo "  Logs:    ./start.sh logs"
        echo "  Stop:    ./start.sh stop"
        echo "  Rebuild: ./start.sh rebuild"
    }
}

mode_stop() {
    show_header
    echo -e "${YELLOW}Parando containers...${NC}"
    check_docker_running || return 1
    DC=$(get_compose) || return 1
    $DC down
    echo -e "${GREEN}✓ Containers parados${NC}"
}

mode_logs() {
    check_docker_running || return 1
    DC=$(get_compose) || return 1
    $DC logs -f --tail=100 app
}

mode_rebuild() {
    show_header
    echo -e "${YELLOW}Forçando rebuild SEM cache...${NC}"
    check_docker_running || return 1
    DC=$(get_compose) || return 1

    $DC down
    $DC build --no-cache --pull
    $DC up -d
    wait_for_healthy
}

mode_clean() {
    show_header
    echo -e "${RED}LIMPEZA: remove containers, volumes anônimos e imagens dangling${NC}"
    read -rp "Continuar? [y/N] " yn
    case "$yn" in
        [Yy]*) ;;
        *) echo "Cancelado."; return 0 ;;
    esac

    check_docker_running || return 1
    DC=$(get_compose) || return 1

    $DC down -v --remove-orphans
    docker image prune -f
    docker builder prune -f
    echo -e "${GREEN}✓ Limpeza concluída${NC}"
}

mode_install() {
    show_header
    echo "Instalando dependências locais..."
    check_python || return 1
    [ -d ".venv" ] || $PY -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Concluído${NC}"
}

mode_bootstrap() {
    show_header
    echo "Criando estrutura de diretórios..."
    check_python || return 1
    $PY main.py --bootstrap-dirs
    echo -e "${GREEN}✓ Concluído${NC}"
}

mode_status() {
    show_header
    check_docker_running || return 1
    DC=$(get_compose) || return 1
    $DC ps
    echo ""
    docker stats --no-stream --format \
        "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" 2>/dev/null \
        | grep -E "xfakesong|NAME" || true
}

# -------------------- menu interativo --------------------

show_menu() {
    show_header
    echo "Selecione uma ação:"
    echo ""
    echo "  ${GREEN}1)${NC} Modo TESTE (Python local — usa GPU se disponível)"
    echo "  ${GREEN}2)${NC} Modo PRODUÇÃO (Docker)"
    echo "  ${GREEN}3)${NC} Modo PRODUÇÃO + GPU (NVIDIA via Docker)"
    echo ""
    echo "  ${BLUE}── GPU local (Linux / WSL2) ──${NC}"
    echo "  ${GREEN}4)${NC} Instalar TF + CUDA local (tensorflow[and-cuda])"
    echo "  ${GREEN}5)${NC} Setup completo WSL2 + GPU (interativo)"
    echo "  ${GREEN}6)${NC} Diagnóstico de GPU (standalone)"
    echo ""
    echo "  ${BLUE}── Docker / runtime ──${NC}"
    echo "  ${GREEN}7)${NC} Stop containers"
    echo "  ${GREEN}8)${NC} Ver logs (follow)"
    echo "  ${GREEN}9)${NC} Rebuild SEM cache"
    echo "  ${GREEN}10)${NC} Status / health"
    echo "  ${GREEN}11)${NC} Limpeza profunda (down -v + prune)"
    echo ""
    echo "  ${BLUE}── Setup ──${NC}"
    echo "  ${GREEN}12)${NC} Instalar dependências locais (CPU)"
    echo "  ${GREEN}13)${NC} Bootstrap diretórios"
    echo ""
    echo "  ${GREEN}0)${NC} Sair"
    echo ""
}

# -------------------- dispatcher --------------------

# Modo argumento direto (auto, p/ CI): ./start.sh prod
case "${1:-}" in
    test)         mode_test; exit $? ;;
    prod)         mode_prod 0; exit $? ;;
    gpu)          mode_prod 1; exit $? ;;
    install-gpu)  mode_install_gpu; exit $? ;;
    wsl-setup)    mode_wsl_setup; exit $? ;;
    gpu-test)     mode_gpu_test; exit $? ;;
    stop)         mode_stop; exit $? ;;
    logs)         mode_logs; exit $? ;;
    rebuild)      mode_rebuild; exit $? ;;
    clean)        mode_clean; exit $? ;;
    install)      mode_install; exit $? ;;
    bootstrap)    mode_bootstrap; exit $? ;;
    status)       mode_status; exit $? ;;
    help|-h|--help)
        cat <<EOF
Uso: $0 <comando>

App:
  test            Modo TESTE local (Python, usa GPU se disponível)
  prod            Modo PRODUÇÃO via Docker
  gpu             Modo PRODUÇÃO + GPU via Docker (NVIDIA)

GPU local (Linux/WSL2):
  install-gpu     Instala tensorflow[and-cuda] no .venv
  wsl-setup       Setup completo WSL2 + GPU (interativo)
  gpu-test        Diagnóstico standalone (sem subir app)

Docker / runtime:
  stop            Parar containers
  logs            Tail dos logs (follow)
  rebuild         Rebuild sem cache
  clean           Limpeza profunda (volumes + prune)
  status          ps + stats

Setup:
  install         Instala dependências CPU (requirements.txt)
  bootstrap       Cria diretórios padrão

Sem argumentos abre o menu interativo.
EOF
        exit 0
        ;;
esac

# Menu interativo
while true; do
    show_menu
    read -rp "Opção: " opt
    case "$opt" in
        1)  mode_test ;;
        2)  mode_prod 0 ;;
        3)  mode_prod 1 ;;
        4)  mode_install_gpu ;;
        5)  mode_wsl_setup ;;
        6)  mode_gpu_test ;;
        7)  mode_stop ;;
        8)  mode_logs ;;
        9)  mode_rebuild ;;
        10) mode_status ;;
        11) mode_clean ;;
        12) mode_install ;;
        13) mode_bootstrap ;;
        0)  exit 0 ;;
        *)  echo -e "${RED}Opção inválida${NC}"; sleep 1 ;;
    esac
    echo ""
    read -rp "Pressione Enter para continuar..." _
done
