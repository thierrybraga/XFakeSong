#!/usr/bin/env bash
# =====================================================================
# Build Smoke Test — XFakeSong (PROD.10)
# =====================================================================
# Verifica que a imagem Docker builda, sobe, healthcheck passa e endpoints
# críticos respondem. Use ANTES de promover uma imagem para produção.
#
# Uso:
#   bash scripts/build_smoke_test.sh                 # build CPU + tests
#   bash scripts/build_smoke_test.sh --tf-cpu        # build com TF_VARIANT=cpu
#   bash scripts/build_smoke_test.sh --gpu           # build + start com GPU
#   bash scripts/build_smoke_test.sh --skip-build    # só roda os testes
# =====================================================================
set -uo pipefail

# Cores
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'

IMAGE_TAG="xfakesong:smoke"
CONTAINER_NAME="xfakesong_smoke"
PORT="${PORT:-7868}"   # porta alternativa para não conflitar com instância rodando
HOST="127.0.0.1"
HEALTH_TIMEOUT=240     # segundos para o healthcheck virar healthy
USE_GPU=0
TF_VARIANT=""
SKIP_BUILD=0

# ----- argv parsing -----
for arg in "$@"; do
    case "$arg" in
        --gpu)        USE_GPU=1 ;;
        --tf-cpu)     TF_VARIANT="cpu" ;;
        --skip-build) SKIP_BUILD=1 ;;
        --port=*)     PORT="${arg#*=}" ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "arg desconhecido: $arg"; exit 1 ;;
    esac
done

step() { echo -e "${B}==>${N} $*"; }
ok()   { echo -e "  ${G}OK${N}  $*"; }
warn() { echo -e "  ${Y}WARN${N} $*"; }
fail() { echo -e "  ${R}FAIL${N} $*"; cleanup; exit 1; }

cleanup() {
    step "Cleanup..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Detecta compose v2 vs v1 (não obrigatório aqui, mas útil)
DC=$(docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# ============================================================
# 1. BUILD
# ============================================================
if [ "$SKIP_BUILD" -eq 0 ]; then
    step "Building image (TF_VARIANT='$TF_VARIANT')..."
    BUILD_ARGS=()
    if [ -n "$TF_VARIANT" ]; then
        BUILD_ARGS+=(--build-arg "TF_VARIANT=$TF_VARIANT")
    fi
    if ! docker build "${BUILD_ARGS[@]}" -t "$IMAGE_TAG" . ; then
        fail "docker build falhou"
    fi
    ok "build concluído"
else
    step "Skipping build (--skip-build)"
    docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 || \
        fail "imagem $IMAGE_TAG não existe, remova --skip-build"
fi

# Tamanho da imagem
SIZE=$(docker image inspect "$IMAGE_TAG" --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo "?")
ok "imagem $IMAGE_TAG (size=$SIZE)"

# ============================================================
# 2. START
# ============================================================
cleanup  # remove resíduo anterior

step "Starting container on port $PORT..."
GPU_FLAGS=()
if [ "$USE_GPU" -eq 1 ]; then
    GPU_FLAGS+=(--gpus all)
    step "GPU mode enabled (--gpus all)"
fi

# Mounts mínimos (read-only para evitar resíduos no host)
docker run -d \
    --name "$CONTAINER_NAME" \
    --rm=false \
    -p "$PORT:7860" \
    -e DEEPFAKE_ENV=production \
    -e DEEPFAKE_LOG_LEVEL=INFO \
    -e ALLOWED_HOSTS=* \
    "${GPU_FLAGS[@]}" \
    "$IMAGE_TAG" >/dev/null \
    || fail "docker run falhou"
ok "container started"

# ============================================================
# 3. WAIT FOR HEALTHY
# ============================================================
step "Aguardando healthcheck virar healthy (max ${HEALTH_TIMEOUT}s)..."
elapsed=0
interval=5
while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
    case "$status" in
        healthy)
            ok "container healthy após ${elapsed}s"
            break
            ;;
        unhealthy)
            echo ""
            docker logs --tail=80 "$CONTAINER_NAME"
            fail "container ficou unhealthy"
            ;;
        "")
            # Sem healthcheck definido ou container parou — verifica running
            running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo "false")
            if [ "$running" != "true" ]; then
                echo ""
                docker logs --tail=80 "$CONTAINER_NAME"
                fail "container parou"
            fi
            ;;
    esac
    printf "\r  status: %-12s (%ds)" "$status" "$elapsed"
    sleep "$interval"
    elapsed=$((elapsed + interval))
done

if [ "$elapsed" -ge "$HEALTH_TIMEOUT" ]; then
    echo ""
    docker logs --tail=80 "$CONTAINER_NAME"
    fail "timeout aguardando healthy"
fi
echo ""

# ============================================================
# 4. ENDPOINT TESTS
# ============================================================
step "Testando endpoints HTTP..."

test_endpoint() {
    local method="$1"
    local path="$2"
    local expected="$3"
    local desc="$4"

    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" \
                --max-time 10 -X "$method" \
                "http://${HOST}:${PORT}${path}")

    if [ "$code" = "$expected" ]; then
        ok "$method $path → $code ($desc)"
        return 0
    else
        warn "$method $path → $code (esperado: $expected) — $desc"
        return 1
    fi
}

errors=0

# System endpoints (críticos)
test_endpoint GET "/api/v1/system/health"    200 "health check" || errors=$((errors+1))
test_endpoint GET "/api/v1/system/status"    200 "status" || errors=$((errors+1))
test_endpoint GET "/api/v1/system/version"   200 "version" || errors=$((errors+1))
test_endpoint GET "/api/v1/system/info"      200 "info snapshot" || errors=$((errors+1))
test_endpoint GET "/api/v1/system/bootstrap" 200 "bootstrap" || errors=$((errors+1))

# Detection (apenas GETs sem upload)
test_endpoint GET "/api/v1/detection/models"        200 "models list" || errors=$((errors+1))
test_endpoint GET "/api/v1/detection/architectures" 200 "architectures" || errors=$((errors+1))

# Features
test_endpoint GET "/api/v1/features/types"   200 "feature types" || errors=$((errors+1))

# Training
test_endpoint GET "/api/v1/training/architectures" 200 "train archs" || errors=$((errors+1))

# History
test_endpoint GET "/api/v1/history/"         200 "history list" || errors=$((errors+1))

# Datasets
test_endpoint GET "/api/v1/datasets/"        200 "datasets list" || errors=$((errors+1))

# Voice profiles
test_endpoint GET "/api/v1/profiles/"        200 "profiles list" || errors=$((errors+1))

# Docs
test_endpoint GET "/api/docs"                200 "swagger docs" || errors=$((errors+1))
test_endpoint GET "/api/openapi.json"        200 "openapi schema" || errors=$((errors+1))

# Gradio root deve retornar HTML
step "Testando renderização do Gradio (raiz)..."
gradio_response=$(curl -s --max-time 15 "http://${HOST}:${PORT}/" | head -200)
if echo "$gradio_response" | grep -qi "gradio\|XfakeSong"; then
    ok "Gradio renderiza HTML na raiz /"
else
    warn "Gradio não retornou HTML esperado na raiz"
    errors=$((errors+1))
fi

# ============================================================
# 5. CHECK CONTAINER STATS
# ============================================================
step "Container stats:"
docker stats --no-stream --format \
    "  cpu: {{.CPUPerc}}  |  mem: {{.MemUsage}}  |  net: {{.NetIO}}" \
    "$CONTAINER_NAME" || true

# ============================================================
# 6. SUMMARY
# ============================================================
echo ""
echo "============================================================="
if [ "$errors" -eq 0 ]; then
    echo -e "${G}  BUILD SMOKE TEST: PASSED${N}"
    echo "  Image: $IMAGE_TAG (size=$SIZE)"
    echo "  All endpoints responded as expected."
    exit_code=0
else
    echo -e "${R}  BUILD SMOKE TEST: FAILED ($errors error(s))${N}"
    echo ""
    echo "Últimas linhas do log do container:"
    docker logs --tail=40 "$CONTAINER_NAME" 2>&1 | sed 's/^/    /'
    exit_code=1
fi
echo "============================================================="

exit $exit_code
