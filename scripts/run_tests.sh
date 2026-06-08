#!/usr/bin/env bash
# ==============================================================================
# Runner padronizado da suíte de testes do XFakeSong.
# Categorias espelham as subpastas de tests/ (marcadas automaticamente pelo
# conftest). Veja docs/06_TESTES.md (caderno de testes).
#
# Uso: ./scripts/run_tests.sh [categoria] [args extras do pytest...]
#   fast | default   suíte rápida (tudo EXCETO smoke) — o run padrão da CI
#   unit             só testes unitários
#   api              só testes de contrato da API
#   functional       só fluxos de frontend
#   integration      só integração (pode treinar modelos — mais lento)
#   smoke            só smoke pesados (TF real)
#   all              TUDO, inclusive smoke
#   cov              suíte rápida + cobertura (app + benchmarks)
#   list             lista os testes que seriam coletados (não executa)
# ==============================================================================
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Usa o python do venv se existir; senão o do PATH.
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    PY=".venv/Scripts/python.exe"
else
    PY="python"
fi
PYTEST=("$PY" -m pytest)

cat="${1:-fast}"
shift || true

case "$cat" in
    fast|default)  "${PYTEST[@]}" tests/ "$@" ;;                 # exclui smoke (addopts)
    unit)          "${PYTEST[@]}" -m unit tests/unit "$@" ;;
    api)           "${PYTEST[@]}" -m api tests/api "$@" ;;
    functional)    "${PYTEST[@]}" -m functional tests/functional "$@" ;;
    integration)   "${PYTEST[@]}" -m integration tests/integration "$@" ;;
    smoke)         "${PYTEST[@]}" -m smoke tests/smoke "$@" ;;   # pesado, TF real
    all)           "${PYTEST[@]}" -m "" tests/ "$@" ;;           # tudo, inclui smoke
    cov)           "${PYTEST[@]}" --cov=app --cov=benchmarks \
                       --cov-report=term-missing tests/ "$@" ;;
    list)          "${PYTEST[@]}" --collect-only -q tests/ "$@" ;;
    -h|--help|help)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
    *)
        echo "Categoria desconhecida: '$cat'" >&2
        echo "Use: fast|unit|api|functional|integration|smoke|all|cov|list" >&2
        exit 2 ;;
esac
