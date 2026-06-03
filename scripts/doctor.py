#!/usr/bin/env python3
"""XFakeSong Doctor — diagnóstico rápido de problemas de instalação/inicialização.

Uso:
    python scripts/doctor.py           # diagnóstico completo
    python scripts/doctor.py --fix     # tenta corrigir automaticamente

Verifica:
- Versão do Python compatível
- Virtualenv ativado / detectado
- Dependências críticas instaladas
- Porta 7860 disponível
- Diretórios writable
- Banco de dados inicializável
- Compatibilidade de versões (gradio × starlette)

Sai com code 0 se OK, 1 se há issues, 2 se há issues críticos.
"""

from __future__ import annotations

import argparse
import importlib
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Força UTF-8 no stdout — em cmd.exe Windows o default é cp1252 que NÃO
# suporta acentos nem caracteres especiais. errors='replace' evita crash
# se algum char não-representável passar.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# O doctor roda de `scripts/` — garante a raiz do projeto no sys.path para que
# `import app.core...` (GPU, database) funcione mesmo invocado por caminho.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ───────────────────────────── helpers ─────────────────────────────

class Color:
    """Códigos ANSI básicos. No Windows com Terminal moderno funcionam."""
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {Color.OK}OK{Color.END}    {msg}")


def warn(msg: str) -> None:
    print(f"  {Color.WARN}WARN{Color.END}  {msg}")


def err(msg: str) -> None:
    print(f"  {Color.ERR}FAIL{Color.END}  {msg}")


def section(title: str) -> None:
    # ASCII puro — cp1252 (default cmd.exe Windows) não tem ─ ═
    print(f"\n{Color.BOLD}-- {title} --{Color.END}")


# ───────────────────────────── checks ─────────────────────────────

def check_python_version() -> Tuple[bool, List[str]]:
    """Python 3.11+ requerido. 3.13 OK desde TF 2.20."""
    section("Python")
    v = sys.version_info
    print(f"  Versão: {v.major}.{v.minor}.{v.micro}")
    print(f"  Executável: {sys.executable}")

    errors = []
    if v.major != 3:
        err(f"Python 3.x esperado, got {v.major}.x")
        errors.append("python_version")
    elif v.minor < 11:
        err(f"Python 3.11+ requerido (got 3.{v.minor})")
        errors.append("python_version")
    elif v.minor >= 14:
        warn(f"Python 3.{v.minor} é muito novo — pacotes podem não ter wheels")
    elif v.minor == 13:
        ok(f"Python 3.13 (precisa TF 2.20+, OK no requirements atual)")
    else:
        ok(f"Python 3.{v.minor} (versão recomendada)")
    return not errors, errors


def check_virtualenv() -> Tuple[bool, List[str]]:
    """Virtualenv ativado ou .venv presente."""
    section("Virtualenv")
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")
    venv_dir = Path(".venv")

    if in_venv:
        ok(f"Rodando em virtualenv ({sys.prefix})")
    elif venv_dir.exists():
        warn(f".venv existe mas não está ativado. "
             f"Ative com: .venv\\Scripts\\activate (Win) ou source .venv/bin/activate")
    else:
        warn("Nenhum venv detectado. Recomendado: python -m venv .venv")
    return True, []  # não-crítico


def check_critical_deps() -> Tuple[bool, List[str]]:
    """Dependências críticas instaladas?"""
    section("Dependências")
    # (import_name, pip_name, desc) — `import X` usa import_name; pip_name é o
    # nome no requirements. Alguns DIFEREM: scikit-learn→sklearn, PyYAML→yaml.
    # Usar o nome errado aqui gerava falsos "AUSENTE" (críticos) em todo SO.
    critical = [
        ("tensorflow", "tensorflow", "ML framework"),
        ("keras", "keras", "ML high-level"),
        ("librosa", "librosa", "Áudio"),
        ("soundfile", "soundfile", "Áudio I/O"),
        ("numpy", "numpy", "Numérico"),
        ("sklearn", "scikit-learn", "ML clássico"),
        ("gradio", "gradio", "UI"),
        ("fastapi", "fastapi", "API"),
        ("starlette", "starlette", "API base"),
        ("uvicorn", "uvicorn", "ASGI server"),
        ("sqlalchemy", "SQLAlchemy", "DB"),
        ("pydantic", "pydantic", "Schemas"),
        ("requests", "requests", "HTTP client"),
        ("yaml", "PyYAML", "YAML"),
        ("matplotlib", "matplotlib", "Plots"),
    ]
    optional = [
        ("datasets", "HF datasets (scripts download)"),
        ("optuna", "Hyperparameter tuning (Sprint 4.2)"),
        ("mlflow", "Experiment tracking (Sprint 4.3)"),
        ("tf2onnx", "ONNX export (Sprint 3.4)"),
        ("onnxruntime", "ONNX runtime (Sprint 3.4)"),
        ("tensorflow_model_optimization", "QAT (Sprint 5.2)"),
    ]

    errors: List[str] = []
    missing_critical: List[str] = []
    broken: List[Tuple[str, str]] = []
    for import_name, pip_name, desc in critical:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            ok(f"{pip_name:<22} {version:<12} {Color.DIM}({desc}){Color.END}")
        except ModuleNotFoundError as e:
            top = (getattr(e, "name", "") or "").split(".")[0]
            if top == import_name:
                # O próprio pacote está ausente
                err(f"{pip_name:<22} {'AUSENTE':<14} {Color.DIM}({desc}){Color.END}")
                errors.append(f"missing_dep:{pip_name}")
                missing_critical.append(pip_name)
            else:
                # Pacote presente, mas uma sub-dependência dele falta
                err(f"{pip_name:<22} {'IMPORT QUEBRADO':<14} {Color.DIM}(falta '{top}'){Color.END}")
                errors.append(f"broken_import:{pip_name}")
                broken.append((pip_name, f"dependência ausente: {top}"))
        except ImportError as e:
            # Presente, mas import falha (ex.: API removida numa sub-dependência —
            # típico de gradio×huggingface_hub incompatíveis: 'cannot import HfFolder')
            err(f"{pip_name:<22} {'IMPORT QUEBRADO':<14} {Color.DIM}({desc}){Color.END}")
            errors.append(f"broken_import:{pip_name}")
            broken.append((pip_name, str(e).splitlines()[0][:80]))

    if missing_critical:
        print()
        err(f"FALTAM {len(missing_critical)} deps críticas: {', '.join(missing_critical)}")
        print(f"  {Color.BOLD}Fix:{Color.END}  pip install -r requirements.txt")

    if broken:
        print()
        err(f"{len(broken)} dep(s) com IMPORT QUEBRADO (instalada, mas falha ao importar):")
        for name, detail in broken:
            print(f"    • {name}: {detail}")
        print(
            f"  {Color.BOLD}Causa comum:{Color.END} versão de uma sub-dependência "
            f"incompatível (ex.: huggingface_hub 1.x quebra gradio 4.x)."
        )
        print(f"  {Color.BOLD}Fix:{Color.END}  pip install -r requirements.txt --upgrade")

    print()
    print(f"  {Color.DIM}Opcionais (não bloqueiam startup):{Color.END}")
    for pkg, desc in optional:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            ok(f"{pkg:<28} {version:<10} {Color.DIM}({desc}){Color.END}")
        except ImportError:
            print(f"  {Color.DIM}— {pkg:<26} (ausente) {desc}{Color.END}")

    return not errors, errors


def check_version_compatibility() -> Tuple[bool, List[str]]:
    """Combinações conhecidas como problemáticas."""
    section("Compatibilidade de versões")
    try:
        from importlib.metadata import version
    except ImportError:
        warn("importlib.metadata não disponível, pulando")
        return True, []

    def safe(pkg: str):
        try:
            return version(pkg)
        except Exception:
            return None

    def parse(v: str) -> tuple:
        return tuple(int(x) if x.isdigit() else 0 for x in v.split(".")[:3])

    g = safe("gradio")
    s = safe("starlette")
    hub = safe("huggingface_hub") or safe("huggingface-hub")

    errors = []
    if g and s:
        if parse(g) < (4, 31, 0) and parse(s) >= (0, 36, 0):
            err(
                f"INCOMPATIBILIDADE: gradio=={g} + starlette=={s}\n"
                f"  → TypeError em runtime na primeira request a '/'.\n"
                f"  Fix: pip install 'gradio>=4.31,<5.0'"
            )
            errors.append("gradio_starlette_incompat")
        else:
            ok(f"gradio={g}, starlette={s} (compatíveis)")
    elif g:
        ok(f"gradio={g}")

    # gradio 4.x importa `HfFolder`, removido no huggingface_hub 1.0 → ImportError
    # que quebra TODA a UI. Pin correto: huggingface_hub>=0.20,<1.0.
    if g and hub:
        if parse(g) < (5, 0, 0) and parse(hub) >= (1, 0, 0):
            err(
                f"INCOMPATIBILIDADE: gradio=={g} + huggingface_hub=={hub}\n"
                f"  → ImportError: cannot import name 'HfFolder' — a UI não sobe.\n"
                f"  Fix: pip install 'huggingface_hub>=0.20,<1.0'"
            )
            errors.append("gradio_hfhub_incompat")
        else:
            ok(f"gradio={g}, huggingface_hub={hub} (compatíveis)")

    return not errors, errors


def check_gpu() -> Tuple[bool, List[str]]:
    """Diagnóstico GPU/CUDA (read-only). GPU é opcional — CPU é suportado.

    Reusa `app.core.gpu.probe_gpu_status()` para mostrar o MESMO diagnóstico
    acionável do app/dashboard, sem alterar o runtime do TensorFlow.
    """
    section("GPU / CUDA")
    try:
        from app.core.gpu import probe_gpu_status
    except Exception as e:
        warn(f"Módulo de GPU indisponível ({type(e).__name__}: {e}) — pulando")
        return True, []

    try:
        st = probe_gpu_status()
    except Exception as e:
        warn(f"Falha ao sondar GPU: {type(e).__name__}: {e}")
        return True, []

    plat = f"{st.get('system', '?')} {st.get('release', '')}".strip()
    if st.get("is_wsl"):
        plat += " (WSL2)"
    print(f"  Plataforma     : {plat}")
    print(
        f"  TensorFlow     : disponível={st.get('tf_available')} "
        f"· com CUDA={st.get('tf_built_with_cuda')}"
    )

    hw = st.get("nvidia_hardware", [])
    if hw:
        for g in hw:
            drv = f" · driver {g['driver_version']}" if g.get("driver_version") else ""
            print(
                f"  Hardware NVIDIA: {g.get('name', '?')} "
                f"({g.get('memory_total_mb', '?')} MB){drv}"
            )
    elif st.get("nvidia_pci_present"):
        print("  Hardware NVIDIA: visível no PCI, porém SEM driver")
    else:
        print("  Hardware NVIDIA: nenhum")

    tf_gpus = st.get("tf_gpus", [])
    diag = st.get("diagnosis", {}) or {}
    code = diag.get("diagnosis", "")

    if tf_gpus:
        names = ", ".join(
            f"{g['name']} (CC {g['compute_capability']}"
            f"{', Tensor Cores' if g.get('tensor_core') else ''})"
            for g in tf_gpus
        )
        ok(f"TensorFlow enxerga {len(tf_gpus)} GPU(s): {names}")
        return True, []

    if code == "no_gpu":
        ok("Sem GPU NVIDIA — modo CPU (suportado; treino/inferência mais lentos)")
        return True, []

    # Há GPU (ou hardware no PCI) mas o TF não a está usando → warning acionável
    warn(st.get("summary", "GPU presente mas o TensorFlow não a utiliza"))
    for h in diag.get("hints", []):
        print(f"    {Color.BOLD}->{Color.END} {h}")
    return False, [f"gpu:{code or 'not_visible'}"]


def check_port(port: int = 7860) -> Tuple[bool, List[str]]:
    """Porta disponível?"""
    section(f"Porta {port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        if result == 0:
            warn(f"Porta {port} está em uso. Possível instância antiga.")
            if sys.platform == "win32":
                print(f"  {Color.BOLD}Fix Windows:{Color.END}")
                print(f"    netstat -ano | findstr :{port}")
                print(f"    taskkill /F /PID <pid>")
            else:
                print(f"  {Color.BOLD}Fix Linux/Mac:{Color.END}")
                print(f"    lsof -i :{port}")
                print(f"    kill -9 <pid>  # OU: fuser -k {port}/tcp")
            return False, ["port_in_use"]
        else:
            ok(f"Porta {port} disponível")
    finally:
        sock.close()
    return True, []


def check_writable_dirs() -> Tuple[bool, List[str]]:
    """Diretórios críticos writable?"""
    section("Diretórios writable")
    dirs = ["logs", "app/models", "app/results", "data"]
    errors = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            warn(f"{d} não existe (será criado em runtime)")
            continue
        if not os.access(p, os.W_OK):
            err(f"{d} não é writable")
            errors.append(f"not_writable:{d}")
        else:
            ok(f"{d} writable")
    return not errors, errors


def check_db_init() -> Tuple[bool, List[str]]:
    """Pode inicializar o banco?"""
    section("Banco de dados")
    try:
        from app.core.database import check_database_health
        if check_database_health():
            ok("DB acessível")
        else:
            warn("check_database_health retornou False")
    except ImportError as e:
        warn(f"app.core.database indisponível (deps faltando?): {e}")
    except Exception as e:
        err(f"Erro DB: {type(e).__name__}: {e}")
        return False, ["db_error"]
    return True, []


def try_fix_install() -> bool:
    """Roda pip install -r requirements.txt."""
    section("Tentando corrigir instalação")
    if not Path("requirements.txt").exists():
        err("requirements.txt não encontrado")
        return False

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        )
        ok("pip install -r requirements.txt concluído")
        return True
    except subprocess.CalledProcessError as e:
        err(f"pip install falhou: {e}")
        return False


# ───────────────────────────── main ─────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true",
                        help="Tenta corrigir problemas automaticamente")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"{Color.BOLD}XFakeSong Doctor{Color.END}\n")

    all_errors: List[str] = []
    has_critical = False

    checks = [
        ("python", check_python_version, True),
        ("venv", check_virtualenv, False),
        ("deps", check_critical_deps, True),
        ("compat", check_version_compatibility, True),
        ("gpu", check_gpu, False),
        ("port", lambda: check_port(args.port), False),
        ("dirs", check_writable_dirs, False),
        ("db", check_db_init, False),
    ]

    for name, fn, is_critical in checks:
        try:
            passed, errors = fn()
            if errors:
                all_errors.extend(errors)
                if is_critical and not passed:
                    has_critical = True
        except Exception as e:
            err(f"Erro inesperado em check '{name}': {e}")
            all_errors.append(f"check_failed:{name}")

    # Summary
    print()
    print("=" * 60)
    if not all_errors:
        print(f"{Color.OK}{Color.BOLD}  STATUS: OK — pronto para iniciar{Color.END}")
        print(f"\n  Iniciar com:  python main.py --gradio --gradio-port {args.port}")
        return 0
    elif has_critical:
        print(f"{Color.ERR}{Color.BOLD}  STATUS: ERRO CRÍTICO — corrigir antes de iniciar{Color.END}")
        print(f"  Problemas: {', '.join(all_errors)}")

        if args.fix and "missing_dep" in str(all_errors):
            try_fix_install()
        else:
            print(f"\n  Tentar fix automático: python scripts/doctor.py --fix")
        return 2
    else:
        print(f"{Color.WARN}{Color.BOLD}  STATUS: WARNINGS — app pode iniciar mas tem ressalvas{Color.END}")
        print(f"  Avisos: {', '.join(all_errors)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
