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
    critical = [
        ("tensorflow", "ML framework", True),
        ("keras", "ML high-level", True),
        ("librosa", "Áudio", True),
        ("soundfile", "Áudio I/O", True),
        ("numpy", "Numérico", True),
        ("scikit-learn", "ML clássico", True),
        ("gradio", "UI", True),
        ("fastapi", "API", True),
        ("starlette", "API base", True),
        ("uvicorn", "ASGI server", True),
        ("sqlalchemy", "DB", True),
        ("pydantic", "Schemas", True),
        ("requests", "HTTP client", True),
        ("pyyaml", "YAML", True),
        ("matplotlib", "Plots", True),
    ]
    optional = [
        ("datasets", "HF datasets (scripts download)"),
        ("optuna", "Hyperparameter tuning (Sprint 4.2)"),
        ("mlflow", "Experiment tracking (Sprint 4.3)"),
        ("tf2onnx", "ONNX export (Sprint 3.4)"),
        ("onnxruntime", "ONNX runtime (Sprint 3.4)"),
        ("tensorflow_model_optimization", "QAT (Sprint 5.2)"),
    ]

    errors = []
    missing_critical = []
    for pkg, desc, _ in critical:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            ok(f"{pkg:<22} {version:<12} {Color.DIM}({desc}){Color.END}")
        except ImportError:
            err(f"{pkg:<22} {'AUSENTE':<12} {Color.DIM}({desc}){Color.END}")
            errors.append(f"missing_dep:{pkg}")
            missing_critical.append(pkg)

    if missing_critical:
        print()
        err(f"FALTAM {len(missing_critical)} deps críticas: {', '.join(missing_critical)}")
        print(f"  {Color.BOLD}Fix:{Color.END}  pip install -r requirements.txt")

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
    return not errors, errors


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
