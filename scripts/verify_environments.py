#!/usr/bin/env python3
"""Verifica que os ambientes Docker da aplicação sobem e têm as dependências.

Para cada ambiente (perfil compose + serviço) o script, opcionalmente, builda a
imagem e então sobe um container efêmero que IMPORTA as bibliotecas-chave da
família e reporta as versões — confirmando que tudo foi instalado com sucesso.
Nos perfis NVIDIA, valida também (opcional) se o TensorFlow enxerga a GPU.

Os limites de CPU/memória passados aos perfis são derivados do `docker info`
(VM do WSL2/Docker), evitando o erro "range of CPUs ..." quando o `.env` pede
mais CPUs do que a VM tem.

Uso:
  python scripts/verify_environments.py                    # todos (assume imagens prontas)
  python scripts/verify_environments.py --build            # builda antes de verificar
  python scripts/verify_environments.py --only benchmark inference-cpu
  python scripts/verify_environments.py --gpu-check        # valida GPU nos perfis nvidia
  python scripts/verify_environments.py --build --gpu-check --require-gpu

Sai com código 0 se TODOS os ambientes selecionados passarem; !=0 caso contrário.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent

# Conjuntos de imports (NOMES DE IMPORT, não de pip) por camada.
_AUDIO = ["numpy", "pandas", "sklearn", "scipy", "librosa", "soundfile", "soxr",
          "joblib", "yaml", "tqdm"]
_TF = ["tensorflow", "keras"]
_TORCH = ["torch", "torchaudio"]
_SSL = ["transformers", "datasets", "huggingface_hub"]
_API = ["gradio", "fastapi", "uvicorn", "pydantic"]


@dataclass(frozen=True)
class Env:
    key: str
    compose_file: str
    service: str
    gpu: bool
    expect: List[str] = field(default_factory=list)


ENVIRONMENTS: List[Env] = [
    Env("inference-cpu", "docker/compose/inference.cpu.yml", "inference-api",
        False, _AUDIO + ["tensorflow", "torch", "transformers"] + _API),
    Env("inference-nvidia", "docker/compose/inference.nvidia.yml", "inference-api",
        True, _AUDIO + ["tensorflow", "torch", "transformers"] + _API),
    Env("train-cpu-classical", "docker/compose/train.cpu.yml", "classical-ml",
        False, _AUDIO + ["jsonschema"]),
    Env("train-nvidia-tensorflow", "docker/compose/train.nvidia.yml", "tensorflow-keras",
        True, _AUDIO + _TF),
    Env("train-nvidia-pytorch", "docker/compose/train.nvidia.yml", "pytorch-audio",
        True, _AUDIO + ["tensorflow"] + _TORCH),
    Env("train-nvidia-ssl", "docker/compose/train.nvidia.yml", "ssl-transformers",
        True, _AUDIO + ["tensorflow"] + _TORCH + _SSL),
    Env("benchmark-nvidia", "docker/compose/benchmark.nvidia.yml", "benchmark",
        True, _AUDIO + ["tensorflow", "keras", "torch", "transformers"] + _API),
]

# Check executado DENTRO do container. argv[1] = lista de imports separada por
# vírgula; passe "--gpu" para também contar GPUs visíveis ao TensorFlow.
_IN_CONTAINER_CHECK = r"""
import importlib, json, sys
mods = [m for m in sys.argv[1].split(",") if m]
res, ok = {}, True
for m in mods:
    try:
        res[m] = str(getattr(importlib.import_module(m), "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        res[m] = "ERRO:%s:%s" % (type(e).__name__, str(e)[:80]); ok = False
gpu = None
if "--gpu" in sys.argv[2:]:
    try:
        import tensorflow as tf
        gpu = len(tf.config.list_physical_devices("GPU"))
    except Exception as e:  # noqa: BLE001
        gpu = "ERRO:%s" % e
print("XFAKE_VERIFY=" + json.dumps({"ok": ok, "mods": res, "gpu": gpu}))
sys.exit(0 if ok else 1)
"""


def _docker() -> str:
    exe = shutil.which("docker")
    if not exe:
        sys.exit("ERRO: 'docker' não encontrado no PATH.")
    return exe


def _safe_limit_env() -> dict:
    """Limites de CPU/mem ≤ o que a VM do Docker tem (evita falha do `run`)."""
    env = dict(os.environ)
    try:
        out = subprocess.run(
            [_docker(), "info", "-f", "{{.NCPU}};{{.MemTotal}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        ncpu_s, mem_s = out.split(";")
        ncpu = max(1, int(ncpu_s))
        mem_g = max(2, int(int(mem_s) / (1024 ** 3)) - 2)  # deixa ~2 GiB de folga
    except Exception:  # noqa: BLE001
        ncpu, mem_g = 2, 4
    for var in ("DOCKER_CPU_LIMIT", "DOCKER_TRAIN_CPU_LIMIT"):
        env[var] = str(ncpu)
    for var in ("DOCKER_MEMORY_LIMIT", "DOCKER_TRAIN_MEMORY_LIMIT"):
        env[var] = f"{mem_g}G"
    return env


def _compose_base(compose_file: str) -> List[str]:
    cmd = [_docker(), "compose", "-f", compose_file]
    env_file = ROOT / ".env"
    if env_file.is_file():
        cmd += ["--env-file", str(env_file)]
    return cmd


def _build(env: Env, run_env: dict, timeout: int) -> Optional[str]:
    cmd = _compose_base(env.compose_file) + ["build", env.service]
    print(f"  [build] {' '.join(cmd[2:])}", flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT), env=run_env,
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        return (p.stderr or p.stdout or "build falhou").strip().splitlines()[-1][:200]
    return None


def verify_one(env: Env, *, build: bool, gpu_check: bool,
               build_timeout: int, run_timeout: int) -> dict:
    run_env = _safe_limit_env()
    result = {"key": env.key, "service": env.service, "ok": False,
              "error": None, "mods": {}, "gpu": None}

    if build:
        err = _build(env, run_env, build_timeout)
        if err:
            result["error"] = f"build: {err}"
            return result

    args = [env.expect and ",".join(env.expect) or "numpy"]
    if gpu_check and env.gpu:
        args.append("--gpu")
    cmd = (_compose_base(env.compose_file)
           + ["run", "--rm", "--no-deps", env.service,
              "python", "-c", _IN_CONTAINER_CHECK, *args])
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=run_env,
                           capture_output=True, text=True, timeout=run_timeout)
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout ({run_timeout}s) ao subir o container"
        return result

    marker = next((ln for ln in (p.stdout or "").splitlines()
                   if ln.startswith("XFAKE_VERIFY=")), None)
    if not marker:
        tail = (p.stderr or p.stdout or "sem saída").strip().splitlines()
        result["error"] = (tail[-1][:200] if tail else "sem marcador de verificação")
        return result
    payload = json.loads(marker[len("XFAKE_VERIFY="):])
    result["mods"] = payload.get("mods", {})
    result["gpu"] = payload.get("gpu")
    result["ok"] = bool(payload.get("ok"))
    return result


def _print_result(r: dict, require_gpu: bool) -> bool:
    status = "OK " if r["ok"] else "FALHOU"
    print(f"\n[{status}] {r['key']} (serviço: {r['service']})")
    if r["error"]:
        print(f"    erro: {r['error']}")
    for mod, ver in r["mods"].items():
        flag = "✗" if str(ver).startswith("ERRO") else "✓"
        print(f"      {flag} {mod:<16} {ver}")
    gpu_ok = True
    if r["gpu"] is not None:
        if isinstance(r["gpu"], int):
            print(f"      GPU visível ao TensorFlow: {r['gpu']}")
            gpu_ok = r["gpu"] > 0
        else:
            print(f"      GPU: {r['gpu']}")
            gpu_ok = False
    passed = r["ok"] and (gpu_ok or not require_gpu)
    return passed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="KEY",
                   help=f"verifica só estes (chaves: {', '.join(e.key for e in ENVIRONMENTS)})")
    p.add_argument("--build", action="store_true", help="builda a imagem antes de verificar")
    p.add_argument("--gpu-check", action="store_true",
                   help="conta GPUs visíveis ao TF nos perfis NVIDIA")
    p.add_argument("--require-gpu", action="store_true",
                   help="trata GPU ausente (nos perfis NVIDIA) como falha")
    p.add_argument("--build-timeout", type=int, default=3600)
    p.add_argument("--run-timeout", type=int, default=600)
    p.add_argument("--list", action="store_true", help="lista os ambientes e sai")
    args = p.parse_args()

    if args.list:
        for e in ENVIRONMENTS:
            print(f"  {e.key:<26} {e.compose_file}  ({e.service}, gpu={e.gpu})")
        return 0

    selected = ENVIRONMENTS
    if args.only:
        keys = set(args.only)
        selected = [e for e in ENVIRONMENTS if e.key in keys]
        unknown = keys - {e.key for e in ENVIRONMENTS}
        if unknown:
            p.error(f"chaves desconhecidas: {', '.join(sorted(unknown))}")

    print("=" * 64)
    print(f" Verificação de ambientes XFakeSong ({len(selected)} alvo(s))")
    print(f" build={args.build} · gpu_check={args.gpu_check} · require_gpu={args.require_gpu}")
    print("=" * 64)

    passed_all = True
    summary = []
    for env in selected:
        r = verify_one(env, build=args.build, gpu_check=args.gpu_check,
                       build_timeout=args.build_timeout, run_timeout=args.run_timeout)
        ok = _print_result(r, args.require_gpu)
        passed_all = passed_all and ok
        summary.append((env.key, ok))

    print("\n" + "=" * 64)
    print(" RESUMO")
    for key, ok in summary:
        print(f"   {'PASS' if ok else 'FAIL'}  {key}")
    print("=" * 64)
    print("Resultado:", "TODOS OK" if passed_all else "HÁ FALHAS")
    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
