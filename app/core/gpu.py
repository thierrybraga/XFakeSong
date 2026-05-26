"""Configuração centralizada de GPU (TensorFlow).

Reúne em UM lugar:
- Detecção de GPUs (via TF E via nvidia-smi/pynvml — captura quando hardware
  existe mas TF não está vendo)
- Memory growth (evita alocar toda VRAM logo)
- Mixed precision auto (FP16 em Tensor Cores — Sprint 3.2)
- XLA hint
- DirectML auto-detect (única opção GPU em Windows nativo com TF ≥ 2.11)
- Sumário amigável (`describe_gpu_setup()`) para Dashboard
- Mensagens **acionáveis** quando há mismatch (ex: GPU NVIDIA presente mas
  TF Windows não tem CUDA → recomenda WSL2 ou tensorflow-directml-plugin)

Uso:
    from app.core.gpu import setup_gpu, describe_gpu_setup

    setup_gpu()                  # uma vez, no startup
    print(describe_gpu_setup())  # snapshot legível

A função é idempotente — múltiplas chamadas não causam efeito colateral.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lock + flag de "já configurado" para idempotência
_setup_lock = threading.Lock()
_setup_done: bool = False
_setup_result: Dict[str, Any] = {}


# =====================================================================
# Helpers internos
# =====================================================================

def _gpu_compute_capability(gpu_device) -> Optional[tuple]:
    """Retorna (major, minor) da CC ou None."""
    try:
        import tensorflow as tf
        details = tf.config.experimental.get_device_details(gpu_device)
        cc = details.get("compute_capability")
        if cc is None:
            return None
        if isinstance(cc, (list, tuple)) and len(cc) >= 2:
            return (int(cc[0]), int(cc[1]))
        # Pode vir como float "7.5"
        s = str(cc)
        if "." in s:
            major, minor = s.split(".", 1)
            return (int(major), int(minor))
        return (int(s), 0)
    except Exception as e:
        logger.debug(f"compute_capability indisponível: {e}")
        return None


def _gpu_name(gpu_device) -> str:
    """Nome legível da GPU (ou device_name TF)."""
    try:
        import tensorflow as tf
        details = tf.config.experimental.get_device_details(gpu_device)
        return details.get("device_name", str(gpu_device))
    except Exception:
        return str(gpu_device)


def _is_tensor_core_capable(major: Optional[int]) -> bool:
    """Tensor Cores requerem Volta+ (CC >= 7.0).
    GPUs Pascal/Maxwell (CC < 7) NÃO se beneficiam de mixed precision.
    """
    return major is not None and major >= 7


def _detect_nvidia_via_smi() -> List[Dict[str, Any]]:
    """Detecta GPUs NVIDIA via `nvidia-smi` (hardware-level, ignora TF).

    Útil quando o usuário tem GPU NVIDIA mas TF não está vendo (típico em
    Windows nativo com TF ≥ 2.11). Retorna lista mesmo que TF veja 0 GPUs.

    Returns:
        Lista de dicts com keys: index, name, uuid, memory_total_mb, driver_version.
        Lista vazia se nvidia-smi não está disponível.
    """
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "driver_version": parts[4],
                })
        return gpus
    except Exception as e:
        logger.debug(f"nvidia-smi falhou: {e}")
        return []


def _detect_nvidia_via_pynvml() -> List[Dict[str, Any]]:
    """Detecta GPUs NVIDIA via pynvml (fallback alternativo)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({
                "index": i,
                "name": name,
                "memory_total_mb": int(mem.total // (1024 * 1024)),
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _detect_nvidia_hardware() -> List[Dict[str, Any]]:
    """Combina nvidia-smi + pynvml para inventário do hardware NVIDIA."""
    gpus = _detect_nvidia_via_smi()
    if not gpus:
        gpus = _detect_nvidia_via_pynvml()
    return gpus


def _detect_directml_plugin() -> bool:
    """Verifica se `tensorflow-directml-plugin` está instalado.

    Em Windows nativo com TF ≥ 2.11, este é o **único caminho** para
    aceleração GPU (CUDA não é mais suportado nativamente).
    """
    try:
        import tensorflow_directml_plugin  # noqa: F401
        return True
    except ImportError:
        return False


def _is_wsl() -> bool:
    """Detecta se estamos rodando em WSL (Linux dentro do Windows)."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            content = f.read().lower()
        return "microsoft" in content or "wsl" in content
    except Exception:
        return False


def _tf_built_with_cuda() -> bool:
    """True se a instalação atual de TF foi compilada com suporte CUDA."""
    try:
        import tensorflow as tf
        return bool(tf.test.is_built_with_cuda())
    except Exception:
        return False


def _diagnose_gpu_situation(
    nvidia_hw: List[Dict[str, Any]],
    tf_gpus_count: int,
) -> Dict[str, Any]:
    """Diagnostica por que TF não está vendo a GPU NVIDIA presente.

    Returns dict com keys:
        diagnosis: str — categoria do problema
        message:   str — mensagem curta para o dashboard
        hints:     list[str] — passos acionáveis
        severity:  "info" | "warning" | "error"
    """
    has_nvidia = bool(nvidia_hw)
    is_windows = platform.system() == "Windows"
    is_wsl = _is_wsl()
    tf_has_cuda = _tf_built_with_cuda()
    has_dml = _detect_directml_plugin()

    if not has_nvidia and tf_gpus_count == 0:
        return {
            "diagnosis": "no_gpu",
            "message": "Sem GPU NVIDIA detectada — usando CPU.",
            "hints": [],
            "severity": "info",
        }

    if has_nvidia and tf_gpus_count > 0:
        return {
            "diagnosis": "ok",
            "message": "GPU NVIDIA detectada e visível ao TensorFlow.",
            "hints": [],
            "severity": "info",
        }

    # Caso problemático: hardware NVIDIA presente mas TF não vê
    if has_nvidia and tf_gpus_count == 0:
        hw_name = nvidia_hw[0].get("name", "?") if nvidia_hw else "?"

        if is_windows and not is_wsl:
            # Windows nativo: única opção é DirectML plugin (TF ≥ 2.11 não tem CUDA Windows)
            if has_dml:
                return {
                    "diagnosis": "windows_dml_installed_but_invisible",
                    "message": (
                        f"{hw_name} detectada e tensorflow-directml-plugin "
                        f"instalado, mas TF não está expondo a GPU. "
                        f"Reinicie o processo Python."
                    ),
                    "hints": [
                        "Feche todas as instâncias Python/Gradio",
                        "Inicie via `start.bat` para garantir env fresh",
                    ],
                    "severity": "warning",
                }
            # Sem DirectML: lista as opções
            tf_ver = "?"
            try:
                import tensorflow as tf
                tf_ver = tf.__version__
            except Exception:
                pass
            return {
                "diagnosis": "windows_native_no_gpu_support",
                "message": (
                    f"{hw_name} presente, mas TF {tf_ver} em Windows nativo "
                    f"não suporta CUDA. GPU não será usada."
                ),
                "hints": [
                    "Opção A (recomendada): WSL2 + Ubuntu — `wsl --install -d Ubuntu`, "
                    "depois `pip install tensorflow[and-cuda]` dentro do WSL",
                    "Opção B (Windows nativo, mais lento): "
                    "`pip install tensorflow-directml-plugin` "
                    "(requer Python 3.7–3.10 e TF 2.10) — incompatível com TF 2.11+",
                    "Opção C: Aceite CPU (lento mas funcional)",
                ],
                "severity": "warning",
            }

        if is_wsl and not tf_has_cuda:
            return {
                "diagnosis": "wsl_no_cuda",
                "message": (
                    f"{hw_name} visível no WSL mas TF não foi compilado com CUDA."
                ),
                "hints": [
                    "Instale TF com suporte CUDA: `pip install tensorflow[and-cuda]`",
                    "Verifique driver NVIDIA Windows ≥ 525.x para WSL2 CUDA",
                ],
                "severity": "warning",
            }

        # Linux ou outros: TF sem CUDA
        if not tf_has_cuda:
            return {
                "diagnosis": "tf_no_cuda",
                "message": (
                    f"{hw_name} presente mas TF não foi compilado com CUDA."
                ),
                "hints": [
                    "Reinstale: `pip install tensorflow[and-cuda]` "
                    "ou use `tensorflow-gpu` em versões antigas",
                ],
                "severity": "warning",
            }

        # CUDA available but TF still doesn't see GPU
        return {
            "diagnosis": "cuda_present_gpu_hidden",
            "message": (
                f"{hw_name} presente, TF compilado com CUDA, mas list_physical_devices "
                f"retornou 0. Possível mismatch de versões CUDA/cuDNN/driver."
            ),
            "hints": [
                "Verifique compatibilidade: TF version × CUDA × cuDNN "
                "(https://www.tensorflow.org/install/source#gpu)",
                "Confira `nvidia-smi` mostra a GPU sem erro",
                "Variável CUDA_VISIBLE_DEVICES pode estar mascarando — unset",
            ],
            "severity": "warning",
        }

    return {
        "diagnosis": "unknown",
        "message": "Estado de GPU desconhecido.",
        "hints": [],
        "severity": "info",
    }


# =====================================================================
# API pública
# =====================================================================

def setup_gpu(
    *,
    memory_growth: bool = True,
    enable_mixed_precision: Optional[bool] = None,
    log_level: int = logging.INFO,
) -> Dict[str, Any]:
    """Configura TF GPU de forma robusta.

    Args:
        memory_growth: se True (default), TF aloca VRAM sob demanda em vez
            de reservar tudo no primeiro `model.fit()`. Recomendado para
            evitar OOM em sistemas multi-app.
        enable_mixed_precision: None=auto (ativa se houver GPU CC>=7);
            True/False força. Sprint 3.2.
        log_level: nível de log para mensagens de configuração.

    Returns:
        Dict com a configuração efetivamente aplicada (idempotente — chamadas
        subsequentes retornam o mesmo dict).
    """
    global _setup_done, _setup_result

    with _setup_lock:
        if _setup_done:
            return _setup_result

        result: Dict[str, Any] = {
            "tf_available": False,
            "gpus_detected": [],          # GPUs que TF está expondo
            "nvidia_hardware": [],        # GPUs físicas via nvidia-smi/pynvml
            "memory_growth_applied": False,
            "mixed_precision_enabled": False,
            "device_policy": "cpu",       # "cpu" | "gpu"
            "tensor_core_capable": False,
            "errors": [],
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "python": sys.version.split()[0],
                "is_wsl": _is_wsl(),
            },
            "tf_built_with_cuda": False,
            "directml_installed": _detect_directml_plugin(),
            "diagnosis": None,            # preenchido ao final
        }

        # Coleta o hardware NVIDIA cedo (independente de TF) — assim mesmo
        # quando TF é CPU-only sabemos se há GPU física disponível.
        result["nvidia_hardware"] = _detect_nvidia_hardware()

        try:
            import tensorflow as tf
        except ImportError as e:
            msg = f"TensorFlow não instalado: {e}"
            logger.warning(msg)
            result["errors"].append(msg)
            # Diagnose mesmo sem TF — usuário sabe se hardware existe
            result["diagnosis"] = _diagnose_gpu_situation(
                result["nvidia_hardware"], tf_gpus_count=0
            )
            _setup_done = True
            _setup_result = result
            return result

        result["tf_available"] = True
        result["tf_built_with_cuda"] = _tf_built_with_cuda()

        # === Detecta GPUs ===
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception as e:
            msg = f"Falha ao listar GPUs: {e}"
            logger.warning(msg)
            result["errors"].append(msg)
            gpus = []

        if not gpus:
            result["device_policy"] = "cpu"
            # Diagnose: hardware NVIDIA presente vs TF cego
            result["diagnosis"] = _diagnose_gpu_situation(
                result["nvidia_hardware"], tf_gpus_count=0
            )
            diag = result["diagnosis"]
            if result["nvidia_hardware"]:
                hw_name = result["nvidia_hardware"][0].get("name", "?")
                logger.warning(
                    f"[GPU] {hw_name} detectada via nvidia-smi mas TF não a expõe. "
                    f"Diagnóstico: {diag['diagnosis']}. {diag['message']}"
                )
                for hint in diag.get("hints", []):
                    logger.warning(f"[GPU]  → {hint}")
            else:
                logger.log(log_level, "[GPU] Nenhuma GPU detectada — usando CPU.")
            _setup_done = True
            _setup_result = result
            return result

        # === Memory growth (deve ser feito ANTES de qualquer operação) ===
        if memory_growth:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                    result["memory_growth_applied"] = True
                except RuntimeError as e:
                    # Pode falhar se TF já inicializou — não-fatal
                    msg = (
                        f"memory_growth não pôde ser aplicado em {g} "
                        f"(TF já inicializado?): {e}"
                    )
                    logger.warning(msg)
                    result["errors"].append(msg)
                except Exception as e:
                    msg = f"Erro inesperado em memory_growth: {e}"
                    logger.warning(msg)
                    result["errors"].append(msg)

        # === Coleta info de cada GPU ===
        any_tensor_core = False
        for idx, g in enumerate(gpus):
            cc = _gpu_compute_capability(g)
            name = _gpu_name(g)
            tc = _is_tensor_core_capable(cc[0] if cc else None)
            if tc:
                any_tensor_core = True
            result["gpus_detected"].append({
                "index": idx,
                "name": name,
                "compute_capability": (
                    f"{cc[0]}.{cc[1]}" if cc else "unknown"
                ),
                "tensor_core": tc,
            })

        result["device_policy"] = "gpu"
        result["tensor_core_capable"] = any_tensor_core

        # === Mixed precision (Sprint 3.2) ===
        if enable_mixed_precision is None:
            should_enable = any_tensor_core
        else:
            should_enable = bool(enable_mixed_precision)

        if should_enable:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                result["mixed_precision_enabled"] = True
                logger.log(
                    log_level,
                    "[GPU] Mixed precision habilitada (mixed_float16) — "
                    "~2× speedup + ~50%% menos VRAM em Tensor Cores",
                )
            except Exception as e:
                msg = f"Mixed precision falhou: {e}"
                logger.warning(msg)
                result["errors"].append(msg)

        # === Logs informativos ===
        logger.log(
            log_level,
            f"[GPU] {len(gpus)} GPU(s) configurada(s) "
            f"(memory_growth={result['memory_growth_applied']}, "
            f"mixed_precision={result['mixed_precision_enabled']})",
        )
        for info in result["gpus_detected"]:
            logger.log(
                log_level,
                f"[GPU] #{info['index']}: {info['name']} "
                f"(CC={info['compute_capability']}, "
                f"tensor_core={info['tensor_core']})",
            )

        # Hint para XLA (Sprint 3.1) — não-obrigatório, mas ajuda no JIT
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=1")

        # Diagnose final (deve dar "ok" no happy path)
        result["diagnosis"] = _diagnose_gpu_situation(
            result["nvidia_hardware"], tf_gpus_count=len(gpus)
        )

        _setup_done = True
        _setup_result = result
        return result


def get_setup_result() -> Dict[str, Any]:
    """Retorna a config aplicada (vazio se setup_gpu ainda não foi chamado)."""
    return dict(_setup_result)


def is_gpu_available() -> bool:
    """True se TF detectou ao menos 1 GPU utilizável."""
    if not _setup_done:
        # Tenta uma detecção rápida sem alterar config global
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices("GPU")) > 0
        except Exception:
            return False
    return _setup_result.get("device_policy") == "gpu"


def describe_gpu_setup() -> str:
    """String legível com snapshot do setup, para Dashboard.

    Quatro casos cobertos:
    1. TF vê GPU → "✓ NVIDIA RTX 4090 · CC 8.9 · Tensor Cores · FP16"
    2. Hardware NVIDIA mas TF não vê → "⚠ NVIDIA RTX 3060 (TF sem CUDA — use WSL2)"
    3. Sem hardware → "✗ CPU only (sem GPU detectada)"
    4. TF indisponível → "? TF indisponível"
    """
    r = _setup_result
    if not r:
        # Setup ainda não rodou — fast path
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                # Tenta detectar hardware mesmo se TF não vê
                hw = _detect_nvidia_hardware()
                if hw:
                    return (
                        f"⚠ {hw[0].get('name', 'NVIDIA GPU')} "
                        f"(TF cego — setup pendente)"
                    )
                return "✗ CPU only (sem GPU detectada)"
            return f"⚠ {len(gpus)} GPU(s) detectada(s) (setup pendente)"
        except Exception:
            return "? TF indisponível"

    if not r.get("tf_available"):
        # Hardware pode existir mesmo sem TF
        hw = r.get("nvidia_hardware", [])
        if hw:
            return f"⚠ {hw[0].get('name', 'NVIDIA GPU')} (TF não instalado)"
        return "? TF não instalado"

    # TF disponível mas não vê GPU
    if r.get("device_policy") == "cpu":
        hw = r.get("nvidia_hardware", [])
        diag = r.get("diagnosis") or {}
        if hw:
            hw_name = hw[0].get("name", "NVIDIA GPU")
            # Curto para a status bar — detalhe completo em get_diagnosis_html()
            d = diag.get("diagnosis", "")
            if d == "windows_native_no_gpu_support":
                return f"⚠ {hw_name} (Windows nativo sem CUDA — use WSL2)"
            if d == "windows_dml_installed_but_invisible":
                return f"⚠ {hw_name} (DirectML instalado — reinicie processo)"
            if d == "wsl_no_cuda":
                return f"⚠ {hw_name} (WSL sem CUDA — `pip install tensorflow[and-cuda]`)"
            if d == "tf_no_cuda":
                return f"⚠ {hw_name} (TF sem CUDA — reinstale com [and-cuda])"
            if d == "cuda_present_gpu_hidden":
                return f"⚠ {hw_name} (CUDA presente mas GPU oculta — checar versões)"
            return f"⚠ {hw_name} (TF não vê — modo CPU)"
        return "✗ CPU only (sem GPU detectada)"

    # GPU detectada e em uso pelo TF
    gpus = r.get("gpus_detected", [])
    if not gpus:
        return "? GPU detectada mas info indisponível"

    first = gpus[0]
    parts = [first["name"]]
    if first["compute_capability"] != "unknown":
        parts.append(f"CC {first['compute_capability']}")
    if first.get("tensor_core"):
        parts.append("Tensor Cores")
    if r.get("mixed_precision_enabled"):
        parts.append("FP16")

    descr = f"✓ {' · '.join(parts)}"
    if len(gpus) > 1:
        descr += f" (+{len(gpus) - 1} GPU(s))"
    return descr


def get_diagnosis_html() -> str:
    """HTML detalhado com hardware NVIDIA + diagnóstico + hints (para Dashboard).

    Retorna painel HTML pronto para `gr.HTML(...)`. Quando há mismatch, lista
    as opções acionáveis (WSL2, DirectML, CPU) que o usuário pode tomar.
    """
    r = _setup_result
    if not r:
        return "<div class='gpu-diag muted'>Setup de GPU ainda não rodou.</div>"

    plat = r.get("platform", {})
    hw = r.get("nvidia_hardware", [])
    diag = r.get("diagnosis") or {}
    sev = diag.get("severity", "info")
    msg = diag.get("message", "")
    hints = diag.get("hints", []) or []

    # Cabeçalho com plataforma
    rows = [
        f"<b>Sistema:</b> {plat.get('system', '?')} {plat.get('release', '')}"
        + (" <span class='badge'>WSL</span>" if plat.get("is_wsl") else "")
        + f" · Python {plat.get('python', '?')}",
        f"<b>TF compilado com CUDA:</b> {r.get('tf_built_with_cuda')}"
        + f" · <b>DirectML:</b> {r.get('directml_installed')}",
    ]

    # Hardware
    if hw:
        rows.append("<b>Hardware NVIDIA detectado:</b>")
        for g in hw:
            line = (
                f"• #{g.get('index', '?')}: {g.get('name', '?')}"
                f" — {g.get('memory_total_mb', '?')} MB"
            )
            if g.get("driver_version"):
                line += f" — driver {g['driver_version']}"
            rows.append(line)
    else:
        rows.append("<b>Hardware NVIDIA:</b> nenhum (ou nvidia-smi indisponível)")

    # Visibilidade pelo TF
    tf_gpus = r.get("gpus_detected", [])
    if tf_gpus:
        rows.append(f"<b>TF vê:</b> {len(tf_gpus)} GPU(s) ✓")
    else:
        rows.append("<b>TF vê:</b> 0 GPU(s)")

    # Diagnóstico + hints
    icon = {"info": "ℹ", "warning": "⚠", "error": "✗"}.get(sev, "ℹ")
    rows.append(f"<div class='gpu-diag-msg gpu-diag-{sev}'>{icon} {msg}</div>")
    if hints:
        rows.append("<b>Como resolver:</b>")
        for h in hints:
            rows.append(f"  • {h}")

    return "<div class='gpu-diag'><pre>" + "\n".join(rows) + "</pre></div>"


__all__ = [
    "setup_gpu",
    "get_setup_result",
    "is_gpu_available",
    "describe_gpu_setup",
    "get_diagnosis_html",
]
