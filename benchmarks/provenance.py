"""Proveniência do run: commit, bibliotecas, checkpoints externos e manifesto.

Sem isto um número do artigo não aponta para uma revisão exata do código nem para
os pesos pré-treinados que entraram no grafo. Saiu de ``runner.py`` (P2-15).
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from benchmarks.common import PROJECT_ROOT


def _git_provenance() -> Dict[str, Any]:
    """Commit que gerou os resultados — sem isto o run não é rastreável.

    Um artigo cujos números não apontam para uma revisão exata do código não é
    reproduzível: a mesma arquitetura pode ter mudado entre execuções (foi o
    caso em 2026-07-27, quando sete arquiteturas mudaram no mesmo dia).
    `dirty=True` sinaliza que havia alterações não commitadas — o run continua,
    mas o registro deixa a ressalva explícita.
    """
    import subprocess

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    status = _run(["git", "status", "--porcelain"])
    return {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def _library_versions() -> Dict[str, Any]:
    """Versões das bibliotecas que influenciam o resultado numérico."""
    versions: Dict[str, Any] = {}
    for module_name, attr in (
        ("tensorflow", "__version__"),
        ("keras", "__version__"),
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("sklearn", "__version__"),
        ("librosa", "__version__"),
        ("torch", "__version__"),
        ("transformers", "__version__"),
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, attr, None)
        except Exception:  # noqa: BLE001
            versions[module_name] = None
    return versions


def _pretrained_checkpoints() -> Dict[str, Any]:
    """Checkpoints pré-treinados que entram no grafo dos modelos.

    AST, WavLM e HuBERT partem de pesos externos: o identificador do checkpoint
    é parte da definição do experimento e precisa constar no artefato.
    """
    checkpoints: Dict[str, Any] = {}
    try:
        from app.domain.models.architectures.ast_pretrained import (
            DEFAULT_AST_CHECKPOINT,
        )

        checkpoints["SpectrogramTransformer"] = DEFAULT_AST_CHECKPOINT
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.domain.models.architectures.registry import architecture_registry

        wavlm = architecture_registry.get_architecture("WavLM").default_params
        hubert = architecture_registry.get_architecture("HuBERT").default_params
        checkpoints["WavLM"] = wavlm.get("wavlm_model")
        checkpoints["HuBERT"] = hubert.get("model_name")
    except Exception:  # noqa: BLE001
        pass
    return checkpoints


def _env_snapshot() -> Dict[str, Any]:
    snap = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "git": _git_provenance(),
        "libraries": _library_versions(),
        "pretrained_checkpoints": _pretrained_checkpoints(),
    }
    try:
        import tensorflow as tf

        snap["tensorflow"] = tf.__version__
        snap["gpu"] = bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        snap["tensorflow"] = None
        snap["gpu"] = False
    try:
        from app.core.gpu import describe_gpu_setup

        snap["device"] = describe_gpu_setup()
    except Exception:
        snap["device"] = "?"
    return snap


def _architecture_provenance(arch: str) -> Dict[str, Any]:
    """Entrada do manifesto oficial correspondente à arquitetura.

    Os rótulos `variant`/`family`/`runner` existiam em `benchmarks/config.py`
    mas NUNCA chegavam a nenhum artefato — proveniência que ninguém lia. Agora
    acompanham cada resultado.
    """
    try:
        from benchmarks.config import (
            EXTENDED_MODEL_MANIFEST,
            OFFICIAL_TCC_MODEL_MANIFEST,
        )
    except Exception:  # noqa: BLE001
        return {}
    for item in list(OFFICIAL_TCC_MODEL_MANIFEST) + list(EXTENDED_MODEL_MANIFEST):
        if item.get("benchmark_name") == arch:
            prov = {
                key: item.get(key)
                for key in ("variant", "family", "runner", "scope", "result_key")
                if item.get(key) is not None
            }
            _apply_ssl_backbone_status(arch, item, prov)
            return prov
    return {}


def _apply_ssl_backbone_status(
    arch: str, item: Dict[str, Any], prov: Dict[str, Any]
) -> None:
    """Substitui o rótulo declarado pelo backbone SSL REALMENTE construído.

    WavLM/HuBERT no caminho Keras degradam para um CNN-1D do zero quando o
    checkpoint não está acessível. Publicar o `variant` do manifesto nesse caso
    faria o artefato alegar backbone pré-treinado onde não houve — erro
    indetectável depois do run. `ssl_utils` registra o que foi montado; aqui
    isso vira proveniência.
    """
    try:
        from app.domain.models.architectures.ssl_utils import (
            get_ssl_backbone_status,
        )

        status = get_ssl_backbone_status(arch)
    except Exception:  # noqa: BLE001
        status = None
    if not status:
        return
    prov["ssl_backbone"] = status
    if not status.get("pretrained") and item.get("fallback_variant"):
        prov["declared_variant"] = prov.get("variant")
        prov["variant"] = item["fallback_variant"]
