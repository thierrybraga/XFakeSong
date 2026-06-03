"""Verificação de versões críticas no startup.

Detecta combinações de versões conhecidas como problemáticas e emite warning
claro ANTES do erro obscuro acontecer em runtime.

Caso mais comum:
    gradio<4.31 + starlette>=0.36 → TypeError: unhashable type: 'dict'
    na primeira request a "/" (Jinja2 cache key inválido).

Uso:
    from app.core.version_check import check_versions
    check_versions()   # chamar no startup do FastAPI/Gradio
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _safe_version(pkg: str) -> Optional[str]:
    """Retorna versão instalada de um pacote ou None."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _parse_version(v: str) -> tuple:
    """Parse 'X.Y.Z' → tuple de ints. Ignora pre-release tags."""
    parts = v.split("+")[0].split("-")[0].split(".")
    out = []
    for p in parts[:3]:
        try:
            out.append(int(p))
        except ValueError:
            try:
                # ex: "0rc1" → 0
                num = ""
                for c in p:
                    if c.isdigit():
                        num += c
                    else:
                        break
                out.append(int(num) if num else 0)
            except ValueError:
                out.append(0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


def check_versions(strict: bool = False) -> List[str]:
    """Verifica compatibilidade de versões críticas.

    Args:
        strict: se True, levanta RuntimeError em vez de apenas logar.

    Returns:
        Lista de mensagens de problemas detectados (vazia se OK).
    """
    issues: List[str] = []

    gradio_v = _safe_version("gradio")
    starlette_v = _safe_version("starlette")
    fastapi_v = _safe_version("fastapi")
    jinja_v = _safe_version("jinja2")
    hub_v = _safe_version("huggingface_hub") or _safe_version("huggingface-hub")

    if gradio_v:
        g = _parse_version(gradio_v)
        # Gradio < 4.31 tem incompatibilidade com Starlette ≥0.36 em
        # templates.TemplateResponse — passa dict como template_name.
        if g < (4, 31, 0):
            if starlette_v and _parse_version(starlette_v) >= (0, 36, 0):
                issues.append(
                    f"INCOMPATIBILIDADE CRÍTICA: gradio=={gradio_v} usa API "
                    f"antiga de TemplateResponse, mas starlette=={starlette_v} "
                    f"requer API nova. A renderização do Gradio em '/' irá "
                    f"falhar com TypeError: unhashable type: 'dict'. "
                    f"FIX: pip install 'gradio>=4.31,<5.0'"
                )

        # Gradio 4.x importa `HfFolder`, removido no huggingface_hub 1.0.
        # Resultado: `ImportError: cannot import name 'HfFolder'` no import do
        # gradio — derruba TODA a UI no boot, antes de qualquer request.
        if g < (5, 0, 0) and hub_v and _parse_version(hub_v) >= (1, 0, 0):
            issues.append(
                f"INCOMPATIBILIDADE CRÍTICA: gradio=={gradio_v} importa "
                f"`HfFolder`, removido no huggingface_hub=={hub_v} (>=1.0). "
                f"O import do gradio falha com ImportError e a UI não sobe. "
                f"FIX: pip install 'huggingface_hub>=0.25,<1.0'"
            )

    # Aviso geral de versões inesperadas
    if jinja_v and _parse_version(jinja_v) < (3, 1, 0):
        issues.append(
            f"jinja2=={jinja_v} é muito antigo. "
            f"Recomendado: pip install 'jinja2>=3.1.5'"
        )

    # Log das versões detectadas (útil para diagnóstico)
    logger.info(
        "Version check: gradio=%s, starlette=%s, fastapi=%s, jinja2=%s, "
        "huggingface_hub=%s",
        gradio_v or "?", starlette_v or "?", fastapi_v or "?", jinja_v or "?",
        hub_v or "?",
    )

    if issues:
        for msg in issues:
            logger.error(msg)
        if strict:
            raise RuntimeError(
                "Versões incompatíveis detectadas:\n  - "
                + "\n  - ".join(issues)
            )

    return issues
