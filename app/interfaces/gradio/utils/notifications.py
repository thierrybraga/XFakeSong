"""Sistema unificado de notificações + feedback (UI Refactor).

Substitui os helpers ad-hoc `notify_error`/`notify_info` por uma API
estruturada com:

- **5 níveis semânticos**: success, info, warning, error, critical
- **History ring buffer**: últimas 50 notificações em memória (deque)
- **ActionableError**: erros com sugestão de fix + link (opcional)
- **Backwards compat**: `notify_error`/`notify_info` continuam funcionando
- **Auto-toast**: integração transparente com `gr.Info`/`gr.Warning`/`gr.Error`

Uso típico:

    from app.interfaces.gradio.utils.notifications import notify, notify_error

    notify("success", "Modelo treinado", "AASIST_v1 — val_acc=0.91")
    notify("warning", "Dataset desbalanceado", "Ratio 1:8. Class weighting será aplicado.")

    # Erro acionável (com hint de fix)
    notify_error(
        "Modelo não encontrado",
        hint="Treine um modelo na aba 🎓 Treinar primeiro.",
        error_code="MODEL_NOT_FOUND",
    )

    # Histórico (para o notification center)
    from app.interfaces.gradio.utils.notifications import get_history
    for note in get_history(limit=10):
        print(note.title, note.timestamp)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from html import escape
from typing import Deque, List, Literal, Optional

from app.core.feedback import (
    clear_feedback_events,
    get_feedback_events,
    get_feedback_unread_count,
    mark_feedback_read,
    publish_feedback,
)

logger = logging.getLogger(__name__)

NotifyLevel = Literal["success", "info", "warning", "error", "critical"]

_LEVEL_ICONS = {
    "success": "✓",
    "info": "ℹ",
    "warning": "⚠",
    "error": "✗",
    "critical": "⛔",
}
_LEVEL_COLORS = {
    "success": "#10b981",
    "info": "#06b6d4",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "critical": "#dc2626",
}

# Ring buffer global (compartilhado entre threads do Gradio queue)
_history_lock = threading.Lock()
_history: Deque["Notification"] = deque(maxlen=50)
_unread_count: int = 0


# =====================================================================
# Modelos
# =====================================================================


@dataclass
class Notification:
    """Uma notificação no sistema."""

    level: NotifyLevel
    title: str
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    error_code: Optional[str] = None
    hint: Optional[str] = None  # sugestão de fix (para erros)
    link: Optional[str] = None  # URL opcional (ex: /api/docs)
    actions: List[dict] = field(default_factory=list)  # botões inline (futuro)

    @property
    def icon(self) -> str:
        return _LEVEL_ICONS.get(self.level, "•")

    @property
    def color(self) -> str:
        return _LEVEL_COLORS.get(self.level, "#94a3b8")

    @property
    def ago(self) -> str:
        """Quanto tempo atrás (relativo)."""
        delta = time.time() - self.timestamp
        if delta < 60:
            return f"{int(delta)}s atrás"
        if delta < 3600:
            return f"{int(delta / 60)}min atrás"
        if delta < 86400:
            return f"{int(delta / 3600)}h atrás"
        return f"{int(delta / 86400)}d atrás"


@dataclass
class ActionableError:
    """Erro com sugestão de fix — separado de Notification para reutilização."""

    title: str
    message: str
    hint: str
    error_code: Optional[str] = None
    link: Optional[str] = None

    def to_notification(self) -> Notification:
        return Notification(
            level="error",
            title=self.title,
            message=self.message,
            error_code=self.error_code,
            hint=self.hint,
            link=self.link,
        )


# =====================================================================
# Erros comuns pré-definidos (acionáveis)
# =====================================================================


class CommonErrors:
    """Catálogo de erros frequentes com hint pronto.

    Reusar evita reinventar mensagens em vários lugares do código.
    """

    @staticmethod
    def model_not_found(model_name: str = "") -> ActionableError:
        return ActionableError(
            title="Modelo não encontrado",
            message=(
                f"O modelo '{model_name}' não está carregado."
                if model_name
                else "Nenhum modelo treinado disponível."
            ),
            hint=(
                "Treine um modelo na aba 🎓 Treinar (Wizard) ou importe um "
                "modelo existente em ⚙ Admin → Datasets."
            ),
            error_code="MODEL_NOT_FOUND",
        )

    @staticmethod
    def no_audio_provided() -> ActionableError:
        return ActionableError(
            title="Nenhum áudio fornecido",
            message="Não foi possível analisar — nenhum arquivo enviado.",
            hint="Arraste e solte um arquivo .wav/.mp3/.flac no campo de upload.",
            error_code="NO_AUDIO",
        )

    @staticmethod
    def unsupported_format(ext: str, supported: List[str]) -> ActionableError:
        return ActionableError(
            title="Formato não suportado",
            message=f"O arquivo '{ext}' não é aceito.",
            hint=f"Use um dos formatos suportados: {', '.join(supported)}",
            error_code="UNSUPPORTED_FORMAT",
        )

    @staticmethod
    def dataset_invalid(path: str, reason: str) -> ActionableError:
        return ActionableError(
            title="Dataset inválido",
            message=f"`{path}`: {reason}",
            hint=(
                "Verifique se a pasta contém subdiretórios para cada classe "
                "(ex: real/, fake/) e arquivos .wav dentro de cada um."
            ),
            error_code="DATASET_INVALID",
        )

    @staticmethod
    def port_in_use(port: int = 7860) -> ActionableError:
        return ActionableError(
            title=f"Porta {port} já em uso",
            message="Outra instância do XFakeSong (ou outro app) está usando essa porta.",
            hint=(
                f"Encerre o processo (Windows: `netstat -ano | findstr :{port}` "
                f"+ `taskkill /F /PID <pid>`) ou rode em outra porta."
            ),
            error_code="PORT_IN_USE",
        )

    @staticmethod
    def db_unavailable(detail: str = "") -> ActionableError:
        return ActionableError(
            title="Banco de dados indisponível",
            message=detail or "Não foi possível conectar ao SQLite.",
            hint=(
                "Verifique se `data/` é writable e se o disco não está cheio. "
                "Em Docker, garanta que o volume `./data:/app/data` está mapeado."
            ),
            error_code="DB_UNAVAILABLE",
        )

    @staticmethod
    def training_failed(reason: str) -> ActionableError:
        return ActionableError(
            title="Treinamento falhou",
            message=reason,
            hint=(
                "Verifique os logs do servidor. Causas comuns:\n"
                "• Labels não-binarizados (use o Wizard com pasta real/fake)\n"
                "• Dataset muito pequeno (<100 samples por classe)\n"
                "• OOM em GPU (reduza batch_size ou habilite memory_growth)"
            ),
            error_code="TRAINING_FAILED",
        )


# =====================================================================
# API pública
# =====================================================================


def notify(
    level: NotifyLevel,
    title: str,
    message: str = "",
    *,
    hint: Optional[str] = None,
    error_code: Optional[str] = None,
    link: Optional[str] = None,
    silent: bool = False,
) -> Notification:
    """Cria notificação, adiciona ao histórico e emite toast Gradio.

    Args:
        level: 'success' | 'info' | 'warning' | 'error' | 'critical'
        title: título curto (visível no toast)
        message: detalhes adicionais (visíveis no notification center)
        hint: sugestão de fix (recomendado para 'error' e 'critical')
        error_code: código curto (ex: 'MODEL_NOT_FOUND')
        link: URL opcional
        silent: se True, NÃO emite toast (só registra no histórico)

    Returns:
        Notification criada.
    """
    note = publish_feedback(
        level,
        title,
        message,
        source="app.interfaces.gradio",
        category="notification",
        hint=hint,
        error_code=error_code,
        link=link,
    )

    # Emite toast no Gradio (com fallback gracioso)
    if not silent:
        _emit_toast(note)

    return note


def _emit_toast(note: Notification) -> None:
    """Mostra toast no Gradio. Sem-op se Gradio indisponível."""
    try:
        import gradio as gr

        # Monta o texto do toast
        text = f"{note.icon} {note.title}"
        if note.message:
            text += f"\n{note.message}"
        if note.hint:
            text += f"\n💡 {note.hint}"

        if note.level in ("error", "critical"):
            # gr.Error é raised → causa toast mas também interrompe.
            # Para notification não-bloqueante, usamos gr.Warning.
            gr.Warning(text)
        elif note.level == "warning":
            gr.Warning(text)
        else:  # info, success
            gr.Info(text)
    except Exception as e:
        logger.debug(f"Toast emit falhou (não-crítico): {e}")


# ── Helpers convenientes (por nível) ─────────────────────────────────


def notify_success(title: str, message: str = "", **kwargs) -> Notification:
    return notify("success", title, message, **kwargs)


def notify_info(
    message: str = "", title: str = "", *, log_info: bool = False, **kwargs
) -> Notification:
    """Compatível com a antiga assinatura `notify_info(message)`.

    Se chamado com 1 argumento posicional, usa-o como TÍTULO para ser
    mais informativo (mensagens antigas como "Toast" cabem como título).
    """
    if not title:
        title = message
        message = ""
    return notify("info", title, message, **kwargs)


def notify_warning(title: str, message: str = "", **kwargs) -> Notification:
    return notify("warning", title, message, **kwargs)


def notify_error(
    message: str = "",
    title: str = "",
    *,
    hint: Optional[str] = None,
    error_code: Optional[str] = None,
    log_exception: bool = True,  # backwards compat
    **kwargs,
) -> Notification:
    """Compatível com `notify_error(message)` legado.

    Args:
        message: pode ser tanto a mensagem (estilo antigo) quanto o título
            se `title` não for fornecido.
        title: título explícito (se omitido, usa message).
        hint: sugestão de fix.
        error_code: código.
        log_exception: backwards compat (sempre logamos errors agora).
    """
    _ = log_exception  # parâmetro mantido por compat
    if not title:
        title = message or "Erro"
        message = ""
    return notify(
        "error",
        title,
        message,
        hint=hint,
        error_code=error_code,
        **kwargs,
    )


def notify_critical(title: str, message: str = "", **kwargs) -> Notification:
    return notify("critical", title, message, **kwargs)


def notify_from_actionable(
    err: ActionableError, *, silent: bool = False
) -> Notification:
    """Atalho: emite notificação a partir de um ActionableError."""
    return notify(
        "error",
        err.title,
        err.message,
        hint=err.hint,
        error_code=err.error_code,
        link=err.link,
        silent=silent,
    )


# ── Histórico / Notification Center ──────────────────────────────────


def get_history(limit: int = 10) -> List[Notification]:
    """Retorna as últimas N notificações (mais recentes primeiro)."""
    return get_feedback_events(limit=limit)


def clear_history() -> int:
    """Limpa o histórico e retorna quantas notificações foram removidas."""
    return clear_feedback_events()


def get_unread_count() -> int:
    """Conta de notificações não lidas (resetado por mark_all_read)."""
    return get_feedback_unread_count()


def mark_all_read() -> None:
    """Marca todas como lidas (zera o counter, mantém histórico)."""
    mark_feedback_read()


# ── Render HTML para Notification Center ─────────────────────────────


def render_notification_center_html(limit: int = 10) -> str:
    """HTML do painel de notificações (para gr.HTML no Dashboard / dropdown)."""
    notes = get_history(limit=limit)
    if not notes:
        return (
            '<div class="notif-empty">'
            '<div class="notif-empty-icon">📭</div>'
            "<div>Nenhuma notificação ainda</div>"
            "</div>"
        )

    items_html = ""
    for n in notes:
        title = escape(str(n.title))
        message = escape(str(n.message)) if n.message else ""
        hint = escape(str(n.hint)) if n.hint else ""
        error_code = escape(str(n.error_code)) if n.error_code else ""
        source = escape(str(getattr(n, "source", "")))
        hint_html = f'<div class="notif-hint">💡 {hint}</div>' if n.hint else ""
        code_html = (
            f'<span class="notif-code">{error_code}</span>' if n.error_code else ""
        )
        source_html = f'<span class="notif-code">{source}</span>' if source else ""
        items_html += f"""
        <div class="notif-item notif-{n.level}">
            <div class="notif-icon">{n.icon}</div>
            <div class="notif-body">
                <div class="notif-header">
                    <span class="notif-title">{title}</span>
                    {code_html}
                    {source_html}
                    <span class="notif-ago">{n.ago}</span>
                </div>
                {f'<div class="notif-msg">{message}</div>' if n.message else ""}
                {hint_html}
            </div>
        </div>
        """

    return f'<div class="notif-list">{items_html}</div>'


__all__ = [
    "Notification",
    "ActionableError",
    "CommonErrors",
    "NotifyLevel",
    "notify",
    "notify_success",
    "notify_info",
    "notify_warning",
    "notify_error",
    "notify_critical",
    "notify_from_actionable",
    "get_history",
    "clear_history",
    "get_unread_count",
    "mark_all_read",
    "render_notification_center_html",
]
