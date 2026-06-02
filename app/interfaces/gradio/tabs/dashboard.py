"""Dashboard — landing page do XFakeSong (UI Fase 1).

Mostra KPIs do sistema, atividade recente, status e notificações.
Todos os service calls têm fallback gracioso para não quebrar a UI quando
algum subsistema (DB, modelos, etc.) está indisponível.
"""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr

logger = logging.getLogger("gradio_dashboard")


# =====================================================================
# Data collectors (todos com fallback)
# =====================================================================


def _count_models() -> int:
    """Conta modelos carregados pelo DetectionService."""
    try:
        from app.dependencies import get_detection_service

        svc = get_detection_service()
        return len(svc.get_available_models() or [])
    except Exception as e:
        logger.debug(f"count_models fallback: {e}")
        return 0


def _count_profiles() -> int:
    """Conta perfis de voz cadastrados."""
    try:
        from app.domain.services.voice_profile_service import VoiceProfileService

        return len(VoiceProfileService().list_profiles() or [])
    except Exception as e:
        logger.debug(f"count_profiles fallback: {e}")
        return 0


def _count_analyses_24h() -> int:
    """Conta análises das últimas 24h no histórico."""
    try:
        from sqlalchemy import func

        from app.core.database import SessionLocal
        from app.domain.models.analysis import AnalysisResult

        with SessionLocal() as db:
            cutoff = datetime.now() - timedelta(hours=24)
            count = (
                db.query(func.count(AnalysisResult.id))
                .filter(AnalysisResult.created_at >= cutoff)
                .scalar()
            )
            return int(count or 0)
    except Exception as e:
        logger.debug(f"count_analyses_24h fallback: {e}")
        return 0


def _datasets_size_gb() -> float:
    """Calcula tamanho total dos diretórios de datasets em GB."""
    total = 0
    for d in [Path("data"), Path("app/datasets"), Path("datasets")]:
        if not d.exists():
            continue
        try:
            for p in d.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except Exception as e:
            logger.debug(f"datasets_size erro em {d}: {e}")
    return round(total / (1024**3), 2)


def _recent_analyses(limit: int = 5) -> list:
    """Últimas N análises do histórico."""
    try:
        from app.core.database import SessionLocal
        from app.domain.models.analysis import AnalysisResult

        with SessionLocal() as db:
            rows = (
                db.query(AnalysisResult)
                .order_by(AnalysisResult.created_at.desc())
                .limit(limit)
                .all()
            )
            out = []
            for r in rows:
                ago = "?"
                try:
                    if r.created_at:
                        delta = datetime.now() - r.created_at.replace(tzinfo=None)
                        if delta.total_seconds() < 60:
                            ago = f"{int(delta.total_seconds())}s atrás"
                        elif delta.total_seconds() < 3600:
                            ago = f"{int(delta.total_seconds() / 60)}min atrás"
                        elif delta.total_seconds() < 86400:
                            ago = f"{int(delta.total_seconds() / 3600)}h atrás"
                        else:
                            ago = f"{int(delta.total_seconds() / 86400)}d atrás"
                except Exception:
                    pass
                label = "FAKE" if r.is_fake else "REAL"
                icon = "⚠" if r.is_fake else "✓"
                out.append(
                    [
                        f"{icon} {label}",
                        r.filename or "unknown",
                        f"{r.confidence:.2f}" if r.confidence is not None else "—",
                        r.model_name or "—",
                        ago,
                    ]
                )
            return out
    except Exception as e:
        logger.debug(f"recent_analyses fallback: {e}")
        return []


def _gpu_status() -> str:
    """Status da GPU usando o módulo central app.core.gpu (GPU.3).

    Mostra nome + CC + Tensor Cores + Mixed Precision em uma linha legível.
    """
    try:
        from app.core.gpu import describe_gpu_setup

        return describe_gpu_setup()
    except Exception as e:
        logger.debug(f"_gpu_status fallback: {e}")
        # Fallback antigo
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return "✗ CPU only"
            return f"✓ {len(gpus)} GPU(s)"
        except Exception:
            return "? TF indisponível"


def _system_status() -> dict:
    """Snapshot consolidado do estado do sistema."""
    status = {
        "version": "1.1.0",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "gpu": _gpu_status(),
        "models": _count_models(),
        "profiles": _count_profiles(),
        "analyses_24h": _count_analyses_24h(),
        "datasets_gb": _datasets_size_gb(),
        "db_ok": False,
    }
    try:
        from app.core.database import check_database_health

        status["db_ok"] = bool(check_database_health())
    except Exception:
        pass
    return status


# =====================================================================
# HTML builders (componentes visuais)
# =====================================================================


def _kpi_card(
    icon: str, value: str, label: str, color_class: str = "kpi-primary"
) -> str:
    """Renderiza um KPI card em HTML."""
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def _render_kpi_row() -> str:
    """Renderiza fileira de 4 KPI cards a partir do estado atual."""
    s = _system_status()
    return f"""
    <div class="kpi-row">
        {_kpi_card("📊", str(s["analyses_24h"]), "Análises últimas 24h", "kpi-primary")}
        {_kpi_card("🎯", str(s["models"]), "Modelos carregados", "kpi-success")}
        {_kpi_card("🎤", str(s["profiles"]), "Perfis de voz", "kpi-info")}
        {_kpi_card("💾", f"{s['datasets_gb']} GB", "Datasets totais", "kpi-neutral")}
    </div>
    """


def _render_system_status() -> str:
    """Painel de status do sistema (DB, modelos, GPU, etc.)."""
    s = _system_status()
    db_icon = "✓" if s["db_ok"] else "⚠"
    db_class = "ok" if s["db_ok"] else "warn"
    models_icon = "✓" if s["models"] > 0 else "⚠"
    models_class = "ok" if s["models"] > 0 else "warn"
    gpu_class = "ok" if s["gpu"].startswith("✓") else "warn"

    return f"""
    <div class="status-panel">
        <div class="status-item {db_class}">
            <span class="status-icon">{db_icon}</span>
            <span class="status-label">Database</span>
            <span class="status-value">{"OK" if s["db_ok"] else "Indisponível"}</span>
        </div>
        <div class="status-item {models_class}">
            <span class="status-icon">{models_icon}</span>
            <span class="status-label">Modelos</span>
            <span class="status-value">{s["models"]} carregado(s)</span>
        </div>
        <div class="status-item {gpu_class}">
            <span class="status-icon">{"✓" if gpu_class == "ok" else "⚠"}</span>
            <span class="status-label">GPU</span>
            <span class="status-value">{s["gpu"]}</span>
        </div>
        <div class="status-item info">
            <span class="status-icon">🐍</span>
            <span class="status-label">Python</span>
            <span class="status-value">{s["python"]} ({s["platform"]})</span>
        </div>
    </div>
    """


def _render_recent_table() -> str:
    """Renderiza tabela HTML de análises recentes (ou empty state)."""
    rows = _recent_analyses(limit=5)
    if not rows:
        return """
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <div class="empty-title">Nenhuma análise ainda</div>
            <div class="empty-desc">
                Faça sua primeira análise na aba <strong>🎯 Detectar</strong>
                ou treine um modelo em <strong>🎓 Treinar</strong>.
            </div>
        </div>
        """

    tbody = ""
    for row in rows:
        result, filename, conf, model, when = row
        result_class = "result-fake" if "FAKE" in result else "result-real"
        tbody += (
            f"<tr>"
            f"<td><span class='{result_class}'>{result}</span></td>"
            f"<td class='mono'>{filename}</td>"
            f"<td>{conf}</td>"
            f"<td>{model}</td>"
            f"<td class='dim'>{when}</td>"
            f"</tr>"
        )

    return f"""
    <table class="recent-table">
        <thead>
            <tr>
                <th>Resultado</th>
                <th>Arquivo</th>
                <th>Confidence</th>
                <th>Modelo</th>
                <th>Quando</th>
            </tr>
        </thead>
        <tbody>
            {tbody}
        </tbody>
    </table>
    """


# =====================================================================
# Build tab
# =====================================================================


def create_dashboard_tab():
    """Constrói a aba 🏠 Dashboard."""
    with gr.Tab("🏠 Painel", id="tab_dashboard"):
        from app.interfaces.gradio.utils.components import page_header

        page_header(
            "🏠",
            "Painel",
            "Visão geral do sistema — modelos, perfis, hardware e "
            "atividade recente.",
        )
        # KPI cards no topo
        kpi_html = gr.HTML(_render_kpi_row(), elem_id="dashboard_kpis")

        # 2 colunas: histórico recente + status do sistema
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Últimas Análises")
                recent_html = gr.HTML(
                    _render_recent_table(), elem_id="dashboard_recent"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Status do Sistema")
                status_html = gr.HTML(
                    _render_system_status(), elem_id="dashboard_status"
                )

        # ─────────────── GPU Diagnóstico (acionável) ───────────────
        # Abre automaticamente quando há mismatch (hardware NVIDIA presente
        # mas TF não está vendo). Caso happy path (TF OK) fica fechado.
        from app.core.gpu import get_diagnosis_html
        from app.core.gpu import get_setup_result as _gpu_get

        try:
            _gpu_diag = (_gpu_get() or {}).get("diagnosis") or {}
            _open_gpu = _gpu_diag.get("severity") in ("warning", "error")
        except Exception:
            _open_gpu = False
        with gr.Accordion(
            "🎮 Diagnóstico de GPU",
            open=_open_gpu,
        ):
            gpu_diag_html = gr.HTML(
                get_diagnosis_html(),
                elem_id="dashboard_gpu_diag",
            )
            refresh_gpu_btn = gr.Button(
                "🔄 Re-detectar GPU",
                size="sm",
                scale=0,
            )

        # ────────────────────────── Handlers ──────────────────────────

        def refresh_dashboard():
            """Re-renderiza os componentes do Dashboard."""
            return (
                _render_kpi_row(),
                _render_recent_table(),
                _render_system_status(),
                get_diagnosis_html(),
            )

        # Re-detectar GPU sem reiniciar app (útil após instalar pynvml/DML plugin)
        def _redetect_gpu():
            # Força nova execução do setup_gpu — _setup_done já está True então
            # essa chamada é idempotente; para forçar nova detecção, resetamos
            # a flag explicitamente.
            from app.core import gpu as _gpu_mod

            _gpu_mod._setup_done = False
            _gpu_mod._setup_result = {}
            _gpu_mod.setup_gpu(log_level=40)  # WARNING+
            from app.interfaces.gradio.utils.notifications import notify_info

            notify_info("Detecção de GPU re-executada")
            return get_diagnosis_html()

        refresh_gpu_btn.click(
            fn=_redetect_gpu,
            inputs=[],
            outputs=[gpu_diag_html],
        )

        # Auto-refresh a cada 30s via Timer (Gradio 4.31+)
        try:
            timer = gr.Timer(30.0)
            timer.tick(
                fn=refresh_dashboard,
                inputs=[],
                outputs=[kpi_html, recent_html, status_html, gpu_diag_html],
            )
        except Exception as e:
            # gr.Timer só existe em versões recentes — degradação graciosa
            logger.debug(f"gr.Timer indisponível: {e}")
