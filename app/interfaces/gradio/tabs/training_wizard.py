"""Wizard de Treinamento — UI Fase 2.

Substitui as 5 sub-tabs aninhadas do training.py original por um workflow
linear de 4 steps. Cada step é um gr.Group cuja visibilidade é controlada
pelo estado atual, dando ao usuário uma sensação clara de progressão.

Steps:
    1. Dataset       — escolher pasta + validação automática de classes
    2. Modelo        — escolher arquitetura + variante (com cards informativos)
    3. Hiperparâmetros — formulário organizado (basic + advanced)
    4. Treinar       — execução com progress visual + logs em tempo real

Backend INTACTO: reutiliza `create_model_by_name` do registry e a mesma
lógica de binarização de labels (BUG.Training.4).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import List

import gradio as gr
from app.core.performance import optimize_tf_dataset
from app.interfaces.gradio.utils.notifications import (
    CommonErrors,
    notify_from_actionable,
    notify_success,
)
from app.interfaces.gradio.utils.plotting import (
    PLOT_ACCENT,
    PLOT_DANGER,
    close_fig,
    safe_tight_layout,
    style_ax,
)

logger = logging.getLogger("gradio_training_wizard")

# Diretório onde os modelos treinados são persistidos (mesmo lido pelo
# ModelLoader / DetectionService em app/dependencies.py).
MODELS_DIR = Path("app/models")

# Holder do ÚLTIMO modelo treinado com sucesso nesta sessão do servidor.
# O modelo Keras é criado dentro do gerador _run_training e seria coletado
# pelo GC ao terminar — guardamos a referência aqui para o botão "Salvar"
# do Step 4 conseguir persisti-lo. (App local single-user: holder de slot único.)
_LAST_TRAINED: dict | None = None


def _slugify_model_name(name: str) -> str:
    """Normaliza o nome do modelo para um nome de arquivo seguro."""
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "modelo"


def _save_trained_model(name: str) -> tuple[bool, str]:
    """Persiste o último modelo treinado em app/models/ (.keras + _config.json).

    O config inclui o input_contract (n_fft/hop/n_mels/sample_rate) para que a
    INFERÊNCIA reproduza exatamente as features usadas no treino — caso
    contrário as predições saem erradas (mismatch train/inference).

    Returns (ok, mensagem_markdown).
    """
    global _LAST_TRAINED
    if not _LAST_TRAINED or _LAST_TRAINED.get("model") is None:
        return False, (
            "⚠️ Nenhum modelo treinado nesta sessão. "
            "Conclua um treino no Step 4 antes de salvar."
        )

    import json

    info = _LAST_TRAINED
    kind = info.get("kind", "keras")
    slug = _slugify_model_name(name)
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Keras → .keras ; sklearn (SVM/RF) → .pkl (+ _scaler.pkl)
        suffix = ".pkl" if kind == "sklearn" else ".keras"
        model_path = MODELS_DIR / f"{slug}{suffix}"
        config_path = MODELS_DIR / f"{slug}_config.json"

        if model_path.exists() or config_path.exists():
            return False, (
                f"⚠️ Já existe um modelo chamado **{slug}**. "
                "Escolha outro nome para não sobrescrever."
            )

        if kind == "sklearn":
            import joblib

            joblib.dump(info["model"], str(model_path))
            if info.get("scaler") is not None:
                joblib.dump(info["scaler"], str(MODELS_DIR / f"{slug}_scaler.pkl"))
        else:
            info["model"].save(str(model_path))

        config = {
            "architecture": info["arch"],
            "input_shape": list(info["input_shape"]),
            "input_contract": info["input_contract"],
            "trained_by": "wizard",
            "model_type": "sklearn" if kind == "sklearn" else "tensorflow",
            "feature_names": info.get("feature_names"),
            "val_accuracy": info.get("val_acc"),
            "val_loss": info.get("val_loss"),
            "epochs": info.get("epochs"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Modelo salvo: {model_path} ({size_mb:.1f} MB)")

        # Recarrega os modelos no DetectionService para uso imediato em Detectar.
        reload_note = " Reinicie ou abra a aba Detectar para carregá-lo."
        try:
            from app.dependencies import get_detection_service

            svc = get_detection_service()
            loader = getattr(svc, "model_loader", None)
            if loader is not None and hasattr(loader, "load_available_models"):
                loader.load_available_models()
                reload_note = " Já disponível na aba **🎯 Detectar**."
        except Exception as e:
            logger.debug(f"Reload do DetectionService pulado: {e}")

        return True, (
            f"✅ Modelo salvo como **{slug}** "
            f"(`{model_path.name}`, {size_mb:.1f} MB).{reload_note}"
        )
    except Exception as e:
        logger.error(f"Falha ao salvar modelo: {e}", exc_info=True)
        return False, f"❌ Erro ao salvar: {e}"


# =====================================================================
# Helpers de feedback ao vivo durante o treinamento (Step 4)
# =====================================================================


def _fmt_secs(s: float) -> str:
    """Formata segundos como mm:ss (ou h:mm:ss se > 1h)."""
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _metric_card(label: str, value, kind: str = "neutral", fmt: str = "{:.4f}") -> str:
    """Card compacto de uma métrica (loss/acc) com cor semântica."""
    import math

    if value is None:
        shown, color = "—", "var(--xf-text-muted, #94a3b8)"
    elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        shown, color = "NaN", "#ef4444"
    else:
        shown = fmt.format(value)
        color = {
            "good": "#10b981",
            "bad": "#ef4444",
            "accent": "#06b6d4",
            "neutral": "var(--xf-text, #f1f5f9)",
        }.get(kind, "var(--xf-text, #f1f5f9)")
    return (
        f'<div class="tl-card">'
        f'<div class="tl-card-label">{label}</div>'
        f'<div class="tl-card-value" style="color:{color}">{shown}</div>'
        f"</div>"
    )


def _history_figure(train_loss, val_loss, train_acc, val_acc):
    """Figura canônica das curvas de treino (Loss | Accuracy).

    Usada tanto no plot AO VIVO quanto no FINAL, para garantir consistência:
    - Eixo X em épocas 1-based ("Época"), com ticks inteiros em treinos curtos.
    - Curva de validação só aparece quando há dados (evita legenda 'val' vazia
      e linha fantasma).
    - Eixo Y de accuracy fixado em [0, 1.02] — a acurácia é uma fração, então a
      auto-escala fazia variações minúsculas parecerem enormes.

    Retorna a `Figure` (o chamador fecha com close_fig para evitar leak).
    """
    import matplotlib.pyplot as plt

    def _clean(seq):
        return [v for v in (seq or []) if v is not None]

    tl, vl = _clean(train_loss), _clean(val_loss)
    ta, va = _clean(train_acc), _clean(val_acc)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss ──
    style_ax(ax[0], fig, "Loss")
    ax[0].plot(range(1, len(tl) + 1), tl, label="treino",
               color=PLOT_ACCENT, marker="o", ms=3)
    if vl:
        ax[0].plot(range(1, len(vl) + 1), vl, label="validação",
                   color=PLOT_DANGER, marker="o", ms=3)
    ax[0].set_xlabel("Época")
    if 1 <= len(tl) <= 20:
        ax[0].set_xticks(list(range(1, len(tl) + 1)))
    ax[0].legend()

    # ── Accuracy ──
    style_ax(ax[1], fig, "Accuracy")
    ax[1].plot(range(1, len(ta) + 1), ta, label="treino",
               color=PLOT_ACCENT, marker="o", ms=3)
    if va:
        ax[1].plot(range(1, len(va) + 1), va, label="validação",
                   color=PLOT_DANGER, marker="o", ms=3)
    ax[1].set_xlabel("Época")
    ax[1].set_ylim(0.0, 1.02)
    if 1 <= len(ta) <= 20:
        ax[1].set_xticks(list(range(1, len(ta) + 1)))
    ax[1].legend()

    safe_tight_layout(fig)
    return fig


def _train_status_html(
    arch: str,
    device: str,
    epoch: int,
    total: int,
    hist: dict,
    elapsed_s: float,
    phase: str = "running",
    note: str = "",
) -> str:
    """Painel HTML rico de progresso do treino ao vivo.

    phase: "preparing" | "running" | "done" | "error"
    hist: {"loss":[...], "acc":[...], "val_loss":[...], "val_acc":[...]}
    """
    import math

    def _last(key):
        seq = hist.get(key) or []
        return seq[-1] if seq else None

    loss, acc = _last("loss"), _last("acc")
    vloss, vacc = _last("val_loss"), _last("val_acc")

    pct = (epoch / total * 100.0) if total else 0.0
    pct = max(0.0, min(100.0, pct))

    # ETA: extrapola pelo tempo médio por época concluída
    eta_txt = "—"
    if phase == "running" and epoch > 0 and epoch < total and elapsed_s > 0:
        per_epoch = elapsed_s / epoch
        eta_txt = _fmt_secs(per_epoch * (total - epoch))

    # Estado/Cabeçalho
    if phase == "preparing":
        bar_color = "linear-gradient(90deg,#3b82f6,#06b6d4)"
        head_icon, head_txt, head_color = "⏳", "Preparando…", "#06b6d4"
        pct_disp = 4.0  # barra indeterminada-ish
    elif phase == "done":
        bar_color = "linear-gradient(90deg,#10b981,#34d399)"
        head_icon, head_txt, head_color = "✓", "Concluído", "#10b981"
        pct_disp = 100.0
    elif phase == "error":
        bar_color = "linear-gradient(90deg,#ef4444,#f87171)"
        head_icon, head_txt, head_color = "✗", "Falhou", "#ef4444"
        pct_disp = 100.0
    else:  # running
        bar_color = "linear-gradient(90deg,#3b82f6,#06b6d4)"
        head_icon, head_txt, head_color = (
            "▶",
            f"Treinando · época {epoch}/{total}",
            "#3b82f6",
        )
        pct_disp = pct

    # cor das métricas
    acc_kind = (
        "good"
        if (isinstance(acc, float) and not math.isnan(acc) and acc >= 0.5)
        else "neutral"
    )
    vacc_kind = (
        "good"
        if (isinstance(vacc, float) and not math.isnan(vacc) and vacc >= 0.5)
        else "neutral"
    )
    loss_kind = (
        "bad"
        if (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss)))
        else "neutral"
    )
    vloss_kind = (
        "bad"
        if (isinstance(vloss, float) and (math.isnan(vloss) or math.isinf(vloss)))
        else "accent"
    )

    note_html = f'<div class="tl-note">{note}</div>' if note else ""

    return f"""
    <div class="train-live">
      <div class="tl-head">
        <span class="tl-head-title" style="color:{head_color}">{head_icon} {head_txt}</span>
        <span class="tl-head-meta">{arch} · {device} · ⏱ {_fmt_secs(elapsed_s)}
          {("· ETA " + eta_txt) if eta_txt != "—" else ""}</span>
      </div>
      <div class="tl-bar-track">
        <div class="tl-bar-fill" style="width:{pct_disp:.1f}%;background:{bar_color}"></div>
      </div>
      <div class="tl-cards">
        {_metric_card("loss", loss, loss_kind)}
        {_metric_card("accuracy", acc, acc_kind)}
        {_metric_card("val_loss", vloss, vloss_kind)}
        {_metric_card("val_accuracy", vacc, vacc_kind)}
      </div>
      {note_html}
    </div>
    """


# Heurística de classificação real/fake reusada (BUG.Training.4)
_REAL_PATTERNS = re.compile(
    r"^(real|bonafide|genuine|original|authentic|natural)",
    re.IGNORECASE,
)


# =====================================================================
# Helpers de dados (validação)
# =====================================================================


def _scan_dataset(path_str: str) -> dict:
    """Inspeciona um diretório de dataset.

    Returns:
        {
            "ok": bool,
            "message": str (mensagem amigável),
            "class_names": list[str],
            "binary_map": dict[int, int],  # idx_dir -> {0 real, 1 fake}
            "file_count": int,
            "files_per_class": dict[str, int],
            "n_real": int,
            "n_fake": int,
        }
    """
    out = {
        "ok": False,
        "message": "",
        "class_names": [],
        "binary_map": {},
        "file_count": 0,
        "files_per_class": {},
        "n_real": 0,
        "n_fake": 0,
    }
    p = Path(path_str)
    if not p.exists():
        out["message"] = f"❌ Pasta `{path_str}` não existe."
        return out
    if not p.is_dir():
        out["message"] = f"❌ `{path_str}` não é um diretório."
        return out

    subdirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not subdirs:
        out["message"] = (
            f"❌ Pasta vazia ou sem subdiretórios de classes.\n\n"
            f"Estrutura esperada:\n"
            f"```\n{path_str}/\n  ├─ real/\n  │   ├─ audio1.wav\n  │   └─ ...\n"
            f"  └─ fake/\n      ├─ audio2.wav\n      └─ ...\n```"
        )
        return out

    # Conta arquivos .wav por subdir
    files_per_class = {}
    binary_map = {}
    n_real = n_fake = 0
    for idx, name in enumerate(subdirs):
        wavs = list((p / name).glob("*.wav"))
        files_per_class[name] = len(wavs)
        is_real = bool(_REAL_PATTERNS.match(name.strip()))
        binary_map[idx] = 0 if is_real else 1
        if is_real:
            n_real += files_per_class[name]
        else:
            n_fake += files_per_class[name]

    total = sum(files_per_class.values())
    if total == 0:
        out["message"] = "❌ Nenhum arquivo .wav encontrado nos subdiretórios."
        return out

    real_dirs = [n for n in subdirs if _REAL_PATTERNS.match(n.strip())]
    fake_dirs = [n for n in subdirs if not _REAL_PATTERNS.match(n.strip())]

    if not real_dirs:
        out["message"] = (
            f"❌ Nenhum subdiretório de áudios REAIS encontrado.\n\n"
            f"Subdiretórios devem começar com: `real`, `bonafide`, `genuine`, "
            f"`original`, `authentic` ou `natural`.\n\n"
            f"Encontrei: {subdirs}"
        )
        return out
    if not fake_dirs:
        out["message"] = (
            f"❌ Nenhum subdiretório de áudios FAKE encontrado.\n\n"
            f"Apenas subdiretórios de áudios reais detectados: {real_dirs}"
        )
        return out

    out.update(
        {
            "ok": True,
            "class_names": subdirs,
            "binary_map": binary_map,
            "file_count": total,
            "files_per_class": files_per_class,
            "n_real": n_real,
            "n_fake": n_fake,
        }
    )

    # Mensagem detalhada
    ratio = n_real / max(n_fake, 1)
    if ratio < 0.5 or ratio > 2:
        balance = f"⚠️ Desbalanceado ({ratio:.2f}:1) — class weighting será aplicado"
    else:
        balance = f"✓ Balanceado ({ratio:.2f}:1)"

    detail = "\n".join(
        f"  • {name} ({'real' if binary_map[idx] == 0 else 'fake'}): "
        f"{files_per_class[name]} arquivos"
        for idx, name in enumerate(subdirs)
    )
    out["message"] = (
        f"✓ **Dataset válido** — {total} arquivos em {len(subdirs)} subdiretórios\n\n"
        f"{detail}\n\n"
        f"**Total**: {n_real} real / {n_fake} fake — {balance}"
    )
    return out


# =====================================================================
# Cards de modelos (Step 2)
# =====================================================================


def _get_model_catalog() -> List[dict]:
    """Catálogo de arquiteturas para o Step 2 com info amigável."""
    try:
        from app.domain.models.architectures.registry import architecture_registry

        archs = architecture_registry.list_architectures()
    except Exception:
        archs = [
            "AASIST",
            "RawGAT-ST",
            "RawNet2",
            "Sonic Sleuth",
            "WavLM",
            "HuBERT",
            "Conformer",
            "Hybrid CNN-Transformer",
            "SpectrogramTransformer",
            "EfficientNet-LSTM",
            "MultiscaleCNN",
            "Ensemble",
        ]

    # Categoria + descrição amigável
    catalog = {
        "AASIST": (
            "🕸️",
            "Graph Attention",
            "Spectro-temporal GAT + HS-GAL. EER 0.83% em ASVspoof. Recomendado para máxima accuracy.",
        ),
        "RawGAT-ST": (
            "🕸️",
            "Graph Attention",
            "Variante do AASIST com foco temporal. Bom para áudios curtos.",
        ),
        "RawNet2": (
            "🌊",
            "Raw audio",
            "SincNet + ResBlocks + GRU. Trabalha direto na waveform.",
        ),
        "Sonic Sleuth": (
            "🎯",
            "Lightweight",
            "Modelo leve (~3M params). 98.27% accuracy. Ideal para edge.",
        ),
        "WavLM": (
            "🤖",
            "SSL Backbone",
            "Self-supervised. Robusto a ruído e canal. Requer mais GPU.",
        ),
        "HuBERT": (
            "🤖",
            "SSL Backbone",
            "Hidden-Unit BERT. Aprende fonemas auto-supervisionado.",
        ),
        "Conformer": (
            "⚡",
            "Transformer + Conv",
            "Conv local + Self-Attention global. Estado-da-arte em speech.",
        ),
        "Hybrid CNN-Transformer": (
            "⚡",
            "Transformer + Conv",
            "CCT. CNN tokenizer + Transformer. 91.47% accuracy.",
        ),
        "SpectrogramTransformer": (
            "🔭",
            "Vision Transformer",
            "ViT adaptado para espectrogramas (AST).",
        ),
        "EfficientNet-LSTM": (
            "📊",
            "Transfer Learning",
            "EfficientNet + Bi-LSTM. Bom baseline com transfer learning.",
        ),
        "MultiscaleCNN": (
            "🔍",
            "CNN multi-escala",
            "Res2Net-50. Multi-scale hierárquico dentro do bloco residual.",
        ),
        "Ensemble": (
            "🎼",
            "Fusão multi-feature",
            "4 branches (Mel+LFCC+CQT+MFCC) + fusão. EER 3%.",
        ),
    }

    out = []
    for arch in archs:
        icon, category, desc = catalog.get(arch, ("🔧", "Outro", f"Arquitetura {arch}"))
        out.append(
            {
                "name": arch,
                "icon": icon,
                "category": category,
                "description": desc,
            }
        )

    # ML clássico (sklearn)
    out.append(
        {
            "name": "SVM",
            "icon": "📐",
            "category": "Classical ML",
            "description": "Support Vector Machine. Baseline rápido com features tabulares.",
        }
    )
    out.append(
        {
            "name": "Random Forest",
            "icon": "🌳",
            "category": "Classical ML",
            "description": "Ensemble de árvores. Robusto, paraleliza em CPU multi-core.",
        }
    )
    return out


def _render_model_cards_html(selected: str = "") -> str:
    """HTML grid de cards de modelos, com card selecionado destacado."""
    cards = _get_model_catalog()
    html = '<div class="model-grid">'
    for m in cards:
        is_sel = "model-card-selected" if m["name"] == selected else ""
        html += f"""
        <div class="model-card {is_sel}" data-arch="{m["name"]}">
            <div class="model-icon">{m["icon"]}</div>
            <div class="model-name">{m["name"]}</div>
            <div class="model-category">{m["category"]}</div>
            <div class="model-desc">{m["description"]}</div>
        </div>
        """
    html += "</div>"
    return html


# =====================================================================
# Step navigation
# =====================================================================


def _step_visibility(current: int):
    """Retorna 4 visibilidades para os 4 gr.Groups dos steps."""
    return [gr.update(visible=(i == current)) for i in range(1, 5)]


def _stepper_html(current: int) -> str:
    """Indicador visual de progresso (passos 1-4)."""
    steps = [
        ("1", "Dataset"),
        ("2", "Modelo"),
        ("3", "Hiperparâmetros"),
        ("4", "Treinar"),
    ]
    html = '<div class="wizard-stepper">'
    for i, (num, label) in enumerate(steps, start=1):
        if i < current:
            state = "done"
            icon = "✓"
        elif i == current:
            state = "active"
            icon = num
        else:
            state = "pending"
            icon = num
        html += f"""
        <div class="step step-{state}">
            <div class="step-circle">{icon}</div>
            <div class="step-label">{label}</div>
        </div>
        """
        if i < len(steps):
            html += '<div class="step-connector"></div>'
    html += "</div>"
    return html


# =====================================================================
# Step 4: training execution
# =====================================================================


def _run_training(
    arch: str,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_class_weighting: bool,
    use_swa: bool,
    use_mixup: bool,
    progress=gr.Progress(),
):
    """Executa o treinamento real e yielda updates de progresso.

    Reusa a mesma lógica de binarização do training.py (BUG.Training.4 fix).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    # Marcador de versão: se você NÃO vê "[treino v3: float32+clipvalue]" nos logs/UI,
    # o app está rodando código ANTIGO — reinicie (Ctrl+C + python main.py --gradio).
    logger.info(
        "[treino v3: float32+clipvalue] iniciando _run_training (anti-NaN robusto)"
    )
    progress(0.05, desc="Carregando dataset...")
    yield (
        "Carregando dataset...",
        f"[treino v3: float32+clipvalue] Iniciando treino: {arch}, "
        f"{epochs} épocas, batch={batch_size}",
        None,
    )

    # Guardará a política de precisão anterior para restaurar no finally.
    _saved_policy = None

    try:
        from app.domain.models.architectures.factory import (
            architecture_factory_registry,
            create_model_by_name,
        )

        # SVM e Random Forest são modelos clássicos (scikit-learn) que operam em
        # FEATURES TABULARES (espectral/cepstral/temporal agregadas), não em
        # espectrogramas/áudio bruto. Têm um pipeline de treino dedicado que
        # extrai as MESMAS features que a inferência usa (contrato garantido).
        _CLASSICAL_ML = {"svm", "random forest", "randomforest", "rf"}
        if arch.strip().lower() in _CLASSICAL_ML:
            yield from _run_classical_training(
                arch=arch,
                dataset_path=dataset_path,
                progress=progress,
            )
            return

        # ─────────────────────────────────────────────────────────────────
        # BUG FIX (NaN mid-época com train_acc alto): treinar em FLOAT32.
        # Sintoma: o modelo treina bem (ex. acc=0.78) e SÓ ENTÃO loss→nan.
        # Causa: política global mixed_float16 (auto em GPUs Tensor Core).
        # O LossScaleOptimizer protege contra UNDERFLOW de gradiente, mas NÃO
        # contra OVERFLOW de ativação no forward — float16 satura em ~65504.
        # Conforme o modelo aprende, ativações crescem e estouram → inf → nan.
        # Treino em float32 elimina esse overflow (a inferência pode continuar
        # em float16 para velocidade). Restaurado no finally.
        try:
            _saved_policy = tf.keras.mixed_precision.global_policy()
            if _saved_policy is not None and _saved_policy.name == "mixed_float16":
                tf.keras.mixed_precision.set_global_policy("float32")
                logger.info(
                    "Treino forçado para float32 (estabilidade numérica). "
                    "mixed_float16 será restaurado após o treino para inferência."
                )
            else:
                # Já é float32 (ou CPU) — nada a restaurar
                _saved_policy = None
        except Exception as e:
            logger.debug(f"Não foi possível ajustar a política de precisão: {e}")
            _saved_policy = None

        SAMPLE_RATE = 16000
        AUDIO_LEN = SAMPLE_RATE * 3  # 3s
        # Parâmetros do front-end espectral (alinhados com audio_preprocessing)
        N_FFT = 512
        HOP = 128  # 48000 / 128 ≈ 375 frames; coerente para Conv2D
        N_MELS = 80
        FMIN, FMAX = 0.0, SAMPLE_RATE / 2
        # Front-end padrão para modelos de espectrograma: LFCC (literatura
        # anti-spoofing — LFCC > mel). N_LFCC=80 preserva o shape (T, 80, 1),
        # então nenhuma arquitetura precisa mudar. Gravado no input_contract
        # para a inferência usar exatamente o mesmo front-end.
        FRONTEND = "lfcc"
        N_LFCC = 80

        # Descobre tipo de input esperado pelo modelo escolhido
        spec = architecture_factory_registry.get_architecture_info(arch)
        if spec is None:
            raise ValueError(f"Arquitetura '{arch}' não registrada no factory.")
        input_type = spec.input_requirements.get("input_type", "spectrogram")

        # BUG FIX: o usuário costuma apontar para `app/datasets/`, que contém
        # subpastas que NÃO são classes: `raw/` (caches de download),
        # `splits/` (duplica real+fake já divididos), `features/`, etc.
        # audio_dataset_from_directory trata CADA subpasta como uma classe →
        # "Found N files belonging to 4 classes" → labels poluídos + dados
        # duplicados (splits/ repete real/ e fake/). Filtramos para usar apenas
        # subpastas que parecem classes de áudio reais.
        from pathlib import Path as _Path

        _EXCLUDE_DIRS = {
            "raw",
            "splits",
            "features",
            "segmented",
            "processed",
            "cache",
            "__pycache__",
            ".git",
            ".ipynb_checkpoints",
            "metadata",
            "tmp",
            "checkpoints",
            "models",
            "logs",
        }
        _ds_root = _Path(dataset_path)
        _explicit_classes = None
        try:
            _subdirs = sorted(
                d.name
                for d in _ds_root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            _valid = [d for d in _subdirs if d.lower() not in _EXCLUDE_DIRS]
            _excluded = [d for d in _subdirs if d.lower() in _EXCLUDE_DIRS]
            if _excluded and _valid:
                _explicit_classes = _valid
                logger.warning(
                    f"Subpastas ignoradas (não são classes): {_excluded}. "
                    f"Usando apenas: {_valid}. "
                    "Para treinar com um conjunto limpo, aponte para uma pasta "
                    "que contenha SOMENTE subpastas de classe (ex.: real/ e fake/)."
                )
        except Exception as e:
            logger.debug(f"Filtragem de classes pulada: {e}")

        _ds_common = dict(
            directory=dataset_path,
            batch_size=batch_size,
            validation_split=0.2,
            seed=42,
            output_sequence_length=AUDIO_LEN,
            label_mode="int",
        )
        if _explicit_classes is not None:
            _ds_common["class_names"] = _explicit_classes

        train_ds = tf.keras.utils.audio_dataset_from_directory(
            subset="training",
            **_ds_common,
        )
        val_ds = tf.keras.utils.audio_dataset_from_directory(
            subset="validation",
            **_ds_common,
        )

        # Binarização de labels (BUG.Training.4)
        class_names = list(train_ds.class_names)
        binary_map = {
            i: 0 if _REAL_PATTERNS.match(n.strip()) else 1
            for i, n in enumerate(class_names)
        }
        _table = tf.constant(
            [binary_map[i] for i in range(len(class_names))],
            dtype=tf.int32,
        )

        # --------------------------------------------------------------
        # Pré-processamento on-the-fly: raw_audio vs spectrogram
        #   - raw_audio   → (T, 1)
        #   - spectrogram → log-mel (T_frames, n_mels, 1)
        # Reaproveita o tf.signal nativo para rodar 100% no graph.
        # --------------------------------------------------------------
        # Mel matrix (cacheada, mesma p/ todos os batches)
        n_freq = N_FFT // 2 + 1
        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS,
            num_spectrogram_bins=n_freq,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=FMIN,
            upper_edge_hertz=FMAX,
        )

        def _to_log_mel(audio):
            # audio: (B, T) ou (B, T, 1) → squeeze
            if audio.shape.rank == 3:
                audio = tf.squeeze(audio, axis=-1)
            # BUG FIX: substituir NaN/Inf em áudio ANTES do STFT.
            # Arquivos corrompidos (silêncio clipped, NaN em WAV/MP3 corrompido)
            # propagam NaN por todo o pipeline → loss: nan na 1ª época.
            audio = tf.where(tf.math.is_finite(audio), audio, tf.zeros_like(audio))
            # Clip extremos (alguns WAVs vêm com amostras fora de [-1, 1])
            audio = tf.clip_by_value(audio, -10.0, 10.0)
            stft = tf.signal.stft(
                audio,
                frame_length=N_FFT,
                frame_step=HOP,
                fft_length=N_FFT,
                window_fn=tf.signal.hann_window,
                pad_end=True,
            )
            mag = tf.abs(stft)  # (B, T_frames, n_freq)
            mel = tf.tensordot(mag, mel_w, axes=1)  # (B, T_frames, n_mels)
            # max() previne mel=0 → log(0)=-inf que TF não captura como NaN mas
            # propaga -inf nos gradientes (NaN ao multiplicar com 0 depois)
            mel = tf.maximum(mel, 1e-10)
            log_mel = tf.math.log(mel)
            # Sanitização final + clip da magnitude (log de valores muito pequenos
            # produz -23 ≈ log(1e-10); clipping em [-15, 15] é gentil e estável)
            log_mel = tf.where(
                tf.math.is_finite(log_mel), log_mel, tf.fill(tf.shape(log_mel), -15.0)
            )
            log_mel = tf.clip_by_value(log_mel, -15.0, 15.0)
            return tf.expand_dims(log_mel, axis=-1)  # (B, T_frames, n_mels, 1)

        # LFCC via núcleo COMPARTILHADO com a inferência (audio_preprocessing.
        # lfcc_from_waveform) — garante paridade train↔test bit-a-bit.
        from app.domain.services.detection.audio_preprocessing import (
            lfcc_from_waveform,
        )

        def _to_lfcc(audio):
            if audio.shape.rank == 3:
                audio = tf.squeeze(audio, axis=-1)
            audio = tf.where(tf.math.is_finite(audio), audio, tf.zeros_like(audio))
            audio = tf.clip_by_value(audio, -10.0, 10.0)
            lfcc = lfcc_from_waveform(
                audio,
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP,
                n_filters=N_LFCC,
                n_lfcc=N_LFCC,
            )  # (B, T_frames, n_lfcc)
            return tf.expand_dims(lfcc, axis=-1)  # (B, T_frames, n_lfcc, 1)

        def _to_spec(audio):
            # Despacha conforme o front-end escolhido (default LFCC).
            return _to_lfcc(audio) if FRONTEND == "lfcc" else _to_log_mel(audio)

        def _prep_raw(audio, label):
            # Garante shape (B, T, 1)
            if audio.shape.rank == 2:
                audio = tf.expand_dims(audio, axis=-1)
            # BUG FIX: sanitizar NaN/Inf em áudio bruto ANTES de chegar ao modelo.
            # Arquivos corrompidos (silêncio clipped, MP3/WAV com NaN) causam
            # NaN imediato nas ativações — o SincConv propaga zeros → AMSoftmax
            # recebe zeros → l2_normalize(0) = NaN → loss NaN na época 1.
            audio = tf.where(tf.math.is_finite(audio), audio, tf.zeros_like(audio))
            # Clip extremos (alguns WAVs vêm com amostras fora de [-1, 1] por escalas
            # erradas ou conversão sem normalização). Esses outliers explodem na
            # primeira camada Conv → propaga +inf → NaN nos gradientes.
            audio = tf.clip_by_value(audio, -10.0, 10.0)
            return audio, tf.gather(_table, label)

        def _prep_spec(audio, label):
            spec_x = _to_spec(audio)
            return spec_x, tf.gather(_table, label)

        # Augmentation (Tak et al. 2022 / Park et al. 2019) — aplicada SOMENTE
        # ao conjunto de treino (val/test ficam intactos):
        #   - raw_audio   → RawBoost (substitui o placeholder de ruído gaussiano)
        #   - spectrogram → SpecAugment (mascaramento tempo/frequência)
        from app.domain.models.training.rawboost import rawboost_tf
        from app.domain.models.training.spec_augment import spec_augment_tf

        def _prep_raw_train(audio, label):
            audio, lab = _prep_raw(audio, label)
            audio = rawboost_tf(audio, sr=SAMPLE_RATE, algo=4, p=0.7)
            return audio, lab

        def _prep_spec_train(audio, label):
            spec_x = _to_spec(audio)
            spec_x = spec_augment_tf(spec_x, p=0.7)
            return spec_x, tf.gather(_table, label)

        if input_type == "raw_audio":
            train_prep, val_prep = _prep_raw_train, _prep_raw
        else:
            train_prep, val_prep = _prep_spec_train, _prep_spec

        train_ds = train_ds.map(train_prep, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(val_prep, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = optimize_tf_dataset(train_ds, cache=False, prefetch=True)
        val_ds = optimize_tf_dataset(val_ds, cache=False, prefetch=True)

        # Computa shape final inspecionando 1 batch (sem consumir o ds em cache)
        for sample_x, _ in train_ds.take(1):
            input_shape = tuple(sample_x.shape[1:])  # ignora dim batch
            break

        progress(0.15, desc=f"Criando modelo {arch}...")
        yield (
            "Criando modelo...",
            f"Input type: {input_type} | shape: {input_shape}\n"
            f"Mapeamento detectado: {len(class_names)} subdirs -> binário\n"
            f"{', '.join(f'{n}->{binary_map[i]}' for i, n in enumerate(class_names))}",
            None,
        )

        model = create_model_by_name(
            arch,
            input_shape=input_shape,
            num_classes=2,
        )

        # Detecta dimensão da saída para escolher loss apropriada:
        #   - 1 unidade (sigmoid) → binary_crossentropy
        #   - 2+ unidades (softmax) → sparse_categorical_crossentropy
        out_units = (
            model.output_shape[-1] if isinstance(model.output_shape, tuple) else 2
        )
        if out_units == 1:
            chosen_loss = "binary_crossentropy"
            chosen_metric = "binary_accuracy"
            # BUG FIX: labels float32 para BCE são corretos, MAS Keras's
            # class_weight não funciona com labels float (exige int para indexar
            # o dict {class_id: weight}). Marcamos para desativar class_weight
            # quando BCE + float labels forem usados (ver abaixo).
            _bce_float_labels = True
            # BUG FIX (rank mismatch): a saída sigmoid de 1 unidade tem shape
            # (B, 1). O label vinha como (B,) → BCE falha com
            # "target and output must have the same rank". Reshape p/ (B, 1).
            # Afeta Sonic Sleuth e Hybrid CNN-Transformer (saída 1-unit).
            train_ds = train_ds.map(
                lambda x, y: (x, tf.reshape(tf.cast(y, tf.float32), (-1, 1))),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            val_ds = val_ds.map(
                lambda x, y: (x, tf.reshape(tf.cast(y, tf.float32), (-1, 1))),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            train_ds = optimize_tf_dataset(train_ds, cache=False, prefetch=True)
            val_ds = optimize_tf_dataset(val_ds, cache=False, prefetch=True)
        else:
            chosen_loss = "sparse_categorical_crossentropy"
            chosen_metric = "accuracy"
            _bce_float_labels = False

        # BUG FIX: Detecta dinamicamente se o modelo emite LOGITS (não probabilidades).
        # AMSoftmaxLayer / Dense linear / Activation('linear') → logits brutos.
        # sparse_categorical_crossentropy padrão (from_logits=False) faz
        # log(y_pred[classe_verdadeira]). Se y_pred < 0 → log(negativo) = NaN.
        # Walk-through das camadas: se a saída é linear/AMSoftmax → from_logits=True.
        def _detect_logit_output(m) -> bool:
            """Detecta se a saída do modelo são logits (não probabilidades)."""
            # 1. Procura AMSoftmaxLayer em qualquer profundidade
            for lyr in m.layers:
                cls_name = type(lyr).__name__
                if cls_name == "AMSoftmaxLayer":
                    return True
            # 2. Verifica a ÚLTIMA camada com peso na saída (ignora Activation
            #    'linear' que é só cast de dtype para float32).
            tail_layers = [
                lyr
                for lyr in m.layers
                if not (
                    type(lyr).__name__ == "Activation"
                    and getattr(lyr.activation, "__name__", "") == "linear"
                )
            ]
            if not tail_layers:
                return False
            last = tail_layers[-1]
            # Dense com activation=None → logits
            act = getattr(last, "activation", None)
            if act is not None and getattr(act, "__name__", "") in ("linear",):
                return True
            return False

        output_is_logits = False
        try:
            output_is_logits = _detect_logit_output(model)
        except Exception as e:
            logger.debug(f"Detecção de logit falhou: {e}")

        if output_is_logits and out_units > 1:
            chosen_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
            logger.info(
                f"{arch}: modelo emite LOGITS — usando "
                "SparseCategoricalCrossentropy(from_logits=True) "
                "(log-sum-exp numericamente estável)."
            )
        elif output_is_logits and out_units == 1:
            chosen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            logger.info(
                f"{arch}: modelo emite LOGITS (1 unit) — "
                "BinaryCrossentropy(from_logits=True)."
            )

        # BUG FIX CRÍTICO: o matching de arquitetura quebrava porque `arch` é o
        # NOME DE EXIBIÇÃO ("SpectrogramTransformer", "Hybrid CNN-Transformer",
        # "RawGAT-ST") mas o set usava snake_case. arch.lower() produzia
        # "spectrogramtransformer"/"hybrid cnn-transformer"/"rawgat-st" que NÃO
        # batiam → LR NÃO era limitado → treino a 8e-4 → gradientes explodem →
        # loss:nan. Só AASIST/Conformer batiam por coincidência (palavra única).
        #
        # Solução: normalizar AMBOS os lados (remove espaços/hífens/underscores).
        def _norm_arch(s: str) -> str:
            return s.lower().replace(" ", "").replace("-", "").replace("_", "")

        _TRANSFORMER_ARCHS = {
            _norm_arch(x)
            for x in (
                "conformer",
                "spectrogram_transformer",
                "hybrid_cnn_transformer",
                "aasist",
                "rawgat_st",
                # Nomes de exibição (defensivo, caso a normalização mude):
                "SpectrogramTransformer",
                "Hybrid CNN-Transformer",
                "RawGAT-ST",
            )
        }
        # LR padrão 8e-4 é alto demais para arquiteturas com atenção sem warmup.
        # Cap em 1e-4 — comprovado estável nos dados reais COM clipvalue
        # (ver bloco do otimizador abaixo). O fator decisivo de estabilidade é
        # clipvalue (não a LR): nos testes, clipnorm quebrava até a 1e-5, mas
        # clipvalue era estável a 1e-4 (loss 0.55) e a 1e-5.
        effective_lr = lr
        if _norm_arch(arch) in _TRANSFORMER_ARCHS and lr > 1e-4:
            effective_lr = min(lr, 1e-4)
            logger.warning(
                f"LR ajustado de {lr:.2e} → {effective_lr:.2e} para {arch} "
                "(arquiteturas com atenção/grafos precisam de LR ≤ 1e-4)"
            )

        # BUG FIX CRÍTICO (explosão de gradiente → 85 pesos NaN no step 1):
        #
        # 1) clipVALUE em vez de clipNORM. clipnorm calcula a norma global do
        #    gradiente e divide por ela; se UM elemento for inf, a norma é inf e
        #    g/inf = NaN em TODOS os elementos → corrompe o modelo inteiro.
        #    clipvalue limita CADA elemento a [-v, v] independentemente:
        #    clip(inf, -v, v) = v (finito). Quebra a cascata de NaN.
        #
        # 2) LossScaleOptimizer SOMENTE em mixed_float16. Em float32 ele é inútil
        #    (não há underflow a corrigir) e PERIGOSO: escala a loss por 32768×,
        #    e em grafos profundos (SincConv+ResBlocks+GAT+HS-GAL) isso empurra
        #    os gradientes para overflow (>3.4e38) → inf → NaN. Verificado: com
        #    LSO em float32 o AASIST quebra no step 1; com Adam puro + clipvalue
        #    é estável.
        base_optimizer = tf.keras.optimizers.Adam(
            learning_rate=effective_lr,
            clipvalue=1.0,  # limita por elemento; lida com inf (clipnorm não)
        )
        _policy_name = tf.keras.mixed_precision.global_policy().name
        if _policy_name == "mixed_float16":
            # Só aqui o loss scaling faz sentido (corrige underflow de gradiente
            # float16). Em float32 NÃO usamos — causaria overflow.
            try:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                logger.info("LossScaleOptimizer ativo (mixed_float16).")
            except Exception as e:
                logger.debug(f"LSO indisponível ({e}); usando Adam puro")
                optimizer = base_optimizer
        else:
            optimizer = base_optimizer
            logger.info(
                f"Otimizador: Adam(lr={effective_lr:.1e}, clipvalue=1.0) em "
                f"{_policy_name} (sem LSO — evita overflow 32768× em float32)."
            )

        # Re-compile se necessário (algumas archs pré-compilam)
        try:
            model.compile(
                optimizer=optimizer,
                loss=chosen_loss,
                metrics=[chosen_metric],
            )
        except Exception as e:
            logger.debug(f"Re-compile pulado: {e}")

        # ─────────────────────────────────────────────────────────────────
        # BUG FIX (Smoke test): Roda 1 batch (forward + loss) ANTES de fit()
        # para detectar NaN/Inf imediatamente, com diagnóstico preciso de ONDE.
        # Evita esperar 1ª época inteira para descobrir que loss=nan.
        # ─────────────────────────────────────────────────────────────────
        try:
            for smoke_x, smoke_y in train_ds.take(1):
                # Estatísticas do batch de input
                x_finite = bool(tf.reduce_all(tf.math.is_finite(smoke_x)).numpy())
                x_min = float(tf.reduce_min(smoke_x).numpy())
                x_max = float(tf.reduce_max(smoke_x).numpy())
                x_mean = float(tf.reduce_mean(smoke_x).numpy())
                x_std = float(tf.math.reduce_std(smoke_x).numpy())

                logger.info(
                    f"[smoke] input shape={smoke_x.shape} finite={x_finite} "
                    f"range=[{x_min:.4f},{x_max:.4f}] mean={x_mean:.4f} std={x_std:.4f}"
                )

                if not x_finite:
                    raise ValueError(
                        "[smoke] Input contém NaN/Inf APÓS sanitização — "
                        "verifique se o dataset tem arquivos válidos. "
                        "Caminho: " + str(dataset_path)
                    )

                # Forward pass
                preds = model(smoke_x, training=False)
                p_finite = bool(tf.reduce_all(tf.math.is_finite(preds)).numpy())
                p_min = float(tf.reduce_min(preds).numpy())
                p_max = float(tf.reduce_max(preds).numpy())
                logger.info(
                    f"[smoke] output shape={preds.shape} finite={p_finite} "
                    f"range=[{p_min:.4f},{p_max:.4f}] "
                    f"is_logits={output_is_logits}"
                )

                if not p_finite:
                    raise ValueError(
                        f"[smoke] Forward pass produziu NaN/Inf — modelo {arch} "
                        f"tem instabilidade numérica nas camadas. "
                        f"Range da saída: [{p_min},{p_max}]. "
                        "Isso normalmente significa: (a) camada custom com divisão "
                        "por zero, (b) BatchNorm em batch degenerado, ou "
                        "(c) input com escala extrema. "
                        "Tente reduzir o batch_size ou usar outra arquitetura."
                    )

                # Loss test
                try:
                    if callable(chosen_loss):
                        loss_val = chosen_loss(smoke_y, preds)
                    else:
                        loss_fn = tf.keras.losses.get(chosen_loss)
                        loss_val = loss_fn(smoke_y, preds)
                    loss_finite = bool(
                        tf.reduce_all(tf.math.is_finite(loss_val)).numpy()
                    )
                    loss_mean = float(tf.reduce_mean(loss_val).numpy())
                    logger.info(
                        f"[smoke] loss finite={loss_finite} mean={loss_mean:.4f} "
                        f"(loss_fn={getattr(chosen_loss, 'name', str(chosen_loss))})"
                    )
                    if not loss_finite:
                        raise ValueError(
                            f"[smoke] Loss produziu NaN/Inf com output finito. "
                            f"Isso indica que from_logits está MAL configurado. "
                            f"output_is_logits={output_is_logits}, "
                            f"chosen_loss={chosen_loss}. "
                            "Solução: verifique se a saída do modelo é "
                            "probabilidade [0,1] ou logit [-inf,+inf]."
                        )
                except ValueError:
                    raise
                except Exception as e:
                    logger.warning(f"[smoke] Loss test pulado: {e}")
                break
        except ValueError as ve:
            # Erro de smoke test → propaga com mensagem clara
            logger.error(str(ve))
            yield (
                "Smoke test falhou",
                f"❌ Treinamento abortado ANTES da época 1:\n\n{ve}",
                None,
            )
            return
        except Exception as e:
            # Erro inesperado no smoke test → log mas continua (não bloquear)
            logger.warning(f"[smoke] Pulado por erro inesperado: {e}")

        # Callbacks
        log_lines = []
        # Nome da métrica de acurácia depende da loss escolhida
        _acc_key = chosen_metric  # "accuracy" ou "binary_accuracy"
        _val_acc_key = f"val_{_acc_key}"

        class _ProgressCb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                pct = 0.2 + 0.75 * (epoch + 1) / epochs
                line = (
                    f"Epoch {epoch + 1}/{epochs} — "
                    f"loss: {logs.get('loss', 0):.4f}, "
                    f"acc: {logs.get(_acc_key, 0):.4f}, "
                    f"val_loss: {logs.get('val_loss', 0):.4f}, "
                    f"val_acc: {logs.get(_val_acc_key, 0):.4f}"
                )
                log_lines.append(line)
                progress(pct, desc=line[:60])

        class _NaNDiagnosticCb(tf.keras.callbacks.Callback):
            """Callback que dá diagnóstico detalhado quando TerminateOnNaN dispara.

            Monitora NaN em DOIS níveis:
              - on_train_batch_end: detecta o BATCH exato onde NaN apareceu
              - on_epoch_end: diagnóstico geral se loss=NaN no fim da época
            """

            def __init__(self):
                super().__init__()
                self._nan_batch_logged = False

            def on_train_batch_end(self, batch, logs=None):
                logs = logs or {}
                loss_val = logs.get("loss", 0.0)
                if loss_val != loss_val and not self._nan_batch_logged:
                    self._nan_batch_logged = True
                    logger.error(
                        f"[NaN] Primeira aparição de loss=NaN no batch #{batch} "
                        f"da época 1.\nIsso significa que o modelo produziu "
                        f"ativações NaN/Inf neste batch específico — provavelmente "
                        f"por (a) gradient explosion (LR muito alto), "
                        f"(b) BatchNorm em batch degenerado (todos samples iguais), "
                        f"ou (c) arquivo de áudio corrompido no batch."
                    )

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                loss_val = logs.get("loss", 0.0)
                if loss_val != loss_val:  # NaN check (NaN != NaN é sempre True)
                    logit_hint = (
                        "\n⚠️  Modelo emite LOGITS — confirmado uso de from_logits=True."
                        if output_is_logits
                        else ""
                    )
                    logger.error(
                        f"[NaN] loss=NaN detectado na época {epoch + 1}.{logit_hint}\n"
                        f"Configuração ativa:\n"
                        f"  • arch={arch}, out_units={out_units}, "
                        f"output_is_logits={output_is_logits}\n"
                        f"  • effective_lr={effective_lr:.2e}, clipvalue=1.0\n"
                        f"  • mixed_precision="
                        f"{tf.keras.mixed_precision.global_policy().name}\n"
                        "Causas prováveis (em ordem de frequência):\n"
                        "  1. Explosão de gradiente (LR alto p/ esta arquitetura)\n"
                        "  2. Dataset extremamente desbalanceado (1 classe << outra)\n"
                        "  3. Dataset poluído (subpastas raw/ splits/ tratadas "
                        "como classes — aponte para uma pasta só com real/ e fake/)\n"
                        "  4. Arquivos de áudio corrompidos/silentes\n"
                        "Diagnósticos sugeridos:\n"
                        "  • Reduza o LR para 1e-6\n"
                        "  • Reduza o batch_size pela metade\n"
                        "  • Tente uma arquitetura mais simples (Sonic Sleuth)\n"
                        "  • Reinicie o app para garantir o código mais recente"
                    )

        class _WeightNaNGuard(tf.keras.callbacks.Callback):
            """Para o treino só se os PESOS ficarem NaN (corrupção REAL).

            Diferente de TerminateOnNaN (que para na 1ª loss NaN), tolera uma
            loss NaN transitória de um batch venenoso — o LossScaleOptimizer já
            pulou esse passo e preservou os pesos. Só aborta se a corrupção for
            real (pesos NaN), o que com o LSO ativo praticamente não ocorre.
            """

            def on_epoch_end(self, epoch, logs=None):
                corrupted = False
                for w in self.model.trainable_variables:
                    if not tf.reduce_all(tf.math.is_finite(w)).numpy():
                        corrupted = True
                        break
                if corrupted:
                    logger.error(
                        f"[WeightGuard] Pesos do modelo ficaram NaN/Inf na época "
                        f"{epoch + 1} — corrupção real. Abortando treino."
                    )
                    self.model.stop_training = True

        log_cb = _ProgressCb()
        nan_diag_cb = _NaNDiagnosticCb()
        callbacks = [
            log_cb,
            nan_diag_cb,
            # Substituímos TerminateOnNaN pelo guard baseado em PESOS: com o
            # LossScaleOptimizer, uma loss NaN transitória não corrompe o modelo,
            # então não devemos abortar por ela. Abortamos só se os pesos forem NaN.
            _WeightNaNGuard(),
        ]

        # BUG FIX: class_weight com BCE + labels float32 é incompatível.
        # Keras aplica class_weight indexando o dict {int_class: weight} com
        # as labels. Quando labels são float32 (0.0/1.0 para BCE), o índice
        # float→int pode produzir comportamento incorreto em algumas versões
        # de TF, causando pesos errados ou NaN na loss ponderada.
        # Solução: desabilitar class_weight quando BCE + float labels são usados.
        # Class weighting (Sprint 1.3)
        class_weight = None
        if use_class_weighting and not _bce_float_labels:
            try:
                from sklearn.utils.class_weight import compute_class_weight

                # BUG FIX: unbatch() produz labels ESCALARES (0-d); np.concatenate
                # de arrays 0-d falha ("zero-dimensional arrays cannot be
                # concatenated"). Iteramos os batches (labels 1-d) e achatamos.
                ys = np.concatenate(
                    [np.asarray(y).reshape(-1) for _, y in train_ds]
                ).astype(int)
                classes = np.unique(ys)
                if len(classes) < 2:
                    logger.warning(
                        f"class_weighting pulado: dataset tem só 1 classe ({classes}). "
                        "Verifique se o caminho aponta para subpastas real/ e fake/."
                    )
                else:
                    cw = compute_class_weight("balanced", classes=classes, y=ys)
                    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
                    logger.info(f"class_weight calculado: {class_weight}")
            except Exception as e:
                logger.warning(f"class_weighting falhou: {e}")
        elif use_class_weighting and _bce_float_labels:
            logger.info(
                "class_weight desabilitado: BCE com labels float32 é incompatível "
                "com class_weight em Keras (use sparse_categorical_crossentropy "
                "para habilitar ponderação de classes)."
            )

        # GPU.10: Escolhe device explícito. Quando TF detectou GPU,
        # força fit() em /GPU:0 (evita silenciosamente cair em CPU se houver
        # ops não-GPU intercaladas — TF normalmente faz placement automático,
        # mas o context explícito serve como "load-bearing" para a UI mostrar
        # ao usuário onde está rodando).
        try:
            import tensorflow as _tf

            _gpus_present = bool(_tf.config.list_physical_devices("GPU"))
        except Exception:
            _gpus_present = False
        train_device = "/GPU:0" if _gpus_present else "/CPU:0"

        # ─────────────────────────────────────────────────────────────────
        # FEEDBACK AO VIVO: roda model.fit() numa THREAD e faz polling de uma
        # fila de épocas. Sem isso, fit() bloqueia e a UI fica "congelada" até
        # o fim — o usuário não vê loss/acc por época nem ETA. Com a thread, o
        # gerador yielda painel de métricas + plot atualizados a cada época.
        # ─────────────────────────────────────────────────────────────────
        import queue as _queue
        import threading as _threading

        _n_train = sum(1 for _ in train_ds)
        _n_val = sum(1 for _ in val_ds)
        header_lines = [
            f"Modelo: {arch} ({model.count_params():,} params)",
            f"Treino: {_n_train} batches · Validação: {_n_val} batches",
            f"Class weight: {class_weight or 'desabilitado'} · Device: {train_device}",
            "─" * 48,
        ]

        # Chaves de métrica reais (callback usa exatamente _acc_key/_val_acc_key)
        hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        epoch_q: "_queue.Queue" = _queue.Queue()

        class _LiveQueueCb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                epoch_q.put(
                    {
                        "epoch": epoch + 1,
                        "loss": logs.get("loss"),
                        "acc": logs.get(_acc_key),
                        "val_loss": logs.get("val_loss"),
                        "val_acc": logs.get(_val_acc_key),
                    }
                )

        fit_box = {"history": None, "error": None}

        def _fit_worker():
            try:
                with tf.device(train_device):
                    fit_box["history"] = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        callbacks=callbacks + [_LiveQueueCb()],
                        class_weight=class_weight,
                        verbose=0,
                    )
            except Exception as exc:  # propaga para o gerador
                fit_box["error"] = exc

        worker = _threading.Thread(target=_fit_worker, daemon=True)
        _t0 = time.time()
        worker.start()

        progress(0.2, desc=f"Treinando {arch} em {train_device}...")
        yield (
            _train_status_html(
                arch,
                train_device,
                0,
                epochs,
                hist,
                0.0,
                phase="preparing",
                note="Compilando o grafo e iniciando a 1ª época "
                "(pode levar alguns segundos)…",
            ),
            "\n".join(header_lines + ["Iniciando treinamento…"]),
            None,
        )

        _live_fig = None
        epoch_lines: List[str] = []
        while worker.is_alive() or not epoch_q.empty():
            try:
                ev = epoch_q.get(timeout=1.0)
            except _queue.Empty:
                # Heartbeat (a cada 1s): mantém a UI VIVA mesmo durante a 1ª época.
                # BUG FIX "trava nos 20%": antes só yieldava se done>0, então
                # durante a 1ª época (que na CPU pode levar minutos) a UI ficava
                # congelada no painel "preparando". Agora o relógio sempre avança.
                done = len(hist["loss"])
                elapsed = time.time() - _t0
                if done > 0:
                    yield (
                        _train_status_html(
                            arch,
                            train_device,
                            done,
                            epochs,
                            hist,
                            elapsed,
                            phase="running",
                        ),
                        "\n".join(header_lines + epoch_lines),
                        gr.update(),  # não recria o plot no heartbeat
                    )
                else:
                    # 1ª época ainda em andamento — relógio vivo + aviso
                    yield (
                        _train_status_html(
                            arch,
                            train_device,
                            0,
                            epochs,
                            hist,
                            elapsed,
                            phase="preparing",
                            note="Treinando a 1ª época… na CPU pode levar "
                            "vários minutos (áudio bruto). O relógio acima "
                            "confirma que está rodando.",
                        ),
                        "\n".join(
                            header_lines
                            + [f"⏱ {_fmt_secs(elapsed)} — 1ª época em andamento…"]
                        ),
                        gr.update(),
                    )
                continue

            # Nova época concluída
            for k in ("loss", "acc", "val_loss", "val_acc"):
                hist[k].append(ev.get(k))
            ep = ev["epoch"]

            def _fmt(v):
                return (
                    "nan"
                    if (v is None or (isinstance(v, float) and v != v))
                    else f"{v:.4f}"
                )

            epoch_lines.append(
                f"Época {ep:>3}/{epochs} — loss {_fmt(ev['loss'])} · "
                f"acc {_fmt(ev['acc'])} · val_loss {_fmt(ev['val_loss'])} · "
                f"val_acc {_fmt(ev['val_acc'])}"
            )
            progress(0.2 + 0.75 * ep / epochs, desc=f"Época {ep}/{epochs}")

            # Plot ao vivo (fecha o anterior p/ evitar leak de memória)
            close_fig(_live_fig)
            _live_fig = _history_figure(
                hist["loss"], hist["val_loss"], hist["acc"], hist["val_acc"]
            )

            yield (
                _train_status_html(
                    arch,
                    train_device,
                    ep,
                    epochs,
                    hist,
                    time.time() - _t0,
                    phase="running",
                ),
                "\n".join(header_lines + epoch_lines),
                _live_fig,
            )

        worker.join()
        if fit_box["error"] is not None:
            raise fit_box["error"]
        history = fit_box["history"]

        # Plot final (canônico, a partir do history completo)
        train_acc_key = _acc_key
        val_acc_key = _val_acc_key
        if history is not None and train_acc_key not in history.history:
            for cand in ("accuracy", "binary_accuracy", "categorical_accuracy"):
                if cand in history.history:
                    train_acc_key = cand
                    val_acc_key = f"val_{cand}"
                    break

        close_fig(_live_fig)
        fig = _history_figure(
            history.history.get("loss"),
            history.history.get("val_loss"),
            history.history.get(train_acc_key),
            history.history.get(val_acc_key),
        )

        progress(1.0, desc="Concluído!")

        # Última métrica VÁLIDA (ignora None). Cai para a métrica de treino se
        # não houver validação, e para NaN se nada existir — em vez de reportar
        # um '0.0' enganoso (o que acontecia com .get(key, [0])[-1]).
        def _last_metric(primary: str, fallback: str = None) -> float:
            for key in (primary, fallback):
                if not key:
                    continue
                seq = [v for v in (history.history.get(key) or []) if v is not None]
                if seq:
                    return float(seq[-1])
            return float("nan")

        final_acc = _last_metric(val_acc_key, train_acc_key)
        final_loss = _last_metric("val_loss", "loss")
        notify_success(
            f"Treino de {arch} concluído",
            message=f"val_acc={final_acc:.4f}, val_loss={final_loss:.4f}",
        )

        # Guarda o modelo treinado para o botão "Salvar" do Step 4.
        # input_contract registra o pré-processamento EXATO usado no treino,
        # garantindo que a inferência reproduza as mesmas features.
        global _LAST_TRAINED
        _LAST_TRAINED = {
            "model": model,
            "arch": arch,
            "input_shape": input_shape,
            "input_contract": {
                "input_type": input_type,
                "n_fft": N_FFT,
                "hop_length": HOP,
                "n_mels": N_MELS,
                # Front-end espectral usado no treino (default LFCC). A inferência
                # lê este campo para reproduzir EXATAMENTE o mesmo front-end.
                "feature_frontend": FRONTEND,
                "n_lfcc": N_LFCC,
                "sample_rate": SAMPLE_RATE,
                "audio_len": AUDIO_LEN,
            },
            "val_acc": float(final_acc) if final_acc == final_acc else None,
            "val_loss": float(final_loss) if final_loss == final_loss else None,
            "epochs": epochs,
        }

        yield (
            _train_status_html(
                arch,
                train_device,
                epochs,
                epochs,
                hist,
                time.time() - _t0,
                phase="done",
                note="✓ Treino concluído — dê um nome e clique em "
                "<b>💾 Salvar Modelo</b> abaixo para usá-lo na detecção.",
            ),
            "\n".join(
                header_lines
                + epoch_lines
                + [
                    "─" * 48,
                    f"✓ Treino concluído — val_accuracy={final_acc:.4f}, "
                    f"val_loss={final_loss:.4f}",
                ]
            ),
            fig,
        )

    except Exception as e:
        logger.error(f"Erro no treino: {e}", exc_info=True)
        # Notify.5: erro acionável com hint específico
        notify_from_actionable(CommonErrors.training_failed(str(e)))
        err_panel = (
            '<div class="train-live">'
            '<div class="tl-head"><span class="tl-head-title" style="color:#ef4444">'
            "✗ Treinamento falhou</span></div>"
            '<div class="tl-note" style="border-left-color:#ef4444;'
            'background:rgba(239,68,68,0.08)">'
            f"{str(e)[:300]}</div></div>"
        )
        yield (
            err_panel,
            f"❌ Erro durante o treino:\n\n{e}\n\n"
            "(Stack trace completo nos logs do servidor.)",
            None,
        )
    finally:
        # Restaura a política de precisão original (mixed_float16 p/ inferência).
        if _saved_policy is not None:
            try:
                tf.keras.mixed_precision.set_global_policy(_saved_policy)
                logger.info(
                    f"Política de precisão restaurada para {_saved_policy.name}."
                )
            except Exception as e:
                logger.debug(f"Falha ao restaurar política de precisão: {e}")


# =====================================================================
# Treinamento de modelos CLÁSSICOS (SVM / Random Forest)
# =====================================================================

# Conjunto de features tabulares — PINADO no input_contract para garantir
# que a inferência (FeaturePreparer) extraia exatamente as mesmas.
# 'spectral' + 'cepstral' (MFCC/LFCC) + 'temporal' são rápidas e robustas;
# 'prosodic' é evitado (pyin é lento).
_CLASSICAL_FEATURE_TYPES = ["spectral", "cepstral", "temporal"]
_CLASSICAL_AGG = "mean"


def _list_class_files(dataset_path: str):
    """Lista (filepath, label) das subpastas de classe, ignorando dirs espúrios.

    Mesma filtragem do caminho DL: ignora raw/, splits/, features/, etc.
    Label: 0 = real, 1 = fake (convenção do projeto).
    """
    _EXCLUDE = {
        "raw",
        "splits",
        "features",
        "segmented",
        "processed",
        "cache",
        "__pycache__",
        ".git",
        ".ipynb_checkpoints",
        "metadata",
        "tmp",
        "checkpoints",
        "models",
        "logs",
    }
    root = Path(dataset_path)
    items = []
    if not root.exists():
        return items
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith(".") or sub.name.lower() in _EXCLUDE:
            continue
        label = 0 if _REAL_PATTERNS.match(sub.name.strip()) else 1
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            for f in sub.glob(ext):
                items.append((f, label))
    return items


def _run_classical_training(arch: str, dataset_path: str, progress):
    """Treina SVM / Random Forest com features tabulares (gerador de feedback).

    Extrai EXATAMENTE as mesmas features que a inferência usará
    (extract_segmented_features + ExtractionConfig pinado no contrato),
    treina o estimador sklearn, calcula val accuracy e popula _LAST_TRAINED
    para o botão Salvar persistir (.pkl + scaler + config).
    """
    import time as _time

    import matplotlib.pyplot as plt
    import numpy as np

    global _LAST_TRAINED
    is_rf = arch.strip().lower() in ("random forest", "randomforest", "rf")
    nice = "Random Forest" if is_rf else "SVM"
    _t0 = _time.time()

    yield (
        _train_status_html(
            nice,
            "CPU",
            0,
            1,
            {},
            0.0,
            phase="preparing",
            note="Coletando arquivos do dataset…",
        ),
        f"[clássico] Iniciando treino {nice} em {dataset_path}",
        None,
    )

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        from app.core.interfaces.audio import AudioData, FeatureType
        from app.domain.services.feature_extraction_service import (
            AudioFeatureExtractionService,
            ExtractionConfig,
        )

        files = _list_class_files(dataset_path)
        n_real = sum(1 for _, y in files if y == 0)
        n_fake = sum(1 for _, y in files if y == 1)
        if len(files) < 10 or n_real == 0 or n_fake == 0:
            yield (
                _train_status_html(
                    nice,
                    "CPU",
                    0,
                    1,
                    {},
                    _time.time() - _t0,
                    phase="error",
                    note="Dataset insuficiente — precisa de áudios "
                    "em subpastas real/ e fake/.",
                ),
                f"❌ Encontrados {n_real} real + {n_fake} fake "
                f"({len(files)} total). Mínimo: 5 por classe.",
                None,
            )
            return

        # A extração tabular calcula ~13 categorias por segmento (inclui prosodic/
        # formant via pyin/LPC — caras: ~1-3s/arquivo na CPU). SVM/RF não precisam
        # de milhares de amostras, então fazemos subamostragem balanceada para o
        # treino concluir em tempo razoável. (Mesma extração da inferência →
        # paridade mantida; só reduzimos a QUANTIDADE de exemplos.)
        import random as _random

        _MAX_PER_CLASS = 500
        reals = [f for f in files if f[1] == 0]
        fakes = [f for f in files if f[1] == 1]
        _rng = _random.Random(42)
        if len(reals) > _MAX_PER_CLASS:
            reals = _rng.sample(reals, _MAX_PER_CLASS)
        if len(fakes) > _MAX_PER_CLASS:
            fakes = _rng.sample(fakes, _MAX_PER_CLASS)
        sampled = reals + fakes
        _rng.shuffle(sampled)
        if len(sampled) < len(files):
            logger.info(
                f"[clássico] Subamostragem: {len(sampled)}/{len(files)} arquivos "
                f"(cap {_MAX_PER_CLASS}/classe) para acelerar a extração."
            )
        files = sampled

        # Reutiliza o feature_service do DetectionService (mesma instância da
        # inferência → contrato idêntico).
        try:
            from app.dependencies import get_detection_service

            feat_svc = get_detection_service().feature_service
        except Exception:
            feat_svc = AudioFeatureExtractionService()

        cfg = ExtractionConfig(
            feature_types=[FeatureType(t) for t in _CLASSICAL_FEATURE_TYPES],
            normalize=True,
            aggregate_method=_CLASSICAL_AGG,
        )

        # ── Extração de features (parte lenta — feedback por arquivo) ──
        X, y, feat_names = [], [], None
        total = len(files)
        skipped = 0
        for i, (fpath, label) in enumerate(files):
            try:
                ad = AudioData.from_file(fpath, sr=16000, mono=True)
                res = feat_svc.extract_segmented_features(ad, cfg)
                vec = res.features.get("combined_features")
                if vec is None or np.asarray(vec).size == 0:
                    skipped += 1
                    continue
                vec = np.asarray(vec, dtype=np.float32).flatten()
                if not np.all(np.isfinite(vec)):
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                X.append(vec)
                y.append(label)
                if feat_names is None:
                    feat_names = res.features.get("feature_names")
            except Exception as e:
                skipped += 1
                logger.debug(f"[clássico] falha em {fpath.name}: {e}")

            if (i + 1) % 10 == 0 or i + 1 == total:
                pct = (i + 1) / total
                progress(0.05 + 0.70 * pct, desc=f"Features {i + 1}/{total}")
                yield (
                    _train_status_html(
                        nice,
                        "CPU",
                        0,
                        1,
                        {},
                        _time.time() - _t0,
                        phase="preparing",
                        note=f"Extraindo features tabulares: {i + 1}/{total} "
                        f"arquivos ({len(X)} OK, {skipped} pulados)…",
                    ),
                    f"[clássico] Extração: {i + 1}/{total} · válidos={len(X)} · "
                    f"pulados={skipped}",
                    None,
                )

        if len(X) < 10:
            yield (
                _train_status_html(
                    nice,
                    "CPU",
                    0,
                    1,
                    {},
                    _time.time() - _t0,
                    phase="error",
                    note="Poucas features válidas.",
                ),
                f"❌ Só {len(X)} amostras válidas após extração. "
                "Verifique se os áudios não estão corrompidos.",
                None,
            )
            return

        X = np.vstack(X)
        y = np.asarray(y, dtype=int)
        n_features = X.shape[1]

        # ── Split + scaler + fit ──
        progress(0.80, desc="Treinando estimador…")
        yield (
            _train_status_html(
                nice,
                "CPU",
                0,
                1,
                {},
                _time.time() - _t0,
                phase="running",
                note=f"Treinando {nice} em {X.shape[0]} amostras × "
                f"{n_features} features…",
            ),
            f"[clássico] Matriz: {X.shape} · treinando {nice}…",
            None,
        )

        strat = y if (np.bincount(y).min() >= 2) else None
        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=strat,
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        if is_rf:
            from sklearn.ensemble import RandomForestClassifier

            est = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            )
        else:
            from sklearn.svm import SVC

            est = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        est.fit(X_tr_s, y_tr)

        val_acc = float(est.score(X_va_s, y_va))
        # F1/confusão simples
        from sklearn.metrics import confusion_matrix

        y_pred = est.predict(X_va_s)
        cm = confusion_matrix(y_va, y_pred, labels=[0, 1])

        # ── Plot: matriz de confusão + (RF) importância das features ──
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        style_ax(ax[0], fig, "Matriz de Confusão (val)")
        ax[0].imshow(cm, cmap="Blues")
        ax[0].set_xticks([0, 1])
        ax[0].set_yticks([0, 1])
        ax[0].set_xticklabels(["real", "fake"])
        ax[0].set_yticklabels(["real", "fake"])
        ax[0].set_xlabel("Predito")
        ax[0].set_ylabel("Verdadeiro")
        cm_max = int(cm.max()) if cm.size and cm.max() > 0 else 1
        for r in range(2):
            for c in range(2):
                # Contraste por célula: texto CLARO nas células escuras (contagem
                # alta) e ESCURO nas claras (contagem baixa). Antes era sempre
                # claro → o número ficava invisível nas células de baixa contagem
                # do colormap Blues.
                txt_color = "#f1f5f9" if cm[r, c] > cm_max / 2 else "#0f172a"
                ax[0].text(
                    c,
                    r,
                    str(cm[r, c]),
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontweight="bold",
                )
        if is_rf and hasattr(est, "feature_importances_"):
            style_ax(ax[1], fig, "Top-15 features (importância)")
            imp = est.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            ax[1].barh(range(len(idx))[::-1], imp[idx], color=PLOT_ACCENT)
            labels = [
                (feat_names[j] if feat_names and j < len(feat_names) else f"f{j}")
                for j in idx
            ]
            ax[1].set_yticks(range(len(idx))[::-1])
            ax[1].set_yticklabels(labels, fontsize=7)
        else:
            style_ax(ax[1], fig, "Distribuição de classes (treino)")
            ax[1].bar(
                ["real", "fake"],
                np.bincount(y_tr, minlength=2),
                color=[PLOT_ACCENT, PLOT_DANGER],
            )
        safe_tight_layout(fig)

        # ── Persistência: popula _LAST_TRAINED (flavor sklearn) ──
        _LAST_TRAINED = {
            "kind": "sklearn",
            "model": est,
            "scaler": scaler,
            "arch": nice,
            "input_shape": (n_features,),
            "feature_names": feat_names,
            "input_contract": {
                "format": "tabular",
                "feature_types": _CLASSICAL_FEATURE_TYPES,
                "aggregate_method": _CLASSICAL_AGG,
                "sample_rate": 16000,
                "scaler_applied": True,
                "n_features": n_features,
            },
            "val_acc": val_acc,
            "val_loss": None,
            "epochs": None,
        }

        notify_success(
            f"Treino de {nice} concluído", message=f"val_accuracy={val_acc:.4f}"
        )
        progress(1.0, desc="Concluído!")
        yield (
            _train_status_html(
                nice,
                "CPU",
                1,
                1,
                {"acc": [val_acc], "val_acc": [val_acc]},
                _time.time() - _t0,
                phase="done",
                note="✓ Treino concluído — dê um nome e clique em "
                "<b>💾 Salvar Modelo</b> abaixo para usá-lo na detecção.",
            ),
            "\n".join(
                [
                    f"✓ {nice} treinado",
                    f"Amostras: {X.shape[0]} ({n_real} real + {n_fake} fake, "
                    f"{skipped} pulados)",
                    f"Features por amostra: {n_features} "
                    f"({'+'.join(_CLASSICAL_FEATURE_TYPES)})",
                    f"Val accuracy: {val_acc:.4f}",
                    f"Matriz de confusão [real,fake]:\n{cm}",
                ]
            ),
            fig,
        )

    except Exception as e:
        logger.error(f"Erro no treino clássico: {e}", exc_info=True)
        notify_from_actionable(CommonErrors.training_failed(str(e)))
        yield (
            (
                '<div class="train-live"><div class="tl-head">'
                '<span class="tl-head-title" style="color:#ef4444">'
                "✗ Treinamento falhou</span></div>"
                f'<div class="tl-note" style="border-left-color:#ef4444">{str(e)[:300]}'
                "</div></div>"
            ),
            f"❌ Erro no treino clássico:\n{e}",
            None,
        )


# =====================================================================
# Tab builder
# =====================================================================


def create_training_wizard_tab():
    """Constrói o wizard de treinamento (Tab principal)."""
    with gr.Tab("🪄 Assistente", id="tab_train_wizard"):
        from app.interfaces.gradio.utils.components import page_header

        page_header(
            "🪄",
            "Assistente de Treinamento",
            "Treine um modelo em 4 passos guiados: dataset, modelo, "
            "hiperparâmetros e execução.",
        )

        # State compartilhado entre steps
        current_step = gr.State(1)
        scan_result = gr.State({})  # dict do _scan_dataset

        # Stepper visual no topo
        stepper = gr.HTML(_stepper_html(1), elem_id="wizard_stepper")

        # ───────────── Step 1: Dataset ─────────────
        with gr.Group(visible=True) as group_s1:
            gr.Markdown("## Step 1 — Escolha do Dataset")
            gr.Markdown(
                "Selecione uma pasta com **subdiretórios para cada classe**. "
                "Subdiretórios começando com `real`, `bonafide`, `genuine`, "
                "`original`, `authentic` ou `natural` são tratados como REAIS. "
                "Os demais (`spoof`, `fake`, etc.) são tratados como FAKE."
            )

            with gr.Row():
                dataset_path_s1 = gr.Textbox(
                    label="Caminho do Dataset",
                    value="app/datasets",
                    placeholder="ex: /data/asvspoof2019",
                    scale=4,
                )
                scan_btn = gr.Button("🔍 Validar", variant="primary", scale=1)

            scan_output = gr.Markdown("*Aguardando validação...*")

            with gr.Row():
                gr.Button("← Voltar", interactive=False, scale=1)  # placeholder
                next_s1_btn = gr.Button(
                    "Próximo →",
                    variant="primary",
                    scale=1,
                    interactive=False,
                )

        # ───────────── Step 2: Modelo ─────────────
        with gr.Group(visible=False) as group_s2:
            gr.Markdown("## Step 2 — Escolha do Modelo")
            gr.Markdown(
                "Selecione a arquitetura. Modelos com **GAT** (AASIST) oferecem maior "
                "accuracy. **Sonic Sleuth** é o mais leve. **Ensemble** é o mais robusto."
            )

            cards_html = gr.HTML(_render_model_cards_html())

            arch_select = gr.Dropdown(
                choices=[m["name"] for m in _get_model_catalog()],
                label="Arquitetura escolhida",
                value="AASIST",
                interactive=True,
            )

            with gr.Row():
                back_s2_btn = gr.Button("← Voltar", scale=1)
                next_s2_btn = gr.Button("Próximo →", variant="primary", scale=1)

        # ───────────── Step 3: Hiperparâmetros ─────────────
        with gr.Group(visible=False) as group_s3:
            gr.Markdown("## Step 3 — Hiperparâmetros")
            gr.Markdown(
                "Os valores padrão funcionam bem na maioria dos casos. "
                "Expanda **Opções Avançadas** para flags dos Sprints 1-5."
            )

            with gr.Row():
                epochs_s3 = gr.Slider(
                    1,
                    200,
                    value=50,
                    step=1,
                    label="Épocas",
                    info="Mais épocas = melhor accuracy mas mais tempo",
                )
                batch_s3 = gr.Slider(
                    4,
                    128,
                    value=16,
                    step=4,
                    label="Batch Size",
                    info="Maior = mais rápido, mas precisa mais RAM/VRAM",
                )

            lr_s3 = gr.Number(
                value=0.0008,
                label="Learning Rate",
                precision=5,
                info="Padrão: 8e-4. Arquiteturas com atenção/grafos "
                "(AASIST, Conformer, Transformers) são auto-limitadas a "
                "1e-4 para estabilidade numérica.",
            )

            with gr.Accordion("Opções Avançadas (Sprints 1-5)", open=False):
                gr.Markdown("### Sprint 1 — Quick wins")
                use_cw = gr.Checkbox(
                    True,
                    label="Class weighting automático (Sprint 1.3)",
                    info="Compensa datasets desbalanceados (recomendado SEMPRE).",
                )
                gr.Checkbox(
                    True,
                    label="Calibração de temperatura (Sprint 1.4)",
                    info="Confidências mais confiáveis pós-treino.",
                )

                gr.Markdown("### Sprint 2 — Treino")
                use_swa = gr.Checkbox(
                    False,
                    label="SWA — Stochastic Weight Averaging (Sprint 2.3)",
                    info="Média móvel dos pesos nas últimas 20% épocas. +0.5–1.5% acc.",
                )
                use_mixup = gr.Checkbox(
                    False,
                    label="Mixup augmentation (Sprint 2.4)",
                    info="Interpola pares no batch. +0.5–1.5% acc. Desabilita class weighting.",
                )

                gr.Markdown("### Sprint 4 — Métricas")
                gr.Checkbox(
                    True,
                    label="EER threshold adaptativo (Sprint 4.5)",
                    info="Calibra threshold de classificação no val set.",
                )

            with gr.Row():
                back_s3_btn = gr.Button("← Voltar", scale=1)
                next_s3_btn = gr.Button(
                    "Iniciar Treinamento →",
                    variant="primary",
                    scale=1,
                )

        # ───────────── Step 4: Treinar ─────────────
        with gr.Group(visible=False) as group_s4:
            gr.Markdown("## Step 4 — Treinamento")

            # Painel de métricas ao vivo (progresso, loss/acc, ETA) — ocupa
            # toda a largura para máxima visibilidade durante o treino.
            status_box = gr.HTML(
                '<div class="train-live"><div class="tl-note">'
                "Aguardando início do treinamento…</div></div>"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    history_plot = gr.Plot(label="Loss & Accuracy (ao vivo)")
                with gr.Column(scale=1):
                    logs_box = gr.TextArea(
                        label="Log por época",
                        interactive=False,
                        lines=16,
                        max_lines=40,
                    )

            # ── Salvar modelo treinado ──────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "### 💾 Salvar modelo\n"
                    "Dê um nome e salve para usar na aba **🎯 Detectar**. "
                    "O pré-processamento do treino é gravado junto (garante que "
                    "a inferência use exatamente as mesmas features)."
                )
                with gr.Row():
                    save_name = gr.Textbox(
                        label="Nome do modelo",
                        placeholder="ex: aasist_ptbr_v1",
                        scale=3,
                    )
                    save_btn = gr.Button(
                        "💾 Salvar Modelo",
                        variant="primary",
                        scale=1,
                    )
                save_status = gr.Markdown("")

            with gr.Row():
                back_s4_btn = gr.Button("← Novo Treino", scale=1)
                gr.HTML('<div style="flex:1"></div>')

        # ────────────────────────── Event handlers ──────────────────────────

        def on_scan(path):
            # Feedback imediato: escanear pode varrer milhares de arquivos.
            # 1º yield mostra estado "validando" + desabilita o avançar;
            # 2º yield entrega o resultado.
            yield (
                gr.update(),
                "⏳ Validando dataset… (varrendo arquivos)",
                gr.update(interactive=False),
            )
            res = _scan_dataset(path)
            yield (
                res,
                res["message"],
                gr.update(interactive=res["ok"]),
            )

        scan_btn.click(
            fn=on_scan,
            inputs=[dataset_path_s1],
            outputs=[scan_result, scan_output, next_s1_btn],
        )

        def goto_step(n):
            return (n, _stepper_html(n), *_step_visibility(n))

        next_s1_btn.click(
            fn=lambda: goto_step(2),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        back_s2_btn.click(
            fn=lambda: goto_step(1),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        next_s2_btn.click(
            fn=lambda: goto_step(3),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        back_s3_btn.click(
            fn=lambda: goto_step(2),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )
        back_s4_btn.click(
            fn=lambda: goto_step(1),
            inputs=[],
            outputs=[current_step, stepper, group_s1, group_s2, group_s3, group_s4],
        )

        # Salvar modelo treinado (Step 4)
        def on_save_model(name):
            # Feedback imediato + desabilita o botão durante a escrita
            yield "⏳ Salvando modelo…", gr.update(interactive=False)
            ok, msg = _save_trained_model(name)
            if ok:
                notify_success("Modelo salvo", message=msg)
            # reabilita o botão (permite renomear/tentar de novo)
            yield msg, gr.update(interactive=True)

        save_btn.click(
            fn=on_save_model,
            inputs=[save_name],
            outputs=[save_status, save_btn],
        )

        # Atualiza preview de cards quando seleciona arquitetura
        arch_select.change(
            fn=_render_model_cards_html,
            inputs=[arch_select],
            outputs=[cards_html],
        )

        # Step 3 → Step 4 + dispara treinamento
        def start_training(
            scan_state,
            arch,
            epochs,
            batch,
            lr,
            use_cw_val,
            use_swa_val,
            use_mixup_val,
        ):
            # Vai para Step 4
            updates_step = [4, _stepper_html(4), *_step_visibility(4)]

            # Streamed do _run_training
            for status, logs, plot in _run_training(
                arch=arch,
                dataset_path=(scan_state or {}).get("class_names")
                and (scan_state.get("path") or "app/datasets")
                or "app/datasets",
                epochs=int(epochs),
                batch_size=int(batch),
                lr=float(lr),
                use_class_weighting=bool(use_cw_val),
                use_swa=bool(use_swa_val),
                use_mixup=bool(use_mixup_val),
            ):
                yield (*updates_step, status, logs, plot)

        # Captura o path do dataset junto com o resultado do scan
        def attach_path_to_scan(scan_state, path):
            if isinstance(scan_state, dict) and scan_state.get("ok"):
                scan_state = {**scan_state, "path": path}
            return scan_state

        # Pequeno hack: quando entra no Step 4, atualizamos o scan_state com o path
        next_s3_btn.click(
            fn=attach_path_to_scan,
            inputs=[scan_result, dataset_path_s1],
            outputs=[scan_result],
        ).then(
            fn=start_training,
            inputs=[
                scan_result,
                arch_select,
                epochs_s3,
                batch_s3,
                lr_s3,
                use_cw,
                use_swa,
                use_mixup,
            ],
            outputs=[
                current_step,
                stepper,
                group_s1,
                group_s2,
                group_s3,
                group_s4,
                status_box,
                logs_box,
                history_plot,
            ],
        )
