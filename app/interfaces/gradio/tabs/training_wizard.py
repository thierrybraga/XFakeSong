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

from app.interfaces.gradio.utils.notifications import (
    CommonErrors,
    notify_error,
    notify_from_actionable,
    notify_info,
    notify_success,
    notify_warning,
)
from app.interfaces.gradio.utils.plotting import (
    PLOT_ACCENT,
    PLOT_DANGER,
    safe_tight_layout,
    style_ax,
)

logger = logging.getLogger("gradio_training_wizard")

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
        "ok": False, "message": "", "class_names": [],
        "binary_map": {}, "file_count": 0, "files_per_class": {},
        "n_real": 0, "n_fake": 0,
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

    out.update({
        "ok": True,
        "class_names": subdirs,
        "binary_map": binary_map,
        "file_count": total,
        "files_per_class": files_per_class,
        "n_real": n_real,
        "n_fake": n_fake,
    })

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
            "AASIST", "RawGAT-ST", "RawNet2", "Sonic Sleuth", "WavLM",
            "HuBERT", "Conformer", "Hybrid CNN-Transformer",
            "SpectrogramTransformer", "EfficientNet-LSTM",
            "MultiscaleCNN", "Ensemble",
        ]

    # Categoria + descrição amigável
    catalog = {
        "AASIST": ("🕸️", "Graph Attention", "Spectro-temporal GAT + HS-GAL. EER 0.83% em ASVspoof. Recomendado para máxima accuracy."),
        "RawGAT-ST": ("🕸️", "Graph Attention", "Variante do AASIST com foco temporal. Bom para áudios curtos."),
        "RawNet2": ("🌊", "Raw audio", "SincNet + ResBlocks + GRU. Trabalha direto na waveform."),
        "Sonic Sleuth": ("🎯", "Lightweight", "Modelo leve (~3M params). 98.27% accuracy. Ideal para edge."),
        "WavLM": ("🤖", "SSL Backbone", "Self-supervised. Robusto a ruído e canal. Requer mais GPU."),
        "HuBERT": ("🤖", "SSL Backbone", "Hidden-Unit BERT. Aprende fonemas auto-supervisionado."),
        "Conformer": ("⚡", "Transformer + Conv", "Conv local + Self-Attention global. Estado-da-arte em speech."),
        "Hybrid CNN-Transformer": ("⚡", "Transformer + Conv", "CCT. CNN tokenizer + Transformer. 91.47% accuracy."),
        "SpectrogramTransformer": ("🔭", "Vision Transformer", "ViT adaptado para espectrogramas (AST)."),
        "EfficientNet-LSTM": ("📊", "Transfer Learning", "EfficientNet + Bi-LSTM. Bom baseline com transfer learning."),
        "MultiscaleCNN": ("🔍", "CNN multi-escala", "Res2Net-50. Multi-scale hierárquico dentro do bloco residual."),
        "Ensemble": ("🎼", "Fusão multi-feature", "4 branches (Mel+LFCC+CQT+MFCC) + fusão. EER 3%."),
    }

    out = []
    for arch in archs:
        icon, category, desc = catalog.get(
            arch, ("🔧", "Outro", f"Arquitetura {arch}")
        )
        out.append({
            "name": arch, "icon": icon,
            "category": category, "description": desc,
        })

    # ML clássico (sklearn)
    out.append({
        "name": "SVM", "icon": "📐", "category": "Classical ML",
        "description": "Support Vector Machine. Baseline rápido com features tabulares.",
    })
    out.append({
        "name": "Random Forest", "icon": "🌳", "category": "Classical ML",
        "description": "Ensemble de árvores. Robusto, paraleliza em CPU multi-core.",
    })
    return out


def _render_model_cards_html(selected: str = "") -> str:
    """HTML grid de cards de modelos, com card selecionado destacado."""
    cards = _get_model_catalog()
    html = '<div class="model-grid">'
    for m in cards:
        is_sel = "model-card-selected" if m["name"] == selected else ""
        html += f"""
        <div class="model-card {is_sel}" data-arch="{m['name']}">
            <div class="model-icon">{m['icon']}</div>
            <div class="model-name">{m['name']}</div>
            <div class="model-category">{m['category']}</div>
            <div class="model-desc">{m['description']}</div>
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

    progress(0.05, desc="Carregando dataset...")
    yield (
        "Carregando dataset...",
        f"Iniciando treino: {arch}, {epochs} épocas, batch={batch_size}",
        None,
    )

    try:
        from app.domain.models.architectures.factory import (
            architecture_factory_registry,
            create_model_by_name,
        )

        SAMPLE_RATE = 16000
        AUDIO_LEN = SAMPLE_RATE * 3  # 3s
        # Parâmetros mel-spectrogram (alinhados com smoke test)
        N_FFT = 512
        HOP = 128  # 48000 / 128 ≈ 375 frames; coerente para Conv2D
        N_MELS = 80
        FMIN, FMAX = 0.0, SAMPLE_RATE / 2

        # Descobre tipo de input esperado pelo modelo escolhido
        spec = architecture_factory_registry.get_architecture_info(arch)
        if spec is None:
            raise ValueError(
                f"Arquitetura '{arch}' não registrada no factory."
            )
        input_type = spec.input_requirements.get("input_type", "spectrogram")

        train_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=dataset_path, batch_size=batch_size,
            validation_split=0.2, subset="training",
            seed=42, output_sequence_length=AUDIO_LEN,
            label_mode="int",
        )
        val_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=dataset_path, batch_size=batch_size,
            validation_split=0.2, subset="validation",
            seed=42, output_sequence_length=AUDIO_LEN,
            label_mode="int",
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
            stft = tf.signal.stft(
                audio, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT,
                window_fn=tf.signal.hann_window, pad_end=True,
            )
            mag = tf.abs(stft)  # (B, T_frames, n_freq)
            mel = tf.tensordot(mag, mel_w, axes=1)  # (B, T_frames, n_mels)
            log_mel = tf.math.log(mel + 1e-6)
            return tf.expand_dims(log_mel, axis=-1)  # (B, T_frames, n_mels, 1)

        def _prep_raw(audio, label):
            # Garante shape (B, T, 1)
            if audio.shape.rank == 2:
                audio = tf.expand_dims(audio, axis=-1)
            return audio, tf.gather(_table, label)

        def _prep_spec(audio, label):
            spec_x = _to_log_mel(audio)
            return spec_x, tf.gather(_table, label)

        prep_fn = _prep_raw if input_type == "raw_audio" else _prep_spec

        train_ds = train_ds.map(prep_fn, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(prep_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

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
            arch, input_shape=input_shape, num_classes=2,
        )

        # Detecta dimensão da saída para escolher loss apropriada:
        #   - 1 unidade (sigmoid) → binary_crossentropy
        #   - 2+ unidades (softmax) → sparse_categorical_crossentropy
        out_units = model.output_shape[-1] if isinstance(
            model.output_shape, tuple
        ) else 2
        if out_units == 1:
            chosen_loss = "binary_crossentropy"
            chosen_metric = "binary_accuracy"
            # Labels precisam ser float para BCE
            train_ds = train_ds.map(
                lambda x, y: (x, tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            val_ds = val_ds.map(
                lambda x, y: (x, tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            chosen_loss = "sparse_categorical_crossentropy"
            chosen_metric = "accuracy"

        # Re-compile se necessário (algumas archs pré-compilam)
        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=chosen_loss,
                metrics=[chosen_metric],
            )
        except Exception as e:
            logger.debug(f"Re-compile pulado: {e}")

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

        log_cb = _ProgressCb()
        callbacks = [
            log_cb,
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        # Class weighting (Sprint 1.3)
        class_weight = None
        if use_class_weighting:
            try:
                from sklearn.utils.class_weight import compute_class_weight
                ys = np.concatenate([y.numpy() for _, y in train_ds.unbatch()])
                cw = compute_class_weight(
                    "balanced", classes=np.unique(ys), y=ys,
                )
                class_weight = {int(c): float(w) for c, w in enumerate(cw)}
            except Exception as e:
                logger.warning(f"class_weighting falhou: {e}")

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

        progress(0.2, desc=f"Treinando {arch} em {train_device}...")
        yield (
            f"Treinando {arch}...",
            "\n".join([
                f"Modelo: {arch} ({model.count_params():,} params)",
                f"Train: {sum(1 for _ in train_ds)} batches",
                f"Val:   {sum(1 for _ in val_ds)} batches",
                f"Class weight: {class_weight or 'desabilitado'}",
                f"Device: {train_device}",
                "",
                "Iniciando model.fit()...",
            ]),
            None,
        )

        with tf.device(train_device):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=0,
            )

        # Plot final
        # Resolve a chave de acurácia conforme a métrica escolhida
        train_acc_key = _acc_key
        val_acc_key = _val_acc_key
        # Fallback caso o histórico use outra chave (algumas archs pré-compilam)
        if train_acc_key not in history.history:
            for cand in ("accuracy", "binary_accuracy", "categorical_accuracy"):
                if cand in history.history:
                    train_acc_key = cand
                    val_acc_key = f"val_{cand}"
                    break

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        style_ax(ax[0], fig, "Loss")
        ax[0].plot(history.history["loss"], label="train", color=PLOT_ACCENT)
        ax[0].plot(history.history.get("val_loss", []), label="val", color=PLOT_DANGER)
        ax[0].legend()
        style_ax(ax[1], fig, "Accuracy")
        ax[1].plot(history.history.get(train_acc_key, []), label="train", color=PLOT_ACCENT)
        ax[1].plot(history.history.get(val_acc_key, []), label="val", color=PLOT_DANGER)
        ax[1].legend()
        safe_tight_layout(fig)

        progress(1.0, desc="Concluído!")

        final_acc = history.history.get(val_acc_key, [0])[-1]
        final_loss = history.history.get("val_loss", [0])[-1]
        notify_success(
            f"Treino de {arch} concluído",
            message=f"val_acc={final_acc:.4f}, val_loss={final_loss:.4f}",
        )

        yield (
            f"✓ Concluído — val_accuracy={final_acc:.4f}, val_loss={final_loss:.4f}",
            "\n".join(log_lines + [
                "",
                f"✓ Treino concluído",
                f"✓ Val accuracy final: {final_acc:.4f}",
                f"✓ Val loss final: {final_loss:.4f}",
                "",
                "Modelo NÃO salvo automaticamente. Use a aba ⚙ Admin > Modelos para gerenciar.",
            ]),
            fig,
        )

    except Exception as e:
        logger.error(f"Erro no treino: {e}", exc_info=True)
        # Notify.5: erro acionável com hint específico
        notify_from_actionable(CommonErrors.training_failed(str(e)))
        yield (
            f"❌ Erro: {str(e)[:100]}",
            f"Stack trace nos logs do servidor.\n\n{e}",
            None,
        )


# =====================================================================
# Tab builder
# =====================================================================

def create_training_wizard_tab():
    """Constrói o wizard de treinamento (Tab principal)."""
    with gr.Tab("🎓 Treinar (Wizard)", id="tab_train_wizard"):
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
                    "Próximo →", variant="primary", scale=1, interactive=False,
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
                    1, 200, value=50, step=1, label="Épocas",
                    info="Mais épocas = melhor accuracy mas mais tempo",
                )
                batch_s3 = gr.Slider(
                    4, 128, value=16, step=4, label="Batch Size",
                    info="Maior = mais rápido, mas precisa mais RAM/VRAM",
                )

            lr_s3 = gr.Number(
                value=0.0008, label="Learning Rate", precision=5,
                info="Padrão: 0.0008 (AdamW recomendado por TCC)",
            )

            with gr.Accordion("Opções Avançadas (Sprints 1-5)", open=False):
                gr.Markdown("### Sprint 1 — Quick wins")
                use_cw = gr.Checkbox(
                    True, label="Class weighting automático (Sprint 1.3)",
                    info="Compensa datasets desbalanceados (recomendado SEMPRE).",
                )
                use_calib = gr.Checkbox(
                    True, label="Calibração de temperatura (Sprint 1.4)",
                    info="Confidências mais confiáveis pós-treino.",
                )

                gr.Markdown("### Sprint 2 — Treino")
                use_swa = gr.Checkbox(
                    False, label="SWA — Stochastic Weight Averaging (Sprint 2.3)",
                    info="Média móvel dos pesos nas últimas 20% épocas. +0.5–1.5% acc.",
                )
                use_mixup = gr.Checkbox(
                    False, label="Mixup augmentation (Sprint 2.4)",
                    info="Interpola pares no batch. +0.5–1.5% acc. Desabilita class weighting.",
                )

                gr.Markdown("### Sprint 4 — Métricas")
                compute_eer = gr.Checkbox(
                    True, label="EER threshold adaptativo (Sprint 4.5)",
                    info="Calibra threshold de classificação no val set.",
                )

            with gr.Row():
                back_s3_btn = gr.Button("← Voltar", scale=1)
                next_s3_btn = gr.Button(
                    "Iniciar Treinamento →", variant="primary", scale=1,
                )

        # ───────────── Step 4: Treinar ─────────────
        with gr.Group(visible=False) as group_s4:
            gr.Markdown("## Step 4 — Treinamento")

            with gr.Row():
                with gr.Column(scale=1):
                    status_box = gr.Textbox(
                        label="Status", interactive=False, lines=1,
                    )
                    logs_box = gr.TextArea(
                        label="Logs", interactive=False,
                        lines=15, max_lines=30,
                    )
                with gr.Column(scale=1):
                    history_plot = gr.Plot(label="Loss & Accuracy")

            with gr.Row():
                back_s4_btn = gr.Button("← Novo Treino", scale=1)
                gr.HTML('<div style="flex:1"></div>')

        # ────────────────────────── Event handlers ──────────────────────────

        def on_scan(path):
            res = _scan_dataset(path)
            return (
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

        # Atualiza preview de cards quando seleciona arquitetura
        arch_select.change(
            fn=_render_model_cards_html,
            inputs=[arch_select],
            outputs=[cards_html],
        )

        # Step 3 → Step 4 + dispara treinamento
        def start_training(
            scan_state, arch, epochs, batch, lr,
            use_cw_val, use_swa_val, use_mixup_val,
        ):
            # Vai para Step 4
            updates_step = [4, _stepper_html(4), *_step_visibility(4)]

            # Streamed do _run_training
            for status, logs, plot in _run_training(
                arch=arch,
                dataset_path=(scan_state or {}).get("class_names") and
                             (scan_state.get("path") or "app/datasets") or "app/datasets",
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
            inputs=[scan_result, arch_select, epochs_s3, batch_s3, lr_s3,
                    use_cw, use_swa, use_mixup],
            outputs=[
                current_step, stepper,
                group_s1, group_s2, group_s3, group_s4,
                status_box, logs_box, history_plot,
            ],
        )
