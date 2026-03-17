"""Aba Gradio para gerenciamento de Perfis de Voz personalizados."""

import json
import logging
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger("gradio_voice_profiles_tab")

# Singleton do serviço
_service_instance = None


def _get_service():
    global _service_instance
    if _service_instance is None:
        from app.domain.services.voice_profile_service import VoiceProfileService
        _service_instance = VoiceProfileService()
    return _service_instance


def _format_duration(seconds: float) -> str:
    """Formata duração em mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _profiles_to_table(profiles) -> list:
    """Converte lista de perfis para dados de tabela."""
    rows = []
    for p in profiles:
        rows.append([
            p.id,
            p.name,
            p.status,
            p.num_samples,
            _format_duration(p.total_duration_seconds),
            p.architecture or "—",
            p.telegram_id or "—",
            p.email or "—",
        ])
    return rows


def _get_profile_choices():
    """Retorna choices para dropdown de perfis."""
    svc = _get_service()
    profiles = svc.list_profiles()
    choices = [(f"{p.name} (ID: {p.id}) [{p.status}]", p.id) for p in profiles]
    return choices


def _refresh_profiles_table():
    """Atualiza tabela e dropdowns."""
    svc = _get_service()
    profiles = svc.list_profiles()
    table_data = _profiles_to_table(profiles)
    choices = [(f"{p.name} (ID: {p.id}) [{p.status}]", p.id) for p in profiles]
    return table_data, gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)


def create_voice_profiles_tab():
    """Cria a aba 'Perfis de Voz' no Gradio."""
    with gr.Tab("🎤 Perfis de Voz", id="tab_voice_profiles"):
        gr.Markdown(
            "### Perfis de Voz Personalizados\n"
            "Crie perfis biométricos vocais com dados pessoais, faça upload de "
            "amostras, treine modelos de detecção específicos e verifique "
            "autenticidade com precisão individualizada."
        )

        with gr.Tabs():
            # ============================================================ #
            #  Sub-tab 1: Gerenciar Perfis                                  #
            # ============================================================ #
            with gr.Tab("📋 Gerenciar Perfis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Criar Novo Perfil")
                        profile_name = gr.Textbox(
                            label="Nome", placeholder="Ex: João Silva"
                        )
                        profile_telegram = gr.Textbox(
                            label="Telegram ID", placeholder="@usuario"
                        )
                        profile_phone = gr.Textbox(
                            label="Telefone", placeholder="+55 11 99999-9999"
                        )
                        profile_email = gr.Textbox(
                            label="Email", placeholder="joao@email.com"
                        )
                        profile_desc = gr.Textbox(
                            label="Descrição",
                            placeholder="Notas sobre o perfil...",
                            lines=2,
                        )
                        profile_arch = gr.Dropdown(
                            label="Arquitetura Padrão",
                            choices=[
                                "sonic_sleuth", "aasist", "rawnet2",
                                "conformer", "efficientnet_lstm",
                                "multiscale_cnn", "ensemble",
                            ],
                            value="sonic_sleuth",
                        )
                        create_btn = gr.Button(
                            "➕ Criar Perfil", variant="primary"
                        )
                        create_status = gr.Textbox(
                            label="Status", interactive=False
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### Perfis Existentes")
                        profiles_table = gr.Dataframe(
                            headers=[
                                "ID", "Nome", "Status", "Amostras",
                                "Duração", "Arquitetura", "Telegram", "Email",
                            ],
                            datatype=[
                                "number", "str", "str", "number",
                                "str", "str", "str", "str",
                            ],
                            interactive=False,
                            label="Perfis",
                        )
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Atualizar Lista")
                            delete_id = gr.Number(
                                label="ID para excluir", precision=0
                            )
                            delete_btn = gr.Button(
                                "🗑️ Excluir Perfil", variant="stop"
                            )
                        delete_status = gr.Textbox(
                            label="Status", interactive=False
                        )

            # ============================================================ #
            #  Sub-tab 2: Dataset de Voz                                    #
            # ============================================================ #
            with gr.Tab("🎵 Dataset de Voz"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_profile_dd = gr.Dropdown(
                            label="Selecionar Perfil",
                            choices=[],
                            interactive=True,
                        )
                        audio_upload = gr.File(
                            label="Upload de Amostras de Áudio",
                            file_count="multiple",
                            file_types=["audio"],
                            type="filepath",
                        )
                        upload_btn = gr.Button(
                            "📤 Adicionar Amostras", variant="primary"
                        )
                        upload_status = gr.Textbox(
                            label="Status do Upload", interactive=False
                        )

                    with gr.Column(scale=2):
                        dataset_info = gr.Markdown("Selecione um perfil para ver o dataset.")
                        samples_table = gr.Dataframe(
                            headers=["Arquivo", "Duração (s)", "Taxa (Hz)", "Tamanho"],
                            datatype=["str", "number", "number", "str"],
                            interactive=False,
                            label="Amostras no Dataset",
                        )
                        with gr.Row():
                            sample_to_remove = gr.Textbox(
                                label="Nome do arquivo para remover"
                            )
                            remove_sample_btn = gr.Button("❌ Remover Amostra")
                        remove_status = gr.Textbox(
                            label="Status", interactive=False
                        )

            # ============================================================ #
            #  Sub-tab 3: Treinar Modelo                                    #
            # ============================================================ #
            with gr.Tab("🧠 Treinar Modelo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        train_profile_dd = gr.Dropdown(
                            label="Selecionar Perfil",
                            choices=[],
                            interactive=True,
                        )
                        train_arch = gr.Dropdown(
                            label="Arquitetura",
                            choices=[
                                "sonic_sleuth", "aasist", "rawnet2",
                                "conformer", "efficientnet_lstm",
                                "multiscale_cnn", "ensemble",
                            ],
                            value="sonic_sleuth",
                        )
                        train_epochs = gr.Slider(
                            label="Épocas", minimum=5, maximum=200,
                            value=30, step=5,
                        )
                        train_batch = gr.Slider(
                            label="Batch Size", minimum=4, maximum=64,
                            value=16, step=4,
                        )
                        train_lr = gr.Slider(
                            label="Learning Rate", minimum=0.0001, maximum=0.01,
                            value=0.001, step=0.0001,
                        )
                        train_btn = gr.Button(
                            "🚀 Iniciar Treinamento", variant="primary"
                        )

                    with gr.Column(scale=2):
                        train_log = gr.Textbox(
                            label="Log de Treinamento",
                            lines=15,
                            interactive=False,
                            autoscroll=True,
                        )
                        with gr.Row():
                            train_loss_plot = gr.Plot(label="Loss")
                            train_acc_plot = gr.Plot(label="Accuracy")
                        train_metrics_json = gr.JSON(
                            label="Métricas Finais"
                        )

            # ============================================================ #
            #  Sub-tab 4: Verificar Voz                                     #
            # ============================================================ #
            with gr.Tab("🔍 Verificar Voz"):
                with gr.Row():
                    with gr.Column(scale=1):
                        verify_profile_dd = gr.Dropdown(
                            label="Selecionar Perfil (com modelo treinado)",
                            choices=[],
                            interactive=True,
                        )
                        verify_audio = gr.Audio(
                            label="Áudio para Verificação",
                            type="filepath",
                        )
                        verify_btn = gr.Button(
                            "🔎 Verificar", variant="primary"
                        )

                    with gr.Column(scale=2):
                        verify_result = gr.Markdown(
                            "Selecione um perfil e faça upload de um áudio."
                        )
                        verify_confidence = gr.Number(
                            label="Confiança (%)", interactive=False
                        )
                        verify_details = gr.JSON(label="Detalhes Técnicos")

        # ================================================================ #
        #  Event Handlers                                                    #
        # ================================================================ #

        def handle_create_profile(name, telegram, phone, email, desc, arch):
            if not name or not name.strip():
                return "❌ Nome é obrigatório.", gr.update(), gr.update(), gr.update(), gr.update()
            try:
                svc = _get_service()
                profile = svc.create_profile(
                    name=name.strip(),
                    telegram_id=telegram.strip() if telegram else None,
                    phone=phone.strip() if phone else None,
                    email=email.strip() if email else None,
                    description=desc.strip() if desc else None,
                    architecture=arch,
                )
                profiles = svc.list_profiles()
                table_data = _profiles_to_table(profiles)
                choices = [(f"{p.name} (ID: {p.id}) [{p.status}]", p.id) for p in profiles]
                return (
                    f"✅ Perfil '{profile.name}' criado (ID: {profile.id})",
                    table_data,
                    gr.update(choices=choices),
                    gr.update(choices=choices),
                    gr.update(choices=choices),
                )
            except Exception as e:
                return f"❌ Erro: {e}", gr.update(), gr.update(), gr.update(), gr.update()

        create_btn.click(
            fn=handle_create_profile,
            inputs=[profile_name, profile_telegram, profile_phone,
                    profile_email, profile_desc, profile_arch],
            outputs=[create_status, profiles_table, dataset_profile_dd,
                     train_profile_dd, verify_profile_dd],
        )

        def handle_refresh():
            svc = _get_service()
            profiles = svc.list_profiles()
            table_data = _profiles_to_table(profiles)
            choices = [(f"{p.name} (ID: {p.id}) [{p.status}]", p.id) for p in profiles]
            return (
                table_data,
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )

        refresh_btn.click(
            fn=handle_refresh,
            outputs=[profiles_table, dataset_profile_dd,
                     train_profile_dd, verify_profile_dd],
        )

        def handle_delete(pid):
            if not pid:
                return "❌ Informe o ID do perfil.", gr.update(), gr.update(), gr.update(), gr.update()
            svc = _get_service()
            ok = svc.delete_profile(int(pid))
            if ok:
                profiles = svc.list_profiles()
                table_data = _profiles_to_table(profiles)
                choices = [(f"{p.name} (ID: {p.id}) [{p.status}]", p.id) for p in profiles]
                return (
                    f"✅ Perfil {int(pid)} removido.",
                    table_data,
                    gr.update(choices=choices),
                    gr.update(choices=choices),
                    gr.update(choices=choices),
                )
            return "❌ Perfil não encontrado.", gr.update(), gr.update(), gr.update(), gr.update()

        delete_btn.click(
            fn=handle_delete,
            inputs=[delete_id],
            outputs=[delete_status, profiles_table, dataset_profile_dd,
                     train_profile_dd, verify_profile_dd],
        )

        # ── Dataset handlers ──

        def handle_load_dataset(profile_id):
            if not profile_id:
                return "Selecione um perfil.", []
            svc = _get_service()
            info = svc.get_dataset_info(int(profile_id))
            rows = []
            for f in info["files"]:
                size_kb = f["size_bytes"] / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
                rows.append([f["filename"], f["duration"], f["sample_rate"], size_str])

            md = (
                f"**Total de amostras:** {info['total_samples']}  \n"
                f"**Duração total:** {_format_duration(info['total_duration'])}"
            )
            return md, rows

        dataset_profile_dd.change(
            fn=handle_load_dataset,
            inputs=[dataset_profile_dd],
            outputs=[dataset_info, samples_table],
        )

        def handle_upload_samples(profile_id, files):
            if not profile_id:
                return "❌ Selecione um perfil primeiro.", gr.update(), gr.update()
            if not files:
                return "❌ Nenhum arquivo selecionado.", gr.update(), gr.update()

            svc = _get_service()
            audio_data = []
            for fpath in files:
                with open(fpath, "rb") as f:
                    audio_data.append((Path(fpath).name, f.read()))

            result = svc.add_audio_samples(int(profile_id), audio_data)

            if result.get("success"):
                added_count = len(result.get("added", []))
                error_count = len(result.get("errors", []))
                status = f"✅ {added_count} amostras adicionadas."
                if error_count:
                    status += f" ({error_count} erros)"

                # Recarregar dataset
                md, rows = handle_load_dataset(profile_id)
                return status, md, rows
            return f"❌ {result.get('error', 'Erro')}", gr.update(), gr.update()

        upload_btn.click(
            fn=handle_upload_samples,
            inputs=[dataset_profile_dd, audio_upload],
            outputs=[upload_status, dataset_info, samples_table],
        )

        def handle_remove_sample(profile_id, filename):
            if not profile_id or not filename:
                return "❌ Selecione perfil e informe o nome do arquivo."
            svc = _get_service()
            ok = svc.remove_audio_sample(int(profile_id), filename.strip())
            if ok:
                return f"✅ '{filename}' removido."
            return "❌ Arquivo não encontrado."

        remove_sample_btn.click(
            fn=handle_remove_sample,
            inputs=[dataset_profile_dd, sample_to_remove],
            outputs=[remove_status],
        )

        # ── Training handlers ──

        def handle_train(profile_id, arch, epochs, batch_size, lr):
            if not profile_id:
                yield "❌ Selecione um perfil.", None, None, None
                return

            svc = _get_service()
            log_lines = []

            def progress_cb(msg):
                log_lines.append(msg)

            log_lines.append(f"Iniciando treinamento para perfil {profile_id}...")
            yield "\n".join(log_lines), None, None, None

            result = svc.train_profile_model(
                profile_id=int(profile_id),
                architecture=arch,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(lr),
                progress_callback=progress_cb,
            )

            if result.get("success"):
                metrics = result.get("metrics", {})
                history = metrics.get("history", {})
                log_lines.append(
                    f"\n✅ Treinamento concluído! "
                    f"Val Accuracy: {metrics.get('val_accuracy', 0):.4f}"
                )

                # Gerar plots
                loss_fig = None
                acc_fig = None

                if history:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')

                        # Design tokens (dark theme)
                        _BG = "#0f172a"
                        _FACE = "#1e293b"
                        _TEXT = "#f1f5f9"
                        _GRID = "#334155"

                        def _style(ax, fig, title):
                            fig.patch.set_facecolor(_BG)
                            ax.set_facecolor(_FACE)
                            ax.set_title(title, color=_TEXT, fontweight="600",
                                         fontsize=12, pad=10)
                            ax.tick_params(colors=_TEXT, labelsize=9)
                            for lbl in (ax.xaxis.label, ax.yaxis.label):
                                lbl.set_color(_TEXT)
                                lbl.set_fontsize(10)
                            for sp in ax.spines.values():
                                sp.set_color(_GRID)
                            ax.grid(True, color=_GRID, alpha=0.3, linewidth=0.5)

                        # Loss plot
                        loss_fig = Figure(figsize=(6, 4))
                        ax = loss_fig.add_subplot(111)
                        _style(ax, loss_fig, "Loss por Época")
                        ax.plot(history.get('loss', []), label='Train Loss',
                                color='#3b82f6', linewidth=2)
                        ax.plot(history.get('val_loss', []), label='Val Loss',
                                color='#ef4444', linewidth=2, linestyle='--')
                        ax.set_xlabel('Época')
                        ax.set_ylabel('Loss')
                        ax.legend(facecolor=_FACE, edgecolor=_GRID,
                                  labelcolor=_TEXT, fontsize=9)
                        loss_fig.tight_layout()

                        # Accuracy plot
                        acc_fig = Figure(figsize=(6, 4))
                        ax2 = acc_fig.add_subplot(111)
                        _style(ax2, acc_fig, "Accuracy por Época")
                        ax2.plot(history.get('accuracy', []), label='Train Acc',
                                 color='#10b981', linewidth=2)
                        ax2.plot(history.get('val_accuracy', []), label='Val Acc',
                                 color='#f59e0b', linewidth=2, linestyle='--')
                        ax2.set_xlabel('Época')
                        ax2.set_ylabel('Accuracy')
                        ax2.legend(facecolor=_FACE, edgecolor=_GRID,
                                   labelcolor=_TEXT, fontsize=9)
                        acc_fig.tight_layout()
                    except Exception as e:
                        log_lines.append(f"⚠️ Erro ao gerar gráficos: {e}")

                # Métricas sem history (muito grande para JSON view)
                display_metrics = {
                    k: v for k, v in metrics.items() if k != "history"
                }

                yield "\n".join(log_lines), loss_fig, acc_fig, display_metrics
            else:
                log_lines.append(f"\n❌ Erro: {result.get('error', 'Falha desconhecida')}")
                yield "\n".join(log_lines), None, None, None

        train_btn.click(
            fn=handle_train,
            inputs=[train_profile_dd, train_arch, train_epochs,
                    train_batch, train_lr],
            outputs=[train_log, train_loss_plot, train_acc_plot,
                     train_metrics_json],
        )

        # ── Verify handlers ──

        def handle_verify(profile_id, audio_path):
            if not profile_id:
                return "❌ Selecione um perfil.", None, None
            if not audio_path:
                return "❌ Faça upload de um áudio.", None, None

            svc = _get_service()
            result = svc.detect_with_profile(int(profile_id), audio_path)

            if result.get("success"):
                is_auth = result["is_authentic"]
                conf = result["confidence"]
                name = result["profile_name"]

                if is_auth:
                    md = (
                        f"## ✅ Voz Autêntica\n\n"
                        f"O áudio **pertence** ao perfil **{name}**.\n\n"
                        f"| Métrica | Valor |\n|---|---|\n"
                        f"| Confiança | **{conf:.1f}%** |\n"
                        f"| Status | Autenticada |\n"
                    )
                else:
                    md = (
                        f"## ⚠️ Voz Não Reconhecida\n\n"
                        f"O áudio **NÃO pertence** ao perfil **{name}** "
                        f"ou é um **deepfake**.\n\n"
                        f"| Métrica | Valor |\n|---|---|\n"
                        f"| Confiança | **{conf:.1f}%** |\n"
                        f"| Status | Rejeitada |\n"
                    )

                details = result.get("details", {})
                details["raw_score"] = result.get("raw_score")
                return md, conf, details
            else:
                return f"❌ {result.get('error', 'Erro')}", None, None

        verify_btn.click(
            fn=handle_verify,
            inputs=[verify_profile_dd, verify_audio],
            outputs=[verify_result, verify_confidence, verify_details],
        )

        # ── Carregar perfis ao iniciar ──

        demo_load = refresh_btn
        demo_load.click(
            fn=handle_refresh,
            outputs=[profiles_table, dataset_profile_dd,
                     train_profile_dd, verify_profile_dd],
        )
