import gradio as gr

from app.domain.models.architectures.registry import architecture_registry
from app.domain.services.detection.utils import get_available_devices
from app.interfaces.gradio.utils.hyperparameters import (
    get_interface_hyperparameters,
    load_defaults,
    optimize_default,
)


def update_device_settings(device_name):
    """Atualiza o dispositivo de inferência no serviço."""
    try:
        # Importação tardia para evitar ciclo
        from app.interfaces.gradio.tabs.detection import get_detection_service
        ds = get_detection_service()
        if ds:
            ds.set_device(device_name)
            return f"✅ Dispositivo alterado para: {device_name}"
        return "❌ Erro: Serviço de detecção não inicializado."
    except Exception as e:
        return f"❌ Erro ao atualizar dispositivo: {str(e)}"


def run_auto_tuning(arch, dataset_path, n_trials, metric, epochs):
    """Roda a busca de hiperparâmetros (Optuna) e devolve (resumo, best, figura).

    Degrada graciosamente: se `optuna` não estiver instalado, ou o dataset não
        existir, retorna uma mensagem clara sem levantar exceção. Cada tentativa treina
    um modelo de verdade via TrainingService — logo pode levar minutos.
    """
    from pathlib import Path

    from app.domain.models.training.hyperparameter_tuning import (
        is_optuna_available,
        suggest_search_space,
        tune_hyperparameters,
    )
    from app.interfaces.gradio.utils.tuning_charts import render_tuning_figure

    if not is_optuna_available():
        return (
            "⚠️ **optuna não está instalado.** A busca automática requer "
            "`pip install optuna` (já em `requirements-dev.txt`).",
            {},
            None,
        )
    if not dataset_path or not Path(str(dataset_path)).exists():
        return (
            f"❌ Dataset `.npz` não encontrado: `{dataset_path}`.\n\n"
            "Forneça um `.npz` com `X_train`/`y_train` (mesmo formato do treino "
            "via serviço).",
            {},
            None,
        )
    try:
        n_trials = max(2, int(n_trials))
        epochs = max(1, int(epochs))
    except (TypeError, ValueError):
        return ("❌ Número de tentativas e épocas devem ser inteiros.", {}, None)

    # val_loss → minimizar; accuracy/f1 → maximizar
    direction = "minimize" if "loss" in str(metric).lower() else "maximize"
    try:
        result = tune_hyperparameters(
            architecture=arch,
            dataset_path=str(dataset_path),
            base_config={"epochs": epochs, "batch_size": 32},
            search_space=suggest_search_space(arch),
            n_trials=n_trials,
            metric=metric,
            direction=direction,
        )
    except Exception as e:  # noqa: BLE001 — superfície de UI: nunca propaga
        return (f"❌ Erro na busca: {e}", {}, None)

    if result.get("status") != "success":
        return (f"❌ Busca falhou: {result.get('errors')}", {}, None)

    study = result.get("study")
    fig = render_tuning_figure(study) if study is not None else None
    best_score = result.get("best_score")
    if best_score is not None:
        summary = (
            f"✅ **Busca concluída** — {result.get('n_trials', 0)} tentativas. "
            f"Melhor **{metric}** = `{best_score:.4f}`."
        )
    else:
        summary = "✅ Busca concluída."
    return summary, result.get("best_params", {}), fig


def create_optimization_tab():
    with gr.Tab("⚡ Otimização"):
        from app.interfaces.gradio.utils.components import page_header

        page_header(
            "⚡",
            "Otimização",
            "Configure hardware, arquiteturas e parâmetros de treinamento "
            "para otimizar a detecção.",
        )

        # --- Seção de Hardware ---
        with gr.Row(elem_classes="responsive-grid"):
            with gr.Column():
                gr.Markdown("#### 🖥️ Configuração de Hardware")
                devices = get_available_devices()
                default_dev = devices[0] if devices else "CPU"

                with gr.Row(elem_classes="action-row"):
                    device_dropdown = gr.Dropdown(
                        choices=devices,
                        label="Dispositivo de Processamento",
                        value=default_dev,
                        interactive=True
                    )
                    apply_dev_btn = gr.Button(
                        "Aplicar Configuração", variant="primary"
                    )

                dev_status = gr.Textbox(
                    label="Status", value="", interactive=False
                )

                apply_dev_btn.click(
                    update_device_settings,
                    inputs=[device_dropdown],
                    outputs=[dev_status]
                )

        gr.Markdown("---")  # Separator

        with gr.Row(elem_classes="responsive-grid"):
            with gr.Column(scale=1):
                arch_choices = architecture_registry.list_architectures()
                default_arch = arch_choices[0] if arch_choices else "MultiscaleCNN"
                initial_hp = get_interface_hyperparameters(default_arch)
                opt_arch = gr.Dropdown(
                    choices=arch_choices,
                    label="Arquitetura",
                    value=default_arch,
                )

                with gr.Group():
                    gr.Markdown("#### Parâmetros Gerais")
                    opt_batch = gr.Number(
                        label="Tamanho do lote", value=int(initial_hp["batch_size"])
                    )
                    opt_epochs = gr.Number(
                        label="Épocas", value=int(initial_hp["epochs"])
                    )
                    opt_lr = gr.Number(
                        label="Taxa de aprendizado",
                        value=float(initial_hp["learning_rate"]),
                        precision=7,
                    )
                    opt_dropout = gr.Number(
                        label="Taxa de dropout",
                        value=float(initial_hp["dropout_rate"]),
                    )
                    opt_l2 = gr.Number(
                        label="Regularização L2",
                        value=float(initial_hp["l2_reg_strength"]),
                        precision=7,
                    )
                    opt_val = gr.Number(
                        label="Proporção de validação",
                        value=float(initial_hp["validation_split"]),
                    )

                # Specific params
                with gr.Group():
                    gr.Markdown("#### Parâmetros Específicos")

                    # Transformer / Spectrogram
                    opt_d_model = gr.Number(label="Dimensão do modelo", visible=False)
                    opt_num_heads = gr.Number(label="Número de cabeças", visible=False)
                    opt_num_blocks = gr.Number(
                        label="Número de blocos", visible=False)
                    opt_patch = gr.Textbox(
                        label="Tamanho do patch (altura × largura)", visible=False)
                    opt_ff_dim = gr.Number(
                        label="Dimensão feed-forward", visible=False)

                    # Multiscale
                    opt_filters = gr.Textbox(
                        label="Filtros (separados por vírgula)", visible=False)
                    opt_kernel_sizes = gr.Textbox(
                        label="Tamanhos de kernel (separados por vírgula)",
                        visible=False,
                    )

                    # Conformer
                    opt_att_heads = gr.Number(
                        label="Cabeças de atenção", visible=False)
                    opt_hid_dim = gr.Number(label="Dimensão oculta", visible=False)
                    opt_n_layers = gr.Number(label="Número de camadas", visible=False)
                    opt_conv_kernel = gr.Textbox(
                        label="Tamanho do kernel convolucional", visible=False)

                    # AASIST / RawGAT
                    opt_hid_base = gr.Number(
                        label="Dimensão oculta (base)", visible=False)
                    opt_n_layers_base = gr.Number(
                        label="Número de camadas (base)", visible=False)

                    # Hybrid
                    opt_base_filters = gr.Number(
                        label="Filtros base", visible=False)
                    opt_res_blocks = gr.Number(
                        label="Blocos residuais", visible=False)
                    opt_trans_layers = gr.Number(
                        label="Camadas Transformer", visible=False)
                    opt_att_heads_h = gr.Number(
                        label="Cabeças de atenção (híbrido)", visible=False)

                    # RawNet2
                    opt_conv_filters_raw = gr.Textbox(
                        label="Filtros convolucionais (áudio bruto)", visible=False)
                    opt_gru_units = gr.Number(label="Unidades GRU", visible=False)
                    opt_dense_units = gr.Number(
                        label="Unidades densas", visible=False)

                opt_save_btn = gr.Button(
                    "💾 Salvar Hiperparâmetros Padrão", variant="primary")

            with gr.Column(scale=1):
                opt_json = gr.JSON(label="Configuração Atual (JSON)")

        # Inputs list for optimize_default
        inputs_list = [
            opt_arch, opt_batch, opt_epochs, opt_lr, opt_dropout, opt_l2,
            opt_val, opt_d_model, opt_num_heads, opt_num_blocks, opt_patch,
            opt_ff_dim, opt_filters, opt_kernel_sizes,
            opt_att_heads, opt_hid_dim, opt_n_layers, opt_conv_kernel,
            opt_hid_base, opt_n_layers_base,
            opt_base_filters, opt_res_blocks, opt_trans_layers,
            opt_att_heads_h, opt_conv_filters_raw, opt_gru_units,
            opt_dense_units
        ]

        # Outputs list (JSON + updates for all inputs)
        outputs_list = [opt_json] + inputs_list[1:]

        # Event: Save/Update
        opt_save_btn.click(
            optimize_default,
            inputs=inputs_list,
            outputs=outputs_list
        )

        # Event: Change Architecture (load defaults)
        opt_arch.change(
            load_defaults,
            inputs=[opt_arch],
            outputs=outputs_list
        )

        gr.Markdown("---")  # Separator

        # --- Busca Automática de Hiperparâmetros (Optuna) ---
        with gr.Accordion(
            "🔍 Busca Automática de Hiperparâmetros (Optuna)", open=False
        ):
            gr.Markdown(
                "Busca Bayesiana (Optuna/TPE) que **treina vários modelos** e "
                "mostra a **convergência** das tentativas e a **importância** de cada "
                "hiperparâmetro. Requer um dataset `.npz` (`X_train`/`y_train`) e "
                "`optuna` instalado. ⚠️ Pode levar **vários minutos** — treina um "
                "modelo por tentativa."
            )
            with gr.Row(elem_classes="responsive-grid"):
                with gr.Column(scale=1):
                    tune_arch = gr.Dropdown(
                        choices=arch_choices,
                        label="Arquitetura",
                        value=arch_choices[0] if arch_choices else "MultiscaleCNN",
                    )
                    tune_dataset = gr.Textbox(
                        label="Conjunto de dados (.npz com X_train/y_train)",
                        placeholder="ex.: data/datasets/features/train.npz",
                    )
                    with gr.Row(elem_classes="responsive-grid"):
                        tune_trials = gr.Number(
                            label="Número de tentativas", value=10, precision=0
                        )
                        tune_epochs = gr.Number(
                            label="Épocas por tentativa", value=5, precision=0
                        )
                    tune_metric = gr.Dropdown(
                        choices=[
                            "val_accuracy",
                            "f1_score",
                            "accuracy",
                            "val_loss",
                        ],
                        label="Métrica objetivo",
                        value="val_accuracy",
                    )
                    tune_btn = gr.Button(
                        "🔍 Buscar melhores hiperparâmetros", variant="primary"
                    )
                with gr.Column(scale=1):
                    tune_status = gr.Markdown("")
                    tune_best = gr.JSON(label="Melhores hiperparâmetros")
            tune_plot = gr.Plot(label="Convergência e importância dos parâmetros")

            tune_btn.click(
                run_auto_tuning,
                inputs=[
                    tune_arch,
                    tune_dataset,
                    tune_trials,
                    tune_metric,
                    tune_epochs,
                ],
                outputs=[tune_status, tune_best, tune_plot],
            )
