import gradio as gr
from app.domain.models.architectures.registry import architecture_registry
from app.interfaces.gradio.utils.hyperparameters import optimize_default, load_defaults


def create_optimization_tab():
    with gr.Tab("Otimização de Hiperparâmetros"):
        gr.Markdown("### ⚙️ Ajuste Fino de Hiperparâmetros")

        with gr.Row():
            with gr.Column(scale=1):
                arch_choices = architecture_registry.list_architectures()
                opt_arch = gr.Dropdown(
                    choices=arch_choices,
                    label="Arquitetura", value=arch_choices[0] if arch_choices else "MultiscaleCNN"
                )

                with gr.Group():
                    gr.Markdown("#### Parâmetros Gerais")
                    opt_batch = gr.Number(label="Batch Size", value=32)
                    opt_epochs = gr.Number(label="Epochs", value=10)
                    opt_lr = gr.Number(
                        label="Learning Rate", value=0.001, precision=5)
                    opt_dropout = gr.Number(label="Dropout Rate", value=0.3)
                    opt_l2 = gr.Number(
                        label="L2 Regularization", value=0.0001, precision=5)
                    opt_val = gr.Number(label="Validation Split", value=0.2)

                # Specific params
                with gr.Group():
                    gr.Markdown("#### Parâmetros Específicos")

                    # Transformer / Spectrogram
                    opt_d_model = gr.Number(label="D-Model", visible=False)
                    opt_num_heads = gr.Number(label="Num Heads", visible=False)
                    opt_num_blocks = gr.Number(
                        label="Num Blocks", visible=False)
                    opt_patch = gr.Textbox(
                        label="Patch Size (HxW)", visible=False)
                    opt_ff_dim = gr.Number(
                        label="Feed Forward Dim", visible=False)

                    # Multiscale
                    opt_filters = gr.Textbox(
                        label="Filters (comma sep)", visible=False)
                    opt_kernel_sizes = gr.Textbox(
                        label="Kernel Sizes (comma sep)", visible=False)

                    # Conformer
                    opt_att_heads = gr.Number(
                        label="Attention Heads", visible=False)
                    opt_hid_dim = gr.Number(label="Hidden Dim", visible=False)
                    opt_n_layers = gr.Number(label="Num Layers", visible=False)
                    opt_conv_kernel = gr.Textbox(
                        label="Conv Kernel Size", visible=False)

                    # AASIST / RawGAT
                    opt_hid_base = gr.Number(
                        label="Hidden Dim (Base)", visible=False)
                    opt_n_layers_base = gr.Number(
                        label="Num Layers (Base)", visible=False)

                    # Hybrid
                    opt_base_filters = gr.Number(
                        label="Base Filters", visible=False)
                    opt_res_blocks = gr.Number(
                        label="Res Blocks", visible=False)
                    opt_trans_layers = gr.Number(
                        label="Transformer Layers", visible=False)
                    opt_att_heads_h = gr.Number(
                        label="Attention Heads (Hybrid)", visible=False)

                    # RawNet2
                    opt_conv_filters_raw = gr.Textbox(
                        label="Conv Filters (Raw)", visible=False)
                    opt_gru_units = gr.Number(label="GRU Units", visible=False)
                    opt_dense_units = gr.Number(
                        label="Dense Units", visible=False)

                opt_save_btn = gr.Button(
                    "💾 Salvar Hiperparâmetros Padrão", variant="primary")

            with gr.Column(scale=1):
                opt_json = gr.JSON(label="Configuração Atual (JSON)")

        # Inputs list for optimize_default
        inputs_list = [
            opt_arch, opt_batch, opt_epochs, opt_lr, opt_dropout, opt_l2, opt_val,
            opt_d_model, opt_num_heads, opt_num_blocks, opt_patch, opt_ff_dim,
            opt_filters, opt_kernel_sizes,
            opt_att_heads, opt_hid_dim, opt_n_layers, opt_conv_kernel,
            opt_hid_base, opt_n_layers_base,
            opt_base_filters, opt_res_blocks, opt_trans_layers, opt_att_heads_h,
            opt_conv_filters_raw, opt_gru_units, opt_dense_units
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
