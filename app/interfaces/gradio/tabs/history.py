import logging
import tempfile

import gradio as gr
import pandas as pd

from app.dependencies import get_analysis_result_repository
from app.interfaces.gradio.utils.components import ui_safe
from app.utils.file_utils import get_gradio_exports_directory

logger = logging.getLogger(__name__)


def get_unique_models():
    """
    Busca os nomes únicos dos modelos salvos no banco de dados.
    """
    try:
        model_list = get_analysis_result_repository().distinct_model_names()
        return ["Todos"] + sorted(model_list)
    except Exception as e:
        logger.error(f"Erro ao buscar modelos: {e}")
        return ["Todos"]


def export_history_csv():
    """
    Exporta o histórico completo para um arquivo CSV temporário.
    """
    try:
        # Busca todos os registros para exportação
        results = get_analysis_result_repository().list_recent()

        data = []
        for r in results:
            data.append({
                "ID": r.id,
                "Data": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Arquivo": r.filename,
                "Resultado": "FAKE" if r.is_fake else "REAL",
                "Confiança": r.confidence,
                "Modelo": r.model_name,
                "Duração (s)": r.duration_seconds,
                "Taxa de Amostragem": r.sample_rate,
                "Detalhes": r.details
            })

        if not data:
            return None

        df = pd.DataFrame(data)

        # Cria arquivo temporário
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.csv',
            prefix='historico_analises_',
            dir=get_gradio_exports_directory(),
        )
        tmp.close()
        df.to_csv(tmp.name, index=False)
        return tmp.name
    except Exception as e:
        logger.error(f"Erro ao exportar CSV: {e}")
        return None


@ui_safe("Falha ao carregar o histórico")
def load_history(filter_model=None, filter_result=None, search_query=None):
    """
    Carrega o histórico de análises do banco de dados com filtros e busca.
    """
    model_name = filter_model if filter_model and filter_model != "Todos" else None
    is_fake = None
    if filter_result and filter_result != "Todos":
        is_fake = filter_result == "Fake"

    results = get_analysis_result_repository().search(
        model_name=model_name,
        is_fake=is_fake,
        filename_contains=search_query or None,
        limit=500,
    )

    data = []
    for r in results:
        data.append([
            r.id,
            r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            r.filename or "N/A",
            "FAKE" if r.is_fake else "REAL",
            f"{r.confidence * 100:.2f}%",
            r.model_name or "N/A",
            f"{r.duration_seconds:.2f}s" if r.duration_seconds else "N/A"
        ])
    return data


def get_details_and_id(evt: gr.SelectData, history_data):
    """
    Retorna os detalhes de uma análise selecionada e seu ID para exclusão.
    """
    if evt is None:
        return (
            "Selecione uma linha para ver detalhes.",
            None,
            gr.update(interactive=False)
        )

    try:
        row_index = evt.index[0]
        # O ID está na primeira coluna (índice 0)
        # history_data é um DataFrame pandas quando retornado pelo componente
        # Dataframe do Gradio em eventos
        analysis_id = int(history_data.iloc[row_index, 0])

        result = get_analysis_result_repository().get(analysis_id)
        if not result:
            return (
                "Análise não encontrada.",
                None,
                gr.update(interactive=False)
            )

        details_str = f"""### Detalhes da Análise #{result.id}

**Data:** {result.created_at.strftime("%Y-%m-%d %H:%M:%S")}
**Arquivo:** {result.filename}
**Modelo:** {result.model_name}
**Resultado:** {"FAKE" if result.is_fake else "REAL"}
**Confiança:** {result.confidence * 100:.4f}%
**Duração:** {result.duration_seconds}s
**Taxa de Amostragem:** {result.sample_rate}Hz

#### Metadados Técnicos:
```json
{result.details}
```
"""
        return details_str, analysis_id, gr.update(interactive=True)
    except Exception as e:
        logger.error(f"Erro ao buscar detalhes: {e}")
        return (
            f"Erro ao carregar detalhes: {str(e)}",
            None,
            gr.update(interactive=False)
        )


def delete_analysis(analysis_id):
    if not analysis_id:
        return "Nenhuma análise selecionada."

    try:
        if get_analysis_result_repository().delete(analysis_id):
            return "Análise excluída com sucesso."
        return "Análise não encontrada."
    except Exception as e:
        return f"Erro ao excluir: {e}"


def clear_all_history():
    try:
        get_analysis_result_repository().delete_all()
        return "Histórico limpo com sucesso."
    except Exception as e:
        return f"Erro ao limpar histórico: {e}"


def create_history_tab():
    with gr.Tab("📜 Histórico", id="tab_history"):
        from app.interfaces.gradio.utils.components import page_header

        page_header(
            "📜",
            "Histórico de Detecções",
            "Consulte, filtre e exporte os resultados de todas as análises "
            "realizadas.",
        )

        with gr.Row(elem_classes="responsive-grid responsive-grid-3"):
            with gr.Column(scale=2):
                search_box = gr.Textbox(
                    label="Buscar por Nome do Arquivo",
                    placeholder="Digite para filtrar...",
                    show_label=True,
                )

            with gr.Column(scale=1):
                filter_model = gr.Dropdown(
                    label="Filtrar por Modelo",
                    choices=["Todos"],
                    value="Todos",
                )

            with gr.Column(scale=1):
                filter_result = gr.Dropdown(
                    label="Filtrar por Resultado",
                    choices=["Todos", "Fake", "Real"],
                    value="Todos",
                )

        with gr.Row(elem_classes="action-row"):
            refresh_btn = gr.Button("Atualizar Lista", variant="primary")
            export_btn = gr.Button("Exportar CSV", variant="secondary")

        download_file = gr.File(label="Download CSV", visible=False)

        history_table = gr.Dataframe(
            headers=[
                "ID", "Data", "Arquivo", "Resultado",
                "Confiança", "Modelo", "Duração"
            ],
            datatype=["number", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            label="Registros de Análise"
        )

        details_view = gr.Markdown(
            "ℹ️ Selecione uma linha na tabela acima para ver os detalhes "
            "completos da análise."
        )

        # Estado para guardar o ID selecionado
        selected_id = gr.State(None)

        with gr.Row(elem_classes="action-row"):
            delete_btn = gr.Button(
                "🗑️ Excluir Análise", variant="stop", interactive=False
            )
            clear_all_btn = gr.Button("⚠️ Limpar Tudo", variant="secondary")

        # --- Event Handlers ---

        @ui_safe("Falha ao recarregar o historico")
        def refresh_data(model, result, search):
            # Atualiza lista de modelos e carrega dados
            models = get_unique_models()
            data = load_history(model, result, search)
            return data, gr.update(choices=models)

        # Atualizar Lista (Botão)
        refresh_btn.click(
            fn=refresh_data,
            inputs=[filter_model, filter_result, search_box],
            outputs=[history_table, filter_model]
        )

        # Filtros Automáticos (Mudança nos inputs)
        search_box.change(
            fn=load_history,
            inputs=[filter_model, filter_result, search_box],
            outputs=[history_table]
        )

        filter_model.change(
            fn=load_history,
            inputs=[filter_model, filter_result, search_box],
            outputs=[history_table]
        )

        filter_result.change(
            fn=load_history,
            inputs=[filter_model, filter_result, search_box],
            outputs=[history_table]
        )

        # Seleção na Tabela (Detalhes + ID para exclusão)
        history_table.select(
            fn=get_details_and_id,
            inputs=[history_table],
            outputs=[details_view, selected_id, delete_btn]
        )

        # Exportação CSV
        @ui_safe("Falha ao exportar")
        def on_export():
            path = export_history_csv()
            if path:
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)

        export_btn.click(
            fn=on_export,
            outputs=[download_file]
        )

        # Exclusão Única
        delete_btn.click(
            fn=delete_analysis,
            inputs=[selected_id],
            outputs=[details_view]
        ).then(
            fn=load_history,
            inputs=[filter_model, filter_result, search_box],
            outputs=[history_table]
        ).then(
            fn=lambda: (None, gr.update(interactive=False)),
            outputs=[selected_id, delete_btn]
        )

        # Limpar Tudo
        clear_all_btn.click(
            fn=clear_all_history,
            outputs=[details_view]
        ).then(
            fn=load_history,
            inputs=[filter_model, filter_result, search_box],
            outputs=[history_table]
        )
