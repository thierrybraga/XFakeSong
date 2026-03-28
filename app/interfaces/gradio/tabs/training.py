import io
import logging
import os
import shutil

# import json
import time
import zipfile
from pathlib import Path

import requests

import gradio as gr

# Configurar logger para capturar logs de treinamento
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
logging.getLogger().addHandler(ch)
logger = logging.getLogger(__name__)


def create_training_tab():
    with gr.Tab("Treinamento"):
        gr.Markdown(
            "### Gestão de Datasets e Treinamento\n"
            "Prepare datasets, extraia features e treine modelos de detecção "
            "de deepfake com diferentes arquiteturas."
        )

        with gr.Tabs():
            # --- TAB DATASET ---
            with gr.Tab("Dataset & Features"):
                gr.Markdown("#### 1. Upload e Preparação do Dataset")

                with gr.Row():
                    with gr.Column():
                        zip_file = gr.File(
                            label="Upload Dataset (.zip)", file_types=[".zip"])
                        gr.Markdown(
                            "*O arquivo zip deve conter pastas "
                            "'real' e 'fake'.*"
                        )
                        upload_btn = gr.Button(
                            "Processar Dataset", variant="primary")

                    with gr.Column():
                        dataset_status = gr.Textbox(
                            label="Status do Dataset", interactive=False)
                        dataset_info = gr.JSON(label="Informações do Dataset")

                gr.Markdown("#### 2. Extração de Features")
                gr.Markdown(
                    "Extrai características (espectrais, temporais, etc.) "
                    "para treinamento de modelos clássicos."
                )

                with gr.Row():
                    feature_extract_btn = gr.Button(
                        "Extrair Features (Batch)", variant="secondary")
                    feature_status = gr.Textbox(
                        label="Status da Extração", interactive=False)

                feature_logs = gr.TextArea(
                    label="Logs da Extração", lines=10, max_lines=20)

                def process_dataset(zip_path):
                    if not zip_path:
                        return "Nenhum arquivo enviado.", {}

                    try:
                        # Diretórios
                        base_dir = Path("app/datasets/raw")
                        if base_dir.exists():
                            shutil.rmtree(base_dir)
                        base_dir.mkdir(parents=True, exist_ok=True)

                        # Extrair
                        with zipfile.ZipFile(zip_path.name, 'r') as zip_ref:
                            zip_ref.extractall(base_dir)

                        # Organizar e Renomear
                        stats = {"real": 0, "fake": 0, "errors": 0}

                        # Tentar identificar pastas real/fake
                        extracted_dirs = [
                            d for d in base_dir.iterdir() if d.is_dir()]
                        # Se extraiu uma pasta raiz, entre nela
                        if len(extracted_dirs) == 1 and (
                                extracted_dirs[0] / "real").exists():
                            root_dir = extracted_dirs[0]
                        else:
                            root_dir = base_dir

                        processed_dir = Path("app/datasets/processed")
                        if processed_dir.exists():
                            shutil.rmtree(processed_dir)
                        processed_dir.mkdir(parents=True, exist_ok=True)

                        (processed_dir / "real").mkdir()
                        (processed_dir / "fake").mkdir()

                        # Processar Real
                        real_src = root_dir / "real"
                        if real_src.exists():
                            for i, f in enumerate(real_src.glob("*")):
                                if f.suffix.lower() in [
                                        '.wav', '.mp3', '.flac']:
                                    new_name = f"real_sample_{i:04d}{f.suffix}"
                                    shutil.copy2(
                                        f, processed_dir / "real" / new_name)
                                    stats["real"] += 1

                        # Processar Fake
                        fake_src = root_dir / "fake"
                        if fake_src.exists():
                            for i, f in enumerate(fake_src.glob("*")):
                                if f.suffix.lower() in [
                                        '.wav', '.mp3', '.flac']:
                                    new_name = f"fake_sample_{i:04d}{f.suffix}"
                                    shutil.copy2(
                                        f, processed_dir / "fake" / new_name)
                                    stats["fake"] += 1

                        return (
                            f"Sucesso! Real: {stats['real']}, "
                            f"Fake: {stats['fake']}",
                            stats
                        )

                    except Exception as e:
                        return f"Erro: {str(e)}", {}

                def extract_features_batch():
                    yield "Iniciando extração...", "Preparando..."

                    try:
                        from app.domain.features.extractors.segmented_feature_extractor import (  # noqa
                            SegmentedFeatureExtractor,
                            SegmentedExtractionConfig
                        )
                        from app.domain.features.exporters.csv_feature_exporter import (  # noqa
                            CSVExportConfig
                        )

                        # Configurar Extrator
                        csv_config = CSVExportConfig(
                            output_base_dir="datasets/features")
                        config = SegmentedExtractionConfig(
                            export_csv=True,
                            csv_config=csv_config,
                            extract_spectral=True,
                            extract_cepstral=True,
                            extract_temporal=True,
                            extract_prosodic=True
                            # Adicione outros conforme necessário, mas cuidado
                            # com performance
                        )
                        extractor = SegmentedFeatureExtractor(config)

                        processed_dir = Path("app/datasets/processed")
                        if not processed_dir.exists():
                            yield (
                                "Erro",
                                "Dataset processado não encontrado. "
                                "Faça upload primeiro."
                            )
                            return

                        logs = []
                        total_files = 0

                        # Labels: 0=Real, 1=Fake
                        for label_name, label_id in [("real", 0), ("fake", 1)]:
                            class_dir = processed_dir / label_name
                            if not class_dir.exists():
                                continue

                            files = list(class_dir.glob("*"))
                            total_class = len(files)

                            for i, f in enumerate(files):
                                msg = (
                                    f"Processando [{label_name}] "
                                    f"{i + 1}/{total_class}: {f.name}"
                                )
                                logs.append(msg)
                                if len(logs) > 20:
                                    logs.pop(0)  # Manter log curto na UI
                                yield (
                                    f"Extraindo... {label_name} "
                                    f"{i + 1}/{total_class}",
                                    "\n".join(logs)
                                )

                                extractor.extract_from_file(
                                    str(f), label=label_id)
                                total_files += 1

                        yield (
                            "Concluído!",
                            f"Extração finalizada. "
                            f"Total arquivos: {total_files}"
                        )

                    except Exception as e:
                        import traceback
                        yield (
                            "Erro Fatal",
                            f"{str(e)}\n{traceback.format_exc()}"
                        )

                upload_btn.click(
                    process_dataset, inputs=[zip_file], outputs=[
                        dataset_status, dataset_info])
                feature_extract_btn.click(
                    extract_features_batch, outputs=[
                        feature_status, feature_logs])

                gr.Markdown("#### 3. Gerenciamento de Datasets via API")
                with gr.Row():
                    api_key = gr.Textbox(
                        label="API Key", value="", type="password"
                    )
                    ds_type = gr.Dropdown(
                        choices=["training", "validation", "test"],
                        value="training", label="Tipo"
                    )
                    ds_name = gr.Textbox(label="Nome do Dataset")
                with gr.Row():
                    list_btn = gr.Button("Listar Datasets (API)")
                    create_btn = gr.Button("Criar Dataset (API)")
                    delete_btn = gr.Button("Excluir Dataset (API)")
                ds_api_result = gr.JSON(label="Resultado API Datasets")

                def list_datasets_api(api_key_val, type_val):
                    base = f"http://localhost:{os.getenv('PORT', '7861')}"
                    headers = {"X-API-Key": api_key_val} if api_key_val else {}
                    r = requests.get(
                        f"{base}/api/v1/datasets/?type={type_val}",
                        headers=headers, timeout=10
                    )
                    return r.json() if r.status_code == 200 else {
                        "error": r.text, "status_code": r.status_code
                    }

                def create_dataset_api(api_key_val, name_val, type_val):
                    base = f"http://localhost:{os.getenv('PORT', '7861')}"
                    headers = {"X-API-Key": api_key_val} if api_key_val else {}
                    data = {"name": name_val, "type": type_val}
                    r = requests.post(
                        f"{base}/api/v1/datasets/",
                        headers=headers, data=data, timeout=10
                    )
                    return r.json() if r.status_code == 200 else {
                        "error": r.text, "status_code": r.status_code
                    }

                def delete_dataset_api(api_key_val, name_val, type_val):
                    base = f"http://localhost:{os.getenv('PORT', '7861')}"
                    headers = {"X-API-Key": api_key_val} if api_key_val else {}
                    r = requests.delete(
                        f"{base}/api/v1/datasets/{name_val}?type={type_val}",
                        headers=headers, timeout=10
                    )
                    return r.json() if r.status_code == 200 else {
                        "error": r.text, "status_code": r.status_code
                    }

                list_btn.click(
                    list_datasets_api,
                    inputs=[api_key, ds_type],
                    outputs=ds_api_result
                )
                create_btn.click(
                    create_dataset_api,
                    inputs=[api_key, ds_name, ds_type],
                    outputs=ds_api_result
                )
                delete_btn.click(
                    delete_dataset_api,
                    inputs=[api_key, ds_name, ds_type],
                    outputs=ds_api_result
                )

            # --- TAB DEEP LEARNING ---
            with gr.Tab("Deep Learning"):
                with gr.Row():
                    with gr.Column():
                        from app.domain.models.architectures.registry import (
                            architecture_registry,
                            get_architecture_info,
                        )

                        arch_choices = (
                            architecture_registry.list_architectures()
                        )
                        dl_arch = gr.Dropdown(
                            choices=arch_choices,
                            label="Arquitetura",
                            value=arch_choices[0]
                            if arch_choices else "MultiscaleCNN"
                        )
                        dl_dataset_path = gr.Textbox(
                            label=(
                                "Caminho do Dataset (Raiz com pastas "
                                "'real' e 'fake')"
                            ),
                            value="app/datasets/processed"
                        )
                        with gr.Row():
                            dl_epochs = gr.Slider(
                                minimum=1, maximum=100, value=5,
                                step=1, label="Épocas"
                            )
                            dl_batch_size = gr.Slider(
                                minimum=2, maximum=128, value=16,
                                step=2, label="Batch Size"
                            )

                        dl_lr = gr.Number(
                            value=0.001, label="Learning Rate", precision=5)

                        with gr.Accordion("Opções Avançadas", open=False):
                            dl_params = gr.JSON(
                                label="Hiperparâmetros da Arquitetura (JSON)",
                                value=(
                                    get_architecture_info(
                                        arch_choices[0]
                                    ).default_params
                                    if arch_choices else {}
                                )
                            )

                        dl_train_btn = gr.Button(
                            "Iniciar Treinamento DL", variant="primary")

                    with gr.Column():
                        dl_status = gr.Textbox(
                            label="Status", interactive=False)
                        dl_logs = gr.TextArea(
                            label="Logs de Treinamento",
                            lines=15,
                            max_lines=30,
                            interactive=False)
                        dl_plot = gr.Plot(label="Curvas de Treinamento")

                        with gr.Row():
                            dl_roc_plot = gr.Plot(label="Curva ROC")
                            dl_cm_plot = gr.Plot(label="Matriz de Confusão")
                            dl_pr_plot = gr.Plot(label="Curva Precisão-Recall")

                        gr.Markdown("#### Analise Avancada de Treinamento")
                        with gr.Row():
                            dl_det_plot = gr.Plot(label="Curva DET / EER")
                            dl_thresh_plot = gr.Plot(label="Otimizacao de Threshold")
                        with gr.Row():
                            dl_class_acc_plot = gr.Plot(label="Acuracia por Classe")
                            dl_tsne_plot = gr.Plot(label="Embedding 2D (t-SNE)")
                        dl_lr_plot = gr.Plot(label="Schedule de Learning Rate")

                def update_arch_params(arch_name):
                    try:
                        from app.domain.models.training.optimized_training_config import (  # noqa
                            load_hyperparameters_json
                        )
                        root_dir = Path(
                            __file__).parent.parent.parent.parent.parent
                        out_dir = root_dir / "app" / "results"

                        # Carrega parâmetros salvos ou defaults
                        hp = load_hyperparameters_json(arch_name, str(out_dir))

                        # Extrair valores comuns para atualizar os sliders
                        epochs = int(hp.get("epochs", 10))
                        batch = int(hp.get("batch_size", 32))
                        lr = float(hp.get("learning_rate", 0.001))

                        # Ajustar máximos se necessário
                        new_max_epochs = max(100, epochs)
                        new_max_batch = max(128, batch)

                        return (
                            hp,
                            gr.update(value=epochs, maximum=new_max_epochs),
                            gr.update(value=batch, maximum=new_max_batch),
                            gr.update(value=lr)
                        )
                    except Exception as e:
                        print(f"Erro ao atualizar params: {e}")
                        return ({}, gr.update(), gr.update(), gr.update())

                dl_arch.change(
                    fn=update_arch_params,
                    inputs=dl_arch,
                    outputs=[dl_params, dl_epochs, dl_batch_size, dl_lr]
                )

                def train_dl_wrapper(arch, dataset_path,
                                     epochs, batch_size, lr, model_params):
                    yield ("Iniciando...", "", None, None, None, None,
                           None, None, None, None, None)

                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        import seaborn as sns
                        import tensorflow as tf
                        from sklearn.metrics import (
                            auc,
                            confusion_matrix,
                            precision_recall_curve,
                            roc_curve,
                        )

                        from app.domain.models.architectures.registry import (
                            create_model_by_name,
                            get_architecture_info,
                        )

                        # Limpar logs anteriores
                        log_capture_string.truncate(0)
                        log_capture_string.seek(0)

                        # Resolver caminho do dataset (relativo ou absoluto)
                        def _resolve_dataset_path(raw_path: str) -> Path | None:
                            """Tenta múltiplos candidatos e retorna o primeiro
                            que contenha subpastas real/ e fake/."""
                            candidates = [Path(raw_path)]
                            # Raiz do projeto (/app ou diretório do arquivo)
                            app_root = Path(__file__).resolve().parents[4]
                            candidates.append(app_root / raw_path)
                            # Também tenta /app/<caminho> fixo no container
                            candidates.append(Path("/app") / raw_path)
                            for p in candidates:
                                try:
                                    p = p.resolve()
                                    if (p.is_dir()
                                            and (p / "real").is_dir()
                                            and (p / "fake").is_dir()):
                                        return p
                                except Exception:
                                    continue
                            # Último recurso: retorna o caminho existente
                            # mesmo sem a estrutura correta
                            for p in candidates:
                                try:
                                    if p.resolve().is_dir():
                                        return p.resolve()
                                except Exception:
                                    continue
                            return None

                        base_path = _resolve_dataset_path(dataset_path)
                        if base_path is None:
                            yield (
                                "Erro",
                                f"Dataset não encontrado: {dataset_path}\n"
                                "Certifique-se de que o caminho existe e "
                                "contém subpastas 'real' e 'fake'.\n"
                                "Execute o script de download:\n"
                                "  python scripts/download_pt_datasets_v2.py "
                                "--all --max-samples 1000",
                                None, None, None, None,
                                None, None, None, None, None
                            )
                            return

                        if not (base_path / "real").is_dir() or \
                                not (base_path / "fake").is_dir():
                            yield (
                                "Erro",
                                f"Dataset em '{base_path}' não tem a estrutura "
                                "esperada.\nCrie subpastas 'real' e 'fake' com "
                                "arquivos .wav dentro.",
                                None, None, None, None,
                                None, None, None, None, None
                            )
                            return

                        # Verificar Requisitos da Arquitetura
                        try:
                            arch_info = get_architecture_info(arch)
                            input_reqs = arch_info.input_requirements
                        except Exception:
                            input_reqs = {}

                        req_type = input_reqs.get("type", "audio")
                        req_format = input_reqs.get("format", "raw")

                        yield (
                            "Configurando Pipeline...",
                            f"Arquitetura: {arch}\n"
                            f"Tipo Entrada: {req_type} ({req_format})",
                            None, None, None, None,
                            None, None, None, None, None
                        )

                        # Configurações de Áudio
                        SAMPLE_RATE = 16000
                        # Duração fixa 3s
                        # (pode ser ajustado via params da arquitetura
                        # futuramente)
                        DURATION = 3
                        AUDIO_LEN = SAMPLE_RATE * DURATION

                        # 1. Carregar Dataset (Raw Audio)
                        try:
                            train_ds = (
                                tf.keras.utils.audio_dataset_from_directory(
                                    directory=str(base_path),
                                    batch_size=batch_size,
                                    validation_split=0.2,
                                    subset='training',
                                    seed=42,
                                    output_sequence_length=AUDIO_LEN,
                                    label_mode='int'
                                )
                            )
                            val_ds = (
                                tf.keras.utils.audio_dataset_from_directory(
                                    directory=str(base_path),
                                    batch_size=batch_size,
                                    validation_split=0.2,
                                    subset='validation',
                                    seed=42,
                                    output_sequence_length=AUDIO_LEN,
                                    label_mode='int'
                                )
                            )
                        except Exception as e:
                            yield (
                                "Erro Dataset",
                                f"Falha ao carregar dataset: {str(e)}",
                                None, None, None, None,
                                None, None, None, None, None
                            )
                            return

                        # 2. Pré-processamento On-the-Fly (se necessário)
                        input_shape = (AUDIO_LEN, 1)

                        if (req_type == "features" and
                                req_format == "spectrogram"):
                            yield (
                                "Pré-processamento...",
                                "Configurando extração de espectrogramas...",
                                None, None, None, None,
                                None, None, None, None, None
                            )

                            # Função de transformação para Espectrograma
                            def get_spectrogram(waveform, label):
                                # Remover dimensão extra se existir
                                # (B, N, 1) -> (B, N)
                                waveform = tf.squeeze(waveform, axis=-1)

                                # STFT
                                spectrogram = tf.signal.stft(
                                    waveform,
                                    frame_length=1024,
                                    frame_step=256,
                                    fft_length=1024
                                )
                                spectrogram = tf.abs(spectrogram)

                                # Mel Spectrogram (simplificado)
                                # Adicionar canal de cor: (Time, Freq, 1)
                                spectrogram = tf.expand_dims(
                                    spectrogram, axis=-1)

                                # Resize para (128, 128)
                                spectrogram = tf.image.resize(
                                    spectrogram, [128, 128])

                                # Log scaling
                                spectrogram = tf.math.log(spectrogram + 1e-6)

                                return spectrogram, label

                            train_ds = train_ds.map(
                                get_spectrogram,
                                num_parallel_calls=tf.data.AUTOTUNE
                            )
                            val_ds = val_ds.map(
                                get_spectrogram,
                                num_parallel_calls=tf.data.AUTOTUNE
                            )

                            # Atualizar input_shape
                            input_shape = (128, 128, 1)

                        elif req_type == "audio":
                            # Garantir shape correto (N, 1)
                            def ensure_channel(waveform, label):
                                return waveform, label

                            train_ds = train_ds.map(ensure_channel)
                            val_ds = val_ds.map(ensure_channel)
                            input_shape = (AUDIO_LEN, 1)

                        # Detectar número real de classes do dataset
                        num_classes = len(train_ds.class_names)

                        # Otimizar performance
                        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
                        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

                        # 3. Criar Modelo
                        yield (
                            "Criando Modelo...",
                            f"Instanciando {arch} com "
                            f"input_shape={input_shape}, "
                            f"num_classes={num_classes} "
                            f"({train_ds.class_names})...",
                            None, None, None, None,
                            None, None, None, None, None
                        )

                        try:
                            # Passar model_params como kwargs
                            model = create_model_by_name(
                                arch,
                                input_shape=input_shape,
                                num_classes=num_classes,
                                **model_params
                            )
                        except Exception as e:
                            yield (
                                "Erro Modelo",
                                f"Falha ao criar modelo: {str(e)}\n"
                                f"Verifique se a arquitetura suporta "
                                f"o input shape {input_shape}",
                                None, None, None, None,
                                None, None, None, None, None
                            )
                            return

                        # 4. Configurar Treinamento
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        model.compile(
                            optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                        class LogCallback(tf.keras.callbacks.Callback):
                            def __init__(self):
                                self.logs_hist = []

                            def on_epoch_end(self, epoch, logs=None):
                                msg = (
                                    f"Epoch {epoch + 1}/{epochs} - "
                                    f"loss: {logs['loss']:.4f} - "
                                    f"accuracy: {logs['accuracy']:.4f} - "
                                    f"val_loss: {logs['val_loss']:.4f} - "
                                    f"val_accuracy: "
                                    f"{logs['val_accuracy']:.4f}\n"
                                )
                                self.logs_hist.append(msg)

                        class LRTrackingCallback(tf.keras.callbacks.Callback):
                            def __init__(self):
                                self.lr_history = []

                            def on_epoch_end(self, epoch, logs=None):
                                lr = float(
                                    self.model.optimizer.learning_rate)
                                self.lr_history.append(lr)

                        log_cb = LogCallback()
                        lr_cb = LRTrackingCallback()

                        yield (
                            "Treinando...",
                            "Iniciando loop de treinamento...",
                            None, None, None, None,
                            None, None, None, None, None
                        )

                        history = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=epochs,
                            callbacks=[log_cb, lr_cb]
                        )

                        # Gerar gráfico de Loss/Acc
                        fig_hist, ax = plt.subplots(1, 2, figsize=(12, 4))
                        ax[0].plot(history.history['loss'], label='Train Loss')
                        ax[0].plot(
                            history.history['val_loss'], label='Val Loss')
                        ax[0].set_title('Loss')
                        ax[0].legend()

                        ax[1].plot(
                            history.history['accuracy'], label='Train Acc')
                        ax[1].plot(
                            history.history['val_accuracy'], label='Val Acc')
                        ax[1].set_title('Accuracy')
                        ax[1].legend()

                        # --- Avaliação Detalhada para Plots ---
                        yield (
                            "Avaliando...",
                            "Gerando gráficos de performance...",
                            fig_hist, None, None, None,
                            None, None, None, None, None
                        )

                        y_true_all = []
                        y_pred_probs_all = []

                        # Iterar sobre dataset de validação
                        for x_batch, y_batch in val_ds:
                            preds = model.predict_on_batch(x_batch)
                            y_true_all.extend(y_batch.numpy())
                            y_pred_probs_all.extend(preds)

                        y_true = np.array(y_true_all)
                        y_probs = np.array(y_pred_probs_all)
                        y_pred_labels = np.argmax(y_probs, axis=1)

                        # Preparar scores para ROC/PR
                        if y_probs.shape[1] == 2:
                            y_scores = y_probs[:, 1]
                        else:
                            y_scores = y_probs.flatten()

                        # 1. Curva ROC
                        fpr, tpr, _ = roc_curve(y_true, y_scores)
                        roc_auc = auc(fpr, tpr)

                        # Dark theme tokens
                        _BG, _FC = "#0f172a", "#1e293b"
                        _TX, _GR = "#f1f5f9", "#334155"

                        def _dstyle(ax, fig, title):
                            fig.patch.set_facecolor(_BG)
                            ax.set_facecolor(_FC)
                            ax.set_title(title, color=_TX, fontweight="600", fontsize=12, pad=10)
                            ax.tick_params(colors=_TX, labelsize=9)
                            for label in (ax.xaxis.label, ax.yaxis.label):
                                label.set_color(_TX)
                                label.set_fontsize(10)
                            for s in ax.spines.values():
                                s.set_color(_GR)
                            ax.grid(True, color=_GR, alpha=0.3, linewidth=0.5)

                        fig_roc, ax_r = plt.subplots(figsize=(8, 6))
                        _dstyle(ax_r, fig_roc, "ROC Curve")
                        ax_r.plot(fpr, tpr, color='#f59e0b', lw=2,
                                  label=f'AUC = {roc_auc:.2f}')
                        ax_r.plot([0, 1], [0, 1], color=_GR, lw=1.5, linestyle='--')
                        ax_r.set_xlim([0.0, 1.0])
                        ax_r.set_ylim([0.0, 1.05])
                        ax_r.set_xlabel('False Positive Rate')
                        ax_r.set_ylabel('True Positive Rate')
                        ax_r.legend(facecolor=_FC, edgecolor=_GR,
                                    labelcolor=_TX, fontsize=9)

                        # 2. Matriz de Confusão
                        cm = confusion_matrix(y_true, y_pred_labels)

                        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                        _dstyle(ax_cm, fig_cm, "Confusion Matrix")
                        sns.heatmap(
                            cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Real', 'Fake'],
                            yticklabels=['Real', 'Fake'], ax=ax_cm,
                            annot_kws={"color": _TX}
                        )
                        ax_cm.set_ylabel('True label')
                        ax_cm.set_xlabel('Predicted label')
                        plt.close(fig_cm)

                        # 3. Curva Precision-Recall
                        precision, recall, _ = precision_recall_curve(
                            y_true, y_scores)

                        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                        _dstyle(ax_pr, fig_pr, "Precision-Recall Curve")
                        ax_pr.plot(recall, precision, color='#06b6d4',
                                   lw=2, label='Precision-Recall')
                        ax_pr.set_xlabel('Recall')
                        ax_pr.set_ylabel('Precision')
                        ax_pr.legend(facecolor=_FC, edgecolor=_GR,
                                     labelcolor=_TX, fontsize=9)
                        plt.close(fig_pr)

                        full_logs = "".join(log_cb.logs_hist)

                        # Salvar modelo
                        from app.core.config.settings import PathConfig
                        paths = PathConfig()
                        save_dir = paths.models_dir
                        save_dir.mkdir(parents=True, exist_ok=True)
                        model_path = save_dir / f"{arch}_{int(time.time())}.h5"
                        model.save(model_path)

                        # --- Gráficos Avançados ---
                        fig_det = None
                        fig_thresh = None
                        fig_class_acc = None
                        fig_tsne = None
                        fig_lr = None

                        try:
                            from app.domain.services.forensic_visualization import (
                                TrainingAnalyticsVisualizer,
                            )
                            tv = TrainingAnalyticsVisualizer()

                            # DET Curve + EER
                            fig_det, eer_val = tv.plot_det_curve_eer(
                                y_true, y_scores)

                            # Threshold optimization
                            fig_thresh = tv.plot_threshold_optimization(
                                y_true, y_scores)

                            # Per-class accuracy (compute from history)
                            real_accs = []
                            fake_accs = []
                            for x_b, y_b in val_ds:
                                preds_b = model.predict_on_batch(x_b)
                                pred_labels_b = np.argmax(preds_b, axis=1)
                                y_np = y_b.numpy()
                                real_mask = y_np == 0
                                fake_mask = y_np == 1
                                if np.sum(real_mask) > 0:
                                    real_accs.append(
                                        float(np.mean(
                                            pred_labels_b[real_mask] == 0)))
                                if np.sum(fake_mask) > 0:
                                    fake_accs.append(
                                        float(np.mean(
                                            pred_labels_b[fake_mask] == 1)))
                            if real_accs and fake_accs:
                                fig_class_acc = tv.plot_per_class_accuracy(
                                    real_accs, fake_accs)

                            # t-SNE of penultimate layer
                            try:
                                # Get penultimate layer output
                                penult_model = tf.keras.Model(
                                    inputs=model.input,
                                    outputs=model.layers[-2].output
                                )
                                embeddings = []
                                labels_emb = []
                                for x_b, y_b in val_ds:
                                    emb = penult_model.predict_on_batch(x_b)
                                    embeddings.append(emb)
                                    labels_emb.extend(y_b.numpy())

                                embeddings = np.concatenate(embeddings, axis=0)
                                labels_emb = np.array(labels_emb)

                                fig_tsne = tv.plot_embedding_2d(
                                    embeddings, labels_emb, method="tsne")
                            except Exception as e:
                                logger.warning(f"t-SNE failed: {e}")

                            # LR schedule
                            if lr_cb.lr_history:
                                fig_lr = tv.plot_lr_schedule(lr_cb.lr_history)

                        except Exception as e:
                            logger.warning(
                                f"Advanced training plots failed: {e}")

                        yield (
                            "Concluído",
                            f"{full_logs}\nModelo salvo em: {model_path}",
                            fig_hist, fig_roc, fig_cm, fig_pr,
                            fig_det, fig_thresh, fig_class_acc,
                            fig_tsne, fig_lr
                        )

                    except Exception as e:
                        import traceback
                        yield (
                            "Erro Fatal",
                            f"{str(e)}\n{traceback.format_exc()}",
                            None, None, None, None,
                            None, None, None, None, None
                        )

                dl_train_btn.click(
                    train_dl_wrapper,
                    inputs=[
                        dl_arch,
                        dl_dataset_path,
                        dl_epochs,
                        dl_batch_size,
                        dl_lr,
                        dl_params],
                    outputs=[
                        dl_status,
                        dl_logs,
                        dl_plot,
                        dl_roc_plot,
                        dl_cm_plot,
                        dl_pr_plot,
                        dl_det_plot,
                        dl_thresh_plot,
                        dl_class_acc_plot,
                        dl_tsne_plot,
                        dl_lr_plot]
                )

                gr.Markdown("#### Progresso de Treinamento (API)")
                with gr.Row():
                    api_arch = gr.Textbox(label="Arquitetura", value="AASIST")
                    api_dataset_path = gr.Textbox(
                        label="Dataset (.npz)", value="datasets/sample.npz"
                    )
                    api_model_name = gr.Textbox(
                        label="Nome do Modelo", value="modelo_treinamento"
                    )
                    api_epochs = gr.Number(
                        label="Epochs", value=5, precision=0
                    )
                    api_bs = gr.Number(
                        label="Batch Size", value=8, precision=0
                    )
                with gr.Row():
                    api_key_tr = gr.Textbox(
                        label="API Key", value="", type="password"
                    )
                    start_api_train = gr.Button(
                        "Iniciar Treinamento (API)", variant="primary"
                    )
                with gr.Row():
                    job_id_box = gr.Textbox(label="Job ID", interactive=False)
                    progress_box = gr.Slider(
                        label="Progresso", minimum=0, maximum=100,
                        value=0, interactive=False
                    )
                    status_box = gr.Textbox(label="Status", interactive=False)
                metrics_json = gr.JSON(label="Métricas")

                def start_training_api(
                    arch, dpath, mname, epochs, bs, api_key_val
                ):
                    base = f"http://localhost:{os.getenv('PORT', '7861')}"
                    headers = {"X-API-Key": api_key_val} if api_key_val else {}
                    payload = {
                        "architecture": arch,
                        "dataset_path": dpath,
                        "model_name": mname,
                        "epochs": int(epochs or 0),
                        "batch_size": int(bs or 0),
                        "parameters": {}
                    }
                    r = requests.post(
                        f"{base}/api/v1/training/start",
                        headers=headers,
                        json=payload,
                        timeout=10
                    )
                    if r.status_code == 200:
                        data = r.json()
                        return (
                            data.get("job_id", ""),
                            0,
                            data.get("status", "pending"),
                            {}
                        )
                    return (
                        "",
                        0,
                        "error",
                        {"error": r.text, "status_code": r.status_code}
                    )

                def poll_status(job_id):
                    base = f"http://localhost:{os.getenv('PORT', '7861')}"
                    if not job_id:
                        yield job_id, 0, "no_job", {}
                        return
                    for _ in range(60):
                        r = requests.get(
                            f"{base}/api/v1/training/status/{job_id}",
                            timeout=10
                        )
                        if r.status_code == 200:
                            data = r.json()
                            yield (
                                job_id,
                                int(data.get("progress", 0)),
                                data.get("status", ""),
                                data.get("metrics", {})
                            )
                            if data.get("status") in [
                                "completed", "failed", "error", "not_found"
                            ]:
                                break
                        else:
                            yield (
                                job_id,
                                0,
                                "error",
                                {
                                    "error": r.text,
                                    "status_code": r.status_code
                                }
                            )
                            break
                        time.sleep(1)

                start_api_train.click(
                    start_training_api,
                    inputs=[
                        api_arch, api_dataset_path, api_model_name,
                        api_epochs, api_bs, api_key_tr
                    ],
                    outputs=[
                        job_id_box, progress_box, status_box, metrics_json
                    ]
                ).then(
                    poll_status,
                    inputs=job_id_box,
                    outputs=[
                        job_id_box, progress_box, status_box, metrics_json
                    ]
                )

            # --- TAB CLASSICAL ML ---
            with gr.Tab("Classical ML"):
                gr.Markdown(
                    "Treinamento de modelos clássicos (SVM, Random Forest) "
                    "usando features extraídas."
                )
                with gr.Row():
                    ml_model_type = gr.Dropdown(
                        choices=[
                            "SVM",
                            "RandomForest"],
                        label="Modelo",
                        value="RandomForest")
                    ml_features_path = gr.Textbox(
                        label="Caminho Features Segmentadas",
                        value="datasets/features/segmented")
                    ml_train_btn = gr.Button("Treinar ML", variant="primary")

                with gr.Row():
                    ml_roc_plot = gr.Plot(label="Curva ROC")
                    ml_cm_plot = gr.Plot(label="Matriz de Confusão")
                    ml_pr_plot = gr.Plot(label="Curva Precisão-Recall")

                ml_output = gr.JSON(label="Resultados Detalhados")

                def train_ml_wrapper(model_type, features_path):
                    try:
                        from app.domain.models.architectures.random_forest import (  # noqa
                            RandomForestModel
                        )
                        from app.domain.models.architectures.svm import (
                            SVMModel
                        )
                        import joblib
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        from sklearn.metrics import (
                            roc_curve,
                            auc,
                            # confusion_matrix,
                            precision_recall_curve
                        )
                        import numpy as np

                        path = Path(features_path)
                        if not path.exists():
                            return {
                                "error": (
                                    f"Diretório não encontrado: "
                                    f"{features_path}. "
                                    f"Extraia features primeiro."
                                )
                            }, None, None, None

                        if model_type == "RandomForest":
                            model = RandomForestModel()
                            model_name = "random_forest"
                        else:
                            model = SVMModel()
                            model_name = "svm"

                        # Treinar
                        results = model.train_with_segmented_features(
                            segmented_path=str(path),
                            feature_types=None  # All
                        )

                        # Salvar modelo
                        save_dir = Path("app/models")
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = save_dir / f"{model_name}.pkl"

                        joblib.dump(model, save_path)
                        results['saved_path'] = str(save_path)
                        results['message'] = (
                            f"Modelo salvo com sucesso em {save_path}"
                        )

                        # --- Geração de Gráficos ---
                        y_test = results.get('y_test', [])
                        y_proba = results.get('y_proba', [])

                        fig_roc = None
                        fig_cm = None
                        fig_pr = None

                        if y_test and y_proba:
                            y_test = np.array(y_test)
                            y_proba = np.array(y_proba)

                            # Se for binário e tiver proba para ambas as
                            # classes
                            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                                y_scores = y_proba[:, 1]
                            else:
                                y_scores = y_proba  # Fallback ou se for 1D

                            # 1. Curva ROC
                            fpr, tpr, _ = roc_curve(y_test, y_scores)
                            roc_auc = auc(fpr, tpr)

                            # Dark theme
                            _BG2, _FC2 = "#0f172a", "#1e293b"
                            _TX2, _GR2 = "#f1f5f9", "#334155"

                            def _ds2(ax, fig, title):
                                fig.patch.set_facecolor(_BG2)
                                ax.set_facecolor(_FC2)
                                ax.set_title(title, color=_TX2, fontweight="600", fontsize=12, pad=10)
                                ax.tick_params(colors=_TX2, labelsize=9)
                                for label in (ax.xaxis.label, ax.yaxis.label):
                                    label.set_color(_TX2)
                                    label.set_fontsize(10)
                                for s in ax.spines.values():
                                    s.set_color(_GR2)
                                ax.grid(True, color=_GR2, alpha=0.3, linewidth=0.5)

                            fig_roc, ax_r2 = plt.subplots(figsize=(8, 6))
                            _ds2(ax_r2, fig_roc, "ROC Curve")
                            ax_r2.plot(fpr, tpr, color='#f59e0b', lw=2,
                                       label=f'AUC = {roc_auc:.2f}')
                            ax_r2.plot([0, 1], [0, 1], color=_GR2, lw=1.5, linestyle='--')
                            ax_r2.set_xlim([0.0, 1.0])
                            ax_r2.set_ylim([0.0, 1.05])
                            ax_r2.set_xlabel('False Positive Rate')
                            ax_r2.set_ylabel('True Positive Rate')
                            ax_r2.legend(facecolor=_FC2, edgecolor=_GR2,
                                         labelcolor=_TX2, fontsize=9)
                            plt.close(fig_roc)

                            # 2. Matriz de Confusão
                            cm = np.array(results['confusion_matrix'])
                            classes = results.get('classes', ['Real', 'Fake'])

                            fig_cm, ax_cm2 = plt.subplots(figsize=(8, 6))
                            _ds2(ax_cm2, fig_cm, "Confusion Matrix")
                            sns.heatmap(
                                cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=classes, yticklabels=classes,
                                ax=ax_cm2, annot_kws={"color": _TX2})
                            ax_cm2.set_ylabel('True label')
                            ax_cm2.set_xlabel('Predicted label')
                            plt.close(fig_cm)

                            # 3. Curva Precision-Recall
                            precision, recall, _ = precision_recall_curve(
                                y_test, y_scores)

                            fig_pr, ax_pr2 = plt.subplots(figsize=(8, 6))
                            _ds2(ax_pr2, fig_pr, "Precision-Recall Curve")
                            ax_pr2.plot(recall, precision, color='#06b6d4',
                                        lw=2, label='Precision-Recall')
                            ax_pr2.set_xlabel('Recall')
                            ax_pr2.set_ylabel('Precision')
                            ax_pr2.legend(facecolor=_FC2, edgecolor=_GR2,
                                          labelcolor=_TX2, fontsize=9)
                            plt.close(fig_pr)

                        return results, fig_roc, fig_cm, fig_pr
                    except Exception as e:
                        import traceback
                        return {
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }, None, None, None

                ml_train_btn.click(
                    train_ml_wrapper,
                    inputs=[ml_model_type, ml_features_path],
                    outputs=[ml_output, ml_roc_plot, ml_cm_plot, ml_pr_plot]
                )
