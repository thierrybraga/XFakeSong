import gradio as gr
import os
from pathlib import Path
import json
import time
import threading
import queue
import logging
import io
import zipfile
import shutil

# Configurar logger para capturar logs de treinamento
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
logging.getLogger().addHandler(ch)


def create_training_tab():
    with gr.Tab("Treinamento"):
        gr.Markdown("### 🏋️ Gestão de Datasets e Treinamento")

        with gr.Tabs():
            # --- TAB DATASET ---
            with gr.Tab("Dataset & Features"):
                gr.Markdown("#### 1. Upload e Preparação do Dataset")

                with gr.Row():
                    with gr.Column():
                        zip_file = gr.File(
                            label="Upload Dataset (.zip)", file_types=[".zip"])
                        gr.Markdown(
                            "*O arquivo zip deve conter pastas 'real' e 'fake'.*")
                        upload_btn = gr.Button(
                            "Processar Dataset", variant="primary")

                    with gr.Column():
                        dataset_status = gr.Textbox(
                            label="Status do Dataset", interactive=False)
                        dataset_info = gr.JSON(label="Informações do Dataset")

                gr.Markdown("#### 2. Extração de Features")
                gr.Markdown(
                    "Extrai características (espectrais, temporais, etc.) para treinamento de modelos clássicos.")

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

                        return f"Sucesso! Real: {stats['real']}, Fake: {stats['fake']}", stats

                    except Exception as e:
                        return f"Erro: {str(e)}", {}

                def extract_features_batch():
                    yield "Iniciando extração...", "Preparando..."

                    try:
                        from app.domain.features.extractors.segmented_feature_extractor import SegmentedFeatureExtractor, SegmentedExtractionConfig
                        from app.domain.features.exporters.csv_feature_exporter import CSVExportConfig

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
                            yield "Erro", "Dataset processado não encontrado. Faça upload primeiro."
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
                                msg = f"Processando [{label_name}] {
                                    i + 1}/{total_class}: {
                                    f.name}"
                                logs.append(msg)
                                if len(logs) > 20:
                                    logs.pop(0)  # Manter log curto na UI
                                yield f"Extraindo... {label_name} {i + 1}/{total_class}", "\n".join(logs)

                                extractor.extract_from_file(
                                    str(f), label=label_id)
                                total_files += 1

                        yield "Concluído!", f"Extração finalizada. Total arquivos: {total_files}"

                    except Exception as e:
                        import traceback
                        yield "Erro Fatal", f"{str(e)}\n{traceback.format_exc()}"

                upload_btn.click(
                    process_dataset, inputs=[zip_file], outputs=[
                        dataset_status, dataset_info])
                feature_extract_btn.click(
                    extract_features_batch, outputs=[
                        feature_status, feature_logs])

            # --- TAB DEEP LEARNING ---
            with gr.Tab("Deep Learning"):
                with gr.Row():
                    with gr.Column():
                        from app.domain.models.architectures.registry import architecture_registry, get_architecture_info

                        arch_choices = architecture_registry.list_architectures()
                        dl_arch = gr.Dropdown(
                            choices=arch_choices,
                            label="Arquitetura", value=arch_choices[0] if arch_choices else "MultiscaleCNN"
                        )
                        dl_dataset_path = gr.Textbox(
                            label="Caminho do Dataset (Raiz com pastas 'real' e 'fake')",
                            value="app/datasets"
                        )
                        with gr.Row():
                            dl_epochs = gr.Slider(
                                minimum=1, maximum=100, value=5, step=1, label="Épocas")
                            dl_batch_size = gr.Slider(
                                minimum=2, maximum=128, value=16, step=2, label="Batch Size")

                        dl_lr = gr.Number(
                            value=0.001, label="Learning Rate", precision=5)

                        with gr.Accordion("Opções Avançadas", open=False):
                            dl_params = gr.JSON(
                                label="Hiperparâmetros da Arquitetura (JSON)",
                                value=get_architecture_info(
                                    arch_choices[0]).default_params if arch_choices else {}
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

                def update_arch_params(arch_name):
                    try:
                        from app.domain.models.training.optimized_training_config import load_hyperparameters_json
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
                    yield "Iniciando...", "", None, None, None, None

                    try:
                        import tensorflow as tf
                        from app.domain.models.architectures.registry import create_model_by_name, get_architecture_info
                        import matplotlib.pyplot as plt
                        from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
                        import seaborn as sns
                        import numpy as np

                        # Limpar logs anteriores
                        log_capture_string.truncate(0)
                        log_capture_string.seek(0)

                        base_path = Path(dataset_path)
                        if not base_path.exists():
                            yield "Erro", f"Dataset não encontrado: {dataset_path}", None, None, None, None
                            return

                        # Verificar Requisitos da Arquitetura
                        try:
                            arch_info = get_architecture_info(arch)
                            input_reqs = arch_info.input_requirements
                        except Exception:
                            input_reqs = {}

                        req_type = input_reqs.get("type", "audio")
                        req_format = input_reqs.get("format", "raw")
                        
                        yield "Configurando Pipeline...", f"Arquitetura: {arch}\nTipo Entrada: {req_type} ({req_format})", None, None, None, None

                        # Configurações de Áudio
                        SAMPLE_RATE = 16000
                        # Duração fixa 3s (pode ser ajustado via params da arquitetura futuramente)
                        DURATION = 3 
                        AUDIO_LEN = SAMPLE_RATE * DURATION

                        # 1. Carregar Dataset (Raw Audio)
                        try:
                            train_ds = tf.keras.utils.audio_dataset_from_directory(
                                directory=str(base_path),
                                batch_size=batch_size,
                                validation_split=0.2,
                                subset='training',
                                seed=42,
                                output_sequence_length=AUDIO_LEN,
                                label_mode='int'
                            )
                            val_ds = tf.keras.utils.audio_dataset_from_directory(
                                directory=str(base_path),
                                batch_size=batch_size,
                                validation_split=0.2,
                                subset='validation',
                                seed=42,
                                output_sequence_length=AUDIO_LEN,
                                label_mode='int'
                            )
                        except Exception as e:
                            yield "Erro Dataset", f"Falha ao carregar dataset: {str(e)}", None, None, None, None
                            return

                        # 2. Pré-processamento On-the-Fly (se necessário)
                        input_shape = (AUDIO_LEN, 1)

                        if req_type == "features" and req_format == "spectrogram":
                            yield "Pré-processamento...", "Configurando extração de espectrogramas on-the-fly...", None, None, None, None
                            
                            # Função de transformação para Espectrograma
                            def get_spectrogram(waveform, label):
                                # Remover dimensão extra se existir (B, N, 1) -> (B, N)
                                waveform = tf.squeeze(waveform, axis=-1)
                                
                                # STFT
                                spectrogram = tf.signal.stft(
                                    waveform, frame_length=1024, frame_step=256, fft_length=1024
                                )
                                spectrogram = tf.abs(spectrogram)
                                
                                # Mel Spectrogram (simplificado: usar apenas Spectrogram ou implementar Mel matrix)
                                # Para simplificar e evitar dependência de torchaudio/librosa dentro do grafo TF,
                                # vamos usar o espectrograma de magnitude logarítmica, que funciona bem para CNNs.
                                # Adicionar canal de cor: (Time, Freq, 1)
                                spectrogram = tf.expand_dims(spectrogram, axis=-1)
                                
                                # Resize para garantir dimensão fixa esperada por alguns modelos (ex: 128x128)
                                # Mas AASIST e outros podem esperar dimensões específicas.
                                # Vamos manter a dimensão natural do STFT: 
                                # 48000 samples / 256 step ~= 187 frames
                                # 1024 fft / 2 + 1 = 513 bins
                                # Shape: (187, 513, 1)
                                
                                # Se a arquitetura exige dimensão fixa (ex: 128 mel bins), seria ideal ter a matriz Mel.
                                # Como fallback robusto, vamos fazer resize para algo quadrado padrão se for CNN 2D genérica,
                                # ou manter se a arquitetura lidar com isso.
                                # O registry diz "feature_dim": 80. Isso implica Mel.
                                
                                # Implementação aproximada de Mel-like via resize se não tivermos tf.signal.linear_to_mel_weight_matrix (TF<2.10 ou complexidade)
                                # Vamos fazer resize para (128, 128) para garantir compatibilidade com CNNs 2D padrão
                                spectrogram = tf.image.resize(spectrogram, [128, 128])
                                
                                # Log scaling
                                spectrogram = tf.math.log(spectrogram + 1e-6)
                                
                                return spectrogram, label

                            train_ds = train_ds.map(get_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
                            val_ds = val_ds.map(get_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
                            
                            # Atualizar input_shape
                            input_shape = (128, 128, 1)

                        elif req_type == "audio":
                            # Garantir shape correto (N, 1)
                            def ensure_channel(waveform, label):
                                return waveform, label
                            
                            train_ds = train_ds.map(ensure_channel)
                            val_ds = val_ds.map(ensure_channel)
                            input_shape = (AUDIO_LEN, 1)

                        # Otimizar performance
                        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
                        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

                        # 3. Criar Modelo
                        yield "Criando Modelo...", f"Instanciando {arch} com input_shape={input_shape}...", None, None, None, None
                        
                        try:
                            # Passar model_params como kwargs
                            model = create_model_by_name(
                                arch, input_shape=input_shape, num_classes=2, **model_params)
                        except Exception as e:
                            yield "Erro Modelo", f"Falha ao criar modelo: {str(e)}\nVerifique se a arquitetura suporta o input shape {input_shape}", None, None, None, None
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
                                msg = f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}\n"
                                self.logs_hist.append(msg)

                        log_cb = LogCallback()

                        yield "Treinando...", "Iniciando loop de treinamento...", None, None, None, None

                        history = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=epochs,
                            callbacks=[log_cb]
                        )

                        # Gerar gráfico de Loss/Acc
                        fig_hist, ax = plt.subplots(1, 2, figsize=(12, 4))
                        ax[0].plot(history.history['loss'], label='Train Loss')
                        ax[0].plot(history.history['val_loss'], label='Val Loss')
                        ax[0].set_title('Loss')
                        ax[0].legend()

                        ax[1].plot(history.history['accuracy'], label='Train Acc')
                        ax[1].plot(history.history['val_accuracy'], label='Val Acc')
                        ax[1].set_title('Accuracy')
                        ax[1].legend()

                        # --- Avaliação Detalhada para Plots ---
                        yield "Avaliando...", "Gerando gráficos de performance...", fig_hist, None, None, None

                        y_true_all = []
                        y_pred_probs_all = []

                        # Iterar sobre dataset de validação
                        for x_batch, y_batch in val_ds:
                            preds = model.predict_on_batch(x_batch)
                            y_true_all.extend(y_batch.numpy())
                            y_pred_probs_all.extend(preds)

                        y_true = np.array(y_true_all)
                        y_probs = np.array(y_pred_probs_all)

                        # Preparar scores para ROC/PR
                        if y_probs.shape[1] == 2:
                            y_scores = y_probs[:, 1]
                        else:
                            y_scores = y_probs.flatten()

                        # 1. Curva ROC
                        fpr, tpr, _ = roc_curve(y_true, y_scores)
                        roc_auc = auc(fpr, tpr)

                        fig_roc = plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='darkorange', lw=2,
                                 label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy',
                                 lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic (ROC)')
                        plt.legend(loc="lower right")
                        
                        yield "Concluído", "Treinamento finalizado com sucesso!", fig_hist, fig_roc, None, None

                    except Exception as e:
                        import traceback
                        yield "Erro Fatal", f"{str(e)}\n{traceback.format_exc()}", None, None, None, None
                        plt.close(fig_roc)

                        # 2. Matriz de Confusão
                        cm = confusion_matrix(y_true, y_pred_labels)

                        fig_cm = plt.figure(figsize=(8, 6))
                        sns.heatmap(
                            cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                                'Real', 'Fake'], yticklabels=[
                                'Real', 'Fake'])
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        plt.title('Confusion Matrix')
                        plt.close(fig_cm)

                        # 3. Curva Precision-Recall
                        precision, recall, _ = precision_recall_curve(
                            y_true, y_scores)

                        fig_pr = plt.figure(figsize=(8, 6))
                        plt.plot(recall, precision, color='blue',
                                 lw=2, label='Precision-Recall curve')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Precision-Recall Curve')
                        plt.legend(loc="lower left")
                        plt.close(fig_pr)

                        full_logs = "".join(log_cb.logs_hist)

                        # Salvar modelo
                        from app.core.config.settings import PathConfig
                        paths = PathConfig()
                        save_dir = paths.models_dir
                        save_dir.mkdir(parents=True, exist_ok=True)
                        model_path = save_dir / f"{arch}_{int(time.time())}.h5"
                        model.save(model_path)

                        yield "Concluído", f"{full_logs}\nModelo salvo em: {model_path}", fig_hist, fig_roc, fig_cm, fig_pr

                    except Exception as e:
                        import traceback
                        yield "Erro Fatal", f"{str(e)}\n{traceback.format_exc()}", None, None, None, None

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
                        dl_pr_plot]
                )

            # --- TAB CLASSICAL ML ---
            with gr.Tab("Classical ML"):
                gr.Markdown(
                    "Treinamento de modelos clássicos (SVM, Random Forest) usando features extraídas.")
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
                        from app.domain.models.architectures.random_forest import RandomForestModel
                        from app.domain.models.architectures.svm import SVMModel
                        import joblib
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
                        import numpy as np

                        path = Path(features_path)
                        if not path.exists():
                            return {
                                "error": f"Diretório não encontrado: {features_path}. Extraia features primeiro."}, None, None, None

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
                        results['message'] = f"Modelo salvo com sucesso em {save_path}"

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

                            fig_roc = plt.figure(figsize=(8, 6))
                            plt.plot(fpr, tpr, color='darkorange', lw=2,
                                     label=f'ROC curve (area = {roc_auc:.2f})')
                            plt.plot([0, 1], [0, 1], color='navy',
                                     lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title(
                                'Receiver Operating Characteristic (ROC)')
                            plt.legend(loc="lower right")
                            plt.close(fig_roc)

                            # 2. Matriz de Confusão
                            cm = np.array(results['confusion_matrix'])
                            classes = results.get('classes', ['Real', 'Fake'])

                            fig_cm = plt.figure(figsize=(8, 6))
                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt='d',
                                cmap='Blues',
                                xticklabels=classes,
                                yticklabels=classes)
                            plt.ylabel('True label')
                            plt.xlabel('Predicted label')
                            plt.title('Confusion Matrix')
                            plt.close(fig_cm)

                            # 3. Curva Precision-Recall
                            precision, recall, _ = precision_recall_curve(
                                y_test, y_scores)

                            fig_pr = plt.figure(figsize=(8, 6))
                            plt.plot(recall, precision, color='blue',
                                     lw=2, label='Precision-Recall curve')
                            plt.xlabel('Recall')
                            plt.ylabel('Precision')
                            plt.title('Precision-Recall Curve')
                            plt.legend(loc="lower left")
                            plt.close(fig_pr)

                        return results, fig_roc, fig_cm, fig_pr
                    except Exception as e:
                        import traceback
                        return {"error": str(
                            e), "traceback": traceback.format_exc()}, None, None, None

                ml_train_btn.click(
                    train_ml_wrapper,
                    inputs=[ml_model_type, ml_features_path],
                    outputs=[ml_output, ml_roc_plot, ml_cm_plot, ml_pr_plot]
                )
