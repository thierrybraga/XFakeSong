"""Treino interativo do assistente: dataset → modelo → treino → avaliação.

Saiu da aba ``app/interfaces/gradio/tabs/training_wizard.py`` (P1-4 do plano
de refatoração): a aba tinha ~1.500 linhas de carga de dados, front-end,
treino, avaliação e persistência misturadas com HTML e gráficos. Aqui fica só
o pipeline, sem Gradio:

- :meth:`InteractiveTrainingService.train` é um gerador de EVENTOS (as
  dataclasses abaixo); a aba os traduz em painel, log e gráficos;
- o progresso fino vai por callback (``on_progress(fração, descrição)``);
- a parada vem de um ``threading.Event`` que a aba controla — o Keras só
  interrompe no fim da época, com o modelo em estado consistente;
- :func:`save_trained_model` grava ``.keras``/``.pkl`` + ``_config.json``.

O pipeline neural é o mesmo de antes, linha a linha: binarização de rótulos
(BUG.Training.4), front-end LFCC compartilhado com a inferência, treino em
float32 com ``clipvalue``, teste de fumaça antes da 1ª época e ``fit`` numa
thread para o feedback por época.
"""

from __future__ import annotations

import json
import logging
import queue as _queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from app.core.performance import optimize_tf_dataset

logger = logging.getLogger(__name__)

#: Subpastas que começam assim são REAIS; as demais, FAKE (BUG.Training.4).
REAL_CLASS_PATTERN = re.compile(
    r"^(real|bonafide|genuine|original|authentic|natural)",
    re.IGNORECASE,
)

#: Subpastas de ``data/datasets/`` que NÃO são classes (caches, splits...).
NON_CLASS_DIRS = frozenset(
    {
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
)

# Conjunto de features tabulares — PINADO no input_contract para garantir
# que a inferência (FeaturePreparer) extraia exatamente as mesmas.
# 'spectral' + 'cepstral' (MFCC/LFCC) + 'temporal' são rápidas e robustas;
# 'prosodic' é evitado (pyin é lento).
CLASSICAL_FEATURE_TYPES = ["spectral", "cepstral", "temporal"]
CLASSICAL_AGGREGATE = "mean"
_CLASSICAL_ARCHS = {"svm", "random forest", "randomforest", "rf"}

ProgressCallback = Callable[[float, str], None]


def _sem_progresso(_fracao: float, _descricao: str) -> None:
    return None


# ─────────────────────────────── resultado ───────────────────────────────


@dataclass
class TrainedModel:
    """Modelo recém-treinado, ainda em memória, pronto para ``save_trained_model``.

    ``input_contract`` registra o pré-processamento EXATO do treino para que a
    inferência reproduza as mesmas features.
    """

    kind: str  # "keras" ou "sklearn"
    model: Any
    arch: str
    input_shape: tuple
    input_contract: Dict[str, Any]
    val_acc: Optional[float]
    val_loss: Optional[float]
    epochs: Optional[int]
    scaler: Any = None
    feature_names: Optional[List[str]] = None


@dataclass(frozen=True)
class SavedModel:
    slug: str
    model_path: Path
    config_path: Path
    size_mb: float


# ──────────────────────────────── eventos ────────────────────────────────


@dataclass(frozen=True)
class DatasetLoading:
    arch: str
    epochs: int
    batch_size: int


@dataclass(frozen=True)
class DatasetReady:
    input_type: str
    input_shape: tuple
    class_names: List[str]
    binary_map: Dict[int, int]


@dataclass(frozen=True)
class SmokeTestFailed:
    """O lote de teste antes da 1ª época produziu NaN/Inf; o treino não começa."""

    message: str


@dataclass(frozen=True)
class FitStarted:
    arch: str
    device: str
    epochs: int
    n_params: int
    n_train_batches: int
    n_val_batches: int
    class_weight: Optional[Dict[int, float]]
    history: Dict[str, list]


@dataclass(frozen=True)
class FitHeartbeat:
    """Nenhuma época nova no último segundo — mantém o relógio da UI vivo."""

    epochs: int
    epochs_done: int
    elapsed: float
    history: Dict[str, list]


@dataclass(frozen=True)
class EpochCompleted:
    epoch: int
    epochs: int
    metrics: Dict[str, Any]  # loss, acc, val_loss, val_acc da época
    history: Dict[str, list]
    elapsed: float


@dataclass(frozen=True)
class FitCompleted:
    trained: TrainedModel
    history: Any  # keras.callbacks.History
    train_acc_key: Optional[str]
    y_true: Any
    y_pred: Any
    y_score: Any
    lr_history: List[float]
    final_acc: float
    final_loss: float
    live_history: Dict[str, list]
    device: str
    epochs: int
    elapsed: float


@dataclass(frozen=True)
class ClassicalStarted:
    arch: str
    dataset_path: str


@dataclass(frozen=True)
class ClassicalDatasetInsufficient:
    arch: str
    n_real: int
    n_fake: int
    total: int
    elapsed: float


@dataclass(frozen=True)
class ClassicalExtractionProgress:
    arch: str
    done: int
    total: int
    valid: int
    skipped: int
    elapsed: float


@dataclass(frozen=True)
class ClassicalTooFewFeatures:
    arch: str
    valid: int
    elapsed: float


@dataclass(frozen=True)
class ClassicalFitting:
    arch: str
    n_samples: int
    n_features: int
    elapsed: float


@dataclass(frozen=True)
class ClassicalCompleted:
    trained: TrainedModel
    is_random_forest: bool
    y_val: Any
    y_pred: Any
    y_score: Any
    confusion: Any
    feature_importances: Any
    train_class_counts: Any
    n_samples: int
    n_features: int
    n_real: int
    n_fake: int
    skipped: int
    val_acc: float
    elapsed: float


@dataclass(frozen=True)
class TrainingFailed:
    error: Exception
    classical: bool = False


# ─────────────────────── helpers de histórico e avaliação ───────────────────────


def history_series(history, *keys):
    """Retorna a primeira série não vazia encontrada no Keras history."""
    data = getattr(history, "history", history) or {}
    for key in keys:
        values = data.get(key)
        if values:
            return values
    return []


def resolve_accuracy_history_keys(history, preferred_train: str = None):
    """Resolve chaves de accuracy gravadas pelo Keras para qualquer arquitetura."""
    data = getattr(history, "history", history) or {}
    candidates = []
    if preferred_train:
        candidates.append(preferred_train)
    candidates.extend(
        [
            "accuracy",
            "binary_accuracy",
            "categorical_accuracy",
            "sparse_categorical_accuracy",
            "acc",
        ]
    )
    for key in candidates:
        if data.get(key):
            val_key = f"val_{key}"
            if data.get(val_key):
                return key, val_key
            return key, None
    for key in data:
        if key.startswith("val_"):
            continue
        if "acc" in key.lower() or "accuracy" in key.lower():
            val_key = f"val_{key}"
            return key, val_key if data.get(val_key) else None
    return None, None


def normalize_true_labels(labels):
    """Converte labels Keras/sparse/one-hot/(B,1) para vetor binário int."""
    import numpy as np

    arr = np.asarray(labels)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        arr = np.argmax(arr, axis=-1)
    return np.asarray(arr).reshape(-1).astype("int32")


def prediction_labels_and_scores(predictions):
    """Converte saída sigmoid/softmax/logits em labels e score da classe fake."""
    import numpy as np

    pred = np.asarray(predictions, dtype="float32")
    if pred.ndim == 1 or (pred.ndim > 1 and pred.shape[-1] == 1):
        scores = pred.reshape(-1)
        if scores.size and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))
        labels = (scores >= 0.5).astype("int32")
        return labels, scores.astype("float32")

    labels = np.argmax(pred, axis=-1).astype("int32")
    scores_src = pred
    row_sums = np.sum(scores_src, axis=-1)
    if scores_src.size and (
        float(scores_src.min()) < 0.0
        or float(scores_src.max()) > 1.0
        or not np.allclose(row_sums, 1.0, atol=1e-3)
    ):
        scores_src = scores_src - np.max(scores_src, axis=-1, keepdims=True)
        exp = np.exp(scores_src)
        scores_src = exp / np.sum(exp, axis=-1, keepdims=True)
    scores = scores_src[:, 1] if scores_src.shape[-1] > 1 else labels.astype("float32")
    return labels, scores.astype("float32")


def collect_validation_predictions(model, val_ds):
    """Coleta y_true, y_pred e y_score do dataset de validação."""
    import numpy as np

    y_true_all = []
    y_pred_all = []
    y_score_all = []
    for x_batch, y_batch in val_ds:
        preds = model.predict_on_batch(x_batch)
        y_true_all.extend(normalize_true_labels(y_batch.numpy()).tolist())
        labels, scores = prediction_labels_and_scores(preds)
        y_pred_all.extend(labels.tolist())
        y_score_all.extend(scores.tolist())
    return (
        np.asarray(y_true_all, dtype="int32"),
        np.asarray(y_pred_all, dtype="int32"),
        np.asarray(y_score_all, dtype="float32"),
    )


def list_class_files(dataset_path: str):
    """Lista (filepath, label) das subpastas de classe, ignorando dirs espúrios.

    Mesma filtragem do caminho DL: ignora raw/, splits/, features/, etc.
    Label: 0 = real, 1 = fake (convenção do projeto).
    """
    root = Path(dataset_path)
    items = []
    if not root.exists():
        return items
    for sub in sorted(root.iterdir()):
        if (
            not sub.is_dir()
            or sub.name.startswith(".")
            or sub.name.lower() in NON_CLASS_DIRS
        ):
            continue
        label = 0 if REAL_CLASS_PATTERN.match(sub.name.strip()) else 1
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            for f in sub.glob(ext):
                items.append((f, label))
    return items


def slugify_model_name(name: str) -> str:
    """Normaliza o nome do modelo para um nome de arquivo seguro."""
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "modelo"


def save_trained_model(
    trained: TrainedModel, name: str, models_dir: Path
) -> SavedModel:
    """Persiste o modelo em ``models_dir`` (.keras|.pkl + _config.json).

    O config inclui o input_contract para que a INFERÊNCIA reproduza
    exatamente as features usadas no treino.

    Raises:
        FileExistsError: já existe modelo com esse nome (nada é sobrescrito).
    """
    slug = slugify_model_name(name)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    # Keras → .keras ; sklearn (SVM/RF) → .pkl (+ _scaler.pkl)
    suffix = ".pkl" if trained.kind == "sklearn" else ".keras"
    model_path = models_dir / f"{slug}{suffix}"
    config_path = models_dir / f"{slug}_config.json"
    if model_path.exists() or config_path.exists():
        raise FileExistsError(slug)

    if trained.kind == "sklearn":
        import joblib

        joblib.dump(trained.model, str(model_path))
        if trained.scaler is not None:
            joblib.dump(trained.scaler, str(models_dir / f"{slug}_scaler.pkl"))
    else:
        # Artefato de inferência sem o estado do otimizador (~3× menor, load
        # mais rápido, saída idêntica). Ver artifacts.save_inference_keras.
        from app.domain.models.training.artifacts import save_inference_keras

        save_inference_keras(trained.model, model_path)

    config = {
        "architecture": trained.arch,
        "input_shape": list(trained.input_shape),
        "input_contract": trained.input_contract,
        "trained_by": "wizard",
        "model_type": "sklearn" if trained.kind == "sklearn" else "tensorflow",
        "feature_names": trained.feature_names,
        "val_accuracy": trained.val_acc,
        "val_loss": trained.val_loss,
        "epochs": trained.epochs,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Modelo salvo: {model_path} ({size_mb:.1f} MB)")
    return SavedModel(slug, model_path, config_path, size_mb)


# ─────────────────────────────── serviço ───────────────────────────────


class InteractiveTrainingService:
    """Treino do assistente, independente da interface que o exibe.

    Args:
        feature_service_factory: devolve o extrator de features do caminho
            clássico. A composição entrega o do ``DetectionService`` — mesma
            instância da inferência, mesmo contrato.
        hyperparameter_plan: ``arch -> hiperparâmetros planejados``. A
            composição entrega ``benchmarks.planning.effective_hyperparameters``;
            o domínio não importa o harness.
    """

    def __init__(
        self,
        feature_service_factory: Optional[Callable[[], Any]] = None,
        hyperparameter_plan: Optional[Callable[[str], Mapping[str, Any]]] = None,
    ) -> None:
        self._feature_service_factory = feature_service_factory
        self._hyperparameter_plan = hyperparameter_plan

    def train(
        self,
        arch: str,
        dataset_path: str,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        use_class_weighting: bool,
        stop_event: Optional[threading.Event] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Iterator[Any]:
        """Treina e gera os eventos de progresso, até o resultado ou a falha."""
        report = on_progress or _sem_progresso
        stop_event = stop_event or threading.Event()
        # Marcador de versão: se você NÃO vê "[treino v3: float32+clipvalue]"
        # nos logs/UI, o app está rodando código ANTIGO — reinicie.
        logger.info(
            "[treino v3: float32+clipvalue] iniciando treino interativo "
            "(anti-NaN robusto)"
        )
        # Um pedido de parada pendente de um treino anterior nao pode matar este.
        stop_event.clear()
        report(0.05, "Carregando dataset...")
        yield DatasetLoading(arch=arch, epochs=epochs, batch_size=batch_size)

        # SVM e Random Forest são modelos clássicos (scikit-learn) que operam
        # em FEATURES TABULARES, não em espectrogramas/áudio bruto. Têm um
        # pipeline dedicado que extrai as MESMAS features que a inferência usa.
        if arch.strip().lower() in _CLASSICAL_ARCHS:
            yield from _train_classical(
                arch,
                dataset_path,
                feature_service_factory=self._feature_service_factory,
                report=report,
            )
            return
        yield from _train_neural(
            arch,
            dataset_path,
            epochs,
            batch_size,
            lr,
            use_class_weighting,
            hyperparameter_plan=self._hyperparameter_plan,
            stop_event=stop_event,
            report=report,
        )


# ─────────────────────────── pipeline neural ───────────────────────────


def _train_neural(
    arch: str,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    use_class_weighting: bool,
    *,
    hyperparameter_plan: Optional[Callable[[str], Mapping[str, Any]]],
    stop_event: threading.Event,
    report: ProgressCallback,
) -> Iterator[Any]:
    """Pipeline Keras do assistente (veja o docstring do módulo)."""
    import numpy as np
    import tensorflow as tf

    # Guardará a política de precisão anterior para restaurar no finally.
    _saved_policy = None

    try:
        from app.domain.models.architectures.factory import (
            architecture_factory_registry,
            create_model_by_name,
        )

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

        # BUG FIX: o usuário costuma apontar para `data/datasets/`, que contém
        # subpastas que NÃO são classes: `raw/` (caches de download),
        # `splits/` (duplica real+fake já divididos), `features/`, etc.
        # audio_dataset_from_directory trata CADA subpasta como uma classe →
        # "Found N files belonging to 4 classes" → labels poluídos + dados
        # duplicados (splits/ repete real/ e fake/). Filtramos para usar apenas
        # subpastas que parecem classes de áudio reais.
        from pathlib import Path as _Path

        _EXCLUDE_DIRS = NON_CLASS_DIRS
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

        _ds_common = {
            "directory": dataset_path,
            "batch_size": batch_size,
            "validation_split": 0.2,
            "seed": 42,
            "output_sequence_length": AUDIO_LEN,
            "label_mode": "int",
        }
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
            i: 0 if REAL_CLASS_PATTERN.match(n.strip()) else 1
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

        # Tier-1 perf: separa FEATURIZAÇÃO (cara: STFT/resample — determinística,
        # cacheável) de AUGMENTATION (estocástica). Cacheamos a featurização para
        # NÃO recalculá-la a cada época, e aplicamos a augmentation DEPOIS do
        # cache, para que RawBoost/SpecAugment continuem variando por época
        # (cachear após a augmentation a congelaria → pioraria a generalização).
        if input_type == "raw_audio":
            _feat_prep = _prep_raw

            def _augment(a, lab):
                return rawboost_tf(a, sr=SAMPLE_RATE, algo=4, p=0.7), lab

        else:
            _feat_prep = _prep_spec

            def _augment(s, lab):
                return spec_augment_tf(s, p=0.7), lab

        train_ds = train_ds.map(_feat_prep, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(_feat_prep, num_parallel_calls=tf.data.AUTOTUNE)

        # Cache em memória só para datasets pequenos/médios (guard de OOM):
        # cardinalidade finita e ≲ 4000 amostras. Datasets grandes seguem sem
        # cache (comportamento anterior, sem risco de estourar a RAM).
        def _maybe_cache(ds):
            try:
                card = int(tf.data.experimental.cardinality(ds))
                if 0 < card * int(batch_size) <= 4000:
                    return ds.cache()
            except Exception:
                pass
            return ds

        train_ds = _maybe_cache(train_ds)  # featurização cacheada (sem aug)
        val_ds = _maybe_cache(val_ds)  # val não tem augmentation → seguro

        # Augmentation DEPOIS do cache → re-randomiza a cada época.
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = optimize_tf_dataset(train_ds, cache=False, prefetch=True)
        val_ds = optimize_tf_dataset(val_ds, cache=False, prefetch=True)

        # Computa shape final inspecionando 1 batch (sem consumir o ds em cache)
        for sample_x, _ in train_ds.take(1):
            input_shape = tuple(sample_x.shape[1:])  # ignora dim batch
            break

        report(0.15, f"Criando modelo {arch}...")
        yield DatasetReady(
            input_type=input_type,
            input_shape=input_shape,
            class_names=class_names,
            binary_map=binary_map,
        )

        # HIPERPARÂMETROS DA ARQUITETURA — resolvidos pelo MESMO caminho do
        # benchmark, não deixados no default do registry.
        #
        # Até 2026-08-20 esta chamada não passava parâmetro nenhum: o assistente
        # construía sempre a topologia de `registry.default_params`, mesmo com o
        # painel de hiperparâmetros exibindo os valores do plano. Para o RawNet2
        # isso produzia a arquitetura ERRADA — o "Improved RawNet" de
        # verificação de locutor (~7,0M parâmetros) com o rótulo "RawNet2" que o
        # TCC usa para o baseline anti-spoofing de Tak et al. (~17,6M) — e o
        # modelo salvo entrava em `data/models/` indistinguível do outro.
        #
        # O FILTRO é o `default_params` da própria arquitetura, não uma lista de
        # exclusão. O plano mistura três naturezas de chave (topologia, regime
        # de treino e metadado como `notes`/`model_family`), e vários
        # construtores aceitam `**kwargs` — passar uma chave estranha levanta
        # TypeError. O registry declara exatamente o que cada `create_model`
        # aceita, então intersectar com ele dá o subconjunto que vai ao
        # construtor; o plano só sobrescreve o VALOR das chaves que já existem
        # lá. Chaves de treino puro (`patience`, `gradient_clip`) saem pelo
        # `_TRAINING_ONLY_PARAM_KEYS`.
        parametros_arquitetura: dict = {}
        try:
            from app.domain.models.architectures.registry import (
                _TRAINING_ONLY_PARAM_KEYS,
                architecture_registry,
            )

            # O plano vem do harness (`benchmarks.planning`) pela composição;
            # o domínio não o importa.
            plano = hyperparameter_plan(arch) if hyperparameter_plan else {}
            _aceitos = set(
                architecture_registry.get_architecture_by_any_name(arch).default_params
                or {}
            ) - set(_TRAINING_ONLY_PARAM_KEYS)
            parametros_arquitetura = {
                chave: valor
                for chave, valor in (plano or {}).items()
                if valor is not None and chave in _aceitos
            }
        except Exception as exc:  # noqa: BLE001 — o treino não pode quebrar por isto
            logger.warning(
                "Hiperparâmetros do plano indisponíveis para %s (%s); "
                "usando os defaults do registry.",
                arch,
                exc,
            )

        model = create_model_by_name(
            arch,
            input_shape=input_shape,
            num_classes=2,
            **parametros_arquitetura,
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
            # Erro de smoke test → treino abortado com mensagem clara
            logger.error(str(ve))
            yield SmokeTestFailed(message=str(ve))
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
                # Cancelamento: o Keras nao expoe interrupcao, mas respeita
                # `stop_training` ao fim da epoca. O treino para no proximo
                # limite de epoca, com o modelo em estado consistente — nao
                # no meio de um passo de gradiente.
                if stop_event.is_set():
                    self.model.stop_training = True
                    log_lines.append(
                        f"[INTERROMPIDO] parada solicitada na epoca {epoch + 1}"
                    )
                    return
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
                report(pct, line[:60])

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
        # Chaves de métrica reais (callback usa exatamente _acc_key/_val_acc_key)
        hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        epoch_q: "_queue.Queue" = _queue.Queue()
        lr_history: List[float] = []

        class _LiveQueueCb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                try:
                    lr = float(
                        tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                    )
                    lr_history.append(lr)
                except Exception:
                    pass
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

        report(0.2, f"Treinando {arch} em {train_device}...")
        yield FitStarted(
            arch=arch,
            device=train_device,
            epochs=epochs,
            n_params=int(model.count_params()),
            n_train_batches=_n_train,
            n_val_batches=_n_val,
            class_weight=class_weight,
            history=hist,
        )

        while worker.is_alive() or not epoch_q.empty():
            try:
                ev = epoch_q.get(timeout=1.0)
            except _queue.Empty:
                # Heartbeat (a cada 1s): mantém a UI VIVA mesmo durante a 1ª
                # época (BUG FIX "trava nos 20%": na CPU ela leva minutos).
                yield FitHeartbeat(
                    epochs=epochs,
                    epochs_done=len(hist["loss"]),
                    elapsed=time.time() - _t0,
                    history=hist,
                )
                continue

            # Nova época concluída
            for k in ("loss", "acc", "val_loss", "val_acc"):
                hist[k].append(ev.get(k))
            ep = ev["epoch"]
            report(0.2 + 0.75 * ep / epochs, f"Época {ep}/{epochs}")
            yield EpochCompleted(
                epoch=ep,
                epochs=epochs,
                metrics=dict(ev),
                history=hist,
                elapsed=time.time() - _t0,
            )

        worker.join()
        if fit_box["error"] is not None:
            raise fit_box["error"]
        history = fit_box["history"]

        # Plot final (canônico, a partir do history completo). Resolve chaves de
        # métrica para qualquer saída de arquitetura: sigmoid usa binary_accuracy;
        # softmax pode registrar accuracy/sparse_categorical_accuracy etc.
        train_acc_key, val_acc_key = resolve_accuracy_history_keys(history, _acc_key)
        report(0.97, "Gerando matriz de confusão...")
        y_true_val, y_pred_val, _y_score_val = collect_validation_predictions(
            model, val_ds
        )

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

        # input_contract registra o pré-processamento EXATO usado no treino,
        # garantindo que a inferência reproduza as mesmas features.
        trained = TrainedModel(
            kind="keras",
            model=model,
            arch=arch,
            input_shape=input_shape,
            input_contract={
                "input_type": input_type,
                "n_fft": N_FFT,
                "hop_length": HOP,
                "n_mels": N_MELS,
                # Front-end espectral usado no treino (default LFCC). A
                # inferência lê este campo para reproduzir o mesmo front-end.
                "feature_frontend": FRONTEND,
                "n_lfcc": N_LFCC,
                "sample_rate": SAMPLE_RATE,
                "audio_len": AUDIO_LEN,
            },
            val_acc=float(final_acc) if final_acc == final_acc else None,
            val_loss=float(final_loss) if final_loss == final_loss else None,
            epochs=epochs,
        )
        yield FitCompleted(
            trained=trained,
            history=history,
            train_acc_key=train_acc_key,
            y_true=y_true_val,
            y_pred=y_pred_val,
            y_score=_y_score_val,
            lr_history=lr_history,
            final_acc=final_acc,
            final_loss=final_loss,
            live_history=hist,
            device=train_device,
            epochs=epochs,
            elapsed=time.time() - _t0,
        )

    except Exception as e:
        logger.error(f"Erro no treino: {e}", exc_info=True)
        yield TrainingFailed(error=e)
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


# ────────────────────────── pipeline clássico ──────────────────────────


def _train_classical(
    arch: str,
    dataset_path: str,
    *,
    feature_service_factory: Optional[Callable[[], Any]],
    report: ProgressCallback,
) -> Iterator[Any]:
    """Treina SVM / Random Forest com features tabulares.

    Extrai EXATAMENTE as mesmas features que a inferência usará
    (extract_segmented_features + ExtractionConfig pinado no contrato), treina
    o estimador sklearn e calcula a acurácia de validação.
    """
    import time as _time

    import numpy as np

    is_rf = arch.strip().lower() in ("random forest", "randomforest", "rf")
    nice = "Random Forest" if is_rf else "SVM"
    _t0 = _time.time()

    yield ClassicalStarted(arch=nice, dataset_path=dataset_path)

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        from app.core.contracts.audio import AudioData, FeatureType
        from app.domain.services.feature_extraction_service import (
            AudioFeatureExtractionService,
            ExtractionConfig,
        )

        files = list_class_files(dataset_path)
        n_real = sum(1 for _, y in files if y == 0)
        n_fake = sum(1 for _, y in files if y == 1)
        if len(files) < 10 or n_real == 0 or n_fake == 0:
            yield ClassicalDatasetInsufficient(
                arch=nice,
                n_real=n_real,
                n_fake=n_fake,
                total=len(files),
                elapsed=_time.time() - _t0,
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

        # A composição entrega o extrator do DetectionService (mesma instância
        # da inferência → contrato idêntico).
        feat_svc = (
            feature_service_factory()
            if feature_service_factory
            else AudioFeatureExtractionService()
        )

        cfg = ExtractionConfig(
            feature_types=[FeatureType(t) for t in CLASSICAL_FEATURE_TYPES],
            normalize=True,
            aggregate_method=CLASSICAL_AGGREGATE,
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
                report(0.05 + 0.70 * pct, f"Características {i + 1}/{total}")
                yield ClassicalExtractionProgress(
                    arch=nice,
                    done=i + 1,
                    total=total,
                    valid=len(X),
                    skipped=skipped,
                    elapsed=_time.time() - _t0,
                )

        if len(X) < 10:
            yield ClassicalTooFewFeatures(
                arch=nice, valid=len(X), elapsed=_time.time() - _t0
            )
            return

        X = np.vstack(X)
        y = np.asarray(y, dtype=int)
        n_features = X.shape[1]

        # ── Split + scaler + fit ──
        report(0.80, "Treinando estimador…")
        yield ClassicalFitting(
            arch=nice,
            n_samples=int(X.shape[0]),
            n_features=int(n_features),
            elapsed=_time.time() - _t0,
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
        if hasattr(est, "predict_proba"):
            y_score = est.predict_proba(X_va_s)[:, 1]
        else:
            y_score = y_pred.astype("float32")
        cm = confusion_matrix(y_va, y_pred, labels=[0, 1])

        trained = TrainedModel(
            kind="sklearn",
            model=est,
            arch=nice,
            input_shape=(n_features,),
            input_contract={
                "format": "tabular",
                "feature_types": CLASSICAL_FEATURE_TYPES,
                "aggregate_method": CLASSICAL_AGGREGATE,
                "sample_rate": 16000,
                "scaler_applied": True,
                "n_features": n_features,
            },
            val_acc=val_acc,
            val_loss=None,
            epochs=None,
            scaler=scaler,
            feature_names=feat_names,
        )
        yield ClassicalCompleted(
            trained=trained,
            is_random_forest=is_rf,
            y_val=y_va,
            y_pred=y_pred,
            y_score=y_score,
            confusion=cm,
            feature_importances=(
                est.feature_importances_
                if is_rf and hasattr(est, "feature_importances_")
                else None
            ),
            train_class_counts=np.bincount(y_tr, minlength=2),
            n_samples=int(X.shape[0]),
            n_features=int(n_features),
            n_real=n_real,
            n_fake=n_fake,
            skipped=skipped,
            val_acc=val_acc,
            elapsed=_time.time() - _t0,
        )

    except Exception as e:
        logger.error(f"Erro no treino clássico: {e}", exc_info=True)
        yield TrainingFailed(error=e, classical=True)
