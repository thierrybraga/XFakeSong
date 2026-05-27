"""Smoke test do pipeline completo do training_wizard.

Cria um dataset fake mínimo (2 classes real/fake com 4 áudios cada) e
executa _run_training por 1 epoch em cada uma das 12 arquiteturas
para validar que:
- Detecção de input_type funciona (raw_audio vs spectrogram)
- Pré-processamento on-the-fly produz shape correto
- Modelo aceita o shape e treina 1 epoch sem erro

Uso:
    python scripts/tests/test_wizard_pipeline.py
    python scripts/tests/test_wizard_pipeline.py --models AASIST,Conformer
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))  # scripts/tests/ → scripts/ → raiz
sys.path.insert(0, PROJECT_ROOT)

# Fake packages para evitar app/__init__ chain (sqlalchemy)
import importlib.util  # noqa: E402
import types  # noqa: E402

for fake_pkg in [
    "app", "app.domain", "app.domain.models", "app.domain.models.architectures",
]:
    if fake_pkg not in sys.modules:
        m = types.ModuleType(fake_pkg)
        m.__path__ = [os.path.join(PROJECT_ROOT, *fake_pkg.split("."))]
        sys.modules[fake_pkg] = m

SAMPLE_RATE = 16000
DURATION = 3
N_FFT = 512
HOP = 128
N_MELS = 80


def make_fake_dataset(root: str, n_per_class: int = 4):
    """Cria 8 .wav (4 real + 4 fake) de 3s @ 16kHz."""
    import soundfile as sf
    real_dir = os.path.join(root, "real")
    fake_dir = os.path.join(root, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n_per_class):
        # Real: senoide pura
        t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * (200 + i * 30) * t)
        sf.write(os.path.join(real_dir, f"r_{i}.wav"), audio, SAMPLE_RATE)
        # Fake: ruído + senoide modulada
        audio = (
            0.4 * np.sin(2 * np.pi * (400 + i * 50) * t)
            + 0.1 * rng.standard_normal(t.shape).astype(np.float32)
        )
        sf.write(os.path.join(fake_dir, f"f_{i}.wav"), audio, SAMPLE_RATE)


def test_one(arch: str, dataset_path: str) -> tuple[bool, str]:
    """Executa o pipeline básico para 1 arquitetura."""
    import tensorflow as tf

    import importlib
    factory_mod = importlib.import_module(
        "app.domain.models.architectures.factory"
    )

    try:
        spec = factory_mod.architecture_factory_registry.get_architecture_info(arch)
        input_type = spec.input_requirements.get("input_type", "spectrogram")

        train_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=dataset_path,
            batch_size=2,
            validation_split=0.5,  # 50/50 com 8 arquivos = 4/4
            subset="training",
            seed=42,
            output_sequence_length=SAMPLE_RATE * DURATION,
            label_mode="int",
        )

        # Binarização (mesma do wizard)
        class_names = list(train_ds.class_names)
        binary_map = {
            i: 0 if "real" in n.lower() else 1
            for i, n in enumerate(class_names)
        }
        _table = tf.constant(
            [binary_map[i] for i in range(len(class_names))],
            dtype=tf.int32,
        )

        # Mel matrix
        n_freq = N_FFT // 2 + 1
        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS,
            num_spectrogram_bins=n_freq,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=0.0,
            upper_edge_hertz=SAMPLE_RATE / 2,
        )

        def _to_log_mel(audio):
            if audio.shape.rank == 3:
                audio = tf.squeeze(audio, axis=-1)
            stft = tf.signal.stft(
                audio, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT,
                window_fn=tf.signal.hann_window, pad_end=True,
            )
            mag = tf.abs(stft)
            mel = tf.tensordot(mag, mel_w, axes=1)
            log_mel = tf.math.log(mel + 1e-6)
            return tf.expand_dims(log_mel, axis=-1)

        def _prep_raw(audio, label):
            if audio.shape.rank == 2:
                audio = tf.expand_dims(audio, axis=-1)
            return audio, tf.gather(_table, label)

        def _prep_spec(audio, label):
            return _to_log_mel(audio), tf.gather(_table, label)

        prep_fn = _prep_raw if input_type == "raw_audio" else _prep_spec
        train_ds = train_ds.map(prep_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

        # Descobre shape
        for sample_x, _ in train_ds.take(1):
            input_shape = tuple(sample_x.shape[1:])
            break

        model = factory_mod.create_model_by_name(
            arch, input_shape=input_shape, num_classes=2,
        )

        # Detecta out_units para escolher loss apropriada
        out_units = model.output_shape[-1]
        if out_units == 1:
            chosen_loss = "binary_crossentropy"
            chosen_metric = "binary_accuracy"
            train_ds = train_ds.map(
                lambda x, y: (x, tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            chosen_loss = "sparse_categorical_crossentropy"
            chosen_metric = "accuracy"

        # Re-compile com loss apropriada
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=chosen_loss,
            metrics=[chosen_metric],
        )

        # 1 epoch (smoke)
        hist = model.fit(train_ds, epochs=1, verbose=0)
        loss = hist.history["loss"][-1]
        acc = hist.history.get(chosen_metric, [0])[-1]
        return True, (
            f"OK input_type={input_type} shape={input_shape} "
            f"out={out_units} loss_fn={chosen_loss} loss={loss:.3f} acc={acc:.3f}"
        )
    except Exception as e:
        return False, f"FAIL: {type(e).__name__}: {str(e)[:200]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="CSV de modelos (vazio = todos)")
    args = parser.parse_args()

    all_models = [
        "AASIST", "RawGAT-ST", "RawNet2", "WavLM", "HuBERT",
        "Sonic Sleuth", "EfficientNet-LSTM", "MultiscaleCNN",
        "SpectrogramTransformer", "Conformer", "Hybrid CNN-Transformer", "Ensemble",
    ]
    targets = (
        [m.strip() for m in args.models.split(",")] if args.models else all_models
    )

    print("=" * 80)
    print("Wizard pipeline smoke test")
    print("=" * 80)

    tmp_dir = tempfile.mkdtemp(prefix="xfake_smoke_")
    print(f"Dataset fake em: {tmp_dir}")
    try:
        make_fake_dataset(tmp_dir, n_per_class=4)

        results = []
        for arch in targets:
            print(f"\n>>> {arch}")
            ok, msg = test_one(arch, tmp_dir)
            results.append((arch, ok, msg))
            print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")

        print("\n" + "=" * 80)
        ok_count = sum(1 for _, ok, _ in results if ok)
        print(f"SUMÁRIO: {ok_count}/{len(results)} treinaram 1 epoch")
        for arch, ok, msg in results:
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] {arch}")

        return 0 if ok_count == len(results) else 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
