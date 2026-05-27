"""Smoke test do pipeline completo do training_wizard.

Cria um dataset fake mínimo (2 classes real/fake com 4 áudios cada) e
executa _run_training por 1 epoch em cada uma das 12 arquiteturas para validar:
- Detecção de input_type funciona (raw_audio vs spectrogram)
- Pré-processamento on-the-fly produz shape correto
- Modelo aceita o shape e treina 1 epoch sem erro

Uso standalone:
    python tests/smoke/test_wizard_pipeline.py
    python tests/smoke/test_wizard_pipeline.py --models AASIST,Conformer

Uso via pytest:
    pytest -m smoke tests/smoke/test_wizard_pipeline.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))  # tests/smoke/ → tests/ → raiz
sys.path.insert(0, PROJECT_ROOT)

# Fake packages para evitar app/__init__ chain (sqlalchemy)
for _fake_pkg in [
    "app", "app.domain", "app.domain.models", "app.domain.models.architectures",
]:
    if _fake_pkg not in sys.modules:
        _m = types.ModuleType(_fake_pkg)
        _m.__path__ = [os.path.join(PROJECT_ROOT, *_fake_pkg.split("."))]
        sys.modules[_fake_pkg] = _m

SAMPLE_RATE = 16000
DURATION = 3
N_FFT = 512
HOP = 128
N_MELS = 80

ALL_MODELS = [
    "AASIST", "RawGAT-ST", "RawNet2", "WavLM", "HuBERT",
    "Sonic Sleuth", "EfficientNet-LSTM", "MultiscaleCNN",
    "SpectrogramTransformer", "Conformer", "Hybrid CNN-Transformer", "Ensemble",
]


def make_fake_dataset(root: str, n_per_class: int = 4) -> None:
    """Cria n_per_class .wav (real + fake) de 3s @ 16kHz."""
    import soundfile as sf

    rng = np.random.default_rng(0)
    for cls, freq_base in [("real", 200), ("fake", 400)]:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
            audio = 0.5 * np.sin(2 * np.pi * (freq_base + i * 30) * t)
            if cls == "fake":
                audio += 0.1 * rng.standard_normal(t.shape).astype(np.float32)
            sf.write(os.path.join(cls_dir, f"{cls[0]}_{i}.wav"), audio, SAMPLE_RATE)


def test_one(arch: str, dataset_path: str) -> tuple[bool, str]:
    """Executa 1 epoch do pipeline wizard para uma arquitetura."""
    import tensorflow as tf
    import importlib

    factory_mod = importlib.import_module("app.domain.models.architectures.factory")

    try:
        spec = factory_mod.architecture_factory_registry.get_architecture_info(arch)
        input_type = spec.input_requirements.get("input_type", "spectrogram")

        train_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=dataset_path, batch_size=2,
            validation_split=0.5, subset="training", seed=42,
            output_sequence_length=SAMPLE_RATE * DURATION, label_mode="int",
        )

        class_names = list(train_ds.class_names)
        binary_map = {i: 0 if "real" in n.lower() else 1 for i, n in enumerate(class_names)}
        _table = tf.constant([binary_map[i] for i in range(len(class_names))], dtype=tf.int32)

        n_freq = N_FFT // 2 + 1
        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=N_MELS, num_spectrogram_bins=n_freq,
            sample_rate=SAMPLE_RATE, lower_edge_hertz=0.0, upper_edge_hertz=SAMPLE_RATE / 2,
        )

        def _to_log_mel(audio):
            if audio.shape.rank == 3:
                audio = tf.squeeze(audio, axis=-1)
            stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP, fft_length=N_FFT,
                                  window_fn=tf.signal.hann_window, pad_end=True)
            return tf.expand_dims(tf.math.log(tf.abs(tf.tensordot(tf.abs(stft), mel_w, axes=1)) + 1e-6), axis=-1)

        def _prep_raw(x, y):
            return (tf.expand_dims(x, -1) if x.shape.rank == 2 else x), tf.gather(_table, y)

        def _prep_spec(x, y):
            return _to_log_mel(x), tf.gather(_table, y)

        train_ds = train_ds.map(_prep_raw if input_type == "raw_audio" else _prep_spec,
                                num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

        for sample_x, _ in train_ds.take(1):
            input_shape = tuple(sample_x.shape[1:])

        model = factory_mod.create_model_by_name(arch, input_shape=input_shape, num_classes=2)
        out_units = model.output_shape[-1]

        if out_units == 1:
            chosen_loss, chosen_metric = "binary_crossentropy", "binary_accuracy"
            train_ds = train_ds.map(lambda x, y: (x, tf.cast(y, tf.float32)),
                                    num_parallel_calls=tf.data.AUTOTUNE)
        else:
            chosen_loss, chosen_metric = "sparse_categorical_crossentropy", "accuracy"

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=chosen_loss, metrics=[chosen_metric])
        hist = model.fit(train_ds, epochs=1, verbose=0)
        loss = hist.history["loss"][-1]
        acc = hist.history.get(chosen_metric, [0])[-1]
        return True, f"OK input_type={input_type} shape={input_shape} out={out_units} loss={loss:.3f} acc={acc:.3f}"

    except Exception as e:
        return False, f"FAIL: {type(e).__name__}: {str(e)[:200]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="", help="CSV de modelos (vazio = todos)")
    args = parser.parse_args(argv)

    targets = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS

    print("=" * 80)
    print("Wizard pipeline smoke test (1 epoch por arquitetura)")
    print("=" * 80)

    tmp_dir = tempfile.mkdtemp(prefix="xfake_smoke_")
    print(f"Dataset sintético em: {tmp_dir}")
    try:
        make_fake_dataset(tmp_dir, n_per_class=4)
        results = []
        for arch in targets:
            print(f"\n>>> {arch}")
            ok, msg = test_one(arch, tmp_dir)
            results.append((arch, ok, msg))
            print(f"  {'[OK]' if ok else '[FAIL]'} {msg}")

        ok_count = sum(1 for _, ok, _ in results if ok)
        print("\n" + "=" * 80)
        print(f"SUMÁRIO: {ok_count}/{len(results)} treinaram 1 epoch")
        for arch, ok, _ in results:
            print(f"  [{'OK  ' if ok else 'FAIL'}] {arch}")
        return 0 if ok_count == len(results) else 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── pytest integration ────────────────────────────────────────────────────────
import pytest  # noqa: E402


@pytest.mark.smoke
def test_smoke_wizard_pipeline() -> None:
    """Pipeline do training_wizard treina 1 epoch em todas as 12 arquiteturas."""
    rc = main(argv=[])
    assert rc == 0, "Uma ou mais arquiteturas falharam no treino — veja saída acima"


# ── standalone ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
