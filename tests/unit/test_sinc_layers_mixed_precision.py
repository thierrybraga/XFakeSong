import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from app.domain.models.architectures.layers import SincConvLayer, SincNetLayer
from app.domain.models.training.augmentation import AudioAugmenter


def test_sinc_layers_accept_mixed_precision_inputs():
    previous_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("mixed_float16")
    try:
        inputs = tf.random.normal((2, 1600, 1), dtype=tf.float16)

        sincnet_output = SincNetLayer(filters=4, kernel_size=31)(inputs)
        sincconv_output = SincConvLayer(n_filters=4, kernel_size=31)(inputs)

        assert sincnet_output.shape == (2, 1600, 4)
        assert sincconv_output.shape == (2, 1600, 4)
        assert np.isfinite(tf.cast(sincnet_output, tf.float32).numpy()).all()
        assert np.isfinite(tf.cast(sincconv_output, tf.float32).numpy()).all()
    finally:
        mixed_precision.set_global_policy(previous_policy)


def test_audio_augmenter_handles_raw_audio_with_single_channel_axis():
    augmenter = AudioAugmenter(
        {
            "noise_factor": 0.01,
            "time_shift_factor": 0.1,
            "frequency_mask_factor": 0.1,
            "time_mask_factor": 0.1,
            "volume_factor": 0.1,
        }
    )

    dataset = augmenter.create_augmented_dataset(
        np.zeros((4, 1600, 1), dtype=np.float32),
        np.array([0, 1, 0, 1], dtype=np.int32),
        batch_size=2,
    )

    batch_x, batch_y = next(iter(dataset))
    assert batch_x.shape == (2, 1600, 1)
    assert batch_y.shape == (2,)
    assert np.isfinite(batch_x.numpy()).all()


def test_stft_e_log_mel_aceitam_mixed_precision():
    """STFT e log-mel precisam sobreviver a `mixed_float16`.

    `tf.signal.stft` chama RFFT, que so aceita float32/float64, e
    `linear_to_mel_weight_matrix` devolve float32 SEMPRE. Sem os casts, o
    caminho de audio bruto nem CONSTROI sob precisao mista:

      RFFT requires tf.float32 or tf.float64 inputs, got: ... dtype=float16
      Input 'y' of 'BatchMatMulV2' Op has type float32 that does not match
      type float16 of argument 'x'

    Encontrado em 2026-08-02 ao instrumentar o MultiscaleCNN. O benchmark nao
    via porque alimenta log-mel ja pronto (`input_domain: spectrogram`), mas
    qualquer consumidor do caminho raw quebrava.
    """
    from app.domain.models.architectures.layers import (
        LogMelFromMagnitudeLayer,
        STFTLayer,
    )

    previous_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("mixed_float16")
    try:
        inputs = tf.random.normal((2, 16000, 1), dtype=tf.float16)

        mag = STFTLayer(
            frame_length=512, frame_step=256, fft_length=512, add_channel_dim=False
        )(inputs)
        # Volta ao dtype da politica: so a FFT sai da precisao mista.
        assert mag.dtype == tf.float16
        assert np.isfinite(tf.cast(mag, tf.float32).numpy()).all()

        log_mel = LogMelFromMagnitudeLayer(
            num_mel_bins=40, num_spectrogram_bins=int(mag.shape[-1])
        )(mag)
        assert log_mel.dtype == tf.float16
        valores = tf.cast(log_mel, tf.float32).numpy()
        # O log roda em float32 de proposito: com mel em float16 o epsilon de
        # 1e-6 sumiria e os bins nulos virariam -inf.
        assert np.isfinite(valores).all()
    finally:
        mixed_precision.set_global_policy(previous_policy)
