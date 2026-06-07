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
