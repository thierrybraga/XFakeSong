"""
Augmentations no domínio espectral (masking).
"""
import tensorflow as tf


def spectral_masking(spectrogram: tf.Tensor,
                     masking_freq: int,
                     masking_time: int,
                     apply_probability: float) -> tf.Tensor:
    """Aplica masking espectral (SpecAugment)."""

    # Proteção contra inputs inválidos (rank < 2)
    rank = tf.rank(spectrogram)

    def apply_freq_mask(spec):
        shape = tf.shape(spec)
        freq_dim = shape[1]

        freq_mask_param = tf.minimum(
            masking_freq,
            freq_dim // 4
        )

        if freq_mask_param > 0:
            f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)

            # Criar máscara
            mask = tf.ones_like(spec)
            indices = tf.range(freq_dim)
            freq_mask = tf.logical_or(indices < f0, indices >= f0 + f)
            freq_mask = tf.expand_dims(freq_mask, 0)  # (1, F)

            # Expandir dimensões extras para broadcasting seguro (ex: channels)
            # Loop seguro para graph mode
            target_rank = tf.rank(spec)

            def cond(m): return tf.rank(m) < target_rank
            def body(m): return tf.expand_dims(m, -1)

            freq_mask = tf.while_loop(
                cond,
                body,
                [freq_mask],
                shape_invariants=[tf.TensorShape(None)]  # Permite rank variar
            )[0]  # while_loop retorna lista de tensores

            mask = mask * tf.cast(freq_mask, spec.dtype)
            spec = spec * mask
        return spec

    def apply_time_mask(spec):
        shape = tf.shape(spec)
        time_dim = shape[0]

        time_mask_param = tf.minimum(
            masking_time,
            time_dim // 4
        )

        if time_mask_param > 0:
            t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)

            # Criar máscara temporal
            mask = tf.ones_like(spec)
            indices = tf.range(time_dim)
            time_mask = tf.logical_or(indices < t0, indices >= t0 + t)
            time_mask = tf.expand_dims(time_mask, 1)  # (T, 1)

            # Expandir dimensões extras
            target_rank = tf.rank(spec)

            def cond(m): return tf.rank(m) < target_rank
            def body(m): return tf.expand_dims(m, -1)

            time_mask = tf.while_loop(
                cond,
                body,
                [time_mask],
                shape_invariants=[tf.TensorShape(None)]
            )[0]

            mask = mask * tf.cast(time_mask, spec.dtype)
            spec = spec * mask
        return spec

    def apply_masking_logic():
        spec = spectrogram
        # Frequency Masking
        spec = tf.cond(
            tf.random.uniform(
                []) < apply_probability,
            lambda: apply_freq_mask(spec),
            lambda: spec)
        # Time Masking
        spec = tf.cond(
            tf.random.uniform(
                []) < apply_probability,
            lambda: apply_time_mask(spec),
            lambda: spec)
        return spec

    # Só aplica se rank >= 2
    return tf.cond(rank >= 2, apply_masking_logic, lambda: spectrogram)
