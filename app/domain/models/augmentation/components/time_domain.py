"""
Augmentations no domínio do tempo (ruído, shift, stretch, volume, smoothing).
"""
import tensorflow as tf


def add_gaussian_noise(audio: tf.Tensor, noise_factor: float,
                       apply_probability: float) -> tf.Tensor:
    """Adiciona ruído gaussiano adaptativo."""
    if tf.random.uniform([]) < apply_probability:
        # Calcular SNR baseado na energia do sinal
        signal_power = tf.reduce_mean(tf.square(audio))
        noise_power = signal_power * noise_factor

        noise = tf.random.normal(
            shape=tf.shape(audio),
            mean=0.0,
            stddev=tf.sqrt(noise_power),
            dtype=audio.dtype
        )
        return audio + noise
    return audio


def time_shift(audio: tf.Tensor, time_shift_factor: float,
               apply_probability: float) -> tf.Tensor:
    """Aplica deslocamento temporal aleatório."""
    if tf.random.uniform([]) < apply_probability:
        audio_length = tf.cast(tf.shape(audio)[0], tf.float32)
        max_shift = tf.cast(audio_length * time_shift_factor, tf.int32)
        shift_amount = tf.random.uniform(
            [],
            -max_shift,
            max_shift,
            dtype=tf.int32
        )
        return tf.roll(audio, shift_amount, axis=0)
    return audio


def time_stretch(audio: tf.Tensor, speed_factor: float,
                 apply_probability: float) -> tf.Tensor:
    """Aplica time stretching (mudança de velocidade)."""
    if tf.random.uniform([]) < apply_probability:
        # Simular time stretching através de resampling
        stretch_factor = tf.random.uniform(
            [],
            1.0 - speed_factor,
            1.0 + speed_factor
        )

        # Calcular novo tamanho
        original_length = tf.shape(audio)[0]
        new_length = tf.cast(
            tf.cast(original_length, tf.float32) / stretch_factor,
            tf.int32
        )

        # Redimensionar usando interpolação linear
        if len(audio.shape) == 1:
            # (T) -> (1, T, 1) -> (H=1, W=T, C=1)
            audio_expanded = tf.expand_dims(tf.expand_dims(audio, 0), -1)
            resized = tf.image.resize(audio_expanded, [1, new_length])
            audio_stretched = tf.squeeze(resized)
        else:
            # (T, C) -> (1, T, C) -> (H=1, W=T, C=C)
            audio_expanded = tf.expand_dims(audio, 0)
            resized = tf.image.resize(audio_expanded, [1, new_length])
            audio_stretched = tf.squeeze(resized, 0)

        # Truncar ou pad para manter o tamanho original
        if new_length > original_length:
            return audio_stretched[:original_length]
        else:
            padding = original_length - new_length
            return tf.pad(audio_stretched, [
                          [0, padding]] + [[0, 0]] * (len(audio.shape) - 1))

    return audio


def volume_perturbation(audio: tf.Tensor, volume_factor: float,
                        apply_probability: float) -> tf.Tensor:
    """Aplica perturbação de volume."""
    if tf.random.uniform([]) < apply_probability:
        factor = tf.random.uniform(
            [],
            1.0 - volume_factor,
            1.0 + volume_factor
        )
        return audio * factor
    return audio


def apply_smoothing(audio: tf.Tensor, apply_probability: float) -> tf.Tensor:
    """Aplica suavização (smoothing) para simular perda de detalhes de alta frequência."""

    def apply_op():
        # Simular filtro média móvel (low-pass simples)
        # Kernel size aleatório entre 3 e 9 (ímpar)
        kernel_size = tf.random.uniform([], 3, 10, dtype=tf.int32)
        kernel_size = kernel_size + (1 - kernel_size % 2)

        rank = tf.rank(audio)
        shape = tf.shape(audio)

        # Garantir acesso seguro aos índices do shape para evitar erro de grafo
        # Concatena 1s ao final para garantir que shape_safe[2] exista mesmo se
        # rank < 3
        shape_safe = tf.concat(
            [shape, tf.ones([4], dtype=shape.dtype)], axis=0)

        # Normalizar para 4D: [Batch, Height=1, Width=Time, Channels]
        # Usar tf.case para reshape correto baseado no rank

        def from_rank_1():
            # (T) -> (1, 1, T, 1)
            return tf.reshape(audio, [1, 1, shape_safe[0], 1])

        def from_rank_2():
            # (T, C) -> (1, 1, T, C)
            return tf.reshape(audio, [1, 1, shape_safe[0], shape_safe[1]])

        def from_rank_3():
            # (B, T, C) -> (B, 1, T, C)
            return tf.reshape(
                audio, [shape_safe[0], 1, shape_safe[1], shape_safe[2]])

        x_4d = tf.case([
            (tf.equal(rank, 1), from_rank_1),
            (tf.equal(rank, 2), from_rank_2),
            (tf.equal(rank, 3), from_rank_3)
        ], default=from_rank_3, exclusive=True)

        # Configurar kernel depthwise
        # x_4d shape: [B, 1, T, C]
        C = tf.shape(x_4d)[-1]
        # Filter: [filter_height, filter_width, in_channels, channel_multiplier]
        # Filter: [1, K, C, 1]
        filter_shape = [1, kernel_size, C, 1]
        kernel = tf.ones(filter_shape, dtype=audio.dtype) / \
            tf.cast(kernel_size, audio.dtype)

        # Aplicar Depthwise Conv2D
        # stride [1, 1, 1, 1] preserva dimensões
        y = tf.nn.depthwise_conv2d(
            x_4d, kernel, strides=[
                1, 1, 1, 1], padding='SAME')

        # Restaurar shape original
        return tf.reshape(y, shape)

    return tf.cond(
        tf.random.uniform([]) < apply_probability,
        apply_op,
        lambda: audio
    )
