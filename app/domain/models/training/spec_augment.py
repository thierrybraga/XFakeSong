"""SpecAugment — mascaramento de tempo/frequência para modelos de espectrograma.

Implementação TensorFlow-nativa (in-graph, `tf.data`) do SpecAugment
(Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic
Speech Recognition", 2019). Aplica bandas de máscara aleatórias nos eixos de
tempo e frequência do espectrograma/LFCC, regularizando CNNs e Transformers e
melhorando a robustez a variações não vistas.

Operações:
  - **Time masking**: zera N bandas contíguas ao longo do eixo de tempo.
  - **Frequency masking**: zera N bandas contíguas ao longo do eixo de freq.

Aplicado SOMENTE ao conjunto de treino (val/test ficam intactos). Máscaras são
geradas **por exemplo** do batch (posição/largura independentes), e a região
mascarada é zerada — após a normalização de entrada do modelo, zero é neutro.

Uso típico (no `tf.data`, sobre o espectrograma já calculado):

    from app.domain.models.training.spec_augment import spec_augment_tf
    ds = ds.map(lambda x, y: (spec_augment_tf(x), y))
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["spec_augment_tf"]


def _mask_axis(y, axis: int, max_pct: float):
    """Zera uma banda contígua aleatória por exemplo ao longo de `axis`.

    `y`: (B, T, F). `axis` = 1 (tempo) ou 2 (frequência).
    """
    b = tf.shape(y)[0]
    length = tf.shape(y)[axis]
    length_f = tf.cast(length, tf.float32)

    # Largura da banda por exemplo: U(0, max_pct) * L
    width = tf.cast(tf.random.uniform([b], 0.0, max_pct) * length_f, tf.int32)
    # Início válido por exemplo: U(0, L - width)
    max_start = tf.maximum(length - width, 1)
    start = tf.cast(
        tf.random.uniform([b]) * tf.cast(max_start, tf.float32), tf.int32
    )

    idx = tf.range(length)[tf.newaxis, :]  # (1, L)
    lo = start[:, tf.newaxis]              # (B, 1)
    hi = (start + width)[:, tf.newaxis]    # (B, 1)
    keep = tf.logical_or(idx < lo, idx >= hi)  # (B, L) — True = mantém
    keep = tf.cast(keep, y.dtype)

    if axis == 1:                 # tempo → (B, T, 1)
        keep = keep[:, :, tf.newaxis]
    else:                         # frequência → (B, 1, F)
        keep = keep[:, tf.newaxis, :]
    return y * keep


def spec_augment_tf(
    spec,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
    time_mask_pct: float = 0.12,
    freq_mask_pct: float = 0.12,
    p: float = 0.7,
):
    """Aplica SpecAugment a um espectrograma batched, com probabilidade `p`.

    Args:
        spec: tensor (B, T, F) ou (B, T, F, 1) float32.
        n_time_masks / n_freq_masks: nº de bandas de máscara por eixo.
        time_mask_pct / freq_mask_pct: largura máxima da banda (fração do eixo).
        p: probabilidade de aplicar (senão retorna o espectrograma original).

    Returns:
        Tensor com o MESMO shape de `spec`, sempre finito.
    """
    has_channel = spec.shape.rank == 4
    x = tf.squeeze(spec, axis=-1) if has_channel else spec  # (B, T, F)
    x = tf.cast(x, tf.float32)

    def _apply():
        y = x
        for _ in range(n_time_masks):
            y = _mask_axis(y, axis=1, max_pct=time_mask_pct)
        for _ in range(n_freq_masks):
            y = _mask_axis(y, axis=2, max_pct=freq_mask_pct)
        y = tf.where(tf.math.is_finite(y), y, tf.zeros_like(y))
        return y

    do_aug = tf.random.uniform([]) < p
    out = tf.cond(do_aug, _apply, lambda: x)
    return tf.expand_dims(out, axis=-1) if has_channel else out
