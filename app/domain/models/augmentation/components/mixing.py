"""
Augmentations de mistura (Mixup, CutMix).
"""
import tensorflow as tf
from typing import Tuple


def mixup_augmentation(batch_x: tf.Tensor, batch_y: tf.Tensor,
                       mixup_alpha: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """Aplica Mixup augmentation."""
    batch_size = tf.shape(batch_x)[0]

    # Gerar lambda da distribuição Beta
    lam = tf.random.gamma([batch_size], mixup_alpha, mixup_alpha)
    lam = tf.maximum(lam, 1 - lam)

    # Embaralhar índices
    indices = tf.random.shuffle(tf.range(batch_size))

    # Expandir lambda para as dimensões corretas (Graph-safe)
    rank = tf.rank(batch_x)
    # Shape [B, 1, 1, ...]
    shape_ones = tf.ones([rank - 1], dtype=tf.int32)
    target_shape = tf.concat([[batch_size], shape_ones], axis=0)
    lam_reshaped = tf.reshape(lam, target_shape)

    # Aplicar Mixup
    mixed_x = lam_reshaped * batch_x + \
        (1 - lam_reshaped) * tf.gather(batch_x, indices)

    # Para labels
    lam_y = lam

    rank_y = tf.rank(batch_y)

    def handle_categorical():
        num_classes = tf.reduce_max(batch_y) + 1
        batch_y_onehot = tf.one_hot(batch_y, num_classes)
        shuffled_y = tf.gather(batch_y_onehot, indices)
        return lam_y[:, None] * batch_y_onehot + \
            (1 - lam_y[:, None]) * shuffled_y

    def handle_onehot():
        shuffled_y = tf.gather(batch_y, indices)
        batch_y_float = tf.cast(batch_y, tf.float32)
        shuffled_y_float = tf.cast(shuffled_y, tf.float32)
        return lam_y[:, None] * batch_y_float + \
            (1 - lam_y[:, None]) * shuffled_y_float

    mixed_y = tf.cond(rank_y == 1, handle_categorical, handle_onehot)

    return mixed_x, mixed_y


def cutmix_augmentation(batch_x: tf.Tensor, batch_y: tf.Tensor,
                        cutmix_alpha: float, mixup_alpha: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """Aplica CutMix augmentation."""

    # Verificar rank para segurança
    rank = tf.rank(batch_x)

    def apply_cutmix_op():
        batch_size = tf.shape(batch_x)[0]
        height = tf.shape(batch_x)[1]
        width = tf.shape(batch_x)[2]

        # Gerar lambda
        lam = tf.random.gamma([batch_size], cutmix_alpha, cutmix_alpha)

        # Calcular tamanho do corte
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(cut_ratio * tf.cast(height, tf.float32), tf.int32)
        cut_w = tf.cast(cut_ratio * tf.cast(width, tf.float32), tf.int32)

        # Posições aleatórias
        cx = tf.random.uniform([batch_size], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([batch_size], 0, height, dtype=tf.int32)

        # Calcular coordenadas do corte
        x1 = tf.maximum(0, cx - cut_w // 2)
        y1 = tf.maximum(0, cy - cut_h // 2)
        x2 = tf.minimum(width, cx + cut_w // 2)
        y2 = tf.minimum(height, cy + cut_h // 2)

        # Embaralhar índices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_x = tf.gather(batch_x, indices)

        # Vectorized Mask Generation (No loop)
        # grid_x: [1, 1, W]
        grid_x = tf.reshape(tf.range(width), [1, 1, width])
        # grid_y: [1, H, 1]
        grid_y = tf.reshape(tf.range(height), [1, height, 1])

        # Expand coordinates to [B, 1, 1]
        x1_e = tf.reshape(x1, [batch_size, 1, 1])
        x2_e = tf.reshape(x2, [batch_size, 1, 1])
        y1_e = tf.reshape(y1, [batch_size, 1, 1])
        y2_e = tf.reshape(y2, [batch_size, 1, 1])

        # Mask: 1 inside box, 0 outside
        mask_x = tf.logical_and(grid_x >= x1_e, grid_x < x2_e)
        mask_y = tf.logical_and(grid_y >= y1_e, grid_y < y2_e)
        mask = tf.logical_and(mask_x, mask_y)  # [B, H, W]

        mask = tf.cast(mask, batch_x.dtype)

        # Expand mask to match channels if needed
        # Assuming batch_x is [B, H, W, C]
        if tf.rank(batch_x) > 3:
            mask = tf.expand_dims(mask, -1)

        # CutMix: Replace region in batch_x with shuffled_x
        # mixed = (1 - mask) * batch_x + mask * shuffled_x
        # Note: shuffled_x is the "patch" source (inside box), batch_x is
        # background
        mixed_x = (1 - mask) * batch_x + mask * shuffled_x

        # Calcular lambda real baseado na área
        cut_area = (x2 - x1) * (y2 - y1)
        total_area = height * width
        lam_real = 1.0 - tf.cast(cut_area, tf.float32) / \
            tf.cast(total_area, tf.float32)

        # Misturar labels
        rank_y = tf.rank(batch_y)

        def handle_categorical_y():
            num_classes = tf.reduce_max(batch_y) + 1
            batch_y_onehot = tf.one_hot(batch_y, num_classes)
            shuffled_y = tf.gather(batch_y_onehot, indices)
            return lam_real[:, None] * batch_y_onehot + \
                (1 - lam_real[:, None]) * shuffled_y

        def handle_onehot_y():
            shuffled_y = tf.gather(batch_y, indices)
            batch_y_float = tf.cast(batch_y, tf.float32)
            shuffled_y_float = tf.cast(shuffled_y, tf.float32)
            return lam_real[:, None] * batch_y_float + \
                (1 - lam_real[:, None]) * shuffled_y_float

        mixed_y = tf.cond(rank_y == 1, handle_categorical_y, handle_onehot_y)

        return mixed_x, mixed_y

    # Fallback para mixup se rank < 3 (não é imagem/espectrograma 2D)
    return tf.cond(rank >= 3, apply_cutmix_op,
                   lambda: mixup_augmentation(batch_x, batch_y, mixup_alpha))
