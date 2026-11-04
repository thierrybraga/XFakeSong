"""Cabeças de classificação e perdas: AM-Softmax, OC-Softmax e suavização de
rótulos. Saiu de ``layers.py`` (P2-15), que continua sendo a porta de entrada e o
ponto de registro no Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers


def create_classification_head(x, num_classes, dropout_rate=0.3, hidden_dims=None):
    """
    Creates a standard classification head with Dense -> BN -> Dropout blocks.
    """
    if hidden_dims is None:
        hidden_dims = [512, 256, 128]
    for i, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation="relu", name=f"classifier_dense{i+1}")(x)
        x = layers.BatchNormalization(name=f"classifier_bn{i+1}")(x)
        # Reduce dropout for deeper layers
        current_dropout = dropout_rate * (0.5 ** (i > 0))  # 0.3, 0.15, 0.15...
        x = layers.Dropout(current_dropout, name=f"classifier_dropout{i+1}")(x)

    # dtype='float32': sob mixed_float16, softmax+crossentropy em float16
    # satura/perde precisão e pode colapsar o treino (rede "morta" após a
    # 1a epoca). Mesma correção já aplicada em AASIST/RawGAT-ST.
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="output", dtype="float32")(
            x
        )
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(
            num_classes, activation="softmax", name="output", dtype="float32"
        )(x)
        loss = "sparse_categorical_crossentropy"

    return outputs, loss


class AMSoftmaxLayer(layers.Layer):
    """Additive Margin Softmax (AM-Softmax / CosFace).

    Replaces Dense+Softmax for more discriminative embeddings in deepfake detection.
    Reference: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition", CVPR 2018

    Applies angular margin penalty: cos(theta) - m, then scales by s.
    During inference (training=False), returns standard cosine similarity logits.

    ⚠️  OUTPUT TYPE: This layer emits RAW LOGITS (scaled cosine similarities),
    NOT probabilities. The loss function MUST use from_logits=True:
        - SparseCategoricalCrossentropy(from_logits=True)
        - categorical_crossentropy(..., from_logits=True)
    Using the string shortcut "sparse_categorical_crossentropy" (which defaults
    to from_logits=False) will compute log(negative_value) → NaN loss.

    scale=15.0 is the safe default (original paper uses 30-64, but smaller values
    prevent overflow in float16 and still provide effective margin training).
    """

    def __init__(self, num_classes, scale=15.0, margin=0.35, **kwargs):
        super(AMSoftmaxLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

    def build(self, input_shape):
        self.W = self.add_weight(
            name="am_softmax_weights",
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(AMSoftmaxLayer, self).build(input_shape)

    def call(self, inputs, labels=None, training=None):
        # BUG FIX: tf.nn.l2_normalize(zero_vector) = 0/0 = NaN when the upstream
        # activations collapse to all-zeros (e.g. due to bad audio normalization or
        # corrupted input). Replace any NaN/Inf embeddings with a unit vector before
        # normalizing so the model degrades gracefully instead of poisoning gradients.
        inputs = tf.where(tf.math.is_finite(inputs), inputs, tf.zeros_like(inputs))

        # L2-normalize embeddings and weights (epsilon=1e-12 in TF prevents /0)
        x_norm = tf.nn.l2_normalize(inputs, axis=-1)
        w_norm = tf.nn.l2_normalize(self.W, axis=0)

        # Cosine similarity
        cosine = tf.matmul(x_norm, w_norm)
        # Clamp to [-1, 1] to guard against numerical noise from float16
        cosine = tf.clip_by_value(cosine, -1.0, 1.0)

        if training and labels is not None:
            # One-hot encode labels
            one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
            # Subtract margin from target class
            cosine = cosine - one_hot * self.margin

        # Scale logits
        logits = self.scale * cosine
        return logits

    def get_config(self):
        config = super(AMSoftmaxLayer, self).get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "scale": self.scale,
                "margin": self.margin,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AMSoftmaxCrossEntropy(tf.keras.losses.Loss):
    """Entropia cruzada com margem aditiva (AM-Softmax/CosFace) NA LOSS.

    A ``AMSoftmaxLayer`` emite ``s·cos(θ)`` SEM margem: no grafo funcional os
    rótulos nunca chegam ao ``call`` da camada, então a margem lá é código
    morto. É aqui, com ``y_true`` disponível, que o AM-Softmax do paper
    acontece::

        logit_alvo ← s·(cos θ − m) = s·cos θ − s·m

    Mantenha ``scale``/``margin`` em sincronia com os da ``AMSoftmaxLayer``.

    Args:
        scale: fator ``s`` usado pela camada (default 15.0).
        margin: margem aditiva ``m`` do CosFace (default 0.35).
        label_smoothing: suavização de rótulo aplicada depois da margem.
    """

    def __init__(
        self,
        scale: float = 15.0,
        margin: float = 0.35,
        label_smoothing: float = 0.1,
        name: str = "am_softmax_ce",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
        self.margin = float(margin)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.cast(tf.one_hot(y_true_int, num_classes), y_pred.dtype)
        # CosFace: subtrai s·m do logit da classe-alvo.
        y_pred = y_pred - one_hot * tf.cast(self.scale * self.margin, y_pred.dtype)
        k = tf.cast(num_classes, y_pred.dtype)
        ls = tf.cast(self.label_smoothing, y_pred.dtype)
        smoothed = one_hot * (1.0 - ls) + ls / k
        return tf.keras.losses.categorical_crossentropy(
            smoothed, y_pred, from_logits=True
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale": self.scale,
                "margin": self.margin,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SparseLabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    """Entropia cruzada com label smoothing para rótulos INTEIROS (esparsos).

    O Keras só oferece `label_smoothing` na versão categórica (one-hot). Esta
    classe faz a conversão internamente, e — por ser uma `Loss` registrada, e
    não uma closure definida dentro do builder — sobrevive a
    `load_model(..., compile=True)`.

    Args:
        label_smoothing: fator de suavização (0.0 desliga).
        from_logits: True se `y_pred` são logits; False para saída softmax.
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        from_logits: bool = False,
        name: str = "sparse_label_smoothing_ce",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        # reshape em vez de squeeze: squeeze total colapsaria batch=1 a escalar.
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        one_hot = tf.cast(tf.one_hot(y_true_int, num_classes), y_pred.dtype)
        k = tf.cast(num_classes, y_pred.dtype)
        ls = tf.cast(self.label_smoothing, y_pred.dtype)
        smoothed = one_hot * (1.0 - ls) + ls / k
        return tf.keras.losses.categorical_crossentropy(
            smoothed, y_pred, from_logits=self.from_logits
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "label_smoothing": self.label_smoothing,
                "from_logits": self.from_logits,
            }
        )
        return config


class OCSoftmaxLayer(layers.Layer):
    """One-Class Softmax (OC-Softmax) — Zhang et al., 2021.

    Objetivo *one-class* para anti-spoofing: aprende UM vetor-centro e produz
    um score escalar de "bonafide-ness" `s = ŵ·x̂` (cosseno, maior = mais
    bonafide). Treinado com `oc_softmax_loss`, compacta os embeddings bonafide
    e afasta os spoof — generaliza melhor a **ataques não vistos** que a
    entropia cruzada binária.

    Saída: `(batch, 1)` (score em [-1, 1]). É uma alternativa OPCIONAL à cabeça
    de 2 unidades softmax (não é default — requer `oc_softmax_loss` e tratamento
    do score escalar na inferência).
    """

    def __init__(self, feat_dim=None, **kwargs):
        super(OCSoftmaxLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            name="oc_center",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(OCSoftmaxLayer, self).build(input_shape)

    def call(self, inputs):
        # NaN-safety (mesma proteção do AMSoftmax)
        inputs = tf.where(tf.math.is_finite(inputs), inputs, tf.zeros_like(inputs))
        x = tf.nn.l2_normalize(inputs, axis=-1)
        w = tf.nn.l2_normalize(self.w, axis=0)
        s = tf.matmul(x, w)  # (batch, 1) — similaridade cosseno com o centro
        return tf.clip_by_value(s, -1.0, 1.0)

    def get_config(self):
        config = super(OCSoftmaxLayer, self).get_config()
        config.update({"feat_dim": self.feat_dim})
        return config


def oc_softmax_loss(m0: float = 0.9, m1: float = 0.2, alpha: float = 20.0):
    """Loss do OC-Softmax (Zhang et al., 2021). `m0 > m1`.

    Convenção de rótulos do SpeechChecker: **0 = real/bonafide** (target, deve ter
    score ≥ m0) e **1 = fake/spoof** (deve ter score ≤ m1).

    `y_pred`: score escalar (batch, 1) da `OCSoftmaxLayer`.
    Forma numericamente estável via `softplus` = log(1 + exp(z)).
    """

    def _loss(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        s = tf.reshape(tf.cast(y_pred, tf.float32), (-1,))
        margin = tf.where(tf.equal(y, 0.0), m0, m1)  # margem por classe
        sign = tf.where(tf.equal(y, 0.0), 1.0, -1.0)  # (-1)^y
        z = alpha * (margin - s) * sign
        return tf.reduce_mean(tf.math.softplus(z))

    return _loss
