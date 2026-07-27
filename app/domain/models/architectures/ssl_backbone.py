"""Backbone SSL (HuBERT / WavLM) pré-treinado, portado para Keras 3.

Por que este módulo existe
--------------------------
O ``transformers`` **não publica HuBERT/WavLM em TensorFlow** utilizável aqui:
seus modelos TF exigem o pacote ``tf-keras`` quando o Keras instalado é 3.x e
sequer importam. O resultado, até 2026-07-27, era que ambas as arquiteturas
caíam num fallback CNN-1D **treinado do zero** — ou seja, os números rotulados
"WavLM"/"HuBERT" no caminho TF não tinham relação com os modelos dos artigos.

A saída: o checkpoint **PyTorch** é legível sem tocar em TensorFlow. Este
módulo lê o ``state_dict`` e reimplementa o forward do backbone com operações
TF, carregando os pesos como variáveis **não-treináveis**. É exatamente a
receita padrão de uso de SSL em tarefas downstream (SUPERB): backbone
congelado como extrator de características, só a cabeça treina.

Cobertura
---------
- **HuBERT** (Hsu et al., 2021): atenção padrão.
- **WavLM** (Chen et al., 2022): idem + **viés posicional relativo com gating**
  (``gru_rel_pos``/``rel_attn_embed``), que é a contribuição arquitetural do
  artigo e está implementada aqui fielmente.

Ambos seguem o esqueleto wav2vec 2.0: extrator convolucional → projeção →
convolução posicional → encoder Transformer (post-LN nos checkpoints *base*).

A fidelidade do port é verificada comparando a saída com a do modelo PyTorch
(ver ``tests/unit/test_ssl_backbone.py``).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

#: Checkpoints padrão de cada família.
DEFAULT_CHECKPOINTS = {
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base",
}


class SSLBackboneUnavailable(RuntimeError):
    """Pesos SSL pré-treinados não puderam ser obtidos."""


def load_pretrained_state_dict(family: str, checkpoint: str) -> Tuple[Dict[str, np.ndarray], Any]:
    """Lê o ``state_dict`` do backbone PyTorch como arrays numpy.

    Não importa TensorFlow nem depende do caminho TF do ``transformers``.
    """
    family = family.lower()
    try:
        if family == "wavlm":
            from transformers import WavLMModel as _Model
        elif family == "hubert":
            from transformers import HubertModel as _Model
        else:
            raise SSLBackboneUnavailable(f"família SSL desconhecida: {family}")
    except SSLBackboneUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SSLBackboneUnavailable(
            f"transformers indisponível para ler o backbone {family}: {exc}"
        ) from exc

    try:
        torch_model = _Model.from_pretrained(checkpoint)
    except Exception as exc:  # noqa: BLE001
        raise SSLBackboneUnavailable(
            f"não foi possível baixar/abrir o checkpoint '{checkpoint}': {exc}"
        ) from exc

    state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
    config = torch_model.config
    del torch_model
    return state, config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class PretrainedSSLBackbone(layers.Layer):
    """Backbone HuBERT/WavLM pré-treinado e CONGELADO, em Keras puro.

    Entrada: áudio bruto ``(B, T)`` ou ``(B, T, 1)`` a 16 kHz.
    Saída: ``(B, L+1, T', H)`` com os hidden states de todas as camadas quando
    ``output_hidden_states=True`` (para a soma ponderada da receita SUPERB), ou
    ``(B, T', H)`` com o último hidden state.

    Todos os pesos são **não-treináveis**: só a cabeça a jusante aprende.
    """

    def __init__(self, family: str = "hubert",
                 checkpoint: Optional[str] = None,
                 output_hidden_states: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.family = family.lower()
        self.checkpoint = checkpoint or DEFAULT_CHECKPOINTS[self.family]
        self.output_hidden_states = bool(output_hidden_states)
        self.trainable = False

        state, config = load_pretrained_state_dict(self.family, self.checkpoint)
        self._state = state
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_layers = int(config.num_hidden_layers)
        self.conv_kernels = list(config.conv_kernel)
        self.conv_strides = list(config.conv_stride)
        self.feat_extract_norm = str(config.feat_extract_norm)
        self.do_stable_layer_norm = bool(config.do_stable_layer_norm)
        self.pos_conv_kernel = int(config.num_conv_pos_embeddings)
        self.pos_conv_groups = int(config.num_conv_pos_embedding_groups)
        self.layer_norm_eps = float(config.layer_norm_eps)
        # Específicos do WavLM (viés posicional relativo).
        self.num_buckets = int(getattr(config, "num_buckets", 320))
        self.max_distance = int(getattr(config, "max_bucket_distance", 800))

        # `do_stable_layer_norm` alterna a ordenação do encoder:
        #   False (checkpoints *-base*)  → post-LN, LN do encoder ANTES do loop
        #   True  (checkpoints *-large*) → pre-LN,  LN do encoder DEPOIS do loop
        # Ambas suportadas — sem isso um checkpoint -large cairia em silêncio no
        # extrator simplificado.

    # ── construção dos pesos ──────────────────────────────────────────────
    def _const(self, name: str, key: str, transform=None) -> tf.Variable:
        value = self._state[key]
        if transform is not None:
            value = transform(value)
        return self.add_weight(
            name=name, shape=value.shape, dtype="float32",
            initializer=tf.keras.initializers.Constant(value),
            trainable=False,
        )

    def build(self, input_shape):
        s = self._state

        # 1) Extrator convolucional — PyTorch (out, in, k) → TF (k, in, out)
        self.conv_w = [
            self._const(f"conv_{i}", f"feature_extractor.conv_layers.{i}.conv.weight",
                        lambda v: np.transpose(v, (2, 1, 0)))
            for i in range(len(self.conv_kernels))
        ]
        # GroupNorm(num_groups=C, C) do 1º bloco ≡ normalizar cada canal no tempo.
        self.gn_w = self._const("gn_w", "feature_extractor.conv_layers.0.layer_norm.weight")
        self.gn_b = self._const("gn_b", "feature_extractor.conv_layers.0.layer_norm.bias")

        # 2) Projeção de características
        self.fp_ln_w = self._const("fp_ln_w", "feature_projection.layer_norm.weight")
        self.fp_ln_b = self._const("fp_ln_b", "feature_projection.layer_norm.bias")
        self.fp_w = self._const("fp_w", "feature_projection.projection.weight",
                                lambda v: v.T)
        self.fp_b = self._const("fp_b", "feature_projection.projection.bias")

        # 3) Convolução posicional (weight norm: W = g · v/‖v‖ sobre dims (0,1))
        g_key = "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        v_key = "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        if g_key not in s:  # transformers < 4.44 usa weight_g/weight_v
            g_key, v_key = ("encoder.pos_conv_embed.conv.weight_g",
                            "encoder.pos_conv_embed.conv.weight_v")
        g, v = s[g_key], s[v_key]
        norm = np.linalg.norm(v.reshape(-1, v.shape[2]), axis=0).reshape(1, 1, -1)
        pos_w = v * (g / (norm + 1e-12))            # (out, in/groups, k)
        self.pos_w = self._const("pos_w", g_key,
                                 lambda _: np.transpose(pos_w, (2, 1, 0)))
        self.pos_b = self._const("pos_b", "encoder.pos_conv_embed.conv.bias")

        self.enc_ln_w = self._const("enc_ln_w", "encoder.layer_norm.weight")
        self.enc_ln_b = self._const("enc_ln_b", "encoder.layer_norm.bias")

        # 4) Camadas do encoder
        self.blocks: List[Dict[str, tf.Variable]] = []
        for i in range(self.num_layers):
            p = f"encoder.layers.{i}."
            blk = {
                "q_w": self._const(f"l{i}_qw", p + "attention.q_proj.weight", lambda v: v.T),
                "q_b": self._const(f"l{i}_qb", p + "attention.q_proj.bias"),
                "k_w": self._const(f"l{i}_kw", p + "attention.k_proj.weight", lambda v: v.T),
                "k_b": self._const(f"l{i}_kb", p + "attention.k_proj.bias"),
                "v_w": self._const(f"l{i}_vw", p + "attention.v_proj.weight", lambda v: v.T),
                "v_b": self._const(f"l{i}_vb", p + "attention.v_proj.bias"),
                "o_w": self._const(f"l{i}_ow", p + "attention.out_proj.weight", lambda v: v.T),
                "o_b": self._const(f"l{i}_ob", p + "attention.out_proj.bias"),
                "ln_w": self._const(f"l{i}_lnw", p + "layer_norm.weight"),
                "ln_b": self._const(f"l{i}_lnb", p + "layer_norm.bias"),
                "fi_w": self._const(f"l{i}_fiw", p + "feed_forward.intermediate_dense.weight",
                                    lambda v: v.T),
                "fi_b": self._const(f"l{i}_fib", p + "feed_forward.intermediate_dense.bias"),
                "fo_w": self._const(f"l{i}_fow", p + "feed_forward.output_dense.weight",
                                    lambda v: v.T),
                "fo_b": self._const(f"l{i}_fob", p + "feed_forward.output_dense.bias"),
                "fln_w": self._const(f"l{i}_flnw", p + "final_layer_norm.weight"),
                "fln_b": self._const(f"l{i}_flnb", p + "final_layer_norm.bias"),
            }
            if self.family == "wavlm":
                blk["gru_const"] = self._const(
                    f"l{i}_gruc", p + "attention.gru_rel_pos_const")
                blk["gru_w"] = self._const(
                    f"l{i}_gruw", p + "attention.gru_rel_pos_linear.weight", lambda v: v.T)
                blk["gru_b"] = self._const(
                    f"l{i}_grub", p + "attention.gru_rel_pos_linear.bias")
                key = p + "attention.rel_attn_embed.weight"
                if key in s:  # só a 1ª camada carrega o embedding (compartilhado)
                    blk["rel_embed"] = self._const(f"l{i}_rel", key)
            self.blocks.append(blk)

        # Liberamos o state_dict: os pesos já vivem nas variáveis.
        self._state = {}
        super().build(input_shape)

    # ── primitivas ────────────────────────────────────────────────────────
    @staticmethod
    def _layer_norm(x, gamma, beta, eps):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + eps) * gamma + beta

    def _grouped_conv1d(self, x, kernel, bias, groups, padding):
        """Conv1D agrupada (o TF cru não expõe `groups`)."""
        in_per_g = x.shape[-1] // groups
        out_per_g = kernel.shape[-1] // groups
        outs = []
        for g in range(groups):
            xi = x[..., g * in_per_g:(g + 1) * in_per_g]
            ki = kernel[..., g * out_per_g:(g + 1) * out_per_g]
            outs.append(tf.nn.conv1d(xi, ki, stride=1, padding=padding))
        return tf.concat(outs, axis=-1) + bias

    def _relative_position_bucket(self, seq_len):
        """Bucketização das posições relativas do WavLM (bidirecional)."""
        num_buckets = self.num_buckets // 2
        pos = tf.range(seq_len)
        rel = pos[None, :] - pos[:, None]                     # (T, T)
        buckets = tf.cast(rel > 0, tf.int32) * num_buckets
        rel = tf.abs(rel)
        max_exact = num_buckets // 2
        is_small = rel < max_exact
        large = tf.math.log(tf.cast(tf.maximum(rel, 1), tf.float32) / max_exact)
        large = large / math.log(self.max_distance / max_exact)
        large = tf.cast(max_exact + large * (num_buckets - max_exact), tf.int32)
        large = tf.minimum(large, num_buckets - 1)
        return buckets + tf.where(is_small, rel, large)

    def _attention(self, h, blk, position_bias):
        b = tf.shape(h)[0]
        t = tf.shape(h)[1]
        hd = self.hidden_size // self.num_heads
        scale = 1.0 / math.sqrt(float(hd))

        def proj(w, bias):
            y = tf.matmul(h, w) + bias
            y = tf.reshape(y, [b, t, self.num_heads, hd])
            return tf.transpose(y, [0, 2, 1, 3])              # (B, H, T, hd)

        q = proj(blk["q_w"], blk["q_b"]) * scale
        k = proj(blk["k_w"], blk["k_b"])
        v = proj(blk["v_w"], blk["v_b"])

        scores = tf.matmul(q, k, transpose_b=True)            # (B, H, T, T)
        if position_bias is not None:
            scores = scores + position_bias

        attn = tf.nn.softmax(scores, axis=-1)
        ctx = tf.matmul(attn, v)                              # (B, H, T, hd)
        ctx = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [b, t, self.hidden_size])
        return tf.matmul(ctx, blk["o_w"]) + blk["o_b"]

    def _gated_position_bias(self, h, blk, base_bias):
        """Viés posicional relativo COM GATING — contribuição do WavLM."""
        b = tf.shape(h)[0]
        t = tf.shape(h)[1]
        hd = self.hidden_size // self.num_heads

        gated = tf.reshape(h, [b, t, self.num_heads, hd])
        gated = tf.transpose(gated, [0, 2, 1, 3])             # (B, H, T, hd)
        proj = tf.matmul(gated, blk["gru_w"]) + blk["gru_b"]  # (B, H, T, 8)
        proj = tf.reduce_sum(tf.reshape(proj, [b, self.num_heads, t, 2, 4]), axis=-1)
        gate = tf.sigmoid(proj)                                # (B, H, T, 2)
        gate_a, gate_b = gate[..., :1], gate[..., 1:]
        gate_out = gate_a * (gate_b * blk["gru_const"] - 1.0) + 2.0   # (B,H,T,1)
        return gate_out * base_bias[None, ...]                 # (B,H,T,T)

    # ── forward ───────────────────────────────────────────────────────────
    def call(self, inputs, training=None):
        x = tf.cast(inputs, tf.float32)
        # Normaliza para (B, T, 1). A checagem NÃO pode depender do valor
        # estático do último eixo: vindo de um `tf.data.Dataset` ele pode ser
        # `None`, e um teste `shape[-1] == 1` falharia em silêncio, deixando o
        # tensor virar rank 4 e estourar lá adiante no `tf.pad`.
        rank = x.shape.rank
        if rank == 3:
            x = x[..., 0]                                      # áudio é mono
        elif rank is not None and rank > 3:
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = tf.expand_dims(x, axis=-1)                         # (B, T, 1)

        # 1) Extrator convolucional
        for i, (kernel, stride) in enumerate(zip(self.conv_kernels, self.conv_strides)):
            x = tf.nn.conv1d(x, self.conv_w[i], stride=stride, padding="VALID")
            if i == 0 and self.feat_extract_norm == "group":
                mean = tf.reduce_mean(x, axis=1, keepdims=True)
                var = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
                x = (x - mean) * tf.math.rsqrt(var + 1e-5) * self.gn_w + self.gn_b
            x = tf.nn.gelu(x, approximate=False)

        # 2) Projeção
        h = self._layer_norm(x, self.fp_ln_w, self.fp_ln_b, self.layer_norm_eps)
        h = tf.matmul(h, self.fp_w) + self.fp_b

        # 3) Convolução posicional + LN do encoder
        pad = self.pos_conv_kernel // 2
        padded = tf.pad(h, [[0, 0], [pad, pad], [0, 0]])
        pos = self._grouped_conv1d(padded, self.pos_w, self.pos_b,
                                   self.pos_conv_groups, padding="VALID")
        if self.pos_conv_kernel % 2 == 0:
            pos = pos[:, :-1, :]                               # SamePadLayer
        h = h + tf.nn.gelu(pos, approximate=False)
        if not self.do_stable_layer_norm:
            # post-LN: o LN do encoder vem ANTES das camadas
            h = self._layer_norm(h, self.enc_ln_w, self.enc_ln_b,
                                 self.layer_norm_eps)

        hidden_states = [h]

        # 4) Camadas do encoder
        base_bias = None
        if self.family == "wavlm":
            bucket = self._relative_position_bucket(tf.shape(h)[1])
            embed = self.blocks[0]["rel_embed"]                # (num_buckets, H)
            base_bias = tf.transpose(tf.gather(embed, bucket), [2, 0, 1])  # (H,T,T)

        for blk in self.blocks:
            if self.do_stable_layer_norm:
                # pre-LN (checkpoints *-large*)
                residual = h
                normed = self._layer_norm(h, blk["ln_w"], blk["ln_b"],
                                          self.layer_norm_eps)
                bias = (self._gated_position_bias(normed, blk, base_bias)
                        if self.family == "wavlm" else None)
                h = residual + self._attention(normed, blk, bias)
                normed = self._layer_norm(h, blk["fln_w"], blk["fln_b"],
                                          self.layer_norm_eps)
                ff = tf.nn.gelu(tf.matmul(normed, blk["fi_w"]) + blk["fi_b"],
                                approximate=False)
                h = h + tf.matmul(ff, blk["fo_w"]) + blk["fo_b"]
            else:
                # post-LN (checkpoints *-base*)
                bias = (self._gated_position_bias(h, blk, base_bias)
                        if self.family == "wavlm" else None)
                attn = self._attention(h, blk, bias)
                h = self._layer_norm(h + attn, blk["ln_w"], blk["ln_b"],
                                     self.layer_norm_eps)
                ff = tf.nn.gelu(tf.matmul(h, blk["fi_w"]) + blk["fi_b"],
                                approximate=False)
                ff = tf.matmul(ff, blk["fo_w"]) + blk["fo_b"]
                h = self._layer_norm(h + ff, blk["fln_w"], blk["fln_b"],
                                     self.layer_norm_eps)
            hidden_states.append(h)

        if self.do_stable_layer_norm:
            # pre-LN: o LN do encoder é aplicado no FIM
            h = self._layer_norm(h, self.enc_ln_w, self.enc_ln_b,
                                 self.layer_norm_eps)
            hidden_states[-1] = h

        if self.output_hidden_states:
            return tf.stack(hidden_states, axis=1)             # (B, L+1, T', H)
        return h

    def compute_output_shape(self, input_shape):
        if self.output_hidden_states:
            return (input_shape[0], self.num_layers + 1, None, self.hidden_size)
        return (input_shape[0], None, self.hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update({
            "family": self.family,
            "checkpoint": self.checkpoint,
            "output_hidden_states": self.output_hidden_states,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class HiddenStateWeightedSum(layers.Layer):
    """Soma ponderada aprendível sobre as camadas do backbone SSL.

    Receita padrão de tarefas downstream com SSL congelado (SUPERB): em vez de
    usar só o último hidden state, aprende-se um peso por camada — camadas
    intermediárias costumam carregar mais informação fonética/artefatual que a
    última. É um dos poucos parâmetros TREINÁVEIS junto com a cabeça.

    Entrada: ``(B, L+1, T, H)`` → saída: ``(B, T, H)``.
    """

    def build(self, input_shape):
        num_layers = int(input_shape[1])
        self.layer_weights = self.add_weight(
            name="layer_weights", shape=(num_layers,),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        w = tf.nn.softmax(self.layer_weights)
        return tf.einsum("bltc,l->btc", inputs, tf.cast(w, inputs.dtype))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[3])

    def get_config(self):
        return super().get_config()
