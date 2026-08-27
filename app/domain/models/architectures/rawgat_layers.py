"""Camadas do RawGAT-ST (Tak et al., 2021) — exclusivas desta arquitetura.

Todas as classes deste modulo pertencem AO RawGAT-ST e a mais nenhuma
arquitetura do projeto. Elas foram extraidas em 2026-08-20 de
``app/domain/models/architectures/layers.py``, onde viviam compartilhadas com o
AASIST (e, no caso do banco sinc, com o RawNet2).

Por que a separacao existe
--------------------------
Enquanto a definicao das camadas era compartilhada, uma correcao de FIDELIDADE
A UM PAPER mudava mais de uma entrada oficial do benchmark de uma vez, em
silencio: mexer no ``GraphReadoutLayer`` ou no ``SincConvLayer`` alterava
AASIST e RawGAT-ST juntos, sem que a mudanca de uma arquitetura declarasse a
outra. Isso e inaceitavel num benchmark comparativo — cada linha da tabela
precisa ser um sistema com historia propria. Com uma copia por arquitetura, uma
correcao ao RawGAT-ST fica contida no RawGAT-ST.

Nomes com prefixo ``RawGat``
----------------------------
``@tf.keras.utils.register_keras_serializable`` registra a classe PELO NOME.
Duas classes com o mesmo nome fariam o segundo registro sobrescrever o
primeiro, e um artefato salvo desserializaria com a classe errada. Por isso
cada copia recebeu um nome novo e unico:

============================  ==========================
Nome em ``layers.py``         Nome aqui
============================  ==========================
``SincConvLayer``             ``RawGatSincConv``
``ResidualBlock1D``           ``RawGatResidualBlock1D``
``ResidualBlock2D``           ``RawGatResidualBlock2D``
``GATConvLayer``              ``RawGatGATConv``
``AASISTGraphAttentionLayer`` ``RawGatGraphAttention``
``GraphPoolLayer``            ``RawGatGraphPool``
``GraphReadoutLayer``         ``RawGatGraphReadout``
============================  ==========================

O corpo das classes e copia fiel da origem — mesma logica, mesmos defaults e
mesmos comentarios, que registram as correcoes de fidelidade de 2026-08-20
(readout ``max(|x|)`` + media; ``top_k(sorted=True)`` para fixar o pareamento
da fusao elemento a elemento; kernel ``(2, 3)`` com ``first=True`` no bloco
residual 2D; banco sinc FIXO). O helper de DSP ``build_sinc_bandpass_filters``
continua vindo de ``layers.py`` de proposito: e matematica generica de
passa-banda, nao uma decisao de arquitetura.
"""

import tensorflow as tf
from tensorflow.keras import layers

from app.domain.models.architectures.layers import build_sinc_bandpass_filters

__all__ = [
    "RawGatSincConv",
    "RawGatResidualBlock1D",
    "RawGatResidualBlock2D",
    "RawGatGATConv",
    "RawGatGraphAttention",
    "RawGatGraphPool",
    "RawGatGraphReadout",
]


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatSincConv(layers.Layer):
    """Learnable sinc-based bandpass filter convolution (from SincNet/RawNet2).

    Implements parameterized sinc filters where low and high cutoff frequencies
    are learnable parameters initialized on the mel scale. Each filter is a
    bandpass filter: sinc(2*pi*f_high*t) - sinc(2*pi*f_low*t), windowed by Hamming.

    Reference: Ravanelli & Bengio, "Speaker Recognition from Raw Waveform with SincNet", 2018
    """

    def __init__(self, n_filters=70, kernel_size=129, sample_rate=16000,
                 min_low_hz=50.0, min_band_hz=50.0, trainable_filters=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        # BANCO FIXO por padrao (2026-08-20), como a referencia.
        #
        # No codigo oficial do RawGAT-ST e do AASIST (classe `CONV`), o banco
        # sinc e montado no forward a partir de pontos mel calculados uma vez
        # no `__init__` e guardado num `torch.Tensor` — NAO um `nn.Parameter`,
        # portanto sem gradiente. Aqui as frequencias de corte eram pesos
        # treinaveis: um grau de liberdade a mais na camada que define o que o
        # modelo enxerga, ausente no baseline com que a tabela do TCC compara.
        # `trainable_filters=True` mantem o comportamento antigo para ablacao.
        self.trainable_filters = bool(trainable_filters)

    def _hz_to_mel(self, hz):
        return 2595.0 * tf.math.log(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def build(self, input_shape):
        # Initialize filter frequencies on mel scale
        low_hz = self.min_low_hz
        high_hz = self.sample_rate / 2.0

        mel_low = self._hz_to_mel(tf.constant(low_hz, dtype=tf.float32))
        mel_high = self._hz_to_mel(tf.constant(high_hz, dtype=tf.float32))

        # n_filters + 1 points on mel scale, then convert back to Hz
        mel_points = tf.linspace(mel_low, mel_high, self.n_filters + 1)
        hz_points = self._mel_to_hz(mel_points)

        # Low frequencies and bandwidths
        init_low = hz_points[:-1]  # (n_filters,)
        init_band = hz_points[1:] - hz_points[:-1]  # (n_filters,)

        self.low_hz_ = self.add_weight(
            name="low_hz",
            shape=(self.n_filters,),
            initializer=tf.keras.initializers.Constant(init_low.numpy()),
            trainable=self.trainable_filters
        )
        self.band_hz_ = self.add_weight(
            name="band_hz",
            shape=(self.n_filters,),
            initializer=tf.keras.initializers.Constant(init_band.numpy()),
            trainable=self.trainable_filters
        )

        # Hamming window (not trainable)
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        self.window_ = 0.54 - 0.46 * tf.cos(2.0 * 3.14159265 * n / (self.kernel_size - 1))

        super().build(input_shape)

    def _sinc(self, x):
        """Normalized sinc function: sin(x) / x, with sinc(0) = 1."""
        safe_x = tf.where(tf.abs(x) < 1e-7, tf.ones_like(x) * 1e-7, x)
        return tf.sin(safe_x) / safe_x

    def call(self, inputs):
        # inputs: (batch, time, 1)
        target_dtype = self.compute_dtype
        inputs = tf.cast(inputs, tf.float32)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Ensure positive frequencies
        min_low_hz = tf.cast(self.min_low_hz, tf.float32)
        min_band_hz = tf.cast(self.min_band_hz, tf.float32)
        nyquist = tf.cast(self.sample_rate / 2.0, tf.float32)
        low = min_low_hz + tf.abs(tf.cast(self.low_hz_, tf.float32))
        high = tf.clip_by_value(
            low + min_band_hz + tf.abs(tf.cast(self.band_hz_, tf.float32)),
            clip_value_min=min_low_hz,
            clip_value_max=nyquist,
        )

        # Eixo temporal centrado, em SEGUNDOS (n − centro)/fs.
        n = tf.cast(tf.range(0, self.kernel_size), tf.float32)
        n = (n - (self.kernel_size - 1.0) / 2.0) / self.sample_rate

        # Construção dos filtros por `build_sinc_bandpass_filters` — a MESMA
        # função usada pelo SincNetLayer (RawNet2). Antes cada camada mantinha
        # sua própria cópia da matemática do sinc; a daqui já teve um bug de
        # argumento (faltava π e sobrava ×fs, degenerando os passa-banda
        # mel-inicializados num banco pseudo-aleatório) que a outra nunca teve.
        # Uma implementação só elimina a classe inteira de divergência.
        # Verificado numericamente: saída idêntica à versão anterior (1e-5).
        band_pass = build_sinc_bandpass_filters(
            low=low,
            high=high,
            n_time=n,
            window=tf.cast(self.window_, tf.float32),
            normalize="l1",   # energia unitária (convenção desta camada)
        )  # (kernel_size, n_filters)

        # Reshape for conv1d: (kernel_size, 1, n_filters)
        filters = tf.expand_dims(band_pass, 1)

        # Apply convolution
        output = tf.nn.conv1d(inputs, filters, stride=1, padding='SAME')

        return tf.cast(output, target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz,
            # Sem isto, um artefato salvo com banco treinavel recarregaria com
            # banco fixo (ou vice-versa) em silencio.
            'trainable_filters': self.trainable_filters,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatResidualBlock1D(layers.Layer):
    """Pre-activation residual block for 1D convolutions.

    Structure: BN -> LeakyReLU -> Conv1D -> BN -> LeakyReLU -> Conv1D + skip.
    Uses 1x1 convolution for skip connection if channel mismatch.

    Reference: RawNet2 (Tak et al., 2021)
    """

    def __init__(self, out_channels, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def build(self, input_shape):
        in_channels = input_shape[-1]

        self.bn1 = layers.BatchNormalization(name=self.name + "_bn1")
        self.conv1 = layers.Conv1D(
            self.out_channels, self.kernel_size, padding='same',
            name=self.name + "_conv1")
        self.bn2 = layers.BatchNormalization(name=self.name + "_bn2")
        self.conv2 = layers.Conv1D(
            self.out_channels, self.kernel_size, padding='same',
            name=self.name + "_conv2")

        self.skip_conv = None
        if in_channels != self.out_channels:
            self.skip_conv = layers.Conv1D(
                self.out_channels, 1, padding='same',
                name=self.name + "_skip")

        super().build(input_shape)

    def call(self, inputs, training=None):
        # Pre-activation residual
        x = self.bn1(inputs, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.3)
        x = self.conv2(x)

        # Skip connection
        shortcut = inputs
        if self.skip_conv is not None:
            shortcut = self.skip_conv(inputs)

        # Ver RawGatResidualBlock2D: mesmo risco de dtype divergente sob mixed
        # precision na reconstrução simbólica do modelo salvo.
        shortcut = tf.cast(shortcut, x.dtype)
        return x + shortcut

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatResidualBlock2D(layers.Layer):
    """Bloco residual 2D para os encoders AASIST/RawGAT-ST fiéis."""

    def __init__(self, out_channels, kernel_size=(2, 3), pool_size=(1, 3),
                 first=False, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        # KERNEL (2, 3), como o `Residual_block` da familia RawGAT-ST/AASIST.
        # O default era (3, 3): 50% mais parametros por convolucao no eixo de
        # frequencia e um campo receptivo diferente do baseline publicado.
        self.kernel_size = tuple(kernel_size)
        self.pool_size = tuple(pool_size)
        # `first=True` PULA a BN + ativacao iniciais, como na referencia. Sem
        # isto, o primeiro bloco da pilha aplicava BN+SELU sobre um tensor que
        # ja vinha normalizado e ativado do front-end sinc — uma segunda
        # saturacao do SELU que o paper nao tem.
        self.first = bool(first)

    def build(self, input_shape):
        # Com `first=True` a pre-ativacao nao roda, entao `bn1` seria um peso
        # orfao (sem gradiente, e o Keras avisa a cada passo).
        self.bn1 = (
            None if self.first
            else layers.BatchNormalization(name=f"{self.name}_bn1")
        )
        self.conv1 = layers.Conv2D(
            self.out_channels, self.kernel_size, padding="same",
            use_bias=False, name=f"{self.name}_conv1",
        )
        self.bn2 = layers.BatchNormalization(name=f"{self.name}_bn2")
        self.conv2 = layers.Conv2D(
            self.out_channels, self.kernel_size, padding="same",
            use_bias=False, name=f"{self.name}_conv2",
        )
        self.skip_conv = None
        if input_shape[-1] != self.out_channels:
            self.skip_conv = layers.Conv2D(
                self.out_channels, 1, padding="same", use_bias=False,
                name=f"{self.name}_skip",
            )
        self.pool = layers.MaxPooling2D(
            pool_size=self.pool_size, strides=self.pool_size,
            padding="same", name=f"{self.name}_pool",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.first:
            x = inputs
        else:
            x = self.bn1(inputs, training=training)
            x = tf.nn.selu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.selu(x)
        x = self.conv2(x)
        shortcut = self.skip_conv(inputs) if self.skip_conv is not None else inputs
        # Sob mixed_float16, a reconstrução simbólica do modelo salvo (Keras 3
        # traça call() com uma policy de dtype diferente da usada no treino)
        # pode entregar `x` e `shortcut` em dtypes distintos, quebrando o Add
        # (float32 x float16). Cast explícito evita depender da policy global.
        shortcut = tf.cast(shortcut, x.dtype)
        return self.pool(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "first": self.first,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatGATConv(layers.Layer):
    """Graph Attention Network layer with ADDITIVE attention (Velickovic et al., 2018).

    Uses the original GAT attention mechanism:
        e_ij = LeakyReLU(a_src^T * W*h_i + a_dst^T * W*h_j)
        alpha_ij = softmax_j(e_ij)
        h'_i = sum_j(alpha_ij * W*h_j)

    This is distinct from the existing GraphAttentionLayer which uses
    scaled dot-product (Transformer-style) attention.

    Reference: Velickovic et al., "Graph Attention Networks", ICLR 2018
    """

    def __init__(self, out_features, num_heads=1, dropout_rate=0.1,
                 concat_heads=True, negative_slope=0.2, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope

    def build(self, input_shape):
        in_features = input_shape[-1]

        # Linear transformation weight: shared across heads but output is out_features * num_heads
        self.W = self.add_weight(
            name="W",
            shape=(in_features, self.out_features * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )

        # Attention vectors: one pair (a_src, a_dst) per head
        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.out_features * self.num_heads if self.concat_heads else self.out_features,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, nodes, in_features)
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]

        # Linear transform: (batch, nodes, out_features * num_heads)
        h = tf.matmul(inputs, self.W)

        # Reshape to (batch, nodes, num_heads, out_features)
        h = tf.reshape(h, (batch_size, num_nodes, self.num_heads, self.out_features))

        # Transpose to (batch, num_heads, nodes, out_features)
        h = tf.transpose(h, [0, 2, 1, 3])

        # Compute attention scores using additive mechanism
        # e_src: (batch, heads, nodes, 1)
        e_src = tf.einsum('bhni,hio->bhno', h, self.a_src)
        # e_dst: (batch, heads, nodes, 1)
        e_dst = tf.einsum('bhni,hio->bhno', h, self.a_dst)

        # e_ij = LeakyReLU(e_src_i + e_dst_j)
        # Broadcasting: (batch, heads, nodes, 1) + (batch, heads, 1, nodes)
        e = e_src + tf.transpose(e_dst, [0, 1, 3, 2])
        e = tf.nn.leaky_relu(e, alpha=self.negative_slope)

        # Attention coefficients
        alpha = tf.nn.softmax(e, axis=-1)  # (batch, heads, nodes, nodes)

        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)

        # Weighted aggregation: (batch, heads, nodes, out_features)
        out = tf.matmul(alpha, h)

        # Reshape back
        # (batch, nodes, heads, out_features)
        out = tf.transpose(out, [0, 2, 1, 3])

        if self.concat_heads:
            # (batch, nodes, heads * out_features)
            out = tf.reshape(out, (batch_size, num_nodes, self.num_heads * self.out_features))
        else:
            # Average heads: (batch, nodes, out_features)
            out = tf.reduce_mean(out, axis=2)

        out = out + self.bias
        return tf.nn.elu(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'concat_heads': self.concat_heads,
            'negative_slope': self.negative_slope
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatGraphAttention(layers.Layer):
    """Atenção de grafo FIEL a RawGAT-ST/AASIST (Tak 2021; Jung ICASSP 2022).

    Diferente do GAT aditivo de Velickovic (``RawGatGATConv``), estes artigos
    derivam o mapa de atenção do **produto elemento a elemento entre pares de
    nós**, projetado e comprimido por ``tanh``, reduzido a um escalar por uma
    direção aprendível e **escalado por uma temperatura** antes do softmax::

        A_ij = softmax_i( w^T · tanh(W_att (h_i ⊙ h_j)) / τ )
        h'   = W_att_proj (A · h) + W_res h          (projeção com e sem atenção)
        h'   = SELU(BN(h'))

    A temperatura é um hiperparâmetro por camada no AASIST (2.0 nos GATs
    espectral/temporal, 100.0 nas HS-GAL) e controla o quanto a atenção se
    aproxima de uma média uniforme.

    NOTA: o softmax é normalizado sobre ``axis=-2`` e a agregação soma sobre o
    último eixo — exatamente como no código de referência dos autores.
    """

    def __init__(self, out_features, temperature=1.0, dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout_rate)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.att_proj = layers.Dense(self.out_features, name="att_proj")
        self.att_weight = self.add_weight(
            name="att_weight", shape=(self.out_features, 1),
            initializer="glorot_uniform", trainable=True,
        )
        self.proj_with_att = layers.Dense(self.out_features, name="proj_with_att")
        self.proj_without_att = layers.Dense(
            self.out_features, name="proj_without_att"
        )
        self.bn = layers.BatchNormalization(name="bn")
        self.input_drop = layers.Dropout(self.dropout_rate)

        # Build EXPLÍCITO das sub-camadas: criadas aqui, elas ficariam com
        # `built=False` na reconstrução do modelo salvo e o Keras 3 aborta o
        # load ("objects could not be loaded ... Dense name=att_proj").
        self.att_proj.build((None, None, None, in_dim))   # tensor par-a-par
        self.proj_with_att.build((None, None, in_dim))
        self.proj_without_att.build((None, None, in_dim))
        self.bn.build((None, None, self.out_features))
        super().build(input_shape)

    def _derive_att_map(self, x):
        # Produto par-a-par: (B, N, 1, C) * (B, 1, N, C) -> (B, N, N, C)
        pairwise = tf.expand_dims(x, 2) * tf.expand_dims(x, 1)
        att = tf.tanh(self.att_proj(pairwise))          # (B, N, N, out)
        att = tf.matmul(att, tf.cast(self.att_weight, att.dtype))  # (B, N, N, 1)
        att = att / tf.cast(self.temperature, att.dtype)
        return tf.nn.softmax(att, axis=-2)

    def call(self, inputs, training=None):
        x = self.input_drop(inputs, training=training)
        att_map = tf.squeeze(self._derive_att_map(x), axis=-1)  # (B, N, N)
        out = self.proj_with_att(tf.matmul(att_map, x)) + self.proj_without_att(x)
        out = self.bn(out, training=training)
        return tf.nn.selu(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_features)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "temperature": self.temperature,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatGraphPool(layers.Layer):
    """Learnable graph pooling via top-k node selection.

    Computes a learned score per node, selects the top-k nodes (k = ratio * N),
    and gates the selected node features by their sigmoid scores.

    Reference: Graph U-Nets (Gao & Ji, 2019)
    """

    def __init__(self, ratio=0.5, target_nodes=None, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        # `target_nodes` seleciona um número ABSOLUTO de nós (top-k), em vez de
        # uma fração. Serve para alinhar dois grafos antes de uma fusão
        # element-wise usando a MESMA primitiva de pooling dos artigos, sem
        # recorrer a uma projeção densa sobre o eixo de nós (que mistura nós
        # arbitrariamente e não é uma operação de grafo).
        self.target_nodes = None if target_nodes is None else int(target_nodes)

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.score_proj = self.add_weight(
            name="score_proj",
            shape=(in_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)
        num_nodes = tf.shape(inputs)[1]
        if self.target_nodes is not None:
            k = tf.minimum(tf.constant(self.target_nodes, tf.int32), num_nodes)
            k = tf.maximum(k, 1)
        else:
            k = tf.maximum(
                tf.cast(tf.cast(num_nodes, tf.float32) * self.ratio, tf.int32), 1
            )

        # Compute scores: (batch, nodes, 1)
        scores = tf.matmul(inputs, self.score_proj)
        scores = tf.squeeze(scores, axis=-1)  # (batch, nodes)

        # Top-k selection.
        #
        # `sorted=True` NÃO é cosmético aqui. Com `sorted=False` o TensorFlow
        # declara a ordem do resultado como não especificada, e medido nesta
        # base (TF 2.21, CPU) ela difere da ordenada em 200 de 200 casos. Essa
        # ordem define QUAIS nós se emparelham na fusão elemento a elemento do
        # RawGAT-ST (`rawgat_st.py`: Multiply entre Gs e Gt), a operação que dá
        # nome à arquitetura — permutar os nós de um dos lados muda o produto
        # em ~130% relativo. Dentro de um run a ordem é estável, então o modelo
        # aprende com o pareamento que recebeu; o risco é de REPRODUTIBILIDADE:
        # outra versão do TF ou o kernel de GPU podem parear diferente e
        # produzir outro modelo a partir do mesmo código.
        #
        # `sorted=True` fixa o pareamento por ranking de score (nó espectral
        # mais saliente com nó temporal mais saliente), que é também a
        # semântica que a referência PyTorch obtém de `torch.topk`.
        _, top_indices = tf.math.top_k(scores, k=k, sorted=True)

        # Gather selected nodes
        batch_size = tf.shape(inputs)[0]

        # Create batch indices for gather_nd
        batch_indices = tf.repeat(
            tf.expand_dims(tf.range(batch_size), 1), k, axis=1
        )  # (batch, k)

        indices = tf.stack([batch_indices, top_indices], axis=-1)  # (batch, k, 2)
        selected_features = tf.gather_nd(inputs, indices)  # (batch, k, features)
        selected_scores = tf.gather_nd(scores, indices)    # (batch, k)

        # Gate by sigmoid of scores
        gate = tf.nn.sigmoid(selected_scores)  # (batch, k)
        gate = tf.expand_dims(gate, -1)        # (batch, k, 1)

        return selected_features * gate

    def compute_output_shape(self, input_shape):
        nodes = input_shape[1]
        if self.target_nodes is not None:
            pooled_nodes = (
                min(self.target_nodes, int(nodes)) if nodes is not None
                else self.target_nodes
            )
        else:
            pooled_nodes = (
                max(int(nodes * self.ratio), 1)
                if nodes is not None
                else None
            )
        return tf.TensorShape((input_shape[0], pooled_nodes, input_shape[-1]))

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio, 'target_nodes': self.target_nodes})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class RawGatGraphReadout(layers.Layer):
    """Readout de grafo: máximo do VALOR ABSOLUTO concatenado com a MÉDIA.

    Formulação de AASIST (Jung et al., ICASSP 2022, §2.4) e RawGAT-ST (Tak et
    al., 2021)::

        T_max = max(|out|, dim=nós)
        T_avg = mean(out, dim=nós)
        readout = concat[T_max, T_avg]

    Saída: ``(batch, 2 * in_features)``.

    CORREÇÃO 2026-08-20. A implementação anterior fazia ``reduce_max`` SEM
    valor absoluto e, no lugar da média, uma soma ponderada por atenção
    aprendida (um peso ``att_w`` de ``in_features`` parâmetros). Divergia dos
    dois papers em duas frentes:

    - **Sem o abs**, os nós chegam aqui depois do SELU, cujo alcance é
      ``(-1.758, +inf)``: um canal fortemente NEGATIVO — evidência tão válida
      quanto uma positiva — era descartado pelo máximo. O paper usa a magnitude
      justamente para não perder esse lado.
    - **Atenção no lugar da média** troca uma estatística fixa por uma
      aprendida, mudando metade do vetor que alimenta a camada de saída.

    O AASIST deste mesmo repositório já monta o readout CORRETO inline
    (``aasist.py``: ``MagnitudeLayer`` + ``GlobalMaxPooling1D`` e
    ``GlobalAveragePooling1D``); era o RawGAT-ST, que consome esta camada
    compartilhada, que ficava com a versão divergente.

    A forma de saída não muda, mas o peso ``att_w`` deixa de existir: artefatos
    salvos com a versão anterior não recarregam sem retreino.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)
        h_max = tf.reduce_max(tf.abs(inputs), axis=1)  # (batch, features)
        h_avg = tf.reduce_mean(inputs, axis=1)         # (batch, features)
        return tf.concat([h_max, h_avg], axis=-1)      # (batch, 2*features)

    def get_config(self):
        return super().get_config()
