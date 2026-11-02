"""Camadas de grafo legadas: desserializam artefatos gravados antes da separação
por arquitetura de 2026-08-20. As em uso estão em ``aasist_layers`` e
``rawgat_layers``. Não corrija aqui o que deveria alcançar um modelo atual. As
``AASIST*`` seguem Jung et al., ICASSP 2022. Saiu de ``layers.py`` (P2-15).
"""

import tensorflow as tf
from tensorflow.keras import layers


class GraphAttentionLayer(layers.Layer):
    """
    Camada de Atenção em Grafos (GAT) simplificada (Self-Attention).
    Combina implementações de AASIST e RawGAT-ST.
    """

    def __init__(self, output_dim: int, num_heads: int = 4, dropout_rate: float = 0.1,
                 use_residual: bool = True, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.kernel = None
        self.bias = None
        self.residual_projection = None

    def build(self, input_shape):
        # Handle both single shape and list of shapes
        if isinstance(input_shape, list):
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape

        input_dim = feature_shape[-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.output_dim * self.num_heads),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim * self.num_heads,),
            initializer="zeros",
            trainable=True
        )

        # Residual connection projection if needed
        if self.use_residual and input_dim != self.output_dim * self.num_heads:
            self.residual_projection = self.add_weight(
                name="residual_projection",
                shape=(input_dim, self.output_dim * self.num_heads),
                initializer="glorot_uniform",
                trainable=True
            )

        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # Handle both single tensor and list inputs
        if isinstance(inputs, list):
            features = inputs[0]
            # adjacency_matrix = inputs[1] # Ignored
        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        num_nodes = tf.shape(features)[1]

        # Linear transformation
        # (batch, nodes, output_dim * num_heads)
        transformed_features = tf.matmul(features, self.kernel)

        # Reshape for multi-head
        # (batch, nodes, heads, output_dim)
        transformed_features = tf.reshape(
            transformed_features, (batch_size, num_nodes, self.num_heads, self.output_dim))

        # Transpose for attention calculation: (batch, heads, nodes, output_dim)
        transformed_features_t = tf.transpose(transformed_features, [0, 2, 1, 3])

        # Scaled Dot-Product Attention
        # (batch, heads, nodes, nodes)
        logits = tf.matmul(transformed_features_t, tf.transpose(transformed_features_t, [0, 1, 3, 2]))
        logits = logits / tf.sqrt(tf.cast(self.output_dim, tf.float32))

        attention_coefs = tf.nn.softmax(logits, axis=-1)

        if training:
            attention_coefs = tf.nn.dropout(attention_coefs, rate=self.dropout_rate)

        # Aggregate
        # (batch, heads, nodes, output_dim)
        outputs = tf.matmul(attention_coefs, transformed_features_t)

        # Reshape back
        # (batch, nodes, heads, output_dim)
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        # (batch, nodes, heads * output_dim)
        outputs = tf.reshape(outputs, (batch_size, num_nodes, self.output_dim * self.num_heads))

        outputs = outputs + self.bias

        # Residual connection
        if self.use_residual:
            if self.residual_projection is not None:
                residual = tf.matmul(features, self.residual_projection)
            else:
                residual = features
            outputs = outputs + residual

        return tf.nn.relu(outputs)

    def get_config(self):
        config = super(GraphAttentionLayer, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual
        })
        return config


class GATConvLayer(layers.Layer):
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
        super(GATConvLayer, self).__init__(**kwargs)
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

        super(GATConvLayer, self).build(input_shape)

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
        config = super(GATConvLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'concat_heads': self.concat_heads,
            'negative_slope': self.negative_slope
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AASISTGraphAttentionLayer(layers.Layer):
    """Atenção de grafo FIEL a RawGAT-ST/AASIST (Tak 2021; Jung ICASSP 2022).

    Diferente do GAT aditivo de Velickovic (``GATConvLayer``), estes artigos
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
class MasterNodeSeed(layers.Layer):
    """Master node treinável ``(1, 1, C)`` replicado para o batch.

    O AASIST declara ``master1``/``master2`` como ``nn.Parameter`` e os injeta
    na primeira HS-GAL de cada ramo. Recebe um tensor apenas para herdar o
    tamanho do batch; o conteúdo dele é ignorado.
    """

    def __init__(self, feature_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)

    def build(self, input_shape):
        self.seed = self.add_weight(
            name="master_seed", shape=(1, 1, self.feature_dim),
            initializer="random_normal", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.tile(tf.cast(self.seed, self.compute_dtype), [batch, 1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, self.feature_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AASISTHtrgGraphAttentionLayer(layers.Layer):
    """HS-GAL heterogênea FIEL ao AASIST (Jung et al., ICASSP 2022, §2.3).

    A contribuição que dá nome à camada são **três conjuntos distintos de
    parâmetros de atenção por TIPO DE ARESTA**: nó-tipo1↔nó-tipo1 (``w11``),
    nó-tipo2↔nó-tipo2 (``w22``) e o par cruzado tipo1↔tipo2 (``w12``,
    compartilhado nas duas direções). Uma atenção homogênea com *type
    embeddings* — como fazia a implementação anterior — não reproduz isso.

    O **master node** agrega o grafo inteiro por uma atenção própria
    (``att_projM``/``att_weightM``) e é devolvido para alimentar a HS-GAL
    seguinte (o "stack"). Quando nenhum master é fornecido, ele é inicializado
    como a MÉDIA de todos os nós, como no código de referência; o AASIST, na
    prática, injeta um master treinável por ramo.

    Entrada: ``[x1, x2]`` ou ``[x1, x2, master]``.
    Saída:   ``(x1', x2', master')``.
    """

    def __init__(self, out_features=32, temperature=100.0, dropout_rate=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        self.dropout_rate = float(dropout_rate)

    def build(self, input_shape):
        in_dim = int(input_shape[0][-1])
        self.proj_type1 = layers.Dense(in_dim, name="proj_type1")
        self.proj_type2 = layers.Dense(in_dim, name="proj_type2")

        self.att_proj = layers.Dense(self.out_features, name="att_proj")
        self.att_projM = layers.Dense(self.out_features, name="att_projM")

        def _att_vec(name):
            return self.add_weight(
                name=name, shape=(self.out_features, 1),
                initializer="glorot_uniform", trainable=True,
            )

        # Três conjuntos por tipo de aresta + o do master (paper §2.3).
        self.att_weight11 = _att_vec("att_weight11")
        self.att_weight22 = _att_vec("att_weight22")
        self.att_weight12 = _att_vec("att_weight12")
        self.att_weightM = _att_vec("att_weightM")

        self.proj_with_att = layers.Dense(self.out_features, name="proj_with_att")
        self.proj_without_att = layers.Dense(
            self.out_features, name="proj_without_att"
        )
        self.proj_with_attM = layers.Dense(
            self.out_features, name="proj_with_attM"
        )
        self.proj_without_attM = layers.Dense(
            self.out_features, name="proj_without_attM"
        )
        self.bn = layers.BatchNormalization(name="bn")
        self.input_drop = layers.Dropout(self.dropout_rate)

        # Build EXPLÍCITO — ver nota em AASISTGraphAttentionLayer.build().
        self.proj_type1.build((None, None, in_dim))
        self.proj_type2.build((None, None, in_dim))
        self.att_proj.build((None, None, None, in_dim))  # tensor par-a-par
        self.att_projM.build((None, None, in_dim))
        self.proj_with_att.build((None, None, in_dim))
        self.proj_without_att.build((None, None, in_dim))
        self.proj_with_attM.build((None, None, in_dim))
        self.proj_without_attM.build((None, None, in_dim))
        self.bn.build((None, None, self.out_features))
        super().build(input_shape)

    def _derive_att_map(self, x, n1):
        pairwise = tf.expand_dims(x, 2) * tf.expand_dims(x, 1)   # (B,N,N,C)
        att = tf.tanh(self.att_proj(pairwise))                    # (B,N,N,out)

        # Blocos por tipo de aresta: 11 (tipo1↔tipo1), 22 (tipo2↔tipo2) e 12
        # (cruzado, compartilhado nas duas direções) — §2.3 do AASIST.
        a11 = tf.matmul(att[:, :n1, :n1, :], tf.cast(self.att_weight11, att.dtype))
        a12 = tf.matmul(att[:, :n1, n1:, :], tf.cast(self.att_weight12, att.dtype))
        a21 = tf.matmul(att[:, n1:, :n1, :], tf.cast(self.att_weight12, att.dtype))
        a22 = tf.matmul(att[:, n1:, n1:, :], tf.cast(self.att_weight22, att.dtype))

        top = tf.concat([a11, a12], axis=2)      # (B, n1, N, 1)
        bottom = tf.concat([a21, a22], axis=2)   # (B, n2, N, 1)
        att_map = tf.concat([top, bottom], axis=1) / tf.cast(
            self.temperature, att.dtype
        )
        return tf.nn.softmax(att_map, axis=-2)

    def _update_master(self, x, master):
        att = tf.tanh(self.att_projM(x * master))                 # (B,N,out)
        att = tf.matmul(att, tf.cast(self.att_weightM, att.dtype))  # (B,N,1)
        att = att / tf.cast(self.temperature, att.dtype)
        att = tf.nn.softmax(att, axis=-2)
        # (B,1,N) @ (B,N,C) -> (B,1,C)
        pooled = tf.matmul(tf.transpose(att, [0, 2, 1]), x)
        return self.proj_with_attM(pooled) + self.proj_without_attM(master)

    def call(self, inputs, training=None):
        if len(inputs) == 3:
            x1, x2, master = inputs
        else:
            x1, x2 = inputs
            master = None

        n1 = x1.shape[1] if x1.shape[1] is not None else tf.shape(x1)[1]

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = tf.concat([x1, x2], axis=1)

        if master is None:
            master = tf.reduce_mean(x, axis=1, keepdims=True)

        x = self.input_drop(x, training=training)
        att_map = tf.squeeze(self._derive_att_map(x, n1), axis=-1)  # (B,N,N)
        master = self._update_master(x, master)

        out = self.proj_with_att(tf.matmul(att_map, x)) + self.proj_without_att(x)
        out = self.bn(out, training=training)
        out = tf.nn.selu(out)

        return out[:, :n1, :], out[:, n1:, :], master

    def compute_output_shape(self, input_shape):
        s1, s2 = input_shape[0], input_shape[1]
        return (
            (s1[0], s1[1], self.out_features),
            (s2[0], s2[1], self.out_features),
            (s1[0], 1, self.out_features),
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "temperature": self.temperature,
            "dropout_rate": self.dropout_rate,
        })
        return config


class GraphPoolLayer(layers.Layer):
    """Learnable graph pooling via top-k node selection.

    Computes a learned score per node, selects the top-k nodes (k = ratio * N),
    and gates the selected node features by their sigmoid scores.

    Reference: Graph U-Nets (Gao & Ji, 2019)
    """

    def __init__(self, ratio=0.5, target_nodes=None, **kwargs):
        super(GraphPoolLayer, self).__init__(**kwargs)
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

        super(GraphPoolLayer, self).build(input_shape)

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
        config = super(GraphPoolLayer, self).get_config()
        config.update({'ratio': self.ratio, 'target_nodes': self.target_nodes})
        return config


class GraphReadoutLayer(layers.Layer):
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
        super(GraphReadoutLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GraphReadoutLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, nodes, features)
        h_max = tf.reduce_max(tf.abs(inputs), axis=1)  # (batch, features)
        h_avg = tf.reduce_mean(inputs, axis=1)         # (batch, features)
        return tf.concat([h_max, h_avg], axis=-1)      # (batch, 2*features)

    def get_config(self):
        return super(GraphReadoutLayer, self).get_config()


class HSGALLayer(layers.Layer):
    """Heterogeneous Stacking Graph Attention Layer.

    The key contribution of AASIST: cross-domain attention between spectral
    and temporal graph nodes, enabling integrated spectro-temporal analysis.

    Takes (spectral_nodes, temporal_nodes) as input and produces updated
    representations for both via cross-attention followed by self-attention
    on the combined heterogeneous graph.

    Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
    Spectro-Temporal Graph Attention Networks", ICASSP 2022
    """

    def __init__(self, out_features, num_heads=2, dropout_rate=0.1, **kwargs):
        super(HSGALLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape is a list of two shapes: [spectral_shape, temporal_shape]
        spec_features = input_shape[0][-1]
        temp_features = input_shape[1][-1]

        # Cross-attention: spectral queries attend to temporal keys/values
        self.W_q_s2t = self.add_weight(
            name="W_q_s2t", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_k_s2t = self.add_weight(
            name="W_k_s2t", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_v_s2t = self.add_weight(
            name="W_v_s2t", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)

        # Cross-attention: temporal queries attend to spectral keys/values
        self.W_q_t2s = self.add_weight(
            name="W_q_t2s", shape=(temp_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_k_t2s = self.add_weight(
            name="W_k_t2s", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)
        self.W_v_t2s = self.add_weight(
            name="W_v_t2s", shape=(spec_features, self.out_features * self.num_heads),
            initializer="glorot_uniform", trainable=True)

        # Attention vectors for additive GAT (per cross-attention direction)
        self.a_s2t = self.add_weight(
            name="a_s2t", shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform", trainable=True)
        self.a_t2s = self.add_weight(
            name="a_t2s", shape=(self.num_heads, self.out_features, 1),
            initializer="glorot_uniform", trainable=True)

        # Self-attention GAT on combined graph
        combined_features = self.out_features * self.num_heads
        self.gat_self = GATConvLayer(
            out_features=self.out_features, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, concat_heads=True,
            name=self.name + "_gat_self")
        self.gat_self.build(tf.TensorShape([None, None, combined_features]))

        # Layer norms
        self.ln_spec = layers.LayerNormalization(name=self.name + "_ln_spec")
        self.ln_temp = layers.LayerNormalization(name=self.name + "_ln_temp")

        super(HSGALLayer, self).build(input_shape)

    def _cross_attention(self, queries, keys_values, W_q, W_k, W_v, a_vec, training=None):
        """Additive cross-attention between two sets of nodes."""
        batch_size = tf.shape(queries)[0]
        n_q = tf.shape(queries)[1]
        n_kv = tf.shape(keys_values)[1]

        Q = tf.matmul(queries, W_q)       # (batch, n_q, out*heads)
        K = tf.matmul(keys_values, W_k)   # (batch, n_kv, out*heads)
        V = tf.matmul(keys_values, W_v)   # (batch, n_kv, out*heads)

        # Reshape to (batch, heads, nodes, out_features)
        Q = tf.reshape(Q, (batch_size, n_q, self.num_heads, self.out_features))
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.reshape(K, (batch_size, n_kv, self.num_heads, self.out_features))
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, (batch_size, n_kv, self.num_heads, self.out_features))
        V = tf.transpose(V, [0, 2, 1, 3])

        # Additive attention: e_ij = LeakyReLU(a^T * (Q_i + K_j))
        # a_vec: (heads, out_features, 1)
        e_q = tf.einsum('bhni,hio->bhno', Q, a_vec)   # (batch, heads, n_q, 1)
        e_k = tf.einsum('bhni,hio->bhno', K, a_vec)   # (batch, heads, n_kv, 1)

        e = e_q + tf.transpose(e_k, [0, 1, 3, 2])  # (batch, heads, n_q, n_kv)
        e = tf.nn.leaky_relu(e, alpha=0.2)

        alpha = tf.nn.softmax(e, axis=-1)

        if training:
            alpha = tf.nn.dropout(alpha, rate=self.dropout_rate)

        # Aggregate: (batch, heads, n_q, out_features)
        out = tf.matmul(alpha, V)
        # (batch, n_q, heads, out_features)
        out = tf.transpose(out, [0, 2, 1, 3])
        # (batch, n_q, heads * out_features)
        out = tf.reshape(out, (batch_size, n_q, self.num_heads * self.out_features))

        return out

    def call(self, inputs, training=None):
        spectral_nodes, temporal_nodes = inputs

        # Cross-attention: spectral attends to temporal
        spec_cross = self._cross_attention(
            spectral_nodes, temporal_nodes,
            self.W_q_s2t, self.W_k_s2t, self.W_v_s2t, self.a_s2t,
            training=training)

        # Cross-attention: temporal attends to spectral
        temp_cross = self._cross_attention(
            temporal_nodes, spectral_nodes,
            self.W_q_t2s, self.W_k_t2s, self.W_v_t2s, self.a_t2s,
            training=training)

        # Layer norm
        spec_cross = self.ln_spec(spec_cross)
        temp_cross = self.ln_temp(temp_cross)

        # Combine into heterogeneous graph for self-attention
        combined = tf.concat([spec_cross, temp_cross], axis=1)  # (batch, n_spec+n_temp, features)
        combined = self.gat_self(combined, training=training)

        # Split back
        n_spec = tf.shape(spec_cross)[1]
        spectral_out = combined[:, :n_spec, :]
        temporal_out = combined[:, n_spec:, :]

        return spectral_out, temporal_out

    def get_config(self):
        config = super(HSGALLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class SpectralPositionEmbedding(layers.Layer):
    """Embedding posicional treinável para os nós espectrais."""

    def build(self, input_shape):
        if input_shape[1] is None or input_shape[-1] is None:
            raise ValueError("SpectralPositionEmbedding exige dimensões estáticas")
        self.position = self.add_weight(
            name="position",
            shape=(1, int(input_shape[1]), int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.cast(self.position, inputs.dtype)


@tf.keras.utils.register_keras_serializable(package="XFakeSong")
class AdaptiveGraphResize(layers.Layer):
    """Projeção aprendível do eixo de nós para tamanho comum S/T.

    LEGADO (desde 2026-07-27) — nenhum builder a instancia. Uma projeção densa
    sobre o eixo de nós MISTURA nós arbitrariamente (não é operação de grafo) e
    não existe em Tak et al. (2021). O alinhamento de Gs/Gt antes da fusão
    element-wise passou a usar ``GraphPoolLayer(target_nodes=...)``, o mesmo
    top-k dos artigos. Mantida só para desserializar checkpoints antigos.
    """

    def __init__(self, target_nodes=12, **kwargs):
        super().__init__(**kwargs)
        self.target_nodes = int(target_nodes)

    def build(self, input_shape):
        if input_shape[1] is None:
            raise ValueError(
                "AdaptiveGraphResize exige número de nós estático"
            )
        self.projection = layers.Dense(
            self.target_nodes,
            use_bias=True,
            name=f"{self.name}_node_projection",
        )
        # Build explícito: como a camada define compute_output_shape(), o
        # Keras 3 pode montar o grafo funcional via inferência de shape sem
        # nunca tracejar call() — o que deixaria `projection` com built=False
        # (sem variáveis) até a primeira chamada real.
        self.projection.build((input_shape[0], input_shape[-1], input_shape[1]))
        super().build(input_shape)

    def call(self, inputs):
        transposed = tf.transpose(inputs, [0, 2, 1])
        projected = self.projection(transposed)
        return tf.transpose(projected, [0, 2, 1])

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            (input_shape[0], self.target_nodes, input_shape[-1])
        )

    def get_config(self):
        config = super().get_config()
        config.update({"target_nodes": self.target_nodes})
        return config
