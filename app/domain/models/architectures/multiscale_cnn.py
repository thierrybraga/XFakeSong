"""Multiscale CNN Architecture Implementation — Res2Net

Paper-faithful implementation based on:
Gao, S.-H.; Cheng, M.-M.; Zhao, K.; Zhang, X.-Y.; Yang, M.-H.; Torr, P.
"Res2Net: A New Multi-Scale Backbone Architecture"
IEEE TPAMI, 2021, 43(2), 652-662. DOI: 10.1109/TPAMI.2019.2938758

Core idea: Hierarchical residual-like connections WITHIN a single residual block.
The Bottle2neck block splits features into s groups (scale dimension) and processes
them sequentially: y_i = K_i(x_i + y_{i-1}), creating multi-scale representations
at a granular level.

Configuration:
- Res2Net-50: [3, 4, 6, 3] Bottle2neck blocks, baseWidth=26, scale=4
- Adapted for 2D spectrogram input (audio deepfake detection)
"""

# Third-party imports
import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from app.core.utils.audio_utils import preprocess_legacy as preprocess
from app.domain.models.architectures.layers import (
    ResizeLayer,
    SqueezeExcitationBlock2D,
    SqueezeExciteBlock,
    STFTLayer,
    ensure_flat_input,
    is_raw_audio,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Res2Net core block: Bottle2neck (Gao et al., 2021)
# ---------------------------------------------------------------------------

class Bottle2neck(layers.Layer):
    """Res2Net Bottle2neck block — paper-faithful implementation.

    Architecture per Gao et al., 2021, Section 3.2:
    1. 1×1 Conv (reduce channels to width * scale)
    2. Split into `scale` groups along channel axis, each with `width` channels
    3. Hierarchical processing:
       - Group 0: pass-through (identity)
       - Group i (i>0): y_i = BN(Conv3x3(x_i + y_{i-1}))
    4. Concatenate all group outputs
    5. 1×1 Conv (expand to planes * expansion)
    6. Residual connection + ReLU

    Args:
        planes: Base number of output channels (before expansion)
        base_width: Width of each scale group (paper default: 26)
        scale: Number of scale groups s (paper default: 4)
        stride: Stride for downsampling (1 or 2)
        expansion: Channel expansion factor (4 for bottleneck)
        stype: 'normal' for standard block, 'stage' for first block of each stage
    """
    expansion = 4

    def __init__(self, planes, base_width=26, scale=4, stride=1,
                 stype='normal', **kwargs):
        super(Bottle2neck, self).__init__(**kwargs)
        self.planes = planes
        self.base_width = base_width
        self.scale = scale
        self.stride = stride
        self.stype = stype

        # Width of each group: w = planes * (baseWidth / 64)
        width = int(tf.math.floor(planes * (base_width / 64.0)))
        self.width = width

        # 1×1 conv: reduce to width * scale channels
        self.conv1 = layers.Conv2D(
            width * scale, (1, 1), use_bias=False, name='conv1'
        )
        self.bn1 = layers.BatchNormalization(name='bn1')

        # s-1 parallel 3×3 convolutions (group 0 is identity/pool)
        self.nums = scale - 1
        self.convs = []
        self.bns_inner = []
        for i in range(self.nums):
            self.convs.append(layers.Conv2D(
                width, (3, 3), strides=(stride, stride),
                padding='same', use_bias=False, name=f'conv_scale_{i}'
            ))
            self.bns_inner.append(layers.BatchNormalization(name=f'bn_scale_{i}'))

        # 1×1 conv: expand to planes * expansion
        self.conv3 = layers.Conv2D(
            planes * self.expansion, (1, 1), use_bias=False, name='conv3'
        )
        self.bn3 = layers.BatchNormalization(name='bn3')

        # SE-block after multi-scale convolutions, before residual addition
        self.se_block = SqueezeExcitationBlock2D(
            reduction=16, name='se_block'
        )

        # Average pool for the pass-through group at stage transitions
        if stype == 'stage':
            self.pool = layers.AveragePooling2D(
                pool_size=(stride, stride), strides=(stride, stride),
                padding='same', name='stage_pool'
            )

        # Downsample for residual connection if needed
        self.downsample = None
        self.downsample_bn = None

    def build(self, input_shape):
        super().build(input_shape)
        in_channels = input_shape[-1]
        out_channels = self.planes * self.expansion
        # Create downsample if dimensions don't match
        if in_channels != out_channels or self.stride != 1:
            self.downsample = layers.Conv2D(
                out_channels, (1, 1), strides=(self.stride, self.stride),
                use_bias=False, name='downsample_conv'
            )
            self.downsample_bn = layers.BatchNormalization(name='downsample_bn')

    def call(self, inputs, training=None):
        identity = inputs

        # 1×1 conv: channel reduction
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        # Split into scale groups along channel axis
        # Each group has `width` channels
        spx = tf.split(out, self.scale, axis=-1)

        # Hierarchical residual processing
        sp_outputs = []
        sp = None
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                # y_i = K_i(x_i + y_{i-1}) — the core Res2Net formula
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns_inner[i](sp, training=training)
            sp = tf.nn.relu(sp)
            sp_outputs.append(sp)

        # Last group: identity (normal) or avg pool (stage transition)
        if self.stype == 'stage':
            sp_outputs.append(self.pool(spx[self.nums]))
        else:
            sp_outputs.append(spx[self.nums])

        # Concatenate all scale groups
        out = layers.Concatenate(axis=-1)(sp_outputs)

        # 1×1 conv: channel expansion
        out = self.conv3(out)
        out = self.bn3(out, training=training)

        # SE-block: channel recalibration after multi-scale convolutions
        out = self.se_block(out, training=training)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(inputs)
            identity = self.downsample_bn(identity, training=training)

        out = out + identity
        out = tf.nn.relu(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'planes': self.planes,
            'base_width': self.base_width,
            'scale': self.scale,
            'stride': self.stride,
            'stype': self.stype,
        })
        return config


# ---------------------------------------------------------------------------
# Safe input handling (kept for compatibility)
# ---------------------------------------------------------------------------

class SafeInputReshapeLayer(layers.Layer):
    """Layer to safely reshape inputs for 2D CNN processing."""

    def __init__(self, input_shape_tuple, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_tuple = input_shape_tuple

    def call(self, x):
        input_shape = self.input_shape_tuple
        if len(input_shape) == 2:
            x = tf.expand_dims(x, axis=-1)
            target_height = max(64, input_shape[0])
            target_width = max(64, input_shape[1])
            x = tf.image.resize(x, (target_height, target_width))
        elif len(input_shape) == 3:
            target_height = max(64, input_shape[0])
            target_width = max(64, input_shape[1])
            x = tf.image.resize(x, (target_height, target_width))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'input_shape_tuple': self.input_shape_tuple})
        return config


# ---------------------------------------------------------------------------
# Res2Net model builder
# ---------------------------------------------------------------------------

def _make_res2net_layer(x, planes, num_blocks, stride=1, base_width=26,
                        scale=4, layer_name='layer'):
    """Build one stage of Res2Net with multiple Bottle2neck blocks.

    Args:
        x: Input tensor
        planes: Base channel count for this stage
        num_blocks: Number of Bottle2neck blocks
        stride: Stride for first block (downsampling)
        base_width: Width per scale group
        scale: Number of scale groups
        layer_name: Name prefix
    """
    # First block may downsample (stride=2) and is 'stage' type
    x = Bottle2neck(
        planes, base_width=base_width, scale=scale,
        stride=stride, stype='stage',
        name=f'{layer_name}_block_0'
    )(x)

    # Remaining blocks are 'normal' type
    for i in range(1, num_blocks):
        x = Bottle2neck(
            planes, base_width=base_width, scale=scale,
            stride=1, stype='normal',
            name=f'{layer_name}_block_{i}'
        )(x)
    return x


def _create_res2net_model(input_shape, num_classes=1, base_width=26, scale=8,
                          layer_config=None, dropout_rate=0.2,
                          architecture='multiscale_cnn'):
    """Create Res2Net model (paper-faithful).

    Res2Net-50 configuration per Gao et al., 2021:
    - Conv1: 7×7, stride 2, 64 filters + BN + ReLU + MaxPool(3×3, stride 2)
    - Layer1: 3× Bottle2neck(64), no downsampling
    - Layer2: 4× Bottle2neck(128), stride 2
    - Layer3: 6× Bottle2neck(256), stride 2
    - Layer4: 3× Bottle2neck(512), stride 2
    - GlobalAvgPool → Dense(num_classes)

    Adapted: uses spectrogram (STFT) front-end for raw audio input.

    Args:
        input_shape: (samples,) for raw audio, (H, W) or (H, W, C) for spectrogram
        num_classes: Number of output classes
        base_width: Width per scale group (paper: 26)
        scale: Number of scale groups (default: 8, paper: 4)
        layer_config: List of block counts per stage (paper Res2Net-50: [3,4,6,3])
        dropout_rate: Dropout before classifier
        architecture: Model name
    """
    if layer_config is None:
        layer_config = [3, 4, 6, 3]  # Res2Net-50

    inputs = layers.Input(shape=input_shape, name='res2net_input')

    # ---- Front-end: handle raw audio ----
    if is_raw_audio(input_shape):
        audio = ensure_flat_input(inputs, input_shape)
        # STFT -> magnitude spectrogram (batch, time, freq, 1)
        x = STFTLayer(name='stft_layer', add_channel_dim=False)(audio)
        # Log-mel spectrogram: apply mel filterbank then log scaling via Lambda
        def apply_log_mel(mag):
            mel_w = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=128,
                num_spectrogram_bins=1025,
                sample_rate=16000,
                lower_edge_hertz=0.0,
                upper_edge_hertz=8000.0,
            )
            mel = tf.matmul(mag, mel_w)       # (batch, time, 128)
            log_mel = tf.math.log(mel + 1e-6)
            return tf.expand_dims(log_mel, axis=-1)  # (batch, time, 128, 1)

        x = layers.Lambda(apply_log_mel, name='log_mel')(x)
        x = ResizeLayer(target_height=128, target_width=128, name='resize_layer')(x)
    else:
        x = SafeInputReshapeLayer(input_shape, name='safe_input_reshape')(inputs)

    # ---- Res2Net stem (Conv1) ----
    # Paper: 7×7 Conv, stride 2, 64 filters → BN → ReLU → MaxPool(3×3, stride 2)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                       use_bias=False, name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # ---- Res2Net stages ----
    # Layer1: planes=64, no downsampling (stride=1)
    x = _make_res2net_layer(x, planes=64, num_blocks=layer_config[0],
                            stride=1, base_width=base_width, scale=scale,
                            layer_name='layer1')

    # Layer2: planes=128, downsampling (stride=2)
    x = _make_res2net_layer(x, planes=128, num_blocks=layer_config[1],
                            stride=2, base_width=base_width, scale=scale,
                            layer_name='layer2')

    # Layer3: planes=256, downsampling (stride=2)
    x = _make_res2net_layer(x, planes=256, num_blocks=layer_config[2],
                            stride=2, base_width=base_width, scale=scale,
                            layer_name='layer3')

    # Layer4: planes=512, downsampling (stride=2)
    x = _make_res2net_layer(x, planes=512, num_blocks=layer_config[3],
                            stride=2, base_width=base_width, scale=scale,
                            layer_name='layer4')

    # ---- Classification head ----
    # Paper: GlobalAvgPool → FC(num_classes)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(dropout_rate, name='classifier_dropout')(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Paper uses SGD with momentum; Adam is a common alternative for audio tasks
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        ),
        loss=loss,
        metrics=['accuracy']
    )

    logger.info(f"Res2Net model created: config={layer_config}, scale={scale}, baseWidth={base_width}, params={model.count_params()}")
    return model


def _create_res2net_lite(input_shape, num_classes=1, architecture='multiscale_cnn_lite'):
    """Lightweight Res2Net variant with fewer blocks and smaller scale.

    Uses Res2Net-26w4s-like configuration for faster training/inference.
    """
    return _create_res2net_model(
        input_shape=input_shape,
        num_classes=num_classes,
        base_width=16,
        scale=4,
        layer_config=[2, 2, 2, 2],  # Res2Net-18 equivalent
        dropout_rate=0.15,
        architecture=architecture
    )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_model(input_shape: Tuple[int, ...], num_classes: int = 1,
                 architecture: str = 'multiscale_cnn', **kwargs) -> models.Model:
    """Factory function to create Res2Net-based multi-scale CNN models.

    Variants:
        'multiscale_cnn': Paper-faithful Res2Net-50 (Gao et al., 2021)
                          scale=8, baseWidth=26, [3,4,6,3] blocks, SE-blocks, log-mel front-end
        'multiscale_cnn_lite': Lightweight Res2Net-18 variant
                               scale=4, baseWidth=16, [2,2,2,2] blocks

    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        architecture: Architecture variant name
    """
    if architecture == 'multiscale_cnn_lite':
        return _create_res2net_lite(input_shape, num_classes, architecture)
    else:
        return _create_res2net_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture,
            **kwargs
        )


# Register custom layers for model save/load compatibility
tf.keras.utils.get_custom_objects().update({
    'Bottle2neck': Bottle2neck,
    'SafeInputReshapeLayer': SafeInputReshapeLayer,
    'SqueezeExciteBlock': SqueezeExciteBlock,
    'SqueezeExcitationBlock2D': SqueezeExcitationBlock2D,
    'preprocess': preprocess,
})
