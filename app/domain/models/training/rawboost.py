"""RawBoost — augmentation para anti-spoofing em áudio bruto (in-graph TF).

Adaptação TensorFlow-nativa do RawBoost (Tak et al., "RawBoost: A Raw Data
Boosting and Augmentation Method applied to Automatic Speaker Verification
Anti-Spoofing", ICASSP 2022). Implementa as três famílias de distorção do
paper de forma que rodam **dentro do grafo** (tf.data), sem scipy:

  1. **LnL convolutive**  — ruído convolutivo linear-e-não-linear (canal FIR
     aleatório + termo não-linear), simula distorções de transmissão/codec.
  2. **ISD additive**     — ruído impulsivo dependente do sinal (impulsos
     esparsos escalados pela amplitude local).
  3. **SSI additive**     — ruído estacionário independente do sinal, colorido
     por um FIR aleatório, somado a um SNR aleatório.

Diferente do placeholder antigo (ruído gaussiano fixo σ=0.01), o RawBoost é a
augmentation que mais melhora a generalização para ataques NÃO vistos em
ASVspoof, sem precisar de dados extras.

A combinação default (`algo=4`) aplica as três em série — recomendada para
o cenário Logical Access (síntese/conversão de voz) no paper.

Uso típico (em `tf.data`):

    from app.domain.models.training.rawboost import rawboost_tf
    ds = ds.map(lambda x, y: (rawboost_tf(x, sr=16000), y))

Todas as operações são NaN-safe e mantêm o shape de entrada.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["rawboost_tf", "lnl_convolutive_noise", "isd_additive_noise",
           "ssi_additive_noise"]


def _to_bt(x):
    """Normaliza entrada para (B, T) e retorna (x_bt, restore_fn)."""
    rank = x.shape.rank
    if rank == 1:  # (T,)
        x_bt = tf.expand_dims(x, 0)
        return x_bt, (lambda y: tf.squeeze(y, 0))
    if rank == 3:  # (B, T, 1)
        x_bt = tf.squeeze(x, axis=-1)
        return x_bt, (lambda y: tf.expand_dims(y, axis=-1))
    return x, (lambda y: y)  # (B, T)


def _rand(minv, maxv, shape=()):
    return tf.random.uniform(shape, minval=minv, maxval=maxv, dtype=tf.float32)


def lnl_convolutive_noise(x_bt, max_taps: int = 20, nl_gain: float = 0.05):
    """Ruído convolutivo linear + não-linear.

    Aplica um filtro FIR aleatório (canal linear) e adiciona um termo
    quadrático filtrado (componente não-linear), normalizado em energia.
    `x_bt`: (B, T) float32.
    """
    b = tf.shape(x_bt)[0]
    t = tf.shape(x_bt)[1]

    # Número de taps do FIR (ímpar para padding SAME simétrico)
    taps = max_taps
    # Kernel FIR aleatório com decaimento (mais peso no centro) — um por batch.
    base = tf.random.normal([taps], stddev=0.5)
    center = taps // 2
    decay = tf.exp(-0.15 * tf.abs(tf.range(taps, dtype=tf.float32) - center))
    fir = base * decay
    fir = fir / (tf.reduce_sum(tf.abs(fir)) + 1e-8)  # normaliza ganho
    fir = tf.reshape(fir, [taps, 1, 1])

    x_in = tf.expand_dims(x_bt, axis=-1)  # (B, T, 1)
    linear = tf.nn.conv1d(x_in, fir, stride=1, padding="SAME")  # (B, T, 1)

    # Componente não-linear: x² filtrado (distorção harmônica) com ganho pequeno
    nl = tf.nn.conv1d(tf.square(x_in), fir, stride=1, padding="SAME")
    out = linear + nl_gain * nl
    out = tf.squeeze(out, axis=-1)  # (B, T)

    # Mistura com o sinal original (mantém inteligibilidade) e renormaliza
    g = _rand(0.4, 1.0, [b, 1])
    out = g * out + (1.0 - g) * x_bt
    del t
    return out


def isd_additive_noise(x_bt, p_impulse: float = 0.02, max_gain: float = 0.7):
    """Ruído impulsivo dependente do sinal (ISD).

    Seleciona ~`p_impulse` das amostras e adiciona impulsos escalados pela
    amplitude local do sinal. `x_bt`: (B, T).
    """
    shape = tf.shape(x_bt)
    mask = tf.cast(tf.random.uniform(shape) < p_impulse, tf.float32)
    impulses = tf.random.uniform(shape, minval=-1.0, maxval=1.0)
    # escala dependente do sinal (amplitude local) + ganho aleatório por exemplo
    local_amp = tf.abs(x_bt) + 0.05
    g = _rand(0.1, max_gain, [shape[0], 1])
    return x_bt + mask * impulses * local_amp * g


def ssi_additive_noise(x_bt, snr_min: float = 10.0, snr_max: float = 40.0,
                       max_taps: int = 16):
    """Ruído estacionário colorido independente do sinal (SSI).

    Gera ruído branco, colore com um FIR aleatório e soma a um SNR sorteado
    por exemplo. `x_bt`: (B, T).
    """
    b = tf.shape(x_bt)[0]
    t = tf.shape(x_bt)[1]

    white = tf.random.normal([b, t, 1])
    taps = max_taps
    fir = tf.random.normal([taps, 1, 1])
    fir = fir / (tf.reduce_sum(tf.abs(fir)) + 1e-8)
    colored = tf.nn.conv1d(white, fir, stride=1, padding="SAME")
    colored = tf.squeeze(colored, axis=-1)  # (B, T)

    # Potências
    sig_pow = tf.reduce_mean(tf.square(x_bt), axis=1, keepdims=True) + 1e-8
    noise_pow = tf.reduce_mean(tf.square(colored), axis=1, keepdims=True) + 1e-8
    snr_db = _rand(snr_min, snr_max, [b, 1])
    snr_lin = tf.pow(10.0, snr_db / 10.0)
    # escala o ruído para atingir o SNR alvo
    scale = tf.sqrt(sig_pow / (noise_pow * snr_lin))
    return x_bt + colored * scale


def rawboost_tf(audio, sr: int = 16000, algo: int = 4, p: float = 0.7):
    """Aplica RawBoost ao áudio bruto (in-graph), com probabilidade `p`.

    Args:
        audio: tensor (T,), (B, T) ou (B, T, 1) float32.
        sr: taxa de amostragem (reservado p/ extensões; não usado nas aprox.).
        algo: combinação das distorções (convenção do paper):
            1 = LnL convolutivo
            2 = ISD impulsivo
            3 = SSI colorido
            4 = série LnL → ISD → SSI (default, recomendado p/ LA)
            5 = série LnL → ISD
            0 = identidade (desliga)
        p: probabilidade de aplicar a augmentation (senão retorna o original).

    Returns:
        Tensor com o MESMO shape de `audio`, sempre finito.
    """
    del sr
    if algo == 0:
        return audio

    x_bt, restore = _to_bt(tf.cast(audio, tf.float32))

    def _apply():
        y = x_bt
        if algo in (1, 4, 5):
            y = lnl_convolutive_noise(y)
        if algo in (2, 4, 5):
            y = isd_additive_noise(y)
        if algo in (3, 4):
            y = ssi_additive_noise(y)
        # sanitiza + limita (este codebase é sensível a NaN/Inf no áudio bruto)
        y = tf.where(tf.math.is_finite(y), y, tf.zeros_like(y))
        y = tf.clip_by_value(y, -10.0, 10.0)
        return y

    # Decisão estocástica por chamada (batch-wide). Mantém grafo simples.
    do_aug = tf.random.uniform([]) < p
    out = tf.cond(do_aug, _apply, lambda: x_bt)
    return restore(out)
