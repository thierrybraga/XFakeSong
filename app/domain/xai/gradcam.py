"""Grad-CAM — mapas de ativação para modelos Keras do benchmark.

Implementação pura TensorFlow do Grad-CAM (Selvaraju et al., 2017): o
gradiente da pontuação de saída em relação a um mapa de características é
agregado por média global (pesos por canal) e combinado com o próprio mapa,
produzindo um heatmap de relevância.

Generalizações necessárias para as arquiteturas deste projeto:

- **mapas 4D** ``(B, H, W, C)`` — caso canônico (Res2Net e demais CNNs 2D);
- **mapas 3D** ``(B, T, C)`` — Conv1D/blocos de tokens (Conformer, CCT, AST):
  o CAM resultante é 1D (relevância por passo temporal/token) e é levado à
  grade da entrada por :func:`heatmap_to_input_grid` (reconstruindo a grade
  de patches quando o número de tokens fatorar, ou como faixa temporal);
- **busca automática de camada** (:func:`compute_gradcam_auto`): camadas de
  reshape próximas à entrada podem ter gradiente nulo/degenerado; a busca
  tenta candidatas da mais profunda para a mais rasa até obter um CAM válido.

As limitações de interpretação (tokenização vs. camadas profundas) são
registradas no relatório do CLI ``scripts/reporting/run_shap_analysis.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def _iter_layers_deep(model: tf.keras.Model):
    """Itera camadas em profundidade (achata submodelos funcionais)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers_deep(layer)
        else:
            yield layer


def _candidate_layers(model: tf.keras.Model) -> list[str]:
    """Camadas candidatas ao Grad-CAM, da mais profunda para a mais rasa.

    Prioriza saídas 4D ``(B,H,W,C)`` e, em seguida, 3D ``(B,T,C)``; ignora
    a camada de entrada.
    """
    rank4: list[str] = []
    rank3: list[str] = []
    for layer in _iter_layers_deep(model):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            shape = layer.output.shape
        except (AttributeError, RuntimeError):
            continue
        if shape is None:
            continue
        if len(shape) == 4:
            rank4.append(layer.name)
        elif len(shape) == 3:
            rank3.append(layer.name)
    return list(reversed(rank4)) + list(reversed(rank3))


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Nome da camada-alvo preferida (última com saída 4D; senão, 3D).

    Raises:
        ValueError: se o modelo não possui camadas com saída 3D/4D.
    """
    candidates = _candidate_layers(model)
    if not candidates:
        raise ValueError(
            "Nenhuma camada com saída 3D/4D encontrada — Grad-CAM não é "
            f"aplicável ao modelo '{model.name}'."
        )
    return candidates[0]


def _spoof_score(preds: tf.Tensor, class_index: Optional[int]) -> tf.Tensor:
    """Pontuação-alvo no espaço de LOGIT: ``log(p/(1-p))`` da classe pedida.

    Modelos muito acurados saturam a sigmoide (``p -> 0/1``), fazendo o
    gradiente da probabilidade colapsar numericamente e o CAM degenerar em
    zero. A transformação logit cancela analiticamente o fator ``p(1-p)`` da
    derivada da sigmoide, recuperando gradiente útil em amostras saturadas.
    """
    if preds.shape[-1] == 1:
        prob = preds[:, 0]
    else:
        index = 1 if class_index is None else int(class_index)
        prob = preds[:, index]
    eps = 1e-7
    return tf.math.log(prob + eps) - tf.math.log(1.0 - prob + eps)


def _score_graph(model: tf.keras.Model) -> tuple[tf.Tensor, bool]:
    """Tensor de pontuação para o Grad-CAM e se ele é logit verdadeiro.

    Quando ``p`` satura em 0/1 EXATOS no float32, o gradiente da
    probabilidade é zero absoluto e nenhuma transformação posterior o
    recupera. Para os padrões usuais de cabeça de classificação, este helper
    reconstrói o logit PRÉ-ativação dentro do grafo funcional:

    - última camada ``Activation('sigmoid'/'softmax')`` → usa o tensor de
      entrada dela (os próprios logits);
    - última camada ``Dense`` com ativação sigmoide/softmax → clona a Dense
      como linear compartilhando ``kernel``/``bias``;
    - caso contrário → devolve ``model.output`` (probabilidades) e o
      chamador aplica a transformação logit aproximada
      (:func:`_spoof_score`).
    """
    last = model.layers[-1]
    act = getattr(last, "activation", None)
    act_name = getattr(act, "__name__", None)

    if isinstance(last, tf.keras.layers.Activation) and act_name in (
        "sigmoid", "softmax",
    ):
        return last.input, True

    if isinstance(last, tf.keras.layers.Dense) and act_name in (
        "sigmoid", "softmax",
    ):
        clone = tf.keras.layers.Dense(
            last.units,
            activation=None,
            use_bias=last.use_bias,
            name=f"{last.name}_gradcam_logits",
        )
        logits = clone(last.input)
        clone.kernel.assign(last.kernel)
        if last.use_bias:
            clone.bias.assign(last.bias)
        return logits, True

    return model.output, False


def compute_gradcam(
    model: tf.keras.Model,
    x: np.ndarray,
    layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
    mode: str = "relu",
) -> np.ndarray:
    """Calcula heatmaps Grad-CAM para um lote de entradas.

    Args:
        model: modelo Keras funcional já carregado (``compile`` opcional).
        x: lote no formato aceito pelo modelo.
        layer_name: camada-alvo; ``None`` usa :func:`find_last_conv_layer`.
        class_index: classe explicada em saídas multiclasse; ``None`` usa a
            classe *spoof* (saída sigmoide ou índice 1).
        mode: ``"relu"`` (Grad-CAM canônico) ou ``"abs"`` (magnitude da
            contribuição, útil quando normalizações deixam a soma ponderada
            aproximadamente constante e o ReLU degenera o mapa).

    Returns:
        ``(B, h, w)`` para camadas 4D ou ``(B, t)`` para camadas 3D, com
        valores normalizados em ``[0, 1]`` por amostra.

    Raises:
        RuntimeError: se o gradiente para a camada-alvo é nulo.
    """
    target_name = layer_name or find_last_conv_layer(model)
    target_layer = None
    for layer in _iter_layers_deep(model):
        if layer.name == target_name:
            target_layer = layer
            break
    if target_layer is None:
        raise ValueError(f"Camada '{target_name}' não encontrada no modelo.")

    score_tensor, is_true_logit = _score_graph(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, score_tensor],
    )

    x_tensor = tf.convert_to_tensor(np.asarray(x, dtype="float32"))
    with tf.GradientTape() as tape:
        conv_out, raw_score = grad_model(x_tensor, training=False)
        if is_true_logit:
            if raw_score.shape[-1] == 1:
                score = raw_score[:, 0]
            else:
                score = raw_score[:, 1 if class_index is None else int(class_index)]
        else:
            score = _spoof_score(raw_score, class_index)

    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError(
            f"Gradiente nulo para a camada '{target_name}' — verifique se a "
            "camada participa do caminho até a saída."
        )

    rank = len(conv_out.shape)
    if rank == 4:
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    elif rank == 3:
        weights = tf.reduce_mean(grads, axis=1, keepdims=True)
    else:  # pragma: no cover - candidatas são sempre 3D/4D
        raise ValueError(f"Posto não suportado para Grad-CAM: {rank}")

    if mode == "abs":
        cam = tf.reduce_sum(tf.abs(grads * conv_out), axis=-1)
    else:
        cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))
    cam = cam.numpy().astype("float64")

    flat = cam.reshape(cam.shape[0], -1)
    maxima = flat.max(axis=1, keepdims=True)
    maxima[maxima == 0.0] = 1.0
    cam = (flat / maxima).reshape(cam.shape)
    logger.debug(
        "Grad-CAM: camada=%s, lote=%d, mapa=%s",
        target_name, cam.shape[0], cam.shape[1:],
    )
    return cam


def compute_gradcam_auto(
    model: tf.keras.Model,
    x: np.ndarray,
    class_index: Optional[int] = None,
) -> tuple[np.ndarray, str]:
    """Grad-CAM com busca automática da camada-alvo.

    Tenta as candidatas (:func:`_candidate_layers`) da mais profunda para a
    mais rasa e retorna o primeiro CAM com gradiente válido e não degenerado
    (variação não nula em ao menos uma amostra).

    Returns:
        ``(cam, nome_da_camada)``.

    Raises:
        RuntimeError: se nenhuma candidata produz CAM válido.
    """
    errors: list[str] = []
    for name in _candidate_layers(model):
        for mode in ("relu", "abs"):
            try:
                cam = compute_gradcam(
                    model, x, layer_name=name, class_index=class_index, mode=mode
                )
            except (RuntimeError, ValueError) as exc:
                errors.append(f"{name}[{mode}]: {exc}")
                break  # erro estrutural: trocar de modo não ajuda
            if float(np.ptp(cam)) > 1e-9:
                label = name if mode == "relu" else f"{name} (abs)"
                return cam, label
            errors.append(f"{name}[{mode}]: CAM degenerado (constante)")
    raise RuntimeError(
        "Nenhuma camada produziu Grad-CAM válido. Tentativas: "
        + "; ".join(errors[:6])
    )


def heatmap_to_input_grid(heatmaps: np.ndarray, height: int, width: int) -> np.ndarray:
    """Leva CAMs à grade da entrada ``(B, height, width)``.

    - CAM 3D ``(B, h, w)``: redimensionamento bilinear direto;
    - CAM 2D ``(B, n)``: se ``n`` fatorar em ``(a, b)`` com aspecto próximo ao
      da entrada (grade de patches/tokens), reconstrói a grade e
      redimensiona; caso contrário, trata como faixa temporal ``(B, 1, n)``
      replicada ao longo do eixo de frequência.
    """
    heatmaps = np.asarray(heatmaps, dtype="float32")
    if heatmaps.ndim == 3:
        grid = heatmaps[..., np.newaxis]
    elif heatmaps.ndim == 2:
        n = heatmaps.shape[1]
        pair = _best_grid(n, height / max(width, 1))
        if pair is not None:
            grid = heatmaps.reshape(-1, pair[0], pair[1])[..., np.newaxis]
        else:
            grid = heatmaps[:, np.newaxis, :, np.newaxis]
    else:
        raise ValueError(f"CAM com posto não suportado: {heatmaps.ndim}")
    resized = tf.image.resize(grid, (height, width), method="bilinear")
    return np.clip(resized.numpy()[..., 0], 0.0, 1.0)


def resize_heatmap(heatmaps: np.ndarray, height: int, width: int) -> np.ndarray:
    """Compatibilidade: alias de :func:`heatmap_to_input_grid` para CAM 3D."""
    return heatmap_to_input_grid(heatmaps, height, width)


def _best_grid(n: int, target_aspect: float) -> Optional[tuple[int, int]]:
    """Fatoração ``(a, b)`` de ``n`` com razão ``a/b`` mais próxima do alvo.

    Retorna ``None`` quando ``n`` é primo/degenerado (sem grade útil além de
    ``1×n``), caso em que o chamador usa a interpretação de faixa temporal.
    """
    best: Optional[tuple[int, int]] = None
    best_err = math.inf
    for a in range(2, int(math.sqrt(n)) + 1):
        if n % a:
            continue
        for pair in ((a, n // a), (n // a, a)):
            err = abs(math.log((pair[0] / pair[1]) / max(target_aspect, 1e-9)))
            if err < best_err:
                best_err = err
                best = pair
    return best
