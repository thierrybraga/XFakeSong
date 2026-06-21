import logging
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from app.core.interfaces.base import ProcessingResult, ProcessingStatus

from .model_loader import ModelInfo
from .utils import prepare_batch_for_model

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Post-hoc temperature scaling for model calibration.

    Adjusts prediction confidence by dividing logits by a learned temperature T.
    Better-calibrated models produce confidence scores that match true accuracy.
    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, model, val_features: np.ndarray, val_labels: np.ndarray):
        """Learn optimal temperature on validation set using NLL minimization."""
        try:
            # Get logits (pre-softmax) from model
            predictions = model.predict(val_features, verbose=0)

            # Grid search for optimal temperature (simple but effective)
            best_temp = 1.0
            best_nll = float('inf')

            for temp in np.arange(0.5, 5.0, 0.1):
                # Apply temperature scaling
                scaled = predictions / temp
                if scaled.shape[-1] > 1:
                    scaled_probs = tf.nn.softmax(scaled).numpy()
                else:
                    scaled_probs = tf.nn.sigmoid(scaled).numpy()

                # Compute NLL
                eps = 1e-7
                scaled_probs = np.clip(scaled_probs, eps, 1.0 - eps)

                if scaled_probs.shape[-1] > 1:
                    nll = -np.mean(np.log(scaled_probs[np.arange(len(val_labels)),
                                                        val_labels.astype(int).flatten()]))
                else:
                    nll = -np.mean(
                        val_labels * np.log(scaled_probs) +
                        (1 - val_labels) * np.log(1 - scaled_probs)
                    )

                if nll < best_nll:
                    best_nll = nll
                    best_temp = temp

            self.temperature = best_temp
            logger.info(f"Temperature calibrated to {self.temperature:.2f} (NLL: {best_nll:.4f})")
        except Exception as e:
            logger.warning(f"Temperature calibration failed: {e}. Using T=1.0")
            self.temperature = 1.0

    def scale(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        if self.temperature == 1.0:
            return predictions
        return predictions / self.temperature


def predict_with_mc_dropout(
    model_info: ModelInfo,
    features: np.ndarray,
    n_samples: int = 20,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Sprint 5.4: Monte Carlo Dropout para quantificação de incerteza.

    Faz N forward passes com dropout ativo (`training=True`) e computa
    estatísticas das predições. Permite distinguir:

    - **Incerteza epistêmica**: alta variância entre forward passes → o modelo
      "não sabe" (área pouco coberta pelo treino, OOD candidate)
    - **Incerteza aleatória**: entropia média alta → ambiguidade inerente
      da amostra (mesmo com mais dados, o modelo não decidiria)

    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation",
    ICML 2016.

    Args:
        model_info: ModelInfo do modelo (deve ter dropout layers)
        features: input (N, *input_shape) ou (*input_shape,) para single sample
        n_samples: número de forward passes MC (típico: 10-50)
        batch_size: batch size para inferência

    Returns:
        Dict com:
            - mean_prediction: (N, K) média sobre os MC samples
            - epistemic_uncertainty: (N,) variância da predição da classe vencedora
            - predictive_entropy: (N,) entropia da média (incerteza total)
            - confidence: (N,) max(mean_prediction)
            - is_uncertain: (N,) bool — flag de alta incerteza
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1 or (
        features.ndim == len(model_info.input_shape) if model_info.input_shape else 0
    ):
        # Single sample → add batch dim
        if model_info.input_shape and features.ndim == len(model_info.input_shape):
            features = features[np.newaxis, ...]

    n_samples = max(2, int(n_samples))

    # Forward passes múltiplos com training=True (mantém dropout ativo)
    keras_model = model_info.model
    all_preds = []
    try:
        # Construct tensor once para evitar overhead
        x_tf = tf.convert_to_tensor(features, dtype=tf.float32)
        for _ in range(n_samples):
            # training=True força dropout estar ativo (e BN usa batch stats —
            # potencialmente problemático; em modelos com BN, MC Dropout puro
            # pode dar resultados ruidosos. Workaround: usar inference_mode
            # alternativo, mas requer custom layer extraction.)
            out = keras_model(x_tf, training=True).numpy()
            all_preds.append(out)
    except Exception as e:
        logger.warning(f"MC Dropout falhou, retornando predict padrão: {e}")
        single = keras_model.predict(features, batch_size=batch_size, verbose=0)
        return {
            'mean_prediction': single,
            'epistemic_uncertainty': np.zeros(len(single), dtype=np.float32),
            'predictive_entropy': np.zeros(len(single), dtype=np.float32),
            'confidence': np.max(single, axis=-1) if single.ndim > 1 else single,
            'is_uncertain': np.zeros(len(single), dtype=bool),
            'n_mc_samples': 0,
            'fallback': True,
        }

    # Stack: (n_samples, batch, K)
    stack = np.stack(all_preds, axis=0)

    # Normaliza shape para (n_samples, N, K)
    if stack.ndim == 2:
        stack = stack[:, :, np.newaxis]

    mean_pred = stack.mean(axis=0)  # (N, K)
    var_pred = stack.var(axis=0)    # (N, K)

    # Epistemic uncertainty: variância da classe predita (argmax da média)
    if mean_pred.shape[-1] > 1:
        pred_class = np.argmax(mean_pred, axis=-1)
        # Variance ao longo da classe predita
        epistemic = np.array([
            var_pred[i, pred_class[i]] for i in range(len(pred_class))
        ], dtype=np.float32)
        confidence = np.max(mean_pred, axis=-1)
    else:
        # Sigmoid (N, 1): variância da única dim
        epistemic = var_pred[:, 0].astype(np.float32)
        confidence = mean_pred[:, 0]

    # Predictive entropy: H(E[p]) — incerteza total
    eps = 1e-7
    if mean_pred.shape[-1] > 1:
        # Multi-class softmax
        p = np.clip(mean_pred, eps, 1.0 - eps)
        entropy = -np.sum(p * np.log(p), axis=-1)
    else:
        # Binary sigmoid
        p = np.clip(mean_pred[:, 0], eps, 1.0 - eps)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    entropy = entropy.astype(np.float32)

    # Flag de alta incerteza: variance > 0.05 OU entropia > log(K)/2
    K = mean_pred.shape[-1] if mean_pred.shape[-1] > 1 else 2
    entropy_threshold = np.log(K) * 0.5
    is_uncertain = (epistemic > 0.05) | (entropy > entropy_threshold)

    return {
        'mean_prediction': mean_pred,
        'epistemic_uncertainty': epistemic,
        'predictive_entropy': entropy,
        'confidence': confidence.astype(np.float32),
        'is_uncertain': is_uncertain,
        'n_mc_samples': n_samples,
        'fallback': False,
    }


def get_jit_predict_fn(model_info: ModelInfo, use_xla: bool = True):
    """Sprint 3.1: retorna função tf.function(jit_compile=True) cacheada para
    inferência acelerada via XLA.

    O retorno é uma `tf.function` que toma um tensor de input e retorna o
    output do modelo. Cacheada em `model_info.jit_predict_fn` na primeira
    chamada (lazy compile). Se `use_xla=False`, retorna função sem JIT
    (fallback para casos onde XLA não suporta ops do modelo).

    Args:
        model_info: ModelInfo com modelo Keras
        use_xla: se True (default), habilita jit_compile=True

    Returns:
        Função callable(input_tensor) -> output_tensor
    """
    keras_model = model_info.model
    if not isinstance(keras_model, tf.keras.Model):
        raise TypeError("JIT requer uma instância tf.keras.Model")

    if getattr(model_info, "jit_predict_fn", None) is not None:
        return model_info.jit_predict_fn

    @tf.function(jit_compile=use_xla, reduce_retracing=True)
    def _jit_predict(x):
        return keras_model(x, training=False)

    try:
        model_info.jit_predict_fn = _jit_predict
    except Exception:
        pass
    logger.debug(
        f"JIT predict function compiled for {model_info.name} "
        f"(use_xla={use_xla})"
    )
    return _jit_predict


def _predict_with_jit(
    model_info: ModelInfo,
    batch_features: np.ndarray,
    use_xla: bool = True,
) -> np.ndarray:
    """Sprint 3.1: predição via tf.function(jit_compile=True) com fallback.

    Se XLA falhar (op não suportada, etc.), faz fallback automático para
    `model.predict(verbose=0)` mantendo correctness.

    Returns:
        np.ndarray com as predições.
    """
    try:
        jit_fn = get_jit_predict_fn(model_info, use_xla=use_xla)
        x = tf.convert_to_tensor(batch_features, dtype=tf.float32)
        return jit_fn(x).numpy()
    except Exception as e:
        # Fallback silencioso para predict padrão (XLA pode não suportar
        # todas as ops, ex: tf.signal.stft em alguns modelos in-graph)
        logger.debug(
            f"JIT predict falhou para {model_info.name}, usando model.predict: {e}"
        )
        return model_info.model.predict(batch_features, verbose=0)


def compute_energy_score(
    predictions: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """OOD score baseado em concentração da distribuição (proxy para energy).

    A formulação original de Energy Score (Liu et al., NeurIPS 2020):
        E(x) = -T * logsumexp(f(x) / T)    # requer LOGITS pré-softmax

    Como o pipeline retorna probabilidades (já normalizadas), `log(p)` perde
    a constante de partição `log Z`, então `logsumexp(log(p))` ≡ 0 e a fórmula
    direta se degenera. Em pipelines onde só temos probabilidades, a literatura
    de OOD detection (Hendrycks & Gimpel 2017, etc.) recomenda usar uma medida
    equivalente baseada em **concentração da distribuição**:

        score = log(K) - H(p)    onde H(p) = -Σ p_i log p_i

    - Distribuição concentrada (one-hot) → H=0 → score=log(K) (alto, in-distribution)
    - Distribuição uniforme → H=log(K) → score=0 (baixo, candidato OOD)

    Quando `temperature ≠ 1.0`, a distribuição é re-suavizada antes do cálculo
    (consistente com o uso pós-calibração).

    Args:
        predictions: array de probabilidades (N, K) softmax ou (N, 1) sigmoid
        temperature: T para suavização (1.0 = sem suavização extra)

    Returns:
        Array de shape (N,) com scores ∈ [0, log K]. Maior = mais in-distribution.
    """
    eps = 1e-7
    T = max(float(temperature), 1e-3)
    p = np.asarray(predictions, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)

    # Normaliza para 2D (N, K)
    if p.ndim == 1:
        # Sigmoid 1D: converte para 2-class
        p_2c = np.stack([1.0 - p, p], axis=-1)
    elif p.ndim > 1 and p.shape[-1] == 1:
        # Sigmoid (N, 1): mesma transformação
        p_2c = np.concatenate([1.0 - p, p], axis=-1)
    else:
        p_2c = p

    # Re-suaviza com temperatura (apenas se T != 1)
    if T != 1.0:
        logits = np.log(p_2c) / T
        max_z = np.max(logits, axis=-1, keepdims=True)
        exp_z = np.exp(logits - max_z)
        p_2c = exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + eps)
        p_2c = np.clip(p_2c, eps, 1.0 - eps)

    K = p_2c.shape[-1]
    # Entropia: H(p) = -Σ p log p
    entropy = -np.sum(p_2c * np.log(p_2c), axis=-1)
    # Score = log(K) - H(p) ∈ [0, log K]
    score = np.log(K) - entropy
    return score.astype(np.float32)


def normalize_logits_to_probs(predictions: np.ndarray) -> np.ndarray:
    """Garante que `predictions` esteja em forma de probabilidades.

    Alguns modelos (ex: AASIST com AMSoftmax + activation='linear') retornam
    **logits brutos** em vez de probabilidades. Outros (RawNet2, WavLM,
    HuBERT, Conformer, etc.) já retornam softmax/sigmoid. Esta função
    detecta o caso e aplica softmax/sigmoid conforme necessário.

    Heurística de detecção (por linha):
    - Se K>1 e (qualquer valor < 0 OR qualquer valor > 1 OR soma ∉ [0.9, 1.1])
      → aplica softmax
    - Se K=1 e (valor < 0 OR valor > 1)
      → aplica sigmoid
    - Caso contrário, retorna como veio.

    Args:
        predictions: shape (N, K) ou (N, 1) ou (N,)

    Returns:
        Array de mesma shape com valores em [0, 1] e somando 1 (no caso K>1).
    """
    p = np.asarray(predictions, dtype=np.float64)

    # Normaliza shape para 2D (N, K)
    if p.ndim == 1:
        p = p[:, np.newaxis]

    K = p.shape[-1]

    if K > 1:
        # Detecta se é logits: qualquer valor fora [0, 1] ou soma != 1 em qualquer linha
        out_of_range = (p < -1e-3).any() or (p > 1.0 + 1e-3).any()
        sums = p.sum(axis=-1)
        not_normalized = ((sums < 0.9) | (sums > 1.1)).any()
        if out_of_range or not_normalized:
            # Aplica softmax linha-a-linha
            max_z = np.max(p, axis=-1, keepdims=True)
            exp_z = np.exp(p - max_z)
            p = exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-12)
    else:
        # K=1: se fora de [0, 1], aplica sigmoid
        if (p < -1e-3).any() or (p > 1.0 + 1e-3).any():
            p = 1.0 / (1.0 + np.exp(-p))

    return p.astype(np.float32)


def apply_temperature_scaling(
    predictions: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Aplica Temperature Scaling em probabilidades (post-hoc).

    Para softmax (output shape (N, K) com K > 1): re-normaliza logits/T.
    Para sigmoid binário (output shape (N, 1) ou (N,)): aplica scaling em
    logit-space e re-converte para probabilidade.

    Args:
        predictions: array de probabilidades (já após softmax/sigmoid)
        temperature: valor T calibrado (1.0 = identidade)

    Returns:
        Array de probabilidades calibradas (mesma shape).
    """
    if temperature is None or temperature == 1.0 or temperature <= 0:
        return predictions

    eps = 1e-7
    p = np.clip(predictions, eps, 1.0 - eps)

    if p.ndim > 1 and p.shape[-1] > 1:
        # Softmax: recupera logits aproximados via log(p) e re-aplica softmax(./T)
        logits = np.log(p) / float(temperature)
        # Subtrai max para estabilidade numérica
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=-1, keepdims=True)
    else:
        # Sigmoid: scaling em logit-space
        logits = np.log(p / (1.0 - p)) / float(temperature)
        return 1.0 / (1.0 + np.exp(-logits))


def _get_numeric_attr(obj: Any, name: str, default: float) -> float:
    value = getattr(obj, name, None)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return value
    return float(default)


def _as_probability(value: Any) -> float:
    prob = float(np.asarray(value).reshape(-1)[0])
    prob = float(np.clip(prob, 0.0, 1.0))
    return round(prob, 6)


class Predictor:
    """Responsável por realizar predições com modelos."""

    # Sprint 3.1: lista de arquiteturas com ops in-graph não-XLA (tf.signal.stft,
    # graph attention dinâmico, etc.) — pulam JIT compile e usam predict padrão.
    _XLA_UNFRIENDLY_ARCHITECTURES = {
        'Sonic Sleuth',           # LFCC/MFCC/CQT in-model via tf.signal
        'Ensemble',               # SharedSTFTLayer + 4 branches
        'WavLM',                  # transformers backbone
        'HuBERT',                 # transformers backbone
        'AASIST',                 # graph attention dinâmico (HSGALLayer)
        'RawGAT-ST',              # graph attention dinâmico
    }

    def __init__(self):
        # Mantido para compatibilidade — calibração principal é per-model
        # via ModelInfo.temperature (Sprint 1.4).
        self.temperature_scaler = TemperatureScaler()

    @classmethod
    def _is_xla_unfriendly(cls, model_info: ModelInfo) -> bool:
        """Sprint 3.1: detecta arquiteturas que XLA não compila bem.

        Retorna True para modelos com ops in-graph que falham em XLA
        (tf.signal.stft em algumas versões, GraphAttention dinâmico, etc.).
        Esses modelos usam predict padrão (sem JIT).
        """
        arch = (model_info.architecture or '').strip()
        return arch in cls._XLA_UNFRIENDLY_ARCHITECTURES

    def predict_with_uncertainty(
        self,
        model_info: ModelInfo,
        features: np.ndarray,
        n_samples: int = 20,
    ) -> ProcessingResult[Dict[str, Any]]:
        """Sprint 5.4: predição com MC Dropout para quantificar incerteza.

        Wrapper conveniente sobre `predict_with_mc_dropout()` que aplica
        também temperatura calibrada (Sprint 1.4) e retorna em formato
        ProcessingResult.

        Args:
            model_info: ModelInfo (deve ser tensorflow, com dropout layers)
            features: input single ou batch
            n_samples: número de MC forward passes (10-50 típico)

        Returns:
            ProcessingResult[Dict] com chaves:
                - is_deepfake (bool)
                - confidence (float)
                - epistemic_uncertainty (float)
                - predictive_entropy (float)
                - is_uncertain (bool) — flag para decisão "abstenha-se"
                - n_mc_samples (int)
        """
        try:
            if model_info.model_type != 'tensorflow':
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["MC Dropout requer modelo tensorflow"],
                )

            from .utils import prepare_batch_for_model
            features_batch = prepare_batch_for_model(
                [features] if features.ndim == len(model_info.input_shape or ()) else features,
                model_info.input_shape,
            )

            mc_result = predict_with_mc_dropout(
                model_info, features_batch, n_samples=n_samples
            )

            # Garante probabilidades antes de temp scaling
            mean_pred_norm = normalize_logits_to_probs(mc_result['mean_prediction'])

            # Aplica temperatura calibrada à média (Sprint 1.4)
            model_temp = float(getattr(model_info, 'temperature', 1.0))
            calibrated_mean = apply_temperature_scaling(mean_pred_norm, model_temp)

            # Determina is_deepfake usando classification threshold (EER ou 0.5)
            fake_threshold = _get_numeric_attr(model_info, 'eer_threshold', 0.5)

            results = []
            for i in range(len(calibrated_mean)):
                pred = calibrated_mean[i]
                # Mesma convenção do _predict_tensorflow_batch: p_fake explícito
                if pred.shape[-1] == 1:
                    p_fake = float(pred[0] if pred.ndim > 0 else pred)
                    p_real = 1.0 - p_fake
                else:
                    p_fake = float(pred[1])
                    p_real = float(pred[0])
                is_fake = bool(p_fake > fake_threshold)
                confidence = p_fake if is_fake else p_real

                results.append({
                    'is_deepfake': bool(is_fake),
                    'confidence': float(confidence),
                    'p_fake': float(p_fake),
                    'p_real': float(p_real),
                    'epistemic_uncertainty': float(mc_result['epistemic_uncertainty'][i]),
                    'predictive_entropy': float(mc_result['predictive_entropy'][i]),
                    'is_uncertain': bool(mc_result['is_uncertain'][i]),
                    'n_mc_samples': int(mc_result['n_mc_samples']),
                    'temperature_applied': model_temp,
                    'classification_threshold': fake_threshold,
                    'mc_fallback': bool(mc_result.get('fallback', False)),
                })

            # Se single sample, retorna dict direto; senão, list
            data = results[0] if len(results) == 1 else results
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, data=data,
            )

        except Exception as e:
            logger.error(f"Erro em predict_with_uncertainty: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"MC Dropout falhou: {str(e)}"],
            )

    def predict(self, model_info: ModelInfo,
                features: np.ndarray,
                device: str = None,
                use_tta: bool = False) -> ProcessingResult[Dict[str, Any]]:
        """Faz predição com um modelo específico (wrapper para single input)."""
        batch_result = self.predict_batch(model_info, [features], device, use_tta=use_tta)

        if batch_result.status != ProcessingStatus.SUCCESS:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=batch_result.errors
            )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=batch_result.data[0]
        )

    def predict_batch(self, model_info: ModelInfo,
                      features_list: List[np.ndarray],
                      device: str = None,
                      use_tta: bool = False) -> ProcessingResult[List[Dict[str, Any]]]:
        """Faz predição em lote, opcionalmente com Test-Time Augmentation."""
        try:
            if not features_list:
                return ProcessingResult(status=ProcessingStatus.SUCCESS, data=[])

            if model_info.model_type == 'tensorflow':
                if use_tta:
                    return self._predict_tensorflow_batch_with_tta(
                        model_info, features_list, device)
                return self._predict_tensorflow_batch(model_info, features_list, device)
            elif model_info.model_type == 'pytorch_transformers':
                return self._predict_pytorch_transformers_batch(
                    model_info, features_list
                )
            elif model_info.model_type == 'sklearn':
                return self._predict_sklearn_batch(model_info, features_list)
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Tipo de modelo não suportado: {model_info.model_type}"]
                )

        except Exception as e:
            logger.error(f"Erro na predição em lote com modelo {model_info.name}: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição: {str(e)}"]
            )

    def _predict_tensorflow_batch(self, model_info: ModelInfo,
                                  features_list: List[np.ndarray],
                                  device: str = None) -> ProcessingResult[List[Dict[str, Any]]]:
        """Predição em lote com modelo TensorFlow."""
        try:
            # Prepara o lote usando o utilitário
            batch_features = prepare_batch_for_model(
                features_list, model_info.input_shape
            )

            # Tier-1 perf: ONNX Runtime quando disponível (FP32, mesmos pesos →
            # mesma saída, porém mais rápido em CPU). Fallback automático p/ TF
            # em qualquer erro (op não suportada, shape, etc.).
            predictions = None
            onnx_sess = getattr(model_info, "onnx_session", None)
            if onnx_sess is not None:
                try:
                    predictions = onnx_sess.predict(batch_features)
                except Exception as e:
                    logger.debug(
                        f"ONNX inferência falhou ({model_info.name}); "
                        f"usando TF: {e}")
                    predictions = None

            if predictions is None:
                # Sprint 3.1: usa JIT (XLA) com fallback automático. Pula XLA p/
                # modelos com ops in-graph não-XLA (Sonic Sleuth, Ensemble usam
                # tf.signal.stft que pode falhar em XLA dependendo da versão TF).
                use_xla = not self._is_xla_unfriendly(model_info)

                if device:
                    # Formatar device string para TensorFlow (ex: /CPU:0, /GPU:0)
                    device_name = device
                    if not device.startswith("/"):
                        device_name = f"/{device}"
                    if device == "CPU":
                        device_name = "/CPU:0"
                    if device.startswith("GPU:") and not device.startswith("/GPU:"):
                        device_name = f"/{device}"

                    with tf.device(device_name):
                        predictions = _predict_with_jit(
                            model_info, batch_features, use_xla=use_xla)
                else:
                    predictions = _predict_with_jit(
                        model_info, batch_features, use_xla=use_xla)

            # Garante que predictions estejam em forma de probabilidades
            # (alguns modelos como AASIST com AMSoftmax + activation='linear'
            # retornam logits brutos). Aplica softmax/sigmoid se necessário.
            predictions = normalize_logits_to_probs(predictions)

            # Sprint 1.4: Aplica temperatura calibrada per-model (post-hoc).
            # Não altera a classe predita, mas re-calibra a confiança para
            # corresponder à acurácia real (Guo et al., ICML 2017).
            model_temp = float(getattr(model_info, 'temperature', 1.0))
            predictions = apply_temperature_scaling(predictions, model_temp)

            # Sprint 2.5: OOD detection via energy score.
            # Computa energy para todas as predições; se threshold disponível
            # no input_contract (calibrado no val set), marca is_ood.
            ood_scores = compute_energy_score(predictions, temperature=model_temp)
            ood_threshold = None
            if model_info.input_contract and isinstance(model_info.input_contract, dict):
                ood_threshold = model_info.input_contract.get('ood_threshold')

            # Sprint 4.5: Threshold de classificação adaptativo per-model.
            # Se model_info.eer_threshold disponível → usa EER threshold,
            # caso contrário → 0.5 fixo (comportamento legado).
            fake_threshold = _get_numeric_attr(model_info, 'eer_threshold', 0.5)

            results = []
            for i in range(len(predictions)):
                pred = predictions[i]
                ood_score = float(ood_scores[i])

                # Interpretar resultado — `confidence` é SEMPRE a probabilidade
                # de FAKE, independente do tipo de output. Para sigmoid 1-unit,
                # é o valor direto. Para softmax 2-unit, é p[1].
                # (Convenção: índice 0 = real, índice 1 = fake.)
                if np.ndim(pred) == 0 or pred.shape[-1] == 1:
                    # Sigmoid binário: pred[0] já é p(fake)
                    p_fake = _as_probability(
                        pred[0] if np.ndim(pred) > 0 else pred
                    )
                    p_real = _as_probability(1.0 - p_fake)
                else:
                    # Softmax: pred[1] = p(fake), pred[0] = p(real)
                    p_fake = _as_probability(pred[1])
                    p_real = _as_probability(pred[0])

                # Decisão usa threshold adaptativo (EER se disponível, senão 0.5)
                is_deepfake = bool(p_fake > fake_threshold)
                # confidence reportada é a probabilidade da CLASSE PREDITA
                # (mais intuitivo para o usuário: "estou 87% confiante de FAKE"
                # ou "estou 87% confiante de REAL", em vez de sempre reportar
                # p_fake mesmo quando a decisão foi "real").
                confidence = _as_probability(p_fake if is_deepfake else p_real)

                # OOD flag: True se energy score abaixo do threshold (= não in-distribution)
                is_ood = (
                    ood_threshold is not None and ood_score < float(ood_threshold)
                )

                results.append({
                    'is_deepfake': is_deepfake,
                    'confidence': float(confidence),
                    'p_fake': float(p_fake),
                    'p_real': float(p_real),
                    'temperature_applied': model_temp,
                    'ood_score': ood_score,
                    'is_ood': bool(is_ood),
                    'ood_threshold': float(ood_threshold) if ood_threshold is not None else None,
                    'classification_threshold': fake_threshold,
                })

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição TensorFlow: {str(e)}"]
            )

    def _predict_pytorch_transformers_batch(
        self,
        model_info: ModelInfo,
        features_list: List[np.ndarray],
    ) -> ProcessingResult[List[Dict[str, Any]]]:
        """Predição em lote com backbones SSL originais em PyTorch."""
        try:
            batch_features = prepare_batch_for_model(
                features_list, model_info.input_shape
            )
            predictions = model_info.model.predict(batch_features)
            predictions = normalize_logits_to_probs(predictions)

            model_temp = float(getattr(model_info, 'temperature', 1.0))
            predictions = apply_temperature_scaling(predictions, model_temp)
            ood_scores = compute_energy_score(predictions, temperature=model_temp)
            ood_threshold = None
            if model_info.input_contract and isinstance(model_info.input_contract, dict):
                ood_threshold = model_info.input_contract.get('ood_threshold')
            fake_threshold = _get_numeric_attr(model_info, 'eer_threshold', 0.5)

            results = []
            for i in range(len(predictions)):
                pred = predictions[i]
                if np.ndim(pred) == 0 or pred.shape[-1] == 1:
                    p_fake = _as_probability(
                        pred[0] if np.ndim(pred) > 0 else pred
                    )
                    p_real = _as_probability(1.0 - p_fake)
                else:
                    p_fake = _as_probability(pred[1])
                    p_real = _as_probability(pred[0])

                is_deepfake = bool(p_fake > fake_threshold)
                confidence = _as_probability(p_fake if is_deepfake else p_real)
                ood_score = float(ood_scores[i])
                is_ood = (
                    ood_threshold is not None and ood_score < float(ood_threshold)
                )
                results.append({
                    'is_deepfake': is_deepfake,
                    'confidence': float(confidence),
                    'p_fake': float(p_fake),
                    'p_real': float(p_real),
                    'temperature_applied': model_temp,
                    'ood_score': ood_score,
                    'is_ood': bool(is_ood),
                    'ood_threshold': (
                        float(ood_threshold) if ood_threshold is not None else None
                    ),
                    'classification_threshold': fake_threshold,
                })

            return ProcessingResult(status=ProcessingStatus.SUCCESS, data=results)
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição PyTorch SSL: {str(e)}"],
            )

    def _predict_sklearn_batch(
        self,
        model_info: ModelInfo,
        features_list: List[np.ndarray]
    ) -> ProcessingResult[List[Dict[str, Any]]]:
        """Predição em lote com modelo sklearn."""
        try:
            results = []
            # Sklearn geralmente espera (n_samples, n_features)
            # Precisamos achatar cada feature vector

            flattened_features = []
            for f in features_list:
                if model_info.scaler:
                     # Scaler espera 2D array (1, n_features) para transform
                     f_reshaped = f.reshape(1, -1)
                     f_scaled = model_info.scaler.transform(f_reshaped)
                     flattened_features.append(f_scaled.flatten())
                else:
                     flattened_features.append(f.flatten())

            X = np.array(flattened_features)

            # Predição em lote
            predictions = model_info.model.predict(X)

            # Probabilidade (convenção: índice 0 = real, índice 1 = fake)
            if hasattr(model_info.model, 'predict_proba'):
                probas = model_info.model.predict_proba(X)
                fake_threshold = _get_numeric_attr(model_info, 'eer_threshold', 0.5)
                for i in range(len(probas)):
                    p_real = float(probas[i][0])
                    p_fake = float(probas[i][1]) if len(probas[i]) > 1 else 1.0 - p_real
                    is_deepfake = bool(p_fake > fake_threshold)
                    confidence = p_fake if is_deepfake else p_real
                    results.append({
                        'is_deepfake': is_deepfake,
                        'confidence': float(confidence),
                        'p_fake': p_fake,
                        'p_real': p_real,
                        'classification_threshold': fake_threshold,
                    })
            else:
                for i in range(len(predictions)):
                    pred_int = int(predictions[i])
                    is_deepfake = pred_int == 1
                    results.append({
                        'is_deepfake': is_deepfake,
                        'confidence': 1.0,
                        'p_fake': 1.0 if is_deepfake else 0.0,
                        'p_real': 0.0 if is_deepfake else 1.0,
                        'classification_threshold': 0.5,
                    })

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro na predição sklearn: {str(e)}"]
            )

    def _predict_tensorflow_batch_with_tta(
        self, model_info: ModelInfo,
        features_list: List[np.ndarray],
        device: str = None,
        n_augmentations: int = 5
    ) -> ProcessingResult[List[Dict[str, Any]]]:
        """Test-Time Augmentation: run multiple augmented copies and average predictions.

        Creates 5 versions of each input (original + 4 perturbations)
        and averages the predictions. Typically improves accuracy by 1-3%.
        """
        try:
            batch_features = prepare_batch_for_model(
                features_list, model_info.input_shape
            )

            all_predictions = []

            # Sprint 3.1: JIT compile (XLA) também no TTA path
            use_xla = not self._is_xla_unfriendly(model_info)

            def _run_prediction(data):
                if device:
                    device_name = device if device.startswith("/") else f"/{device}"
                    if device == "CPU":
                        device_name = "/CPU:0"
                    with tf.device(device_name):
                        return _predict_with_jit(model_info, data, use_xla=use_xla)
                return _predict_with_jit(model_info, data, use_xla=use_xla)

            # 1. Original
            all_predictions.append(_run_prediction(batch_features))

            # 2. Small Gaussian noise (positive)
            if n_augmentations >= 2:
                noisy = batch_features + np.random.normal(
                    0, 0.005, batch_features.shape).astype(np.float32)
                all_predictions.append(_run_prediction(noisy))

            # 3. Small Gaussian noise (negative)
            if n_augmentations >= 3:
                noisy_neg = batch_features - np.random.normal(
                    0, 0.005, batch_features.shape).astype(np.float32)
                all_predictions.append(_run_prediction(noisy_neg))

            # 4. Time shift (small circular shift)
            if n_augmentations >= 4 and batch_features.ndim >= 2:
                shift = max(1, batch_features.shape[1] // 50)
                shifted = np.roll(batch_features, shift, axis=1)
                all_predictions.append(_run_prediction(shifted))

            # 5. Volume perturbation
            if n_augmentations >= 5:
                vol = batch_features * np.random.uniform(0.95, 1.05)
                all_predictions.append(_run_prediction(vol.astype(np.float32)))

            # Average predictions
            avg_predictions = np.mean(all_predictions, axis=0)

            # Sprint 1.4: Calibração per-model (preferida) com fallback ao
            # temperature_scaler global (compatibilidade com código antigo).
            model_temp = float(getattr(model_info, 'temperature', 1.0))
            if model_temp == 1.0 and self.temperature_scaler.temperature != 1.0:
                model_temp = float(self.temperature_scaler.temperature)
            avg_predictions = apply_temperature_scaling(avg_predictions, model_temp)

            results = []
            for i in range(len(avg_predictions)):
                pred = avg_predictions[i]
                if pred.shape[-1] == 1:
                    confidence = float(pred[0] if pred.ndim > 0 else pred)
                    is_deepfake = confidence > 0.5
                else:
                    confidence = float(np.max(pred))
                    is_deepfake = np.argmax(pred) == 1

                results.append({
                    'is_deepfake': is_deepfake,
                    'confidence': confidence,
                    'tta_applied': True,
                    'n_augmentations': len(all_predictions),
                    'temperature_applied': model_temp,
                })

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            logger.warning(f"TTA failed, falling back to standard prediction: {e}")
            return self._predict_tensorflow_batch(model_info, features_list, device)
