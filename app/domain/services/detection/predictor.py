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


class Predictor:
    """Responsável por realizar predições com modelos."""

    def __init__(self):
        self.temperature_scaler = TemperatureScaler()

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

            # Predição
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
                    predictions = model_info.model.predict(batch_features, verbose=0)
            else:
                predictions = model_info.model.predict(batch_features, verbose=0)

            results = []
            for i in range(len(predictions)):
                pred = predictions[i]

                # Interpretar resultado
                if pred.shape[-1] == 1:
                    # Saída binária
                    confidence = float(pred[0] if pred.ndim > 0 else pred)
                    is_deepfake = confidence > 0.5
                else:
                    # Saída categórica
                    confidence = float(np.max(pred))
                    is_deepfake = np.argmax(pred) == 1

                results.append({
                    'is_deepfake': is_deepfake,
                    'confidence': confidence
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

            # Probabilidade
            if hasattr(model_info.model, 'predict_proba'):
                probas = model_info.model.predict_proba(X)
                for i in range(len(probas)):
                    confidence = float(np.max(probas[i]))
                    is_deepfake = np.argmax(probas[i]) == 1
                    results.append({
                        'is_deepfake': is_deepfake,
                        'confidence': confidence
                    })
            else:
                for i in range(len(predictions)):
                    is_deepfake = int(predictions[i]) == 1
                    results.append({
                        'is_deepfake': is_deepfake,
                        'confidence': 1.0
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

            def _run_prediction(data):
                if device:
                    device_name = device if device.startswith("/") else f"/{device}"
                    if device == "CPU":
                        device_name = "/CPU:0"
                    with tf.device(device_name):
                        return model_info.model.predict(data, verbose=0)
                return model_info.model.predict(data, verbose=0)

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

            # Apply temperature scaling if calibrated
            if self.temperature_scaler.temperature != 1.0:
                avg_predictions = self.temperature_scaler.scale(avg_predictions)

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
                    'n_augmentations': len(all_predictions)
                })

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results
            )

        except Exception as e:
            logger.warning(f"TTA failed, falling back to standard prediction: {e}")
            return self._predict_tensorflow_batch(model_info, features_list, device)
