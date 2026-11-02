"""Calibração pós-treino sobre a validação: conjunto sob ruído, temperatura,
limiar de OOD, limiar de EER e métricas finais.

Saiu de ``trainer.py`` (P2-15). O ``ModelTrainer`` chama estas funções no
fim de ``train`` e as expõe pelos mesmos métodos privados de antes, que o
``TrainingService`` usa para recalibrar sobre os pesos restaurados.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


def add_awgn(X: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """AWGN a um SNR alvo (por amostra). Paridade EXATA com
    benchmarks/data.add_awgn: a potência REALIZADA do ruído é normalizada
    por amostra (antes só a potência esperada era calibrada — SNR
    realizado desviava ~0,4% em janelas de 80k amostras; agora o SNR
    realizado é idêntico ao alvo, como no protocolo do benchmark)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype="float32")
    flat = X.reshape(len(X), -1)
    sig_power = np.mean(flat ** 2, axis=1, keepdims=True)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)

    unit_noise = rng.standard_normal(flat.shape).astype("float32")
    unit_power = np.mean(unit_noise ** 2, axis=1, keepdims=True)
    target_power = sig_power / max(snr_lin, 1e-12)
    scale = np.sqrt(target_power / np.maximum(unit_power, 1e-12))
    noise = unit_noise * scale
    return (flat + noise).reshape(X.shape).astype("float32")


def looks_like_waveform(X: np.ndarray) -> bool:
    """Heurística de forma de onda (paridade com benchmarks.data).

    (N, T) ou (N, T, 1) com T >= 1000 amostras. Espectrogramas do projeto
    têm (T~100, F~80) e nunca batem esse critério.
    """
    arr = np.asarray(X)
    if arr.ndim == 2:
        return arr.shape[1] >= 1000
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr.shape[1] >= 1000
    return False


def build_calibration_set(validation_data, config):
    """Val limpo + cópias com AWGN para calibrar sob ruído.

    Gated por `calibrate_under_noise` (default True) nos SNRs de
    `calibration_snr_db`. Retorna `(X, y)`; em falha, devolve o val original.

    Protocolo AWGN (2026-07-14): o ruído só é fisicamente válido na FORMA
    DE ONDA. Para modelos espectrais/tabulares (val já em log-mel ou
    features), adicionar AWGN aqui perturbaria o domínio errado — nesses
    casos a calibração usa apenas o val limpo (mesma regra que o benchmark
    aplica ao desativar calibrate_under_noise).
    """
    if validation_data is None:
        return validation_data
    if not getattr(config, "calibrate_under_noise", True):
        return validation_data
    snrs = list(getattr(config, "calibration_snr_db", []) or [])
    if not snrs:
        return validation_data
    if not looks_like_waveform(validation_data[0]):
        logger.info(
            "Calibração sob ruído pulada: entrada de validação não é forma "
            "de onda (AWGN só é aplicado no domínio do waveform; ver "
            "protocolo 2026-07-12). Calibrando com val limpo."
        )
        return validation_data
    try:
        X_val, y_val = validation_data
        X_val = np.asarray(X_val, dtype="float32")
        y_val = np.asarray(y_val)
        extra = [add_awgn(X_val, snr, seed=1234 + i)
                 for i, snr in enumerate(snrs)]
        X_cal = np.concatenate([X_val, *extra], axis=0)
        y_cal = np.concatenate([y_val] * (1 + len(snrs)), axis=0)
        logger.info(
            "Calibração sob ruído: val %d → %d amostras (SNRs=%s)",
            len(y_val), len(y_cal), snrs,
        )
        return (X_cal, y_cal)
    except Exception as e:  # noqa: BLE001
        logger.warning("Falha ao montar calibração com ruído: %s", e)
        return validation_data


def calibrate_temperature(
    model: tf.keras.Model,
    validation_data: Tuple[np.ndarray, np.ndarray],
    config,
) -> float:
    """Calibração post-hoc de temperatura via grid search no conjunto de validação.

    Implementa Temperature Scaling (Guo et al., ICML 2017): busca o T que
    minimiza NLL nas predições do val set. O valor é salvo no input_contract
    e aplicado automaticamente pelo Predictor na inferência.

    Returns:
        Temperatura calibrada (1.0 se desabilitado/falhar).
    """
    default_t = 1.0
    if not getattr(config, "auto_calibrate_temperature", True):
        return default_t

    try:
        X_val, y_val = validation_data
        min_samples = getattr(config, "calibration_min_samples", 50)
        if len(X_val) < min_samples:
            logger.info(
                f"Calibração de temperatura pulada: val set tem "
                f"{len(X_val)} amostras (mínimo {min_samples})"
            )
            return default_t

        # Converte y para índices de classe (compatível com calibrate())
        y_for_calib = np.asarray(y_val)
        if y_for_calib.ndim > 1 and y_for_calib.shape[-1] > 1:
            y_for_calib = np.argmax(y_for_calib, axis=-1)
        else:
            y_for_calib = y_for_calib.ravel().astype(int)

        # Importação tardia para evitar ciclo (predictor importa do trainer)
        from app.domain.services.detection.predictor import TemperatureScaler

        scaler = TemperatureScaler()
        scaler.calibrate(model, X_val, y_for_calib)
        temperature = float(scaler.temperature)
        logger.info(
            f"Temperatura calibrada: T={temperature:.3f} "
            f"({len(X_val)} amostras de val)"
        )
        return temperature

    except Exception as e:
        logger.warning(f"Erro na calibração de temperatura: {e}")
        return default_t


def compute_ood_threshold(
    model: tf.keras.Model,
    validation_data: Tuple[np.ndarray, np.ndarray],
    config,
    temperature: float = 1.0,
) -> Optional[float]:
    """Sprint 2.5: Calibra threshold de OOD detection no val set.

    Computa energy scores para todas as amostras de validação (que são
    consideradas in-distribution por construção) e usa o quantil
    (1 - ood_quantile) como threshold. Amostras com energy score abaixo
    desse threshold serão flagged como OOD na inferência.

    Args:
        model: modelo treinado
        validation_data: (X_val, y_val)
        temperature: T calibrado (para consistência com energia)

    Returns:
        Threshold (float) ou None se desabilitado/falhar.
    """
    if not getattr(config, "compute_ood_threshold", True):
        return None

    try:
        from app.domain.services.detection.predictor import (
            apply_temperature_scaling,
            compute_energy_score,
            normalize_logits_to_probs,
        )

        X_val, _ = validation_data
        predictions = model.predict(
            X_val, batch_size=config.batch_size, verbose=0
        )
        # Aplica mesma temperatura que será usada em inferência
        predictions = normalize_logits_to_probs(predictions)
        predictions = apply_temperature_scaling(predictions, temperature)
        energy_scores = compute_energy_score(predictions, temperature=temperature)

        # Threshold = quantil inferior. ood_quantile=0.95 → 5% das amostras
        # in-distribution com menores energy scores serão falsos positivos OOD.
        q = 1.0 - float(getattr(config, "ood_quantile", 0.95))
        threshold = float(np.quantile(energy_scores, q))
        logger.info(
            f"OOD threshold calibrado: {threshold:.4f} "
            f"(quantile {q:.2f} de {len(energy_scores)} amostras val | "
            f"range=[{energy_scores.min():.3f}, {energy_scores.max():.3f}])"
        )
        return threshold

    except Exception as e:
        logger.warning(f"Erro ao calibrar OOD threshold: {e}")
        return None


def final_metrics(
    model: tf.keras.Model,
    validation_data: Tuple[np.ndarray, np.ndarray],
    config,
    metrics_calculator: MetricsCalculator,
) -> Dict[str, float]:
    """Calcula métricas finais do modelo."""
    X_val, y_val = validation_data

    # Predições
    y_pred = model.predict(X_val, batch_size=config.batch_size, verbose=0)
    from app.domain.services.detection.predictor import (
        normalize_logits_to_probs,
    )
    y_pred = normalize_logits_to_probs(y_pred)
    # Suporta saídas (N,1) sigmoid e (N,K) softmax
    y_pred_classes = (
        np.argmax(y_pred, axis=1)
        if (y_pred.ndim > 1 and y_pred.shape[-1] > 1)
        else (y_pred.ravel() > 0.5).astype(int)
    )

    # Calcular métricas
    return metrics_calculator.calculate_all_metrics(
        y_val, y_pred_classes, y_pred
    )


def compute_eer_threshold(
    model: tf.keras.Model,
    validation_data: Tuple[np.ndarray, np.ndarray],
    config,
    metrics_calculator: MetricsCalculator,
    temperature: float = 1.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Sprint 4.5: Calibra Equal Error Rate threshold no val set.

    EER é o ponto onde FPR == FNR — métrica padrão em anti-spoofing
    (ASVspoof). O threshold associado é frequentemente mais informativo
    que o 0.5 fixo, pois balanceia falsos positivos e falsos negativos.

    Args:
        model: modelo treinado
        validation_data: (X_val, y_val)
        temperature: T calibrado (Sprint 1.4) para consistência

    Returns:
        (eer_threshold, eer_value), ambos float ou None se falhar.
        - eer_threshold: score acima do qual classificar como fake
        - eer_value: valor de EER (taxa de erro no ponto FPR=FNR)
    """
    try:
        from app.domain.services.detection.predictor import (
            apply_temperature_scaling,
            normalize_logits_to_probs,
        )

        X_val, y_val = validation_data
        predictions = model.predict(
            X_val, batch_size=config.batch_size, verbose=0
        )
        predictions = normalize_logits_to_probs(predictions)
        predictions = apply_temperature_scaling(predictions, temperature)

        # Extrai score de probabilidade da classe "fake" (índice 1)
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            scores = predictions[:, 1]
        elif predictions.ndim > 1 and predictions.shape[-1] == 1:
            scores = predictions[:, 0]
        else:
            scores = predictions.ravel()

        # y_val pode ser sparse ou one-hot
        y_arr = np.asarray(y_val)
        if y_arr.ndim > 1 and y_arr.shape[-1] > 1:
            y_true = np.argmax(y_arr, axis=-1)
        else:
            y_true = y_arr.ravel().astype(int)

        # Usa MetricsCalculator.calculate_eer (já existente)
        eer_value, eer_threshold = metrics_calculator.calculate_eer(
            y_true, scores
        )
        logger.info(
            f"EER threshold calibrado: T={eer_threshold:.4f}, EER={eer_value:.4f} "
            f"({len(scores)} amostras val) — alternativa ao threshold 0.5"
        )
        return float(eer_threshold), float(eer_value)

    except Exception as e:
        logger.warning(f"Erro ao calibrar EER threshold: {e}")
        return None, None
