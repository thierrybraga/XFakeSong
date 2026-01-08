"""
Extração de Características de Modelagem Preditiva
=================================================

Este módulo implementa características baseadas em modelagem preditiva linear,
incluindo LPC, LSF, PARCOR, coeficientes de reflexão e análise de erro de
predição. Refatorado para usar sub-módulos.
"""

import numpy as np
from typing import Dict

from .components.lpc import extract_lpc_features, extract_lsf_features
from .components.coeffs import extract_reflection_features, extract_parcor_features
from .components.error import extract_prediction_error_features
from .components.stability import extract_stability_features


class PredictiveFeatureExtractor:
    """
    Extrator de características de modelagem preditiva.
    """

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, lpc_order: int = 12):
        """
        Inicializa o extrator de características preditivas.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            lpc_order: Ordem do modelo LPC
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.lpc_order = lpc_order

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características de modelagem preditiva.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características preditivas
        """
        features = {}

        # === CARACTERÍSTICAS LPC ===
        lpc_features = extract_lpc_features(
            y, self.sr, self.frame_length, self.hop_length, self.lpc_order
        )
        features.update(lpc_features)

        # === CARACTERÍSTICAS LSF/LSP ===
        lsf_features = extract_lsf_features(
            y, self.sr, self.frame_length, self.hop_length, self.lpc_order
        )
        features.update(lsf_features)

        # === COEFICIENTES DE REFLEXÃO ===
        reflection_features = extract_reflection_features(
            y, self.frame_length, self.hop_length, self.lpc_order
        )
        features.update(reflection_features)

        # === COEFICIENTES PARCOR ===
        parcor_features = extract_parcor_features(
            y, self.frame_length, self.hop_length, self.lpc_order
        )
        features.update(parcor_features)

        # === ANÁLISE DE ERRO DE PREDIÇÃO ===
        prediction_error_features = extract_prediction_error_features(
            y, self.frame_length, self.hop_length, self.lpc_order
        )
        features.update(prediction_error_features)

        # === CARACTERÍSTICAS DE ESTABILIDADE ===
        stability_features = extract_stability_features(
            y, self.frame_length, self.hop_length, self.lpc_order
        )
        features.update(stability_features)

        return features


def test_predictive_features():
    """Testa as características de modelagem preditiva."""
    # Gerar sinais de teste com diferentes características preditivas
    duration = 1.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))

    print("🔮 Testando Características de Modelagem Preditiva:")

    # === SINAL AUTOREGRESSIVO (alta predibilidade) ===
    # Gerar sinal AR usando filtro IIR
    # y[n] = 0.8*y[n-1] - 0.3*y[n-2] + noise[n]

    noise = np.random.randn(len(t)) * 0.1
    y_ar = np.zeros_like(t)

    for i in range(2, len(y_ar)):
        y_ar[i] = 0.8 * y_ar[i - 1] - 0.3 * y_ar[i - 2] + noise[i]

    extractor = PredictiveFeatureExtractor(sr=sr, lpc_order=8)
    features_ar = extractor.extract_features(y_ar)

    print("\n📊 Sinal Autoregressivo (alta predibilidade):")
    key_features = ['prediction_efficiency', 'prediction_snr',
                    'stability_ratio', 'lpc_gain_mean',
                    'reflection_stability_ratio']

    for feature in key_features:
        if feature in features_ar:
            print(f"  {feature}: {features_ar[feature]:.3f}")

    # === SINAL HARMÔNICO (estrutura preditiva) ===
    f0 = 220  # Hz
    y_harmonic = (np.sin(2 * np.pi * f0 * t) +
                  0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                  0.3 * np.sin(2 * np.pi * 3 * f0 * t))

    # Adicionar pouco ruído
    y_harmonic += 0.05 * np.random.randn(len(t))

    features_harmonic = extractor.extract_features(y_harmonic)

    print("\n🎵 Sinal Harmônico (estrutura preditiva):")
    for feature in key_features:
        if feature in features_harmonic:
            print(f"  {feature}: {features_harmonic[feature]:.3f}")

    # === RUÍDO BRANCO (baixa predibilidade) ===
    y_noise = np.random.randn(len(t))

    features_noise = extractor.extract_features(y_noise)

    print("\n🔊 Ruído Branco (baixa predibilidade):")
    for feature in key_features:
        if feature in features_noise:
            print(f"  {feature}: {features_noise[feature]:.3f}")

    # Comparar predibilidade
    print("\n📈 Comparação de Predibilidade:")

    signals = {
        'Autoregressivo': features_ar,
        'Harmônico': features_harmonic,
        'Ruído': features_noise
    }

    comparison_features = [
        'prediction_efficiency',
        'prediction_snr',
        'stability_ratio']

    for feature in comparison_features:
        print(f"\n  {feature}:")
        for signal_name, features in signals.items():
            if feature in features:
                print(f"    {signal_name}: {features[feature]:.3f}")

    # Mostrar alguns coeficientes LPC
    print("\n🔧 Coeficientes LPC (primeiros 4):")
    for signal_name, features in signals.items():
        print(f"  {signal_name}:")
        for i in range(1, 5):
            coeff_name = f'lpc_coeff_{i}_mean'
            if coeff_name in features:
                print(f"    a{i}: {features[coeff_name]:.3f}")


if __name__ == "__main__":
    test_predictive_features()
