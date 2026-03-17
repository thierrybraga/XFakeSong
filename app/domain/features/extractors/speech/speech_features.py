"""
Extração de Características Específicas para Fala
===============================================

Este módulo implementa características específicas para análise de fala,
incluindo características linguísticas, articulatórias e temporais.
"""

from typing import Dict

import numpy as np

from .measures import (
    extract_articulatory_features,
    extract_fluency_features,
    extract_linguistic_features,
    extract_temporal_speech_features,
    extract_vocal_quality_features,
)


class SpeechFeatureExtractor:
    """
    Extrator de características específicas para fala.
    """

    def __init__(self, sr: int = 22050, frame_length: int = 2048,
                 hop_length: int = 512, f0_min: float = 80,
                 f0_max: float = 400):
        """
        Inicializa o extrator de características de fala.

        Args:
            sr: Taxa de amostragem
            frame_length: Comprimento da janela
            hop_length: Salto entre janelas
            f0_min: Frequência fundamental mínima
            f0_max: Frequência fundamental máxima
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características específicas para fala.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características de fala
        """
        features = {}

        # === CARACTERÍSTICAS LINGUÍSTICAS ===
        features.update(
            extract_linguistic_features(
                y,
                self.sr,
                self.frame_length,
                self.hop_length))

        # === CARACTERÍSTICAS ARTICULATÓRIAS ===
        features.update(
            extract_articulatory_features(
                y,
                self.sr,
                self.frame_length,
                self.hop_length))

        # === CARACTERÍSTICAS TEMPORAIS DE FALA ===
        features.update(
            extract_temporal_speech_features(
                y,
                self.sr,
                self.frame_length,
                self.hop_length))

        # === CARACTERÍSTICAS DE FLUÊNCIA ===
        features.update(
            extract_fluency_features(
                y,
                self.sr,
                self.frame_length,
                self.hop_length))

        # === CARACTERÍSTICAS DE QUALIDADE VOCAL ===
        features.update(
            extract_vocal_quality_features(
                y,
                self.sr,
                self.frame_length,
                self.hop_length))

        return features


def test_speech_features():
    """Testa as características específicas para fala."""
    # Gerar sinais de teste com diferentes características de fala
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(duration * sr))

    print("🗣️ Testando Características Específicas para Fala:")

    # === SINAL DE FALA SINTÉTICA ===
    # Simular fala com segmentos vocálicos e consonantais

    # Segmento vocálico (harmônico)
    f0 = 150  # Hz
    vowel_segment = (np.sin(2 * np.pi * f0 * t[:len(t) // 3]) +
                     0.5 * np.sin(2 * np.pi * 2 * f0 * t[:len(t) // 3]) +
                     0.3 * np.sin(2 * np.pi * 3 * f0 * t[:len(t) // 3]))

    # Segmento consonantal (ruído filtrado)
    consonant_segment = np.random.randn(len(t) // 3) * 0.3

    # Pausa
    pause_segment = np.zeros(len(t) // 3)

    # Combinar segmentos
    y_speech = np.concatenate(
        [vowel_segment, consonant_segment, pause_segment])

    # Adicionar envelope para simular articulação
    envelope = np.exp(-0.5 * (t - duration / 2)**2 / (duration / 4)**2)
    y_speech *= envelope

    extractor = SpeechFeatureExtractor(sr=sr)
    features_speech = extractor.extract_features(y_speech)

    print("\n📊 Sinal de Fala Sintética:")
    key_features = ['speech_rate', 'vowel_count', 'consonant_segment_count',
                    'pause_count', 'syllables_per_second', 'speech_continuity_ratio']

    for feature in key_features:
        if feature in features_speech:
            print(f"  {feature}: {features_speech[feature]:.3f}")

    # === SINAL COM PAUSAS FREQUENTES ===
    # Simular fala hesitante

    segments = []

    for i in range(5):
        # Segmento de fala curto
        speech_duration = 0.2
        speech_samples = int(speech_duration * sr)
        speech_t = np.linspace(0, speech_duration, speech_samples)

        speech_seg = np.sin(2 * np.pi * 200 * speech_t) * np.exp(-speech_t * 2)
        segments.append(speech_seg)

        # Pausa
        pause_duration = 0.3
        pause_samples = int(pause_duration * sr)
        pause_seg = np.zeros(pause_samples)
        segments.append(pause_seg)

    y_hesitant = np.concatenate(segments)

    # Ajustar comprimento
    if len(y_hesitant) > len(t):
        y_hesitant = y_hesitant[:len(t)]
    else:
        y_hesitant = np.pad(y_hesitant, (0, len(t) - len(y_hesitant)))

    features_hesitant = extractor.extract_features(y_hesitant)

    print("\n🤔 Sinal com Pausas Frequentes (hesitante):")
    for feature in key_features:
        if feature in features_hesitant:
            print(f"  {feature}: {features_hesitant[feature]:.3f}")

    # === SINAL FLUENTE ===
    # Simular fala fluente e contínua

    # Variação suave de frequência para simular prosódia
    f0_variation = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # Variação prosódica

    y_fluent = np.zeros_like(t)

    for i, freq in enumerate(f0_variation):
        y_fluent[i] = np.sin(2 * np.pi * freq * t[i])

    # Adicionar harmônicos
    y_fluent += 0.5 * np.sin(2 * np.pi * 2 * f0_variation * t)
    y_fluent += 0.3 * np.sin(2 * np.pi * 3 * f0_variation * t)

    # Envelope suave
    y_fluent *= 0.8

    features_fluent = extractor.extract_features(y_fluent)

    print("\n🎯 Sinal Fluente e Contínuo:")
    for feature in key_features:
        if feature in features_fluent:
            print(f"  {feature}: {features_fluent[feature]:.3f}")

    # Comparar características entre tipos de fala
    print("\n📈 Comparação entre Tipos de Fala:")

    signals = {
        'Sintética': features_speech,
        'Hesitante': features_hesitant,
        'Fluente': features_fluent
    }

    comparison_features = [
        'speech_rate',
        'pause_count',
        'speech_continuity_ratio']

    for feature in comparison_features:
        print(f"\n  {feature}:")
        for signal_name, features in signals.items():
            if feature in features:
                print(f"    {signal_name}: {features[feature]:.3f}")

    # Mostrar características articulatórias estimadas
    print("\n🗣️ Características Articulatórias Estimadas:")
    articulatory_features = ['vocal_tract_length_mean', 'jaw_openness_estimate',
                             'tongue_position_estimate']

    for signal_name, features in signals.items():
        print(f"  {signal_name}:")
        for feature in articulatory_features:
            if feature in features:
                print(f"    {feature}: {features[feature]:.3f}")


if __name__ == "__main__":
    test_speech_features()
