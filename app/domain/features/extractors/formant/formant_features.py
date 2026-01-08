"""
Extração de Características de Formantes
=======================================

Implementa extração de formantes F1-F4, larguras de banda e características derivadas.
"""

import numpy as np
import librosa
import warnings
from typing import Dict, Any

from app.domain.features.interfaces import IFeatureExtractor
from app.domain.features.types import FeatureType
from app.core.interfaces.audio import AudioData, AudioFeatures
from app.core.interfaces.base import ProcessingStatus, ProcessingResult

from .components.lpc import preemphasis, levinson_durbin_autocorr
from .components.formants import find_formants_from_lpc
from .components.metrics import (
    compute_formant_trajectories,
    compute_vowel_space_area,
    compute_formant_dispersion,
    compute_effective_f2,
    compute_formant_ratios
)


class FormantFeatureExtractor(IFeatureExtractor):
    """Extrator de características de formantes."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048, hop_length: int = 512,
                 n_formants: int = 4, max_freq: float = 8000.0, window: str = 'hamming',
                 preemphasis: float = 0.97, fmin: float = 80.0):
        """
        Inicializa o extrator de formantes.

        Args:
            sr: Taxa de amostragem
            frame_length: Tamanho da janela
            hop_length: Tamanho do salto
            n_formants: Número de formantes a extrair
            max_freq: Frequência máxima para análise (padrão: 8000Hz)
            window: Tipo de janela
            preemphasis: Coeficiente de pré-ênfase
            fmin: Frequência mínima (padrão: 80Hz)
        """
        # Validação de parâmetros
        if sr <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")
        if frame_length <= 0:
            raise ValueError("Tamanho da janela deve ser positivo")
        if hop_length <= 0:
            raise ValueError("Tamanho do salto deve ser positivo")
        if n_formants <= 0:
            raise ValueError("Número de formantes deve ser positivo")
        if max_freq <= 0 or max_freq > sr / 2:
            raise ValueError(
                "Frequência máxima deve ser positiva e menor que Nyquist")
        if not 0 <= preemphasis <= 1:
            raise ValueError(
                "Coeficiente de pré-ênfase deve estar entre 0 e 1")

        if hop_length > frame_length:
            warnings.warn(
                "hop_length maior que frame_length pode causar perda de informação")

        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_formants = n_formants
        self.max_freq = max_freq
        self.window = window
        self.preemphasis = preemphasis
        self.fmin = fmin

        # Parâmetros para análise LPC
        self.lpc_order = min(12, sr // 1000)  # Ordem típica para formantes

    def extract(
            self, audio_data: AudioData) -> ProcessingResult[AudioFeatures]:
        """
        Extrai características de formantes do áudio.

        Args:
            audio_data: Dados de áudio

        Returns:
            ProcessingResult com características de formantes
        """
        try:
            if audio_data.samples is None or len(audio_data.samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=["Dados de áudio vazios"]
                )

            samples = audio_data.samples
            if audio_data.sample_rate != self.sr:
                # Reamostrar se necessário
                samples = librosa.resample(
                    samples, orig_sr=audio_data.sample_rate, target_sr=self.sr)

            features_dict = self.extract_features(samples)

            # Converter características para formato AudioFeatures
            features_arrays = {}
            for name, value in features_dict.items():
                if isinstance(value, (list, np.ndarray)):
                    if len(np.array(value).shape) == 0:
                        features_arrays[name] = np.array([float(value)])
                    else:
                        features_arrays[name] = np.array(value).flatten()
                else:
                    features_arrays[name] = np.array([float(value)])

            audio_features = AudioFeatures(
                features=features_arrays,
                feature_type=FeatureType.FORMANT,
                extraction_params=self.get_extraction_params(),
                audio_metadata={'sample_rate': audio_data.sample_rate}
            )

            return ProcessingResult.success(
                data=audio_features
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[str(e)]
            )

    def get_feature_type(self) -> FeatureType:
        """Retorna o tipo de características extraídas."""
        return FeatureType.FORMANT

    def get_feature_names(self) -> list[str]:
        """Retorna os nomes das características extraídas."""
        names = []
        # Características básicas dos formantes
        for i in range(self.n_formants):
            formant_name = f'F{i + 1}'
            names.extend([
                f'formant_{formant_name}_freq',
                f'formant_{formant_name}_bandwidth',
                f'formant_{formant_name}_amplitude',
                f'{formant_name}_trajectory_mean',
                f'{formant_name}_trajectory_std'
            ])

        # Características derivadas
        names.extend([
            'vowel_space_area', 'formant_dispersion', 'effective_f2',
            'F2_F1_ratio', 'F3_F2_ratio', 'F4_F3_ratio'
        ])

        return names

    def get_extraction_params(self) -> Dict[str, Any]:
        """Retorna os parâmetros de extração."""
        return {
            'sr': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'n_formants': self.n_formants,
            'max_freq': self.max_freq,
            'window': self.window,
            'preemphasis': self.preemphasis,
            'fmin': self.fmin,
            'lpc_order': self.lpc_order
        }

    def extract_features(self, y: np.ndarray) -> Dict:
        """
        Extrai todas as características de formantes.

        Args:
            y: Sinal de áudio

        Returns:
            Dicionário com características de formantes
        """
        # Validação de entrada
        if not isinstance(y, np.ndarray):
            raise TypeError("Entrada deve ser um array numpy")

        if len(y) == 0:
            warnings.warn("Sinal de áudio vazio")
            return self._get_default_formant_features()

        # Verificar se o sinal é muito curto
        min_length = self.frame_length
        if len(y) < min_length:
            warnings.warn(
                f"Sinal muito curto ({
                    len(y)} samples), fazendo padding para {min_length}")
            y = np.pad(y, (0, min_length - len(y)), mode='constant')

        # Normalizar sinal
        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max

        try:
            features = {}

            # === CARACTERÍSTICAS DOS FORMANTES ===

            # Extrair formantes usando LPC
            formants_data = self._extract_formants_lpc(y)

            # Frequências dos formantes
            for i in range(self.n_formants):
                formant_name = f'F{i + 1}'
                features[f'formant_{formant_name}_freq'] = formants_data[f'{formant_name}_freq']
                features[f'formant_{formant_name}_bandwidth'] = formants_data[f'{formant_name}_bw']
                features[f'formant_{formant_name}_amplitude'] = formants_data[f'{formant_name}_amp']

            # === CARACTERÍSTICAS DERIVADAS ===

            # Trajetórias dos formantes (velocidade de mudança)
            trajectories = compute_formant_trajectories(
                formants_data, self.n_formants)
            features.update(trajectories)

            # Área do espaço vocálico
            features['vowel_space_area'] = compute_vowel_space_area(
                formants_data['F1_freq'], formants_data['F2_freq'])

            # Dispersão dos formantes
            features['formant_dispersion'] = compute_formant_dispersion(
                formants_data, self.n_formants)

            # F2' efetivo (formante ajustado)
            features['effective_F2'] = compute_effective_f2(
                formants_data['F1_freq'], formants_data['F2_freq'], formants_data['F3_freq'])

            # Razões entre formantes
            formant_ratios = compute_formant_ratios(formants_data)
            features.update(formant_ratios)

            # === MAPEAMENTO PARA NOMES ESPERADOS ===
            # Mapear características de formantes para os nomes esperados
            features['f1_mean'] = np.nanmean(
                formants_data['F1_freq']) if len(
                formants_data['F1_freq']) > 0 else 0
            features['f2_mean'] = np.nanmean(
                formants_data['F2_freq']) if len(
                formants_data['F2_freq']) > 0 else 0
            features['f3_mean'] = np.nanmean(
                formants_data['F3_freq']) if len(
                formants_data['F3_freq']) > 0 else 0
            features['f4_mean'] = np.nanmean(
                formants_data['F4_freq']) if len(
                formants_data['F4_freq']) > 0 else 0
            features['f1_bandwidth'] = np.nanmean(
                formants_data['F1_bw']) if len(
                formants_data['F1_bw']) > 0 else 0
            features['f2_bandwidth'] = np.nanmean(
                formants_data['F2_bw']) if len(
                formants_data['F2_bw']) > 0 else 0
            features['f3_bandwidth'] = np.nanmean(
                formants_data['F3_bw']) if len(
                formants_data['F3_bw']) > 0 else 0
            features['f4_bandwidth'] = np.nanmean(
                formants_data['F4_bw']) if len(
                formants_data['F4_bw']) > 0 else 0

            return features

        except Exception as e:
            warnings.warn(
                f"Erro na extração de características de formantes: {e}")
            return self._get_default_formant_features()

    def _get_default_formant_features(self) -> Dict:
        """Retorna características padrão em caso de erro."""
        features = {}

        # Características básicas dos formantes
        for i in range(self.n_formants):
            formant_name = f'F{i + 1}'
            features[f'formant_{formant_name}_freq'] = np.array([0.0])
            features[f'formant_{formant_name}_bandwidth'] = np.array([0.0])
            features[f'formant_{formant_name}_amplitude'] = np.array([0.0])
            features[f'{formant_name}_trajectory_mean'] = 0.0
            features[f'{formant_name}_trajectory_std'] = 0.0

        # Características derivadas
        features['vowel_space_area'] = 0.0
        features['formant_dispersion'] = 0.0
        features['effective_F2'] = np.array([0.0])
        features['F2_F1_ratio_mean'] = 0.0
        features['F2_F1_ratio_std'] = 0.0
        features['F3_F2_ratio_mean'] = 0.0
        features['F3_F2_ratio_std'] = 0.0

        return features

    def _extract_formants_lpc(self, y: np.ndarray) -> Dict:
        """Extrai formantes usando Linear Predictive Coding."""
        # Parâmetros LPC
        lpc_order = min(2 + self.sr // 1000, 16)  # Regra empírica

        # Dividir em frames
        frames = librosa.util.frame(y, frame_length=self.frame_length,
                                    hop_length=self.hop_length)

        formants_data = {
            'F1_freq': [], 'F1_bw': [], 'F1_amp': [],
            'F2_freq': [], 'F2_bw': [], 'F2_amp': [],
            'F3_freq': [], 'F3_bw': [], 'F3_amp': [],
            'F4_freq': [], 'F4_bw': [], 'F4_amp': []
        }

        for frame in frames.T:
            # Pré-ênfase
            preemphasized = preemphasis(frame, self.preemphasis)

            # Aplicar janela
            windowed = preemphasized * np.hamming(len(preemphasized))

            # Calcular coeficientes LPC
            lpc_coeffs = levinson_durbin_autocorr(windowed, lpc_order)

            # Encontrar formantes a partir dos coeficientes LPC
            formants = find_formants_from_lpc(lpc_coeffs, self.sr)

            # Armazenar resultados (preencher com NaN se não encontrados)
            for i in range(self.n_formants):
                formant_key = f'F{i + 1}'
                if i < len(formants):
                    freq, bw, amp = formants[i]
                    formants_data[f'{formant_key}_freq'].append(freq)
                    formants_data[f'{formant_key}_bw'].append(bw)
                    formants_data[f'{formant_key}_amp'].append(amp)
                else:
                    formants_data[f'{formant_key}_freq'].append(np.nan)
                    formants_data[f'{formant_key}_bw'].append(np.nan)
                    formants_data[f'{formant_key}_amp'].append(np.nan)

        # Converter para arrays numpy
        for key in formants_data:
            formants_data[key] = np.array(formants_data[key])

        return formants_data


# Para manter compatibilidade se o arquivo for executado diretamente
if __name__ == "__main__":
    def test_formant_features():
        """Testa as características de formantes."""
        # Gerar sinal de teste simulando vogal
        duration = 1.0
        sr = 22050
        t = np.linspace(0, duration, int(duration * sr))

        # Simular vogal /a/ com formantes aproximados
        f0 = 200  # Fundamental
        f1 = 730  # F1 para /a/
        f2 = 1090  # F2 para /a/
        f3 = 2440  # F3 para /a/

        # Gerar sinal com harmônicos e formantes
        y = np.zeros_like(t)

        # Fundamental e harmônicos
        for harmonic in range(1, 8):
            freq = harmonic * f0
            if freq < sr / 2:
                amplitude = 1.0 / harmonic  # Amplitude decresce com harmônico

                # Modelar formantes como filtros passa-banda
                if abs(freq - f1) < 100:
                    amplitude *= 3  # Amplificar F1
                elif abs(freq - f2) < 150:
                    amplitude *= 2  # Amplificar F2
                elif abs(freq - f3) < 200:
                    amplitude *= 1.5  # Amplificar F3

                y += amplitude * np.sin(2 * np.pi * freq * t)

        # Adicionar modulação leve nos formantes
        # Modulação lenta
        # f1_mod = f1 + 20 * np.sin(2 * np.pi * 3 * t)
        # f2_mod = f2 + 30 * np.sin(2 * np.pi * 2 * t)

        # Sintetizar sinal (modelo de fonte-filtro simplificado)

        # Aplicar envelope
        envelope = np.exp(-2 * t) + 0.3  # Decaimento com sustain
        y = y * envelope

        # Adicionar ruído leve
        y += 0.05 * np.random.randn(len(y))

        # Extrair características
        extractor = FormantFeatureExtractor(sr=sr)
        features = extractor.extract_features(y)

        print("🎵 Características de Formantes Extraídas:")
        for name, values in features.items():
            if isinstance(values, np.ndarray):
                if values.ndim == 1:
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        print(f"  {name}: média={np.mean(valid_values):.1f}, "
                              f"std={np.std(valid_values):.1f}")
                    else:
                        print(f"  {name}: sem valores válidos")
                else:
                    print(f"  {name}: shape={values.shape}")
            else:
                print(f"  {name}: {values:.1f}")

    test_formant_features()
