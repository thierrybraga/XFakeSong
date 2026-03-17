#!/usr/bin/env python3
"""
Serviço de Segmentação de Áudio
==============================

Este módulo implementa a segmentação de arquivos de áudio em intervalos de 1 segundo
para extração de features mais granular e consistente.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import librosa
import numpy as np

from app.core.interfaces.audio import AudioData
from app.core.interfaces.base import ProcessingResult, ProcessingStatus


@dataclass
class AudioSegment:
    """Representa um segmento de áudio de 1 segundo"""
    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    duration: float
    segment_id: int
    source_file: str

    def to_audio_data(self) -> AudioData:
        """Converte para AudioData"""
        return AudioData(
            samples=self.samples,
            sample_rate=self.sample_rate,
            duration=self.duration
        )


@dataclass
class SegmentationConfig:
    """Configuração para segmentação de áudio"""
    segment_duration: float = 1.0  # Duração do segmento em segundos
    # Sobreposição entre segmentos (0.0 = sem sobreposição)
    overlap_ratio: float = 0.0
    # Duração mínima para considerar um segmento válido
    min_segment_duration: float = 0.5
    target_sample_rate: int = 16000    # Taxa de amostragem alvo
    normalize_segments: bool = True    # Normalizar cada segmento
    remove_silence: bool = False       # Remover segmentos silenciosos
    silence_threshold: float = 0.01    # Limiar para detecção de silêncio


class AudioSegmentationService:
    """Serviço para segmentação de arquivos de áudio em intervalos de 1 segundo"""

    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()
        self.logger = logging.getLogger(__name__)

        # Validar configuração
        self._validate_config()

    def _validate_config(self):
        """Valida a configuração de segmentação"""
        if self.config.segment_duration <= 0:
            raise ValueError("Duração do segmento deve ser positiva")

        if not (0.0 <= self.config.overlap_ratio < 1.0):
            raise ValueError("Taxa de sobreposição deve estar entre 0.0 e 1.0")

        if self.config.min_segment_duration <= 0:
            raise ValueError("Duração mínima do segmento deve ser positiva")

        if self.config.target_sample_rate <= 0:
            raise ValueError("Taxa de amostragem deve ser positiva")

    def segment_audio_file(
            self, file_path: str) -> ProcessingResult[List[AudioSegment]]:
        """Segmenta um arquivo de áudio em intervalos de 1 segundo"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Arquivo não encontrado: {file_path}"]
                )

            # Carregar áudio
            samples, sr = librosa.load(
                str(file_path),
                sr=self.config.target_sample_rate
            )

            if len(samples) == 0:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Arquivo de áudio vazio: {file_path}"]
                )

            # Segmentar áudio
            segments = self._create_segments(
                samples=samples,
                sample_rate=sr,
                source_file=str(file_path)
            )

            if not segments:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    errors=[f"Nenhum segmento válido criado para: {file_path}"]
                )

            self.logger.info(
                f"Arquivo {
                    file_path.name} segmentado em {
                    len(segments)} segmentos de {
                    self.config.segment_duration}s"
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=segments
            )

        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                errors=[f"Erro ao segmentar {file_path}: {str(e)}"]
            )

    def segment_audio_data(self, audio_data: AudioData,
                           source_file: str = "unknown") -> List[AudioSegment]:
        """Segmenta dados de áudio já carregados"""
        return self._create_segments(
            samples=audio_data.samples,
            sample_rate=audio_data.sample_rate,
            source_file=source_file
        )

    def _create_segments(self, samples: np.ndarray, sample_rate: int,
                         source_file: str) -> List[AudioSegment]:
        """Cria segmentos de áudio"""
        segments = []

        # Calcular parâmetros de segmentação
        segment_samples = int(self.config.segment_duration * sample_rate)
        overlap_samples = int(segment_samples * self.config.overlap_ratio)
        hop_samples = segment_samples - overlap_samples

        # Criar segmentos
        segment_id = 0
        start_sample = 0

        while start_sample < len(samples):
            end_sample = min(start_sample + segment_samples, len(samples))
            segment_samples_data = samples[start_sample:end_sample]

            # Verificar duração mínima
            segment_duration = len(segment_samples_data) / sample_rate
            if segment_duration < self.config.min_segment_duration:
                break

            # Verificar se não é silêncio (se configurado)
            if self.config.remove_silence and self._is_silence(
                    segment_samples_data):
                start_sample += hop_samples
                continue

            # Normalizar segmento (se configurado)
            if self.config.normalize_segments:
                segment_samples_data = self._normalize_segment(
                    segment_samples_data)

            # Criar segmento
            segment = AudioSegment(
                samples=segment_samples_data,
                sample_rate=sample_rate,
                start_time=start_sample / sample_rate,
                end_time=end_sample / sample_rate,
                duration=segment_duration,
                segment_id=segment_id,
                source_file=source_file
            )

            segments.append(segment)
            segment_id += 1
            start_sample += hop_samples

        return segments

    def _is_silence(self, samples: np.ndarray) -> bool:
        """Verifica se um segmento é silencioso"""
        rms_energy = np.sqrt(np.mean(samples ** 2))
        return rms_energy < self.config.silence_threshold

    def _normalize_segment(self, samples: np.ndarray) -> np.ndarray:
        """Normaliza um segmento de áudio"""
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            return samples / max_val
        return samples

    def segment_batch(
            self, file_paths: List[str]) -> ProcessingResult[Dict[str, List[AudioSegment]]]:
        """Segmenta múltiplos arquivos de áudio"""
        results = {}
        errors = []

        for file_path in file_paths:
            result = self.segment_audio_file(file_path)

            if result.status == ProcessingStatus.SUCCESS:
                results[file_path] = result.data
            else:
                errors.extend(result.errors)

        if errors:
            return ProcessingResult(
                status=ProcessingStatus.PARTIAL_SUCCESS if results else ProcessingStatus.ERROR,
                data=results,
                errors=errors
            )

        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=results
        )

    def get_segment_iterator(
            self, file_paths: List[str]) -> Iterator[Tuple[str, AudioSegment]]:
        """Retorna iterador para processar segmentos em lotes"""
        for file_path in file_paths:
            result = self.segment_audio_file(file_path)

            if result.status == ProcessingStatus.SUCCESS:
                for segment in result.data:
                    yield file_path, segment
            else:
                self.logger.warning(
                    f"Falha ao segmentar {file_path}: {
                        result.errors}")

    def get_statistics(self, segments: List[AudioSegment]) -> Dict[str, Any]:
        """Calcula estatísticas dos segmentos"""
        if not segments:
            return {}

        durations = [seg.duration for seg in segments]
        energies = [np.sqrt(np.mean(seg.samples ** 2)) for seg in segments]

        stats = {
            'total_segments': len(segments),
            'total_duration': sum(durations),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'avg_energy': np.mean(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'sample_rate': segments[0].sample_rate,
            'unique_sources': len({seg.source_file for seg in segments})
        }

        return stats

    def save_segments_info(
            self, segments: List[AudioSegment], output_file: str):
        """Salva informações dos segmentos em arquivo JSON"""
        import json

        segments_info = {
            'segmentation_config': {
                'segment_duration': self.config.segment_duration,
                'overlap_ratio': self.config.overlap_ratio,
                'min_segment_duration': self.config.min_segment_duration,
                'target_sample_rate': self.config.target_sample_rate,
                'normalize_segments': self.config.normalize_segments,
                'remove_silence': self.config.remove_silence
            },
            'statistics': self.get_statistics(segments),
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'source_file': seg.source_file,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'sample_count': len(seg.samples),
                    'rms_energy': float(np.sqrt(np.mean(seg.samples ** 2)))
                }
                for seg in segments
            ],
            'timestamp': datetime.now().isoformat()
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments_info, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Informações dos segmentos salvas em: {output_path}")


def main():
    """Demonstração do serviço de segmentação"""
    print("🎵 SERVIÇO DE SEGMENTAÇÃO DE ÁUDIO")
    print("=" * 50)

    # Configuração
    config = SegmentationConfig(
        segment_duration=1.0,
        overlap_ratio=0.0,
        min_segment_duration=0.5,
        target_sample_rate=16000,
        normalize_segments=True,
        remove_silence=False
    )

    # Criar serviço
    segmentation_service = AudioSegmentationService(config)

    # Testar com arquivos de exemplo
    samples_dir = Path("datasets/raw")

    if samples_dir.exists():
        # Buscar arquivos de teste
        real_files = list((samples_dir / "real").glob("*.wav"))[:3]
        fake_files = list((samples_dir / "fake").glob("*.wav"))[:3]

        test_files = [str(f) for f in real_files + fake_files]

        if test_files:
            print(f"\n📁 Testando com {len(test_files)} arquivos...")

            # Segmentar arquivos
            result = segmentation_service.segment_batch(test_files)

            if result.status == ProcessingStatus.SUCCESS:
                total_segments = sum(len(segments)
                                     for segments in result.data.values())
                print("\n✅ Segmentação concluída!")
                print(f"   Total de segmentos: {total_segments}")

                # Mostrar estatísticas por arquivo
                for file_path, segments in result.data.items():
                    file_name = Path(file_path).name
                    stats = segmentation_service.get_statistics(segments)

                    print(f"\n📊 {file_name}:")
                    print(f"   Segmentos: {stats['total_segments']}")
                    print(f"   Duração total: {stats['total_duration']:.2f}s")
                    print(f"   Energia média: {stats['avg_energy']:.4f}")

                # Salvar informações
                all_segments = []
                for segments in result.data.values():
                    all_segments.extend(segments)

                output_file = "datasets/features/segmentation_info.json"
                segmentation_service.save_segments_info(
                    all_segments, output_file)

            else:
                print(f"\n❌ Erro na segmentação: {result.errors}")
        else:
            print("\n❌ Nenhum arquivo de áudio encontrado para teste")
    else:
        print(f"\n❌ Diretório de amostras não encontrado: {samples_dir}")


if __name__ == "__main__":
    main()
