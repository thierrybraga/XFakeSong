"""Streaming Inference API (Sprint 5.3).

Provê detecção contínua de deepfake em áudio streaming (microfone,
WebSocket, gravação ao vivo). Mantém estado entre chunks via buffer
circular + smoothing exponencial dos scores.

Princípios:
- **Sliding window**: cada predição cobre uma janela de N segundos
- **Hop overlap**: chunks adjacentes têm sobreposição (>= 50%) para suavizar
- **EMA smoothing**: score exponencial-moving-average elimina ruído frame-a-frame
- **Stateful**: classe `StreamingDetector` mantém histórico entre chamadas
- **Async-friendly**: API compatível com asyncio/WebSocket

Uso típico (file stream / microfone):
    detector = StreamingDetector(
        detection_service=service,
        model_name='AASIST_v1',
        window_seconds=3.0,
        hop_seconds=1.0,
        sample_rate=16000,
    )

    for chunk in audio_chunks:  # chunks de ~100ms
        result = detector.push(chunk)
        if result is not None:
            print(f"t={result['timestamp']:.2f}s "
                  f"is_fake={result['is_fake']} "
                  f"confidence={result['smoothed_confidence']:.3f}")

Uso típico (WebSocket — pseudo-código):
    async with websocket.connect('ws://...') as ws:
        detector = StreamingDetector(...)
        async for chunk in ws:
            result = detector.push(np.frombuffer(chunk, dtype=np.float32))
            if result:
                await ws.send_json(result)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuração de streaming inference."""
    # Janela de áudio analisada por inferência (segundos)
    window_seconds: float = 3.0
    # Hop entre janelas consecutivas — se < window_seconds há overlap (recomendado)
    hop_seconds: float = 1.0
    # Taxa de amostragem (Hz) — deve corresponder ao modelo
    sample_rate: int = 16000
    # EMA smoothing factor para confidence (0=sem smoothing, 1=máximo)
    ema_alpha: float = 0.3
    # Manter histórico de N últimos resultados
    history_size: int = 100
    # Buffer max em segundos (descarta amostras mais antigas)
    max_buffer_seconds: float = 30.0
    # Threshold de classificação (None = usa default do modelo / 0.5)
    classification_threshold: Optional[float] = None
    # Mínimo de amostras antes da primeira predição
    min_samples_for_inference: Optional[int] = None  # default: window * sr


@dataclass
class StreamingResult:
    """Resultado de uma inferência streaming."""
    timestamp: float  # tempo (s) desde o início do stream
    is_fake: bool
    raw_confidence: float       # confidence desta janela
    smoothed_confidence: float  # EMA das últimas inferências
    window_start_s: float
    window_end_s: float
    samples_processed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingDetector:
    """Detector de deepfake em streaming.

    Args:
        detection_service: instância de DetectionService (já carregado)
        model_name: nome do modelo a usar (deve estar em service.loaded_models)
        config: StreamingConfig (default: 3s window, 1s hop, 16kHz)
    """

    def __init__(
        self,
        detection_service: Any,
        model_name: str,
        config: Optional[StreamingConfig] = None,
    ):
        self.detection_service = detection_service
        self.model_name = model_name
        self.config = config or StreamingConfig()

        # Buffer circular de áudio (samples) e índice de tempo
        self._buffer: Deque[float] = deque(
            maxlen=int(self.config.max_buffer_seconds * self.config.sample_rate)
        )
        self._stream_start_time = time.time()
        self._total_samples_processed = 0
        self._last_inference_sample = 0  # índice do último hop

        # EMA state
        self._ema_confidence: Optional[float] = None

        # Histórico
        self._history: Deque[StreamingResult] = deque(
            maxlen=self.config.history_size
        )

        # Pre-compute thresholds
        self._window_samples = int(
            self.config.window_seconds * self.config.sample_rate
        )
        self._hop_samples = int(
            self.config.hop_seconds * self.config.sample_rate
        )
        self._min_samples = (
            self.config.min_samples_for_inference or self._window_samples
        )

    def push(self, samples: np.ndarray) -> Optional[StreamingResult]:
        """Adiciona novas amostras de áudio ao buffer. Inferência só roda
        se houver `min_samples` no buffer E o último hop foi atingido.

        Args:
            samples: ndarray 1D de float32 (mono) ou 2D (será convertido para mono)

        Returns:
            StreamingResult se uma nova inferência foi feita, None caso contrário.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples[:, 0]  # pega primeiro canal

        # Adiciona ao buffer
        self._buffer.extend(samples.tolist())
        self._total_samples_processed += len(samples)

        # Decide se faz inferência: precisa min_samples e ter passado hop
        if len(self._buffer) < self._min_samples:
            return None

        samples_since_last = self._total_samples_processed - self._last_inference_sample
        if samples_since_last < self._hop_samples:
            return None

        # Extrai janela mais recente
        return self._run_inference()

    def _run_inference(self) -> Optional[StreamingResult]:
        """Executa inferência na janela atual e atualiza estado."""
        # Pega últimos `window_samples` do buffer
        window = np.array(
            list(self._buffer)[-self._window_samples:],
            dtype=np.float32,
        )

        # Normalização leve (alinha com expectativa do modelo)
        max_abs = float(np.max(np.abs(window)))
        if max_abs > 0:
            window = window / max_abs

        # Wrap em AudioData
        try:
            from app.core.interfaces.audio import AudioData
            audio_data = AudioData(
                samples=window,
                sample_rate=self.config.sample_rate,
                channels=1,
                duration=self.config.window_seconds,
            )
        except Exception as e:
            logger.error(f"Falha ao criar AudioData: {e}")
            return None

        # Inferência via DetectionService
        try:
            result = self.detection_service.detect_single(
                audio_data, model_name=self.model_name
            )
            from app.core.interfaces.base import ProcessingStatus
            if result.status != ProcessingStatus.SUCCESS:
                logger.debug(f"Predição falhou: {result.errors}")
                return None
            pred = result.data
        except Exception as e:
            logger.warning(f"Erro em detect_single (streaming): {e}")
            return None

        raw_conf = float(pred.confidence)

        # EMA smoothing
        alpha = self.config.ema_alpha
        if self._ema_confidence is None:
            self._ema_confidence = raw_conf
        else:
            self._ema_confidence = (
                alpha * raw_conf + (1 - alpha) * self._ema_confidence
            )

        # Threshold de classificação
        threshold = (
            self.config.classification_threshold
            if self.config.classification_threshold is not None
            else 0.5
        )

        # Timestamp (segundos relativos ao início do stream)
        elapsed = time.time() - self._stream_start_time
        window_end_s = self._total_samples_processed / self.config.sample_rate
        window_start_s = window_end_s - self.config.window_seconds

        sr = StreamingResult(
            timestamp=elapsed,
            is_fake=bool(self._ema_confidence > threshold),
            raw_confidence=raw_conf,
            smoothed_confidence=float(self._ema_confidence),
            window_start_s=max(0.0, window_start_s),
            window_end_s=window_end_s,
            samples_processed=self._total_samples_processed,
            metadata={
                'model_name': self.model_name,
                'classification_threshold': threshold,
                'ema_alpha': alpha,
                'raw_is_fake': bool(pred.is_fake),
                'per_window_metadata': pred.metadata,
            },
        )

        self._history.append(sr)
        self._last_inference_sample = self._total_samples_processed
        return sr

    def get_history(self) -> List[StreamingResult]:
        """Retorna histórico ordenado dos últimos `history_size` resultados."""
        return list(self._history)

    def get_aggregate_score(self, last_n: int = 5) -> Optional[float]:
        """Retorna média da smoothed_confidence das últimas N inferências.

        Útil para emitir uma "decisão final" após coletar evidência
        suficiente do stream.
        """
        if not self._history:
            return None
        recent = list(self._history)[-last_n:]
        return float(np.mean([r.smoothed_confidence for r in recent]))

    def reset(self) -> None:
        """Reinicia o estado (útil quando muda o speaker/áudio)."""
        self._buffer.clear()
        self._stream_start_time = time.time()
        self._total_samples_processed = 0
        self._last_inference_sample = 0
        self._ema_confidence = None
        self._history.clear()

    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do detector (para monitoring)."""
        n_inferences = len(self._history)
        fake_count = sum(1 for r in self._history if r.is_fake)
        return {
            'model_name': self.model_name,
            'samples_processed': self._total_samples_processed,
            'duration_processed_s': self._total_samples_processed / self.config.sample_rate,
            'n_inferences': n_inferences,
            'fake_inferences': fake_count,
            'fake_ratio': fake_count / n_inferences if n_inferences > 0 else 0.0,
            'current_ema_confidence': self._ema_confidence,
            'aggregate_score_last_5': self.get_aggregate_score(5),
        }
