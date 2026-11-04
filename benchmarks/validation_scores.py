"""Partição de validação para recalibrar e escolher fusão sem treinar.

Lê só `X_val`, `y_val` e a fatia de validação de `speaker_ids`: `X_test` nunca sai do arquivo.
A ordem é a do benchmark: correção de banda, depois ruído, depois o front-end do modelo.

As janelas do NPZ têm o tamanho da entrada dos modelos (48.000 amostras). Os três recortes da
avaliação coincidem, e uma passagem basta para reproduzir o score do run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.domain.features.benchmark_frontend import apply_band_correction
from benchmarks.data import BenchmarkData, prepare_input_for_architecture

# Fora dos intervalos usados no treino (+10000) e na avaliação de teste (+20000).
SEMENTE_RUIDO_VALIDACAO = 90000


@dataclass
class ValidationSet:
    waveforms: np.ndarray
    labels: np.ndarray
    speakers: Optional[np.ndarray] = None
    _corrigidas: Dict[Optional[float], np.ndarray] = field(default_factory=dict)

    def corrected(self, cutoff_hz: Optional[float]) -> np.ndarray:
        """Formas de onda com a política de banda do contrato, calculada uma vez."""
        if cutoff_hz not in self._corrigidas:
            self._corrigidas[cutoff_hz] = (
                apply_band_correction(self.waveforms, float(cutoff_hz))
                if cutoff_hz
                else self.waveforms
            )
        return self._corrigidas[cutoff_hz]


def load_validation(dataset_path: Path | str) -> ValidationSet:
    """Carrega a validação de um NPZ canônico, sem abrir a partição de teste."""
    with np.load(dataset_path, allow_pickle=False) as npz:
        waveforms = np.asarray(npz["X_val"], dtype="float32")
        labels = np.asarray(npz["y_val"]).astype(int).ravel()
        speakers = None
        if "speaker_ids" in npz.files:
            inicio = len(npz["y_train"])
            fatia = np.asarray(npz["speaker_ids"]).astype(str).ravel()
            if len(fatia) >= inicio + len(labels):
                speakers = fatia[inicio : inicio + len(labels)]
    if waveforms.ndim == 3 and waveforms.shape[-1] == 1:
        waveforms = waveforms[..., 0]
    return ValidationSet(waveforms, labels, speakers)


def prepared_input(
    validation: ValidationSet, info: Any, snr_db: Optional[float] = None
) -> np.ndarray:
    """Entrada do modelo: banda do contrato, ruído opcional e o front-end da arquitetura."""
    contrato = info.input_contract if isinstance(info.input_contract, dict) else {}
    banda = contrato.get("band_correction") or {}
    ondas = validation.corrected(banda.get("cutoff_hz"))
    if snr_db is not None:
        ondas = BenchmarkData.add_awgn(
            ondas, float(snr_db), seed=SEMENTE_RUIDO_VALIDACAO + int(snr_db)
        )
    return prepare_input_for_architecture(ondas, info.architecture)[0]
