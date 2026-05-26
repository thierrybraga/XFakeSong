"""Stochastic Weight Averaging (SWA) callback para Keras 3.

Implementa SWA (Izmailov et al., "Averaging Weights Leads to Wider Optima
and Better Generalization", UAI 2018) como Keras Callback.

SWA mantém uma média móvel dos pesos do modelo nas últimas N épocas e
aplica essa média no final do treinamento. Pesquisa empírica mostra
ganhos de +0.5–1.5% accuracy em diversas tarefas, incluindo deepfake
detection, sem custo adicional de inferência.

Uso típico:
    swa = SWACallback(start_epoch=80, swa_freq=1)
    model.fit(..., callbacks=[swa, ...])
    # Ao final do treino, swa.apply() é chamado automaticamente em on_train_end()
    # e os pesos do modelo são substituídos pela média SWA.

Importante:
    - SWA não funciona bem com BatchNorm sem recompute (precisamos rodar 1 forward
      pass nos dados de treino após aplicar pesos médios para atualizar estatísticas).
    - Esta implementação faz `apply_swa()` automaticamente em `on_train_end` e,
      se um `bn_update_data` for fornecido, atualiza as BNs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import tensorflow as tf


class SWACallback(tf.keras.callbacks.Callback):
    """Callback de Stochastic Weight Averaging.

    Args:
        start_epoch: Época a partir da qual começa a acumular pesos
                     (0-indexed; default: usa 80% das épocas como ponto de partida).
        swa_freq: Frequência de coleta de pesos (1 = toda época após start_epoch).
        bn_update_data: Dataset opcional (tf.data.Dataset) ou tupla (X, y)
                        para recomputar estatísticas de BatchNorm após SWA.
        verbose: 1 = log de progresso, 0 = silencioso.
    """

    def __init__(
        self,
        start_epoch: int = -1,
        swa_freq: int = 1,
        bn_update_data=None,
        verbose: int = 1,
    ):
        super().__init__()
        self.start_epoch = start_epoch
        self.swa_freq = max(1, int(swa_freq))
        self.bn_update_data = bn_update_data
        self.verbose = verbose

        self._swa_weights: Optional[List[np.ndarray]] = None
        self._n_collected: int = 0
        self._logger = logging.getLogger(__name__)
        self._total_epochs: Optional[int] = None

    def _resolve_start_epoch(self, total_epochs: Optional[int]) -> int:
        """Resolve start_epoch=-1 para 80% do total de épocas."""
        if self.start_epoch >= 0:
            return self.start_epoch
        if total_epochs is None or total_epochs <= 0:
            return 0
        # Default: começa nas últimas ~20% das épocas
        return max(0, int(total_epochs * 0.8))

    def on_train_begin(self, logs=None):
        # Tenta extrair total de épocas dos params
        try:
            self._total_epochs = int(self.params.get('epochs', 0)) if self.params else None
        except Exception:
            self._total_epochs = None
        resolved = self._resolve_start_epoch(self._total_epochs)
        if self.verbose:
            self._logger.info(
                f"SWA habilitado: começa coleta na época {resolved} "
                f"(total={self._total_epochs}), freq={self.swa_freq}"
            )

    def on_epoch_end(self, epoch, logs=None):
        start = self._resolve_start_epoch(self._total_epochs)
        if epoch < start:
            return
        # Coleta apenas em intervalos de swa_freq
        if (epoch - start) % self.swa_freq != 0:
            return

        current_weights = self.model.get_weights()
        if self._swa_weights is None:
            # Primeira coleta: inicializa com cópias
            self._swa_weights = [np.array(w, copy=True) for w in current_weights]
            self._n_collected = 1
        else:
            # Atualiza média móvel: avg = avg + (w - avg) / (n + 1)
            self._n_collected += 1
            for i, w in enumerate(current_weights):
                self._swa_weights[i] += (w - self._swa_weights[i]) / self._n_collected

        if self.verbose:
            self._logger.info(
                f"SWA: peso coletado #{self._n_collected} na época {epoch}"
            )

    def apply_swa(self) -> bool:
        """Aplica os pesos SWA acumulados ao modelo.

        Returns:
            True se foi aplicado com sucesso, False se nenhuma coleta ocorreu.
        """
        if self._swa_weights is None or self._n_collected == 0:
            if self.verbose:
                self._logger.warning(
                    "SWA: nenhum peso foi coletado; mantendo pesos originais"
                )
            return False

        try:
            self.model.set_weights(self._swa_weights)
            if self.verbose:
                self._logger.info(
                    f"SWA: pesos médios aplicados ({self._n_collected} épocas)"
                )

            # Recomputa estatísticas de BatchNorm se possível
            if self.bn_update_data is not None:
                self._update_bn_stats()

            return True
        except Exception as e:
            self._logger.error(f"SWA: falha ao aplicar pesos médios: {e}")
            return False

    def _update_bn_stats(self):
        """Atualiza estatísticas de BatchNorm com 1 pass nos dados.

        Necessário porque BN guarda mean/var das batches durante o treino;
        após SWA, esses valores estão dessincronizados com os pesos médios.
        """
        try:
            data = self.bn_update_data
            # Cria modelo temporário com training=True para atualizar BN
            # Faz isso via prediction loop manual com training=True flag

            # Heurística simples: 1 forward pass com training=True
            if isinstance(data, tuple) and len(data) == 2:
                X, _ = data
                # Pega só os primeiros 1000 samples para velocidade
                X_sub = X[:1000] if len(X) > 1000 else X
                # batch_size compatível
                batch_size = 32
                for i in range(0, len(X_sub), batch_size):
                    batch = X_sub[i:i + batch_size]
                    self.model(batch, training=True)
            elif hasattr(data, 'take'):
                # tf.data.Dataset
                for batch in data.take(30):  # ~30 batches
                    if isinstance(batch, tuple):
                        x = batch[0]
                    else:
                        x = batch
                    self.model(x, training=True)

            if self.verbose:
                self._logger.info("SWA: estatísticas de BatchNorm atualizadas")
        except Exception as e:
            self._logger.warning(f"SWA: falha ao atualizar BN stats: {e}")

    def on_train_end(self, logs=None):
        # Aplica SWA automaticamente ao fim do treino
        self.apply_swa()
