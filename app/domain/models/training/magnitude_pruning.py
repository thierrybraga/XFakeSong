"""Magnitude-based weight pruning (P3 — MultiscaleCNN).

Implementa pruning por magnitude via TensorFlow Model Optimization Toolkit
(tfmot). O MultiscaleCNN é forte (~99,7%) porém grande (~188 MB); pruning poda
os pesos de menor magnitude (zera-os) com retomada de fine-tuning, reduzindo
o tamanho efetivo (com compressão) e o custo de inferência com perda mínima.

Fluxo típico:

| Etapa                    | Tamanho | Accuracy preservada |
|--------------------------|---------|---------------------|
| FP32 denso (baseline)    | 100%    | 100%                |
| Pruning 50% + fine-tune  | ~50% NZ | ~98-99% ✓           |
| Pruning 80% + fine-tune  | ~20% NZ | ~95-98%             |

(NZ = pesos não-zero; o ganho em disco vem da compressão do .tflite/.zip,
pois zeros comprimem muito.)

Dependência opcional (degradação graciosa se ausente):
    pip install tensorflow-model-optimization

Uso:
    from app.domain.models.training.magnitude_pruning import (
        apply_pruning, strip_pruning, prune_and_finetune
    )
    pruned = prune_and_finetune(model, X_train, y_train,
                                validation_data=(X_val, y_val),
                                target_sparsity=0.5, epochs=5)
    final = strip_pruning(pruned)  # remove wrappers → modelo denso podado
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def is_tfmot_available() -> bool:
    """Verifica se tensorflow_model_optimization está instalado."""
    try:
        import tensorflow_model_optimization  # noqa: F401
        return True
    except ImportError:
        return False


def apply_pruning(
    model: Any,
    target_sparsity: float = 0.5,
    begin_step: int = 0,
    end_step: int = 1000,
    initial_sparsity: float = 0.0,
) -> Optional[Any]:
    """Embrulha o modelo com pruning por magnitude (PolynomialDecay).

    Args:
        model: tf.keras.Model treinado (denso).
        target_sparsity: fração final de pesos zerados (0.5 = 50%).
        begin_step / end_step: janela (em passos de otimização) do ramp de
            esparsidade. Dimensione end_step ≈ (n_amostras/batch)*épocas.
        initial_sparsity: esparsidade no início do ramp.

    Returns:
        Modelo pronto para fine-tuning de pruning, ou None se indisponível.
        IMPORTANTE: o fine-tuning exige o callback UpdatePruningStep
        (ver `pruning_callbacks`).
    """
    if not (0.0 < target_sparsity < 1.0):
        raise ValueError("target_sparsity deve estar em (0, 1)")
    if not is_tfmot_available():
        logger.warning(
            "tensorflow_model_optimization não instalado. Instale com: "
            "pip install tensorflow-model-optimization"
        )
        return None

    try:
        import tensorflow_model_optimization as tfmot

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=target_sparsity,
            begin_step=begin_step,
            end_step=end_step,
        )
        pruned = prune_low_magnitude(model, pruning_schedule=schedule)
        logger.info(
            "Pruning aplicado: sparsity alvo=%.0f%%, ramp steps=[%d, %d]",
            target_sparsity * 100, begin_step, end_step,
        )
        return pruned
    except Exception as e:
        logger.error(f"Falha ao aplicar pruning: {e}", exc_info=True)
        return None


def pruning_callbacks(log_dir: Optional[Union[str, Path]] = None) -> list:
    """Callbacks obrigatórios para o fine-tuning de pruning.

    UpdatePruningStep é OBRIGATÓRIO — sem ele o cronograma de esparsidade não
    avança e o pruning não acontece. Retorna lista vazia se tfmot ausente.
    """
    if not is_tfmot_available():
        return []
    import tensorflow_model_optimization as tfmot

    cbs = [tfmot.sparsity.keras.UpdatePruningStep()]
    if log_dir is not None:
        cbs.append(
            tfmot.sparsity.keras.PruningSummaries(log_dir=str(log_dir))
        )
    return cbs


def strip_pruning(pruned_model: Any) -> Optional[Any]:
    """Remove os wrappers de pruning, devolvendo um modelo denso já podado.

    Deve ser chamado APÓS o fine-tuning, antes de salvar/exportar — os zeros
    permanecem, mas sem o overhead das máscaras de pruning.
    """
    if not is_tfmot_available():
        return pruned_model
    try:
        import tensorflow_model_optimization as tfmot

        return tfmot.sparsity.keras.strip_pruning(pruned_model)
    except Exception as e:
        logger.error(f"Falha ao remover wrappers de pruning: {e}")
        return pruned_model


def compute_sparsity(model: Any) -> float:
    """Fração global de pesos exatamente zero (mede o efeito do pruning)."""
    import numpy as np

    total, zeros = 0, 0
    for w in model.get_weights():
        arr = np.asarray(w)
        total += arr.size
        zeros += int((arr == 0).sum())
    return (zeros / total) if total else 0.0


def prune_and_finetune(
    model: Any,
    X_train: Any,
    y_train: Any,
    validation_data: Optional[Tuple[Any, Any]] = None,
    target_sparsity: float = 0.5,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Optional[Any]:
    """Aplica pruning + fine-tuning e devolve o modelo denso podado (stripped).

    Recompila com um LR baixo (o modelo já está treinado; só recupera a perda
    induzida pela poda). Retorna None se tfmot indisponível ou em falha.
    """
    if not is_tfmot_available():
        logger.warning("Pruning pulado: tensorflow_model_optimization ausente.")
        return None

    try:
        import numpy as np
        import tensorflow as tf

        n = len(np.asarray(y_train))
        steps_per_epoch = max(1, n // max(1, batch_size))
        end_step = steps_per_epoch * max(1, epochs)

        pruned = apply_pruning(
            model, target_sparsity=target_sparsity, end_step=end_step
        )
        if pruned is None:
            return None

        # Preserva a loss original; recompila só para o fine-tuning de pruning.
        loss = getattr(model, "loss", None) or "sparse_categorical_crossentropy"
        pruned.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )
        pruned.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=pruning_callbacks(),
            verbose=0,
        )
        final = strip_pruning(pruned)
        logger.info(
            "Pruning concluído: sparsity efetiva=%.1f%%",
            compute_sparsity(final) * 100,
        )
        return final
    except Exception as e:
        logger.error(f"Falha em prune_and_finetune: {e}", exc_info=True)
        return None
