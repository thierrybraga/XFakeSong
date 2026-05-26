"""Quantization-Aware Training (Sprint 5.2).

Implementa QAT via TensorFlow Model Optimization Toolkit (tfmot). QAT insere
fake-quantize ops durante o treinamento, fazendo o modelo aprender a operar
em INT8 sem perder muita acurácia.

Comparação com Post-Training Quantization (PTQ — Sprint 3.4):

| Método           | Tamanho | Speedup CPU | Accuracy preservada |
|------------------|---------|-------------|---------------------|
| FP32 baseline    | 100%    | 1×          | 100%                |
| PTQ INT8         | 25%     | 2-3×        | ~85-95%             |
| QAT INT8         | 25%     | 2-3×        | ~95-99% ✓ MELHOR    |

Dependência opcional (degradação graciosa se não instalada):
    pip install tensorflow-model-optimization

Uso típico:
    from app.domain.models.training.quantization_aware import (
        apply_qat, convert_qat_to_tflite_int8
    )

    qat_model = apply_qat(trained_model)
    qat_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    tflite_int8 = convert_qat_to_tflite_int8(qat_model, 'model_qat_int8.tflite')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


def is_tfmot_available() -> bool:
    """Verifica se tensorflow_model_optimization está instalado."""
    try:
        import tensorflow_model_optimization  # noqa: F401
        return True
    except ImportError:
        return False


def apply_qat(
    model: Any,
    layers_to_quantize: Optional[List[str]] = None,
    quantize_all: bool = True,
) -> Optional[Any]:
    """Aplica Quantization-Aware Training a um modelo Keras.

    Args:
        model: tf.keras.Model treinado em FP32
        layers_to_quantize: lista de nomes de layers para quantizar.
            Se None e quantize_all=True, quantiza o modelo inteiro.
        quantize_all: se True, usa `quantize_model` (quantize todas layers compatíveis)

    Returns:
        Modelo com fake-quantize ops, pronto para fine-tuning. Ou None se falhar.
    """
    if not is_tfmot_available():
        logger.warning(
            "tensorflow_model_optimization não instalado. Instale com: "
            "pip install tensorflow-model-optimization"
        )
        return None

    try:
        import tensorflow_model_optimization as tfmot

        quantize_model_fn = tfmot.quantization.keras.quantize_model
        quantize_apply = tfmot.quantization.keras.quantize_apply
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer

        if quantize_all or layers_to_quantize is None:
            qat_model = quantize_model_fn(model)
            logger.info(
                f"QAT aplicado a TODO o modelo: {qat_model.count_params()} params "
                f"(incluindo fake-quantize ops)"
            )
        else:
            # Quantize seletivo: anota só as layers especificadas
            def _annotate_layer(layer):
                if layer.name in layers_to_quantize:
                    return quantize_annotate_layer(layer)
                return layer

            import tensorflow as tf
            annotated = tf.keras.models.clone_model(
                model, clone_function=_annotate_layer
            )
            qat_model = quantize_apply(annotated)
            logger.info(
                f"QAT aplicado seletivamente: {len(layers_to_quantize)} layers"
            )
        return qat_model

    except Exception as e:
        logger.error(f"Falha ao aplicar QAT: {e}", exc_info=True)
        return None


def fine_tune_qat(
    qat_model: Any,
    X_train,
    y_train,
    validation_data: Optional[tuple] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
) -> Optional[Any]:
    """Fine-tuning do modelo QAT (deve usar LR menor que treino original).

    QAT precisa de algumas épocas de fine-tuning para calibrar os pontos
    de quantização. LR baixo evita destruir os pesos pré-treinados.

    Args:
        qat_model: output de `apply_qat()`
        X_train, y_train: dados de treino
        validation_data: opcional (X_val, y_val)
        epochs: typically 5-20 épocas suficiente
        batch_size: tamanho do batch
        learning_rate: 10× menor que o treino original (default 1e-5)

    Returns:
        Modelo QAT fine-tuned, ou None em caso de falha.
    """
    if qat_model is None:
        return None

    try:
        import tensorflow as tf

        # Recompila com LR menor (preserva loss/metrics se já compilado)
        if qat_model.optimizer is None:
            logger.warning(
                "qat_model não está compilado; compilando com AdamW + BCE default"
            )
            qat_model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
        else:
            # Atualiza LR do optimizer existente
            try:
                qat_model.optimizer.learning_rate.assign(learning_rate)
            except (AttributeError, ValueError):
                pass  # alguns optimizers/schedules não suportam assign direto

        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        if validation_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True,
            ))

        qat_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(f"QAT fine-tuning concluído ({epochs} épocas)")
        return qat_model

    except Exception as e:
        logger.error(f"Falha em fine_tune_qat: {e}", exc_info=True)
        return None


def convert_qat_to_tflite_int8(
    qat_model: Any,
    output_path: Union[str, Path],
    representative_dataset=None,
) -> Optional[Path]:
    """Converte modelo QAT para TFLite INT8 (deploy mobile/edge).

    QAT models são treinados com fake-quantize, então a conversão preserva
    bem a accuracy. Para deploy em CPU/desktop, prefira ONNX (Sprint 3.4).

    Args:
        qat_model: modelo QAT fine-tuned
        output_path: caminho do .tflite de saída
        representative_dataset: generator de dados para calibração final
            (não obrigatório em QAT, mas melhora ranges de quantização)

    Returns:
        Path do .tflite gerado, ou None em caso de falha.
    """
    try:
        import tensorflow as tf

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"TFLite INT8 (QAT) salvo: {output_path} ({size_mb:.2f} MB)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Falha ao converter QAT → TFLite: {e}", exc_info=True)
        return None


def create_representative_dataset(
    X: Any,
    n_samples: int = 100,
    batch_size: int = 1,
):
    """Helper para criar `representative_dataset` callable do TFLite.

    Args:
        X: array (N, *input_shape) com dados de treino/val
        n_samples: amostras para usar na calibração (100 é suficiente)
        batch_size: tamanho do batch (1 para máxima granularidade)

    Returns:
        Callable generator compatível com `converter.representative_dataset`.
    """
    import numpy as np
    X_arr = np.asarray(X, dtype=np.float32)
    n = min(n_samples, len(X_arr))

    def _gen():
        for i in range(0, n, batch_size):
            batch = X_arr[i:i + batch_size]
            yield [batch.astype(np.float32)]

    return _gen
