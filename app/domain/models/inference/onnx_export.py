"""ONNX Export + INT8 Quantization (Sprint 3.4).

Permite exportar modelos Keras para ONNX (Open Neural Network Exchange) e
opcionalmente quantizar para INT8. Útil para deploy em produção:

- **Menor footprint**: INT8 reduz tamanho em ~4× (32-bit float → 8-bit int)
- **Mais rápido em CPU**: 2–3× speedup usando ONNX Runtime
- **Cross-platform**: ONNX roda em CPU, GPU, mobile (CoreML, NNAPI), edge devices
- **Sem dependência de TensorFlow** no deploy

Dependências (opcionais — degradação graciosa se não instaladas):
    pip install tf2onnx onnxruntime          # FP32
    pip install onnxruntime-extensions       # quantization tools

Uso típico:
    from app.domain.models.inference.onnx_export import export_to_onnx, quantize_int8

    # Export FP32
    onnx_path = export_to_onnx(keras_model, 'model.onnx', input_shape=(128, 80))

    # Quantize INT8
    int8_path = quantize_int8(onnx_path, 'model_int8.onnx')

    # Inferência ONNX
    from app.domain.models.inference.onnx_export import OnnxInferenceSession
    session = OnnxInferenceSession('model_int8.onnx')
    predictions = session.predict(features)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _check_tf2onnx() -> bool:
    """Verifica se tf2onnx está instalado."""
    try:
        import tf2onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _check_onnxruntime() -> bool:
    """Verifica se onnxruntime está instalado."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def export_to_onnx(
    keras_model: Any,
    output_path: Union[str, Path],
    input_shape: Optional[Tuple[int, ...]] = None,
    opset: int = 13,
    dynamic_batch: bool = True,
) -> Optional[Path]:
    """Exporta modelo Keras para ONNX.

    Args:
        keras_model: modelo `tf.keras.Model` treinado
        output_path: caminho do arquivo .onnx de saída
        input_shape: shape de entrada SEM batch dim (ex: (128, 80) para
            espectrograma 128×80). Se None, infere de `keras_model.input_shape`.
        opset: versão do ONNX opset (13 = compatível com onnxruntime 1.7+)
        dynamic_batch: se True, batch dimension é dinâmica (None)

    Returns:
        Path do arquivo gerado, ou None em caso de falha.
    """
    if not _check_tf2onnx():
        logger.warning(
            "tf2onnx não instalado. Pule export ONNX ou instale com: "
            "pip install tf2onnx"
        )
        return None

    try:
        import tensorflow as tf
        import tf2onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determinar shape do input signature
        if input_shape is None:
            inferred = keras_model.input_shape
            # Remove batch dim
            input_shape = tuple(d for d in inferred[1:])

        batch_dim = None if dynamic_batch else 1
        full_shape = (batch_dim,) + tuple(input_shape)

        # Cria input signature
        input_signature = [
            tf.TensorSpec(full_shape, tf.float32, name="input")
        ]

        model_proto, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=input_signature,
            opset=opset,
            output_path=str(output_path),
        )

        logger.info(
            f"Modelo exportado para ONNX: {output_path} "
            f"(opset={opset}, shape={full_shape})"
        )
        return output_path

    except Exception as e:
        logger.error(f"Falha ao exportar ONNX: {e}", exc_info=True)
        return None


def quantize_int8(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    calibration_data: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Quantiza modelo ONNX para INT8.

    Suporta dois modos:
    1. **Dynamic quantization** (default, sem `calibration_data`):
       Quantiza pesos para INT8 dinamicamente. Mais rápido, menos preciso.
    2. **Static quantization** (com `calibration_data`):
       Usa amostras representativas para calibrar pontos de quantização.
       Mais preciso, mas requer dados de calibração.

    Args:
        onnx_path: caminho do .onnx FP32 de entrada
        output_path: caminho do .onnx INT8 de saída
        calibration_data: ndarray (N, *input_shape) para calibração estática.
            Se None, usa quantização dinâmica.

    Returns:
        Path do arquivo INT8, ou None em caso de falha.
    """
    if not _check_onnxruntime():
        logger.warning(
            "onnxruntime não instalado. Instale com: pip install onnxruntime"
        )
        return None

    try:
        from onnxruntime.quantization import (
            QuantType,
            quantize_dynamic,
        )

        onnx_path = Path(onnx_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if calibration_data is None:
            # Quantização dinâmica (sem dados de calibração)
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )
            logger.info(
                f"Modelo quantizado para INT8 (dynamic): {output_path}"
            )
        else:
            # Quantização estática (precisa de calibration_data)
            try:
                from onnxruntime.quantization import (
                    CalibrationDataReader,
                    QuantFormat,
                    quantize_static,
                )

                class _NpyCalibReader(CalibrationDataReader):
                    def __init__(self, data: np.ndarray, batch_size: int = 1):
                        self.data = data
                        self.batch_size = batch_size
                        self.idx = 0

                    def get_next(self):
                        if self.idx >= len(self.data):
                            return None
                        batch = self.data[self.idx:self.idx + self.batch_size]
                        self.idx += self.batch_size
                        return {"input": batch.astype(np.float32)}

                quantize_static(
                    model_input=str(onnx_path),
                    model_output=str(output_path),
                    calibration_data_reader=_NpyCalibReader(calibration_data),
                    quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                )
                logger.info(
                    f"Modelo quantizado para INT8 (static, "
                    f"{len(calibration_data)} samples): {output_path}"
                )
            except ImportError:
                logger.warning("Static quantization indisponível, usando dynamic")
                quantize_dynamic(
                    model_input=str(onnx_path),
                    model_output=str(output_path),
                    weight_type=QuantType.QInt8,
                )

        # Compara tamanhos
        size_orig = onnx_path.stat().st_size / (1024 * 1024)
        size_int8 = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Tamanho: {size_orig:.2f} MB → {size_int8:.2f} MB "
            f"(reduzido {(1 - size_int8/size_orig) * 100:.1f}%)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Falha ao quantizar ONNX para INT8: {e}", exc_info=True)
        return None


class OnnxInferenceSession:
    """Wrapper de sessão ONNX Runtime para inferência.

    Provê API similar ao `model.predict()` do Keras, facilitando substituição
    em deploys onde TensorFlow não está disponível.

    Uso:
        session = OnnxInferenceSession('model_int8.onnx')
        predictions = session.predict(features)  # (N, K)
        session.close()  # ou usar como context manager
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: Optional[list] = None,
    ):
        """
        Args:
            onnx_path: caminho do .onnx
            providers: providers ONNX Runtime na ordem de preferência.
                Default: ['CUDAExecutionProvider', 'CPUExecutionProvider']
                (CUDA tentado primeiro, fallback CPU).
        """
        if not _check_onnxruntime():
            raise ImportError(
                "onnxruntime não instalado. pip install onnxruntime "
                "(ou onnxruntime-gpu para CUDA)"
            )

        import onnxruntime as ort

        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in (
                'CUDAExecutionProvider', 'CPUExecutionProvider'
            ) if p in available]

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            f"ONNX session iniciada: {onnx_path} | providers={providers} | "
            f"input_shape={self.input_shape}"
        )

    def predict(
        self,
        features: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Inferência em batch.

        Args:
            features: (N, *input_shape) ou (*input_shape,) para single sample
            batch_size: tamanho do batch para inferência

        Returns:
            np.ndarray (N, *output_shape) com predições
        """
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == len(self.input_shape) - 1:
            features = features[np.newaxis, ...]

        outputs = []
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            out = self.session.run([self.output_name], {self.input_name: batch})[0]
            outputs.append(out)

        return np.concatenate(outputs, axis=0) if outputs else np.array([])

    def close(self):
        """Libera recursos da sessão (no-op em ORT atual, mas explícito)."""
        self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def is_onnx_available() -> bool:
    """Retorna True se tf2onnx + onnxruntime estão disponíveis."""
    return _check_tf2onnx() and _check_onnxruntime()
