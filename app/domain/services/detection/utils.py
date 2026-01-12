import numpy as np
from typing import Optional, Tuple, Union, List
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def pad_or_truncate(data: np.ndarray, target_length: int, axis: int = 0, mode: str = 'constant', constant_values: float = 0.0) -> np.ndarray:
    """
    Ajusta o tamanho do array para target_length ao longo do eixo especificado.
    Se for maior, trunca. Se for menor, preenche (padding).
    """
    current_length = data.shape[axis]
    
    if current_length == target_length:
        return data
    
    if current_length > target_length:
        # Truncar
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, target_length)
        return data[tuple(slices)]
    
    # Padding
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, target_length - current_length)
    
    return np.pad(data, pad_width, mode=mode, constant_values=constant_values)

def prepare_batch_for_model(features_list: List[np.ndarray], target_shape: Optional[Tuple] = None) -> np.ndarray:
    """
    Prepara uma lista de features para processamento em lote.
    Garante que todos tenham o mesmo shape.
    """
    if not features_list:
        return np.array([])
        
    # Se target_shape não for fornecido, usar o shape do primeiro elemento ou o maior
    if target_shape is None:
        # Assumindo que queremos alinhar pela primeira dimensão (tempo/sequência)
        max_len = max(f.shape[0] for f in features_list)
        base_shape = list(features_list[0].shape)
        base_shape[0] = max_len
        target_shape = tuple(base_shape)
        
    processed = []
    for f in features_list:
        # Ajustar primeira dimensão
        if len(target_shape) >= 1 and target_shape[0] is not None:
             f = pad_or_truncate(f, target_shape[0], axis=0)
        processed.append(f)
        
    return np.stack(processed)

def get_available_devices() -> List[str]:
    """Retorna lista de dispositivos disponíveis (CPU, GPU:0, etc)."""
    devices = ["CPU"]
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            devices.extend([f"GPU:{i}" for i in range(len(gpus))])
    except Exception as e:
        logger.warning(f"Erro ao listar dispositivos GPU: {e}")
    return devices

def set_memory_growth():
    """Configura crescimento de memória para GPUs."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        logger.warning(f"Erro ao configurar memory growth: {e}")
