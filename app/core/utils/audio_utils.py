import numpy as np
import logging
import librosa

logger = logging.getLogger(__name__)

def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to a target dB level.
    
    Args:
        audio: Input audio array
        target_db: Target level in dB
        
    Returns:
        Normalized audio array
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
        
    scalar = 10 ** (target_db / 20) / (rms + 1e-9)
    return audio * scalar

def pad_or_truncate(audio: np.ndarray, max_len: int) -> np.ndarray:
    """
    Pad or truncate audio to a specific length.
    
    Args:
        audio: Input audio array
        max_len: Maximum length in samples
        
    Returns:
        Processed audio array
    """
    if len(audio) > max_len:
        return audio[:max_len]
    else:
        return np.pad(audio, (0, max_len - len(audio)), mode='constant')

def preprocess_legacy(x):
    """DEPRECATED: Use secure preprocessing pipeline instead."""
    logger.warning(
        "Global preprocess function is DEPRECATED and may cause data leakage. "
        "Use the secure training pipeline for preprocessing."
    )
    # Minimal processing to avoid breaking existing models
    return x
