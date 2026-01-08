import numpy as np
from typing import Optional, Dict, List, Tuple
from .components.correlation_dimension import (
    compute_correlation_dimension,
    compute_correlation_dimension_optimized
)
from .components.lyapunov import (
    compute_lyapunov_exponent,
    compute_lyapunov_exponent_optimized
)
from .components.rqa import (
    compute_rqa_features,
    compute_rqa_features_optimized,
    _find_diagonal_lines,
    _find_diagonal_lines_optimized,
    _find_vertical_lines,
    _find_vertical_lines_optimized
)

# Re-exporting functions for backward compatibility
__all__ = [
    'compute_correlation_dimension',
    'compute_correlation_dimension_optimized',
    'compute_lyapunov_exponent',
    'compute_lyapunov_exponent_optimized',
    'compute_rqa_features',
    'compute_rqa_features_optimized',
    '_find_diagonal_lines',
    '_find_diagonal_lines_optimized',
    '_find_vertical_lines',
    '_find_vertical_lines_optimized'
]
