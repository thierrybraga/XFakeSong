import numpy as np
from typing import Optional, Dict
from .components.box_counting import (
    compute_fractal_dimension,
    compute_fractal_dimension_optimized
)
from .components.hurst import (
    compute_hurst_exponent,
    compute_hurst_exponent_optimized
)
from .components.higuchi import (
    compute_higuchi_fractal,
    compute_higuchi_fractal_optimized
)
from .components.dfa import (
    compute_dfa_exponent,
    compute_dfa_exponent_optimized
)

# Re-exporting functions for backward compatibility
__all__ = [
    'compute_fractal_dimension',
    'compute_fractal_dimension_optimized',
    'compute_hurst_exponent',
    'compute_hurst_exponent_optimized',
    'compute_higuchi_fractal',
    'compute_higuchi_fractal_optimized',
    'compute_dfa_exponent',
    'compute_dfa_exponent_optimized'
]
