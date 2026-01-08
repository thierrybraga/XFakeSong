import numpy as np
from typing import Optional, Dict
from .components.approximate import (
    compute_approximate_entropy,
    compute_approximate_entropy_optimized
)
from .components.sample import (
    compute_sample_entropy,
    compute_sample_entropy_optimized
)
from .components.permutation import (
    compute_permutation_entropy,
    compute_permutation_entropy_optimized
)
from .components.multiscale import (
    compute_multiscale_entropy,
    compute_multiscale_entropy_optimized
)

# Re-exporting functions for backward compatibility
__all__ = [
    'compute_approximate_entropy',
    'compute_approximate_entropy_optimized',
    'compute_sample_entropy',
    'compute_sample_entropy_optimized',
    'compute_permutation_entropy',
    'compute_permutation_entropy_optimized',
    'compute_multiscale_entropy',
    'compute_multiscale_entropy_optimized'
]
