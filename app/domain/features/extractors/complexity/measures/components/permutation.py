import numpy as np
from scipy.stats import entropy
from typing import Optional, Dict


def compute_permutation_entropy(
        y: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Calcula a entropia de permutação."""
    N = len(y)
    permutations = {}

    for i in range(N - delay * (order - 1)):
        # Extrair padrão
        pattern = []
        for j in range(order):
            pattern.append(y[i + j * delay])

        # Obter permutação
        sorted_indices = sorted(range(len(pattern)), key=lambda k: pattern[k])
        perm = tuple(sorted_indices)

        if perm in permutations:
            permutations[perm] += 1
        else:
            permutations[perm] = 1

    # Calcular entropia
    total = sum(permutations.values())
    probs = [count / total for count in permutations.values()]

    return entropy(probs, base=2)


def compute_permutation_entropy_optimized(
        y: np.ndarray, order: int = 3, delay: int = 1,
        cache: Optional[Dict] = None) -> float:
    """Versão otimizada da entropia de permutação."""
    N = len(y)

    # Limitar tamanho
    if N > 5000:
        step = N // 5000
        y = y[::step][:5000]
        N = len(y)

    if N < order * delay:
        return np.nan

    # Cache key
    cache_key = f"permen_{hash(y.tobytes())}_{order}_{delay}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Implementação vetorizada
    patterns = np.zeros((N - delay * (order - 1), order))
    for j in range(order):
        patterns[:, j] = y[j * delay:N - delay * (order - 1) + j * delay]

    # Obter permutações usando argsort
    sorted_indices = np.argsort(patterns, axis=1)

    # Converter para tuplas para contagem
    permutations = {}
    for perm in sorted_indices:
        perm_tuple = tuple(perm)
        permutations[perm_tuple] = permutations.get(perm_tuple, 0) + 1

    # Calcular entropia
    total = len(sorted_indices)
    probs = np.array(list(permutations.values())) / total
    result = entropy(probs, base=2)

    # Cache resultado
    if cache is not None:
        cache[cache_key] = result

    return result
