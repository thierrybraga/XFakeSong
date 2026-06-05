"""Dataset do benchmark: carga, splits estratificados e ruído AWGN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class BenchmarkData:
    """Conjunto (X, y) homogêneo usado pelo benchmark.

    X tem forma (N, *input_shape) — espectrograma (T, F) para modelos neurais,
    achatado para (N, T*F) pelos modelos clássicos (SVM/RF) no runner.
    y é 1D com rótulos {0=real/bonafide, 1=fake/spoof}.
    """

    X: np.ndarray
    y: np.ndarray
    name: str = "dataset"

    @classmethod
    def synthetic(cls, n: int = 360, shape: Tuple[int, int] = (32, 16),
                  seed: int = 42) -> "BenchmarkData":
        """Dataset sintético separável (verificação do harness, sem áudio).

        A classe fake recebe um deslocamento de média → linearmente separável
        em poucas épocas. NÃO substitui dados reais; serve para testar o pipeline.
        """
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n,) + tuple(shape)).astype("float32")
        y = rng.integers(0, 2, n).astype("int64")
        X[y == 1] += 0.6
        return cls(X=X, y=y, name="synthetic")

    @classmethod
    def from_npz(cls, path: str) -> "BenchmarkData":
        """Carrega de um .npz. Concatena X_train/X_val/X_test se presentes e
        re-divide de forma estratificada (test set controlado e reprodutível).
        """
        data = np.load(path, allow_pickle=False)
        xs, ys = [], []
        for xk, yk in (("X_train", "y_train"), ("X_val", "y_val"),
                       ("X_test", "y_test"), ("X", "y")):
            if xk in data and yk in data:
                xs.append(np.asarray(data[xk], dtype="float32"))
                ys.append(np.asarray(data[yk]))
        if not xs:
            raise ValueError(
                f"{path}: esperado X_train/y_train (ou X/y) no .npz; "
                f"chaves encontradas: {list(data.keys())}"
            )
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        # Normaliza rótulos para {0,1} (1D, inteiro)
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[-1] > 1:
            y = np.argmax(y, axis=-1)
        y = y.ravel().astype("int64")
        return cls(X=X, y=y, name=Path(path).stem)

    def stratified_split(
        self, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15
    ):
        """Divisão estratificada 70/15/15 (preserva a proporção de classes)."""
        try:
            from sklearn.model_selection import train_test_split

            idx = np.arange(len(self.y))
            train_idx, temp_idx = train_test_split(
                idx, test_size=val_frac + test_frac,
                stratify=self.y, random_state=seed,
            )
            rel_test = test_frac / (val_frac + test_frac)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=rel_test,
                stratify=self.y[temp_idx], random_state=seed,
            )
        except Exception:
            # Fallback sem estratificação (datasets minúsculos)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(self.y))
            n_test = max(1, int(len(idx) * test_frac))
            n_val = max(1, int(len(idx) * val_frac))
            test_idx, val_idx, train_idx = (
                idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
            )
        return (
            self.X[train_idx], self.y[train_idx],
            self.X[val_idx], self.y[val_idx],
            self.X[test_idx], self.y[test_idx],
        )

    @staticmethod
    def add_awgn(X: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
        """Adiciona ruído gaussiano branco aditivo a um SNR alvo (por amostra).

        Para modelos raw-audio (X = forma de onda) isto é AWGN no áudio; para
        modelos de espectrograma (X = log-mel/LFCC) é uma aproximação do AWGN
        em espaço de entrada — escolha deliberada para um teste uniforme e
        reprodutível em todas as arquiteturas (documentado no relatório).
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype="float32")
        flat = X.reshape(len(X), -1)
        sig_power = np.mean(flat ** 2, axis=1, keepdims=True)  # (N,1)
        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        noise_std = np.sqrt(sig_power / max(snr_lin, 1e-12))
        noise = rng.standard_normal(flat.shape).astype("float32") * noise_std
        return (flat + noise).reshape(X.shape).astype("float32")
