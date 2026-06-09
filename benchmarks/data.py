"""Dataset do benchmark: carga, splits estratificados e ruído AWGN."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

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
    metadata: Dict[str, Any] | None = None

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
        return cls(X=X, y=y, name="synthetic", metadata={"source": "synthetic"})

    @classmethod
    def from_npz(cls, path: str) -> "BenchmarkData":
        """Carrega de um .npz. Concatena X_train/X_val/X_test se presentes e
        re-divide de forma estratificada (test set controlado e reprodutível).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {path}")
        data = np.load(p, allow_pickle=False)
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
        metadata: Dict[str, Any] = {"source": str(p)}
        if "metadata_json" in data:
            try:
                raw_meta = data["metadata_json"]
                if hasattr(raw_meta, "item"):
                    raw_meta = raw_meta.item()
                metadata.update(json.loads(str(raw_meta)))
                metadata["npz_path"] = str(p)
            except Exception:
                metadata["metadata_parse_error"] = True
        loaded = cls(X=X, y=y, name=p.stem, metadata=metadata)
        loaded.validate()
        return loaded

    def validate(self, min_per_class: int = 2) -> None:
        """Valida sanidade básica antes de treinar/avaliar."""
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        if X.ndim < 2:
            raise ValueError(f"{self.name}: X deve ter batch + features, shape={X.shape}")
        if len(X) != len(y):
            raise ValueError(f"{self.name}: len(X)={len(X)} difere de len(y)={len(y)}")
        if not np.isfinite(X).all():
            raise ValueError(f"{self.name}: X contém NaN ou Inf")
        if not np.isfinite(y).all():
            raise ValueError(f"{self.name}: y contém NaN ou Inf")
        labels, counts = np.unique(y.ravel().astype("int64"), return_counts=True)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"{self.name}: labels esperados {{0,1}}, encontrados {labels.tolist()}")
        too_small = {int(k): int(v) for k, v in zip(labels, counts) if v < min_per_class}
        if too_small:
            raise ValueError(
                f"{self.name}: classes com amostras insuficientes para split: {too_small}"
            )

    def prepare_for_architecture(self, architecture: str) -> "BenchmarkData":
        """Retorna uma visão de X compatível com o contrato da arquitetura.

        O benchmark aceita datasets `.npz` homogêneos, mas o preset completo
        mistura arquiteturas raw, espectrograma e clássicas. Esta etapa adapta
        a forma do tensor para cada família mantendo `y` e a ordem das amostras.
        """
        input_type, requirements = _architecture_input_contract(architecture)
        if _is_classical_arch(architecture):
            prepared = _to_tabular_features(self.X)
            actual_type = (
                "tabular_audio_features"
                if _looks_like_raw_audio(self.X)
                else "tabular_flattened"
            )
        elif input_type == "raw_audio":
            prepared = _to_raw_audio(self.X, requirements)
            actual_type = "raw_audio"
        elif input_type == "spectrogram":
            prepared = _to_spectrogram(self.X, requirements)
            actual_type = "spectrogram"
        else:
            prepared = np.asarray(self.X, dtype="float32")
            actual_type = input_type or "unchanged"

        meta = dict(self.metadata or {})
        meta.update(
            {
                "architecture": architecture,
                "input_type": actual_type,
                "original_shape": list(np.asarray(self.X).shape[1:]),
                "prepared_shape": list(np.asarray(prepared).shape[1:]),
            }
        )
        return BenchmarkData(
            X=np.asarray(prepared, dtype="float32"),
            y=np.asarray(self.y, dtype="int64"),
            name=self.name,
            metadata=meta,
        )

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


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_classical_arch(name: str) -> bool:
    return _slug(name) in {"svm", "randomforest"}


def _architecture_input_contract(architecture: str) -> tuple[str, Dict[str, Any]]:
    if _is_classical_arch(architecture):
        return "tabular", {}
    try:
        from app.domain.models.architectures.factory import (
            architecture_factory_registry,
        )
        from app.domain.models.architectures.registry import normalize_arch_name

        normalized = normalize_arch_name(architecture)
        spec = architecture_factory_registry.get_architecture_info(normalized)
        if spec:
            req = dict(spec.input_requirements or {})
            return str(req.get("input_type", "spectrogram")), req
    except Exception:
        pass
    return "spectrogram", {}


def _fit_length(flat: np.ndarray, target_len: int) -> np.ndarray:
    if flat.shape[1] == target_len:
        return flat
    if flat.shape[1] > target_len:
        return flat[:, :target_len]
    repeats = int(np.ceil(target_len / max(1, flat.shape[1])))
    return np.tile(flat, (1, repeats))[:, :target_len]


def _resize_axis(X: np.ndarray, target: int, axis: int) -> np.ndarray:
    current = X.shape[axis]
    if current == target:
        return X
    if current > target:
        slices = [slice(None)] * X.ndim
        slices[axis] = slice(0, target)
        return X[tuple(slices)]
    pad_width = [(0, 0)] * X.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(X, pad_width, mode="edge")


def _normalize_per_sample(X: np.ndarray) -> np.ndarray:
    flat = X.reshape(len(X), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    return ((flat - mean) / np.maximum(std, 1e-6)).reshape(X.shape).astype("float32")


def _looks_like_raw_audio(X: np.ndarray) -> bool:
    arr = np.asarray(X)
    if arr.ndim == 2:
        sample_len = arr.shape[1]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        sample_len = arr.shape[1]
    else:
        return False
    return sample_len >= 1000


def _audio_flat(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype="float32").reshape(len(X), -1)


def _to_tabular_features(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    if not _looks_like_raw_audio(arr):
        return arr

    flat = _audio_flat(arr)
    feats = [
        flat.mean(axis=1),
        flat.std(axis=1),
        np.mean(np.abs(flat), axis=1),
        np.sqrt(np.mean(flat ** 2, axis=1)),
        flat.min(axis=1),
        flat.max(axis=1),
        np.percentile(flat, 25, axis=1),
        np.percentile(flat, 50, axis=1),
        np.percentile(flat, 75, axis=1),
        np.mean(np.diff(flat, axis=1) ** 2, axis=1),
        np.mean(np.signbit(flat[:, 1:]) != np.signbit(flat[:, :-1]), axis=1),
    ]
    try:
        import librosa

        mfcc_stats = []
        for y in flat:
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
            mfcc_stats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
        feats.append(np.asarray(mfcc_stats, dtype="float32").T)
    except Exception:
        pass

    return np.vstack(feats).T.astype("float32")


def _to_raw_audio(X: np.ndarray, requirements: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    flat = arr.reshape(len(arr), -1)
    target_len = int(
        requirements.get("min_sequence_length")
        or requirements.get("sample_rate", 16000)
    )
    raw = _fit_length(flat, max(1, target_len))
    raw = _normalize_per_sample(raw)
    return raw[..., np.newaxis]


def _raw_audio_to_logmel(X: np.ndarray, requirements: Dict[str, Any]) -> np.ndarray:
    import librosa

    flat = _audio_flat(X)
    sample_rate = int(requirements.get("sample_rate") or 16000)
    feature_dim = int(requirements.get("feature_dim") or 80)
    time_steps = int(requirements.get("min_sequence_length") or 100)
    hop_length = max(64, int(np.ceil(flat.shape[1] / max(time_steps, 1))))

    specs = []
    for y in flat:
        mel = librosa.feature.melspectrogram(
            y=y.astype("float32"),
            sr=sample_rate,
            n_fft=512,
            hop_length=hop_length,
            n_mels=feature_dim,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max).T
        mel_db = _resize_axis(mel_db[np.newaxis, ...], max(1, time_steps), axis=1)[0]
        specs.append(mel_db[:, :feature_dim])
    return _normalize_per_sample(np.asarray(specs, dtype="float32"))


def _to_spectrogram(X: np.ndarray, requirements: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    if _looks_like_raw_audio(arr):
        return _raw_audio_to_logmel(arr, requirements)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        # Vetores tabulares viram grade T x F por repetição/truncamento.
        feature_dim = int(requirements.get("feature_dim") or 80)
        time_steps = int(requirements.get("min_sequence_length") or 100)
        flat = _fit_length(arr.reshape(len(arr), -1), time_steps * feature_dim)
        arr = flat.reshape(len(arr), time_steps, feature_dim)
    elif arr.ndim == 3:
        # Piso defensivo: se o spec não declara o alvo, garante uma grade grande
        # o bastante para sobreviver às camadas de pooling (evita "Negative
        # dimension" em arquiteturas profundas com entradas sintéticas pequenas).
        time_steps = int(requirements.get("min_sequence_length") or max(arr.shape[1], 64))
        feature_dim = int(requirements.get("feature_dim") or max(arr.shape[2], 64))
        arr = _resize_axis(arr, max(1, time_steps), axis=1)
        arr = _resize_axis(arr, max(1, feature_dim), axis=2)
    else:
        flat = arr.reshape(len(arr), -1)
        feature_dim = int(requirements.get("feature_dim") or 80)
        time_steps = int(requirements.get("min_sequence_length") or 100)
        flat = _fit_length(flat, time_steps * feature_dim)
        arr = flat.reshape(len(arr), time_steps, feature_dim)
    return _normalize_per_sample(arr)
