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
    # P0 — fonte/gerador por amostra (alinhado a X/y). Usado para splits
    # disjuntos por grupo e para o protocolo cross-generator. None quando o
    # dataset não carrega proveniência.
    groups: np.ndarray | None = None

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
        used_keys: list[tuple[str, str]] = []
        for xk, yk in (("X_train", "y_train"), ("X_val", "y_val"),
                       ("X_test", "y_test"), ("X", "y")):
            if xk in data and yk in data:
                xs.append(np.asarray(data[xk], dtype="float32"))
                ys.append(np.asarray(data[yk]))
                used_keys.append((xk, yk))
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

        # P0 — proveniência por amostra (fonte/gerador). Preferimos uma chave
        # explícita `groups` no .npz; senão derivamos do nome de arquivo nos
        # `paths` do metadata (mesma ordem de concatenação: train→val→test→X).
        groups = cls._extract_groups(data, metadata, used_keys, n=len(y))
        loaded = cls(X=X, y=y, name=p.stem, metadata=metadata, groups=groups)
        loaded.validate()
        return loaded

    @staticmethod
    def _derive_group(path: str) -> str:
        """Extrai o identificador de fonte/gerador do nome do arquivo.

        Os nomes seguem `<fonte>_NNNNN.wav` (ex.: brspeech, cvpt, fkvoice). Não
        há ID de falante embutido, então o grupo mais fino disponível é a
        FONTE/GERADOR — suficiente para o reteste cross-generator (XTTS=fkvoice).
        """
        base = str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()
        m = re.match(r"([a-z]+)", base)
        return m.group(1) if m else "unknown"

    @classmethod
    def _extract_groups(
        cls,
        data: Any,
        metadata: Dict[str, Any],
        used_keys: list[tuple[str, str]],
        n: int,
    ) -> np.ndarray | None:
        """Constrói o array de grupos alinhado a X/y, ou None se indisponível."""
        if "groups" in getattr(data, "files", []):
            g = np.asarray(data["groups"]).astype(str).ravel()
            return g if len(g) == n else None

        splits = (metadata or {}).get("splits") or {}
        key_to_split = {"X_train": "train", "X_val": "val", "X_test": "test"}
        groups: list[str] = []
        for xk, _yk in used_keys:
            split_name = key_to_split.get(xk)
            paths = (splits.get(split_name) or {}).get("paths") if split_name else None
            if not paths:
                return None  # sem proveniência completa → não arrisca desalinhar
            groups.extend(cls._derive_group(p) for p in paths)
        if len(groups) != n:
            return None
        return np.asarray(groups, dtype=object).astype(str)

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
            groups=self.groups,
        )

    def stratified_split(
        self,
        seed: int = 42,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        group_split: bool = False,
        holdout_generator: str | None = None,
    ):
        """Divisão 70/15/15. Suporta três modos:

        - **estratificada** (default): preserva a proporção de classes.
        - **group_split**: mantém fonte/gerador DISJUNTO entre train/val/test
          (anti-vazamento), quando `groups` está disponível.
        - **holdout_generator**: protocolo cross-generator — segura um gerador
          fora do treino e o usa (só ele) como teste.
        """
        if holdout_generator is not None and self.groups is not None:
            return self._cross_generator_split(
                holdout_generator, seed, val_frac
            )
        if group_split and self.groups is not None:
            return self._grouped_split(seed, val_frac, test_frac)
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

    def _select(self, idx: np.ndarray):
        idx = np.asarray(idx, dtype=int)
        return self.X[idx], self.y[idx]

    def _grouped_split(self, seed: int, val_frac: float, test_frac: float):
        """Split disjunto por grupo via StratifiedGroupKFold (anti-vazamento).

        Mantém cada fonte/gerador inteiramente em um único conjunto. Como há
        poucos grupos correlacionados à classe, isto pode desbalancear classes
        — é o trade-off honesto para eliminar vazamento de fonte.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        groups = np.asarray(self.groups)
        idx = np.arange(len(self.y))
        n_groups = len(np.unique(groups))
        # nº de folds limitado pelo nº de grupos; teste = 1 fold.
        n_splits = max(2, min(round(1.0 / max(test_frac, 1e-6)), n_groups))
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        trainval_idx, test_idx = next(sgkf.split(idx, self.y, groups))
        # Val a partir do trainval, ainda disjunto por grupo quando possível.
        g_tv = groups[trainval_idx]
        if len(np.unique(g_tv)) >= 2:
            rel_val = val_frac / (1.0 - test_frac)
            inner_splits = max(2, min(round(1.0 / max(rel_val, 1e-6)),
                                      len(np.unique(g_tv))))
            sgkf2 = StratifiedGroupKFold(
                n_splits=inner_splits, shuffle=True, random_state=seed
            )
            tr_rel, val_rel = next(
                sgkf2.split(trainval_idx, self.y[trainval_idx], g_tv)
            )
            train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[val_rel]
        else:
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(trainval_idx)
            n_val = max(1, int(len(shuffled) * val_frac))
            val_idx, train_idx = shuffled[:n_val], shuffled[n_val:]
        Xtr, ytr = self._select(train_idx)
        Xv, yv = self._select(val_idx)
        Xte, yte = self._select(test_idx)
        return Xtr, ytr, Xv, yv, Xte, yte

    def _cross_generator_split(
        self, holdout_generator: str, seed: int, val_frac: float
    ):
        """Protocolo cross-generator: treina SEM `holdout_generator`, testa NELE.

        Teste = todas as amostras do gerador segurado + reais não vistos no
        treino (para manter ambas as classes no teste). Train/val saem do
        restante, estratificados por classe.
        """
        from sklearn.model_selection import train_test_split

        groups = np.asarray(self.groups)
        held = np.char.lower(groups.astype(str)) == holdout_generator.lower()
        if not held.any():
            # gerador inexistente → cai no split estratificado padrão
            return self.stratified_split(seed=seed, val_frac=val_frac)

        idx = np.arange(len(self.y))
        held_idx = idx[held]
        rest_idx = idx[~held]
        y_rest = self.y[rest_idx]

        # Reais ficam disponíveis no restante; reservamos uma fração de reais
        # para compor o teste cross-generator (classe 0 inédita no treino).
        real_rest = rest_idx[y_rest == 0]
        rng = np.random.default_rng(seed)
        real_rest = rng.permutation(real_rest)
        n_real_test = min(len(real_rest), max(1, len(held_idx)))
        real_test_idx = real_rest[:n_real_test]
        test_idx = np.concatenate([held_idx, real_test_idx])

        trainval_idx = np.setdiff1d(rest_idx, real_test_idx, assume_unique=False)
        y_tv = self.y[trainval_idx]
        try:
            tr_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_frac, stratify=y_tv,
                random_state=seed,
            )
        except Exception:
            shuffled = rng.permutation(trainval_idx)
            n_val = max(1, int(len(shuffled) * val_frac))
            val_idx, tr_idx = shuffled[:n_val], shuffled[n_val:]

        Xtr, ytr = self._select(tr_idx)
        Xv, yv = self._select(val_idx)
        Xte, yte = self._select(test_idx)
        return Xtr, ytr, Xv, yv, Xte, yte

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
        start = max(0, (flat.shape[1] - target_len) // 2)
        return flat[:, start : start + target_len]
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

    # P2 — RASTA-PLP: características robustas a ruído/canal para os modelos
    # clássicos (SVM/RF), que colapsavam a ~50% sob ruído. O filtro RASTA
    # remove variações lentas (efeitos de canal), tornando o vetor de features
    # mais estável sob degradação. Reusa o extrator do domínio.
    feats.append(_rasta_plp_stats(flat))

    return np.vstack(feats).T.astype("float32")


def _rasta_plp_stats(flat: np.ndarray, n_plp: int = 13) -> np.ndarray:
    """Estatísticas (média/desvio por coeficiente) de RASTA-PLP por amostra.

    Retorna shape (2*n_plp, N) para empilhar com as demais features. Degrada
    para zeros se a extração falhar (mantém o vetor de features consistente).
    """
    try:
        from app.domain.features.extractors.cepstral.components.plp import (
            extract_rasta_plp_features,
        )
    except Exception:
        return np.zeros((2 * n_plp, len(flat)), dtype="float32")

    rows = []
    for y in flat:
        try:
            feats = extract_rasta_plp_features(
                y.astype("float32"), sr=16000, frame_length=512,
                hop_length=256, n_plp=n_plp,
            )
            rp = np.asarray(feats.get("rasta_plp"))
            if rp.ndim != 2 or rp.shape[0] != n_plp:
                raise ValueError("forma RASTA-PLP inesperada")
            rows.append(np.concatenate([rp.mean(axis=1), rp.std(axis=1)]))
        except Exception:
            rows.append(np.zeros(2 * n_plp, dtype="float32"))
    arr = np.nan_to_num(np.asarray(rows, dtype="float32"), nan=0.0,
                        posinf=0.0, neginf=0.0)
    return arr.T


def _to_raw_audio(X: np.ndarray, requirements: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype="float32")
    flat = arr.reshape(len(arr), -1)
    min_len = int(
        requirements.get("min_sequence_length")
        or requirements.get("sample_rate", 16000)
    )
    # `min_sequence_length` is a lower bound. Architectures may request an
    # explicit `target_sequence_length` when full clips are too memory-heavy.
    # Without an explicit target, preserve long raw clips instead of truncating
    # them to the minimum.
    target_len = int(
        requirements.get("sequence_length")
        or requirements.get("target_sequence_length")
        or max(flat.shape[1], min_len)
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
