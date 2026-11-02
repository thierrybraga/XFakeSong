"""Auditorias das partições: sobreposição binária, impressões digitais,
locutor/fonte entre partições e o atalho fonte→rótulo. Saiu de ``runner.py``
(P2-15); o runner SSL usa as mesmas funções.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np

from benchmarks.data import BenchmarkData


def _audit_split_overlap(raw_splits, fail_on_overlap: bool = True) -> Dict[str, Any]:
    """Detecta amostras binariamente idênticas entre treino, validação e teste."""

    names = ("train", "val", "test")
    arrays = (raw_splits[0], raw_splits[2], raw_splits[4])
    labels = (raw_splits[1], raw_splits[3], raw_splits[5])
    fingerprints: Dict[str, set[str]] = {}
    split_fingerprints: Dict[str, Any] = {
        "method": "sha256_ordered_blake2b_sample_hashes_and_labels"
    }
    for name, values, y in zip(names, arrays, labels):
        current: set[str] = set()
        ordered = hashlib.sha256()
        y_arr = np.ascontiguousarray(np.asarray(y, dtype="int64"))
        ordered.update(str(np.asarray(values).shape).encode("ascii"))
        ordered.update(y_arr.view(np.uint8))
        for sample in np.asarray(values):
            contiguous = np.ascontiguousarray(sample)
            digest_bytes = hashlib.blake2b(
                contiguous.view(np.uint8), digest_size=16
            ).digest()
            current.add(digest_bytes.hex())
            ordered.update(digest_bytes)
        fingerprints[name] = current
        split_fingerprints[name] = {
            "n": int(len(y_arr)),
            "sha256": ordered.hexdigest(),
        }

    pair_counts = {
        "train_val": len(fingerprints["train"] & fingerprints["val"]),
        "train_test": len(fingerprints["train"] & fingerprints["test"]),
        "val_test": len(fingerprints["val"] & fingerprints["test"]),
    }
    total = sum(pair_counts.values())
    audit = {
        "method": "blake2b-128_exact_array_bytes",
        "pairwise_overlap_counts": pair_counts,
        "split_fingerprints": split_fingerprints,
        "passed": total == 0,
    }
    if total and fail_on_overlap:
        raise ValueError(
            "Contaminação entre partições: amostras idênticas detectadas "
            f"{pair_counts}. Corrija o NPZ antes do benchmark."
        )
    return audit


def _split_fingerprint(raw_splits) -> Dict[str, Any]:
    """SHA-256 determinístico da identidade e ordem de cada partição."""

    result: Dict[str, Any] = {"method": "sha256_ordered_array_bytes_and_labels"}
    for name, X, y in (
        ("train", raw_splits[0], raw_splits[1]),
        ("val", raw_splits[2], raw_splits[3]),
        ("test", raw_splits[4], raw_splits[5]),
    ):
        digest = hashlib.sha256()
        y_arr = np.ascontiguousarray(np.asarray(y, dtype="int64"))
        digest.update(str(np.asarray(X).shape).encode("ascii"))
        digest.update(y_arr.view(np.uint8))
        for sample in np.asarray(X):
            contiguous = np.ascontiguousarray(sample)
            digest.update(contiguous.view(np.uint8))
        result[name] = {"n": int(len(y_arr)), "sha256": digest.hexdigest()}
    return result


def _audit_split_provenance(data: BenchmarkData) -> Dict[str, Any]:
    """Relata sobreposição de fonte/locutor nas partições efetivamente usadas."""

    indices = data.last_split_indices or {}
    required = {"train", "val", "test"}
    if not required.issubset(indices):
        return {"available": False, "reason": "no_effective_split_indices"}
    result: Dict[str, Any] = {"available": True}
    for field, values in (("groups", data.groups), ("speakers", data.speakers)):
        if values is None:
            result[field] = {"available": False}
            continue
        sets = {
            split: set(np.asarray(values)[indices[split]].astype(str).tolist())
            for split in ("train", "val", "test")
        }
        intersections = {
            "train_val": sorted(sets["train"] & sets["val"]),
            "train_test": sorted(sets["train"] & sets["test"]),
            "val_test": sorted(sets["val"] & sets["test"]),
        }
        result[field] = {
            "available": True,
            "unique_counts": {key: len(value) for key, value in sets.items()},
            "overlap_counts": {key: len(value) for key, value in intersections.items()},
            "train_test_examples": intersections["train_test"][:10],
            "disjoint": all(not value for value in intersections.values()),
        }
    return result


def _audit_source_label_shortcut(
    data: BenchmarkData,
    *,
    threshold: float = 0.55,
    fail: bool = False,
) -> Dict[str, Any]:
    """Measure how accurately a source-majority oracle predicts the label."""
    if data.groups is None:
        audit = {
            "available": False,
            "passed": False,
            "reason": "source_ids_missing",
        }
        if fail:
            raise ValueError("Auditoria fonte-rotulo exige source_ids explicitos")
        return audit

    groups = np.asarray(data.groups).astype(str)
    labels = np.asarray(data.y).astype(int)
    counts: Dict[str, Dict[str, int]] = {}
    correct = 0
    for source in sorted(set(groups.tolist())):
        source_labels = labels[groups == source]
        real_n = int(np.sum(source_labels == 0))
        fake_n = int(np.sum(source_labels == 1))
        counts[source] = {"real": real_n, "fake": fake_n}
        correct += max(real_n, fake_n)
    accuracy = correct / max(len(labels), 1)
    audit = {
        "available": True,
        "method": "source_majority_oracle",
        "accuracy": accuracy,
        "threshold": float(threshold),
        "counts": counts,
        "passed": bool(accuracy <= threshold),
    }
    if fail and not audit["passed"]:
        raise ValueError(
            f"Atalho fonte-rotulo: oraculo={accuracy:.4f} > limite={threshold:.4f}"
        )
    return audit
