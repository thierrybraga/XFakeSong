"""Regressões dos controles metodológicos do benchmark acadêmico."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.data import BenchmarkData
from benchmarks.runner import (
    _audit_split_overlap,
    _audit_split_provenance,
    _split_fingerprint,
)
from scripts.benchmark.run_models_sequential import (
    _inspect_npz,
    _sha256_file,
    _validate_test_lock,
)
from scripts.reporting.consolidate_results import _confusion_matrix_for_row


def _write_predefined_npz(path) -> None:
    rng = np.random.default_rng(7)
    arrays = {
        "X_train": rng.normal(size=(8, 160, 1)).astype("float32"),
        "y_train": np.array([0, 1] * 4),
        "X_val": rng.normal(size=(4, 160, 1)).astype("float32"),
        "y_val": np.array([0, 1] * 2),
        "X_test": rng.normal(size=(4, 160, 1)).astype("float32"),
        "y_test": np.array([0, 1] * 2),
        "groups": np.array(["train"] * 8 + ["val"] * 4 + ["test"] * 4),
        "speaker_ids": np.array([f"speaker-{idx}" for idx in range(16)]),
    }
    np.savez_compressed(path, **arrays)


def test_predefined_test_is_frozen_across_training_seeds(tmp_path) -> None:
    dataset = tmp_path / "fixture.npz"
    _write_predefined_npz(dataset)
    data = BenchmarkData.from_npz(str(dataset))

    first = data.stratified_split(seed=42, preserve_predefined=True)
    first_hash = _split_fingerprint(first)["test"]["sha256"]
    second = data.stratified_split(seed=999, preserve_predefined=True)
    second_hash = _split_fingerprint(second)["test"]["sha256"]

    assert first_hash == second_hash
    assert data.last_split_indices is not None
    assert _audit_split_overlap(second)["passed"] is True


def test_provenance_audit_uses_effective_split(tmp_path) -> None:
    dataset = tmp_path / "fixture.npz"
    _write_predefined_npz(dataset)
    data = BenchmarkData.from_npz(str(dataset))
    data.stratified_split(seed=42, preserve_predefined=True)

    audit = _audit_split_provenance(data)

    assert audit["available"] is True
    assert audit["groups"]["disjoint"] is True


def test_npz_preflight_requires_and_counts_predefined_splits(tmp_path) -> None:
    dataset = tmp_path / "fixture.npz"
    _write_predefined_npz(dataset)

    inspection = _inspect_npz(dataset)

    assert inspection["predefined_splits"] is True
    assert inspection["sample_count"] == 16
    assert inspection["test_archive_identity_sha256"]


def test_academic_lock_validates_dataset_and_test_identity(tmp_path) -> None:
    import json

    dataset = tmp_path / "fixture.npz"
    lock = tmp_path / "fixture.npz.test-lock.json"
    _write_predefined_npz(dataset)
    inspection = _inspect_npz(dataset)
    lock.write_text(
        json.dumps(
            {
                "protocol_version": "xfakesong-test-lock-v1",
                "dataset_size_bytes": dataset.stat().st_size,
                "dataset_sha256": _sha256_file(dataset),
                "test_archive_identity_sha256": inspection[
                    "test_archive_identity_sha256"
                ],
                "declared_untouched": True,
                "created_before_training": True,
            }
        ),
        encoding="utf-8",
    )

    validated = _validate_test_lock(dataset, inspection, lock)

    assert validated["validated"] is True


def test_academic_lock_is_mandatory(tmp_path) -> None:
    dataset = tmp_path / "fixture.npz"
    _write_predefined_npz(dataset)

    with pytest.raises(ValueError, match="selo do teste não encontrado"):
        _validate_test_lock(
            dataset,
            _inspect_npz(dataset),
            tmp_path / "missing.test-lock.json",
        )

def test_confusion_matrix_uses_official_decision_threshold() -> None:
    row = {
        "slug": "model",
        "decision_threshold": 0.5,
        "robustness": {"clean": {"eer_threshold": 0.95}},
    }
    extras = {"model": {"scores_clean": [0.4, 0.6], "y_test": [0, 1]}}

    matrix = _confusion_matrix_for_row(row, extras)

    assert matrix.tolist() == [[1, 0], [0, 1]]


def test_exact_train_test_duplicate_aborts() -> None:
    sample = np.ones((1, 16, 1), dtype="float32")
    labels = np.array([0])
    splits = (sample, labels, np.zeros_like(sample), labels, sample.copy(), labels)

    with pytest.raises(ValueError, match="Contaminação"):
        _audit_split_overlap(splits, fail_on_overlap=True)