"""Regressoes do protocolo academico de dataset v2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.domain.dataset_metadata import speaker_manifest
from benchmarks.data import BenchmarkData
from benchmarks.evaluate import evaluate_grouped_scores, evaluate_scores
from benchmarks.runner import _audit_source_label_shortcut
from scripts.benchmark.run_tcc_pipeline import _load_wav
from scripts.dataset import build_dataset


def _touch_wavs(directory: Path, prefix: str, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{prefix}_{index:05d}.wav").write_bytes(b"RIFF")


def test_balance_refuses_to_shrink_target(tmp_path, monkeypatch) -> None:
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    _touch_wavs(real_dir, "brspeech", 9)
    _touch_wavs(fake_dir, "brspeech", 10)
    monkeypatch.setattr(build_dataset, "REAL_DIR", real_dir)
    monkeypatch.setattr(build_dataset, "FAKE_DIR", fake_dir)

    with pytest.raises(RuntimeError, match="Dataset incompleto"):
        build_dataset.step_balance(target_per_class=10)


def test_source_quota_selection_is_seeded_and_exact() -> None:
    files = [Path(f"brspeech_{i:05d}.wav") for i in range(10)]
    files += [Path(f"fkvoice_{i:05d}.wav") for i in range(10)]
    quotas = {"brspeech": 4, "fkvoice": 4}

    removed_a = build_dataset._excess_round_robin(
        files.copy(), 8, seed=7, quotas=quotas
    )
    removed_b = build_dataset._excess_round_robin(
        files.copy(), 8, seed=7, quotas=quotas
    )
    kept = set(files) - set(removed_a)

    assert removed_a == removed_b
    assert sum(path.stem.startswith("brspeech") for path in kept) == 4
    assert sum(path.stem.startswith("fkvoice") for path in kept) == 4


def test_strict_speaker_id_never_falls_back_to_source(monkeypatch) -> None:
    monkeypatch.setattr(speaker_manifest, "load_manifest", lambda: {})
    with pytest.raises(speaker_manifest.MissingSampleMetadataError):
        speaker_manifest.speaker_for_path("brspeech_00001.wav", strict=True)


def test_short_audio_tiles_and_long_audio_center_crops(monkeypatch) -> None:
    import librosa

    monkeypatch.setattr(
        librosa,
        "load",
        lambda *_args, **_kwargs: (np.array([1.0, 2.0, 3.0]), 16_000),
    )
    short, original, start = _load_wav(Path("short.wav"), 16_000, 8)
    assert original == 3
    assert start == 0
    assert short.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]

    monkeypatch.setattr(
        librosa,
        "load",
        lambda *_args, **_kwargs: (np.arange(10, dtype="float32"), 16_000),
    )
    long, original, start = _load_wav(Path("long.wav"), 16_000, 4)
    assert original == 10
    assert start == 3
    assert long.tolist() == [3.0, 4.0, 5.0, 6.0]


def test_speaker_protocol_without_explicit_coverage_fails() -> None:
    data = BenchmarkData(
        X=np.zeros((12, 8, 1), dtype="float32"),
        y=np.array([0, 1] * 6),
        speakers=np.array([f"speaker-{i}" for i in range(12)]),
    )
    with pytest.raises(ValueError, match="speaker_known"):
        data.stratified_split(speaker_split=True, preserve_predefined=False)


def test_cluster_bootstrap_and_group_metrics_are_reported() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    clusters = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    groups = np.array(["source-a"] * 4 + ["source-b"] * 4)

    metrics = evaluate_scores(y, p, n_bootstrap=20, cluster_ids=clusters)
    grouped = evaluate_grouped_scores(y, p, groups)

    assert metrics["bootstrap_unit"] == "cluster"
    assert metrics["bootstrap_clusters"] == 4
    assert grouped["n_groups"] == 2
    assert grouped["macro_accuracy"] == pytest.approx(1.0)
    assert grouped["worst_group_accuracy"] == pytest.approx(1.0)


def test_source_majority_oracle_can_fail_academic_guard() -> None:
    data = BenchmarkData(
        X=np.zeros((8, 4, 1), dtype="float32"),
        y=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        groups=np.array(["real-only"] * 4 + ["fake-only"] * 4),
    )

    audit = _audit_source_label_shortcut(data, threshold=0.55, fail=False)
    assert audit["accuracy"] == pytest.approx(1.0)
    assert audit["passed"] is False
    with pytest.raises(ValueError, match="Atalho fonte-rotulo"):
        _audit_source_label_shortcut(data, threshold=0.55, fail=True)


def test_predefined_split_with_single_class_fails() -> None:
    data = BenchmarkData(
        X=np.zeros((8, 4, 1), dtype="float32"),
        y=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        predefined_split_indices={
            "train": np.array([0, 1]),
            "val": np.array([2, 4]),
            "test": np.array([3, 5]),
        },
    )

    with pytest.raises(ValueError, match="train.*real e fake"):
        data.stratified_split()


def test_predefined_split_overlap_fails() -> None:
    data = BenchmarkData(
        X=np.zeros((6, 4, 1), dtype="float32"),
        y=np.array([0, 1, 0, 1, 0, 1]),
        predefined_split_indices={
            "train": np.array([0, 3]),
            "val": np.array([1, 4]),
            "test": np.array([2, 3]),
        },
    )

    with pytest.raises(ValueError, match="sobrepostas"):
        data.stratified_split()


def test_manifest_keys_separate_real_and_fake_homonyms() -> None:
    real = speaker_manifest._manifest_key("splits/train/real/brspeech_00001.wav")
    fake = speaker_manifest._manifest_key("splits/test/fake/brspeech_00001.wav")
    assert real == "real/brspeech_00001.wav"
    assert fake == "fake/brspeech_00001.wav"
