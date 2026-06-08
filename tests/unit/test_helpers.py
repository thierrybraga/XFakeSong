"""Cobertura das funções utilitárias de `app/core/utils/helpers.py`.

Complementa `test_core_utils.py` (que cobre safe_filename, get_file_hash,
format_file_size e format_duration) exercitando o restante: I/O JSON,
decorators, merge/flatten/chunk, normalização de array, validação de áudio,
IDs únicos e o ProgressTracker.
"""

import logging

import numpy as np
import pytest

import app.core.utils.helpers as helpers
from app.core.utils.helpers import (
    ProgressTracker,
    chunk_list,
    create_unique_id,
    deep_merge_dicts,
    ensure_directory,
    flatten_dict,
    format_duration,
    format_file_size,
    get_file_size,
    get_timestamp,
    load_json,
    normalize_array,
    parse_timestamp,
    retry_decorator,
    save_json,
    timing_decorator,
    validate_audio_file,
)


# ── filesystem ────────────────────────────────────────────────────────────

def test_ensure_directory_creates_and_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    out = ensure_directory(target)
    assert out.is_dir()
    # chamar de novo não falha
    assert ensure_directory(target) == target


def test_get_file_size(tmp_path):
    f = tmp_path / "s.bin"
    f.write_bytes(b"12345")
    assert get_file_size(f) == 5


def test_format_file_size_bytes_and_tb():
    assert format_file_size(512) == "512.0 B"
    assert format_file_size(1024 ** 4) == "1.0 TB"


def test_format_duration_minutes_and_hours():
    assert format_duration(90).endswith("m")
    assert format_duration(7200).endswith("h")


# ── timestamps ────────────────────────────────────────────────────────────

def test_timestamp_aware_roundtrip():
    aware = get_timestamp(timezone_aware=True)
    assert "T" in aware
    assert parse_timestamp(aware).tzinfo is not None


def test_timestamp_naive_roundtrip():
    naive = get_timestamp(timezone_aware=False)
    assert parse_timestamp(naive).tzinfo is None


# ── decorators ────────────────────────────────────────────────────────────

def test_timing_decorator_returns_and_reraises():
    @timing_decorator
    def ok():
        return 42

    @timing_decorator
    def boom():
        raise ValueError("x")

    assert ok() == 42
    with pytest.raises(ValueError):
        boom()


def test_retry_decorator_succeeds_after_failures(monkeypatch):
    monkeypatch.setattr(helpers.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    @retry_decorator(max_attempts=3, delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_decorator_exhausts_and_raises_last(monkeypatch):
    monkeypatch.setattr(helpers.time, "sleep", lambda *_: None)

    @retry_decorator(max_attempts=2, delay=0.01)
    def always():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        always()


# ── JSON ──────────────────────────────────────────────────────────────────

def test_save_and_load_json_roundtrip_creates_parent(tmp_path):
    p = tmp_path / "sub" / "data.json"
    assert save_json({"a": 1, "b": [1, 2]}, p) is True
    assert load_json(p) == {"a": 1, "b": [1, 2]}


def test_load_json_missing_with_default():
    assert load_json("/nonexistent/xyz.json", default={"d": 1}) == {"d": 1}


def test_load_json_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_json("/nonexistent/xyz.json")


def test_load_json_invalid_with_default(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert load_json(bad, default=[]) == []


# ── dict helpers ──────────────────────────────────────────────────────────

def test_deep_merge_dicts_nested_without_mutation():
    a = {"x": {"a": 1, "b": 2}, "y": 1}
    b = {"x": {"b": 3, "c": 4}, "z": 9}
    out = deep_merge_dicts(a, b)
    assert out == {"x": {"a": 1, "b": 3, "c": 4}, "y": 1, "z": 9}
    # entrada original preservada
    assert a == {"x": {"a": 1, "b": 2}, "y": 1}


def test_flatten_dict_default_and_custom_sep():
    d = {"a": {"b": {"c": 1}}, "x": 2}
    assert flatten_dict(d) == {"a.b.c": 1, "x": 2}
    assert flatten_dict(d, sep="/") == {"a/b/c": 1, "x": 2}


def test_chunk_list_even_uneven_empty():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 3) == []


# ── normalize_array ───────────────────────────────────────────────────────

def test_normalize_array_minmax():
    out = normalize_array(np.array([0.0, 1, 2, 3, 4]), "minmax")
    assert np.allclose(out, [0, 0.25, 0.5, 0.75, 1.0])


def test_normalize_array_zscore_zero_mean():
    out = normalize_array(np.array([1.0, 2, 3]), "zscore")
    assert np.isclose(out.mean(), 0.0)


def test_normalize_array_robust_shape():
    out = normalize_array(np.array([1.0, 2, 3, 100]), "robust")
    assert out.shape == (4,)


@pytest.mark.parametrize("method", ["minmax", "zscore", "robust"])
def test_normalize_array_zero_variance_returns_zeros(method):
    out = normalize_array(np.array([5.0, 5, 5]), method)
    assert np.all(out == 0)


def test_normalize_array_unknown_method_raises():
    with pytest.raises(ValueError):
        normalize_array(np.array([1.0, 2]), "bogus")


# ── validação de áudio ────────────────────────────────────────────────────

def test_validate_audio_file_cases(tmp_path):
    good = tmp_path / "a.wav"
    good.write_bytes(b"RIFF....data")
    assert validate_audio_file(good) is True

    bad_ext = tmp_path / "a.txt"
    bad_ext.write_bytes(b"x")
    assert validate_audio_file(bad_ext) is False

    empty = tmp_path / "e.wav"
    empty.write_bytes(b"")
    assert validate_audio_file(empty) is False

    assert validate_audio_file(tmp_path / "missing.wav") is False


# ── IDs únicos ────────────────────────────────────────────────────────────

def test_create_unique_id_prefix_length_uniqueness():
    plain = create_unique_id()
    assert len(plain) == 8

    prefixed = create_unique_id(prefix="job", length=6)
    assert prefixed.startswith("job_")
    assert len(prefixed.split("_", 1)[1]) == 6

    assert create_unique_id() != create_unique_id()


# ── ProgressTracker ───────────────────────────────────────────────────────

def test_progress_tracker_update_clamp_and_str():
    pt = ProgressTracker(total=4, description="treino")
    pt.update()
    pt.update(2)
    info = pt.get_progress()
    assert info["current"] == 3
    assert 70 < info["percentage"] < 80
    assert info["description"] == "treino"

    pt.update(10)  # não ultrapassa o total
    assert pt.get_progress()["current"] == 4
    assert "treino" in str(pt)


def test_progress_tracker_zero_total_no_div_by_zero():
    pt = ProgressTracker(total=0)
    assert pt.get_progress()["percentage"] == 0
