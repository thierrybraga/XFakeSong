"""Regressoes do pipeline de dataset: balanceamento, janela e proveniencia.`n`nExercitam a maquinaria compartilhada (composicao, manifesto de falante,`ncarregamento no benchmark), independente do protocolo vigente.`n"""

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
from scripts.dataset.preprocess_dataset import _content_aware_class_allocation


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


def test_content_allocation_does_not_chase_irrelevant_class_deficit() -> None:
    """Regressao: um split com deficit enorme de fake mas real ja excedente
    nao deve atrair grupos de conteudo puro-real so por ter pontuacao total
    inflada pelo deficit da OUTRA classe. Reproduz o desbalanceamento real
    observado em producao (val: real=1433/fake=904, ratio 1.585 - fora da
    faixa 0.8-1.25 de docs/data/public-datasets.md), causado pela formula
    antiga somar (nr - n_real) + (nf - n_fake) mesmo quando o grupo so
    continha uma das duas classes."""
    n_real_groups = 300
    n_fake_groups = 300
    idx = np.arange(n_real_groups + n_fake_groups)
    labels = np.array([0] * n_real_groups + [1] * n_fake_groups)
    content_groups = np.array([f"g{i}" for i in idx], dtype=object)

    # val ja tem excesso de real (deficit negativo) mas falta muito fake.
    need = {"train": (50, -50), "val": (-50, 200), "test": (50, -50)}

    train_idx, val_idx, test_idx = _content_aware_class_allocation(
        idx, labels, content_groups, need, seed=42
    )

    val_real = int((labels[val_idx] == 0).sum())
    # A formula antiga despejava ~200 grupos puro-real em val mesmo com
    # deficit negativo; a correta mantem bem abaixo disso.
    assert val_real < 100, f"val recebeu {val_real} grupos reais que nao precisava"

    placed = sorted(train_idx.tolist() + val_idx.tolist() + test_idx.tolist())
    assert placed == list(idx)


def test_content_allocation_keeps_content_groups_atomic() -> None:
    """Um grupo de conteudo (texto/enunciado) com multiplas amostras nunca
    pode ser fatiado entre splits - isso constituiria vazamento de conteudo,
    exatamente o que create_splits audita via `content_leakage`."""
    rng = np.random.default_rng(3)
    group_sizes = rng.integers(1, 6, size=50)
    idx: list[int] = []
    labels_list: list[int] = []
    groups_list: list[str] = []
    cursor = 0
    for gi, size in enumerate(group_sizes):
        label = gi % 2
        for _ in range(int(size)):
            idx.append(cursor)
            labels_list.append(label)
            groups_list.append(f"g{gi}")
            cursor += 1

    idx_arr = np.array(idx)
    labels = np.array(labels_list)
    content_groups = np.array(groups_list, dtype=object)
    need = {"train": (60, 60), "val": (15, 15), "test": (15, 15)}

    train_idx, val_idx, test_idx = _content_aware_class_allocation(
        idx_arr, labels, content_groups, need, seed=42
    )

    placed = sorted(train_idx.tolist() + val_idx.tolist() + test_idx.tolist())
    assert placed == list(idx_arr)

    split_of = {}
    for name, sel in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in sel.tolist():
            split_of[i] = name

    by_group: dict[str, set[str]] = {}
    for i, key in zip(idx_arr.tolist(), content_groups.tolist()):
        by_group.setdefault(key, set()).add(split_of[i])

    violations = {g: s for g, s in by_group.items() if len(s) > 1}
    assert not violations, f"grupos fatiados entre splits: {violations}"


def test_manifest_keys_separate_real_and_fake_homonyms() -> None:
    real = speaker_manifest._manifest_key("splits/train/real/brspeech_00001.wav")
    fake = speaker_manifest._manifest_key("splits/test/fake/brspeech_00001.wav")
    assert real == "real/brspeech_00001.wav"
    assert fake == "fake/brspeech_00001.wav"
