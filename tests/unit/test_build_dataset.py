from scripts import build_dataset


def _touch_wavs(directory, prefix, count):
    directory.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (directory / f"{prefix}_{idx:05d}.wav").write_bytes(b"RIFF")


def test_step_balance_archives_excess_without_deleting(tmp_path, monkeypatch):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    overflow_dir = tmp_path / "overflow"
    _touch_wavs(real_dir, "real", 104)
    _touch_wavs(fake_dir, "fake", 103)

    monkeypatch.setattr(build_dataset, "REAL_DIR", real_dir)
    monkeypatch.setattr(build_dataset, "FAKE_DIR", fake_dir)
    monkeypatch.setattr(build_dataset, "OVERFLOW_DIR", overflow_dir)

    real_final, fake_final = build_dataset.step_balance(
        target_per_class=101,
        delete_excess=False,
    )

    assert real_final == 101
    assert fake_final == 101
    assert len(list(real_dir.glob("*.wav"))) == 101
    assert len(list(fake_dir.glob("*.wav"))) == 101
    assert len(list((overflow_dir / "real").glob("*.wav"))) == 3
    assert len(list((overflow_dir / "fake").glob("*.wav"))) == 2


def test_public_real_source_prefixes_include_common_voice_and_fleurs():
    public_real = build_dataset.COMPOSITION["sources"]["real"][1]

    assert "cvpt" in public_real["file_prefix"]
    assert "fleurs" in public_real["file_prefix"]
    assert "--common-voice-pt" in public_real["args"]
    assert "--fleurs" in public_real["args"]
