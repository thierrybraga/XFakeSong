"""Cobertura de `app/core/utils/system_utils.py` (filesystem com tmp_path)."""

import os
import time
from pathlib import Path

from app.core.utils.system_utils import bootstrap_dirs, cleanup_workspace

_SIXTY_DAYS = 60 * 60 * 24 * 60


def _make_old(path: Path) -> None:
    past = time.time() - _SIXTY_DAYS
    os.utime(path, (past, past))


def test_bootstrap_dirs_creates_and_is_idempotent(tmp_path):
    bootstrap_dirs(tmp_path)
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "results").is_dir()
    # segunda chamada não levanta
    bootstrap_dirs(tmp_path)


def test_cleanup_dry_run_keeps_everything(tmp_path):
    (tmp_path / "app").mkdir()
    old = tmp_path / "old.log"
    old.write_text("x")
    _make_old(old)

    cleanup_workspace(tmp_path, days=30, dry_run=True)
    assert old.exists()  # dry-run não remove nada


def test_cleanup_removes_old_log_and_pycache_keeps_recent(tmp_path):
    appd = tmp_path / "app"
    appd.mkdir()
    (appd / "results").mkdir()

    old = tmp_path / "old.log"
    old.write_text("x")
    _make_old(old)

    pyc = appd / "sub" / "__pycache__"
    pyc.mkdir(parents=True)
    (pyc / "x.pyc").write_text("y")

    recent = tmp_path / "recent.log"
    recent.write_text("z")  # mtime atual → preservado

    cleanup_workspace(tmp_path, days=30, dry_run=False)

    assert not old.exists()
    assert not pyc.exists()
    assert recent.exists()


def test_cleanup_delete_datasets_flag(tmp_path):
    (tmp_path / "app").mkdir()
    ds = tmp_path / "datasets"
    ds.mkdir()
    (ds / "a.wav").write_text("x")

    cleanup_workspace(tmp_path, days=30, dry_run=False, delete_datasets=True)
    assert not ds.exists()
