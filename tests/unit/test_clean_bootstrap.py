"""Primeira execução sem artefatos e bootstrap idempotente."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.bootstrap import (
    BootstrapError,
    DatabaseState,
    OperationalPaths,
    ResourceState,
    discover_resources,
    ensure_operational_directories,
    inspect_database,
)
from app.domain.services.detection.model_loader import ModelLoader

ROOT = Path(__file__).resolve().parents[2]


def test_empty_resource_inventory_is_valid(tmp_path):
    paths = OperationalPaths.resolve(tmp_path)
    ensure_operational_directories(paths)
    inventory = discover_resources(paths)
    assert inventory.datasets == ResourceState.EMPTY
    assert inventory.models == ResourceState.EMPTY
    assert inventory.benchmarks == ResourceState.EMPTY
    assert inventory.reports == ResourceState.EMPTY


def test_incomplete_and_complete_dataset_are_distinct(tmp_path):
    paths = OperationalPaths.resolve(tmp_path)
    ensure_operational_directories(paths)
    (paths.dataset_real / "real.wav").write_bytes(b"RIFF")
    assert discover_resources(paths).datasets == ResourceState.INCOMPLETE
    (paths.dataset_fake / "fake.wav").write_bytes(b"RIFF")
    assert discover_resources(paths).datasets == ResourceState.AVAILABLE


def test_model_loader_does_not_create_fake_models(tmp_path):
    loader = ModelLoader(tmp_path / "models")
    loader.load_available_models()
    assert loader.get_available_models() == []
    assert loader.default_model is None
    assert list((tmp_path / "models").iterdir()) == []


def test_database_states_do_not_hide_corruption(tmp_path):
    database = tmp_path / "app.db"
    assert inspect_database(database) == DatabaseState.MISSING
    database.touch()
    assert inspect_database(database) == DatabaseState.NEW
    database.write_bytes(b"not-a-sqlite-database")
    assert inspect_database(database) == DatabaseState.INVALID


def test_existing_database_data_is_preserved(tmp_path):
    database = tmp_path / "app.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE preserved (value TEXT)")
        connection.execute("INSERT INTO preserved VALUES ('keep')")
        connection.execute("PRAGMA user_version=2")
    assert inspect_database(database) == DatabaseState.COMPATIBLE
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM preserved").fetchone() == ("keep",)


def test_permission_failure_is_actionable(tmp_path, monkeypatch):
    paths = OperationalPaths.resolve(tmp_path)

    def deny(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", deny)
    with pytest.raises(BootstrapError, match="Sem permissão"):
        ensure_operational_directories(paths)


def test_main_bootstrap_twice_from_another_cwd(tmp_path):
    storage = tmp_path / "storage"
    cwd = tmp_path / "outside-project"
    cwd.mkdir()
    env = {
        **os.environ,
        "XFAKE_STORAGE_DIR": str(storage),
        "PYTHONPATH": os.pathsep.join(
            [
                str(ROOT),
                str(ROOT / ".venv" / "Lib" / "site-packages"),
            ]
        ),
    }
    env.pop("DATABASE_URL", None)
    command = [sys.executable, str(ROOT / "main.py"), "--bootstrap-dirs"]
    first = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    second = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (storage / "datasets" / "real").is_dir()
    assert (storage / "datasets" / "fake").is_dir()
    assert inspect_database(storage / "app.db") == DatabaseState.COMPATIBLE
