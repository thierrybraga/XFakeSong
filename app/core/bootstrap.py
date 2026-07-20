"""Bootstrap canônico e idempotente do XFakeSong.

Recursos científicos ausentes são estados válidos. Somente falhas estruturais
(permissão, banco corrompido ou configuração obrigatória inválida) interrompem
a inicialização.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
MODEL_SUFFIXES = {".keras", ".h5", ".pkl", ".pt", ".onnx"}


class ResourceState(str, Enum):
    MISSING = "missing"
    EMPTY = "empty"
    INCOMPLETE = "incomplete"
    AVAILABLE = "available"
    INVALID = "invalid"


class DatabaseState(str, Enum):
    MISSING = "missing"
    NEW = "new"
    COMPATIBLE = "compatible"
    MIGRATION_REQUIRED = "migration_required"
    INVALID = "invalid"


class BootstrapError(RuntimeError):
    """Falha estrutural que impede uma inicialização segura."""


@dataclass(frozen=True)
class OperationalPaths:
    root: Path
    data: Path
    datasets: Path
    dataset_real: Path
    dataset_fake: Path
    models: Path
    checkpoints: Path
    results: Path
    benchmark_results: Path
    reporting_results: Path
    logs: Path
    cache: Path
    temp: Path
    uploads: Path
    database: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None) -> "OperationalPaths":
        project_root = Path(root or PROJECT_ROOT).expanduser().resolve()
        storage_raw = (
            None
            if root is not None
            else os.getenv("XFAKE_STORAGE_DIR") or os.getenv("DEEPFAKE_STORAGE_DIR")
        )
        storage = (
            Path(storage_raw).expanduser() if storage_raw else project_root / "data"
        )
        if not storage.is_absolute():
            storage = project_root / storage
        data = storage.resolve()
        datasets = data / "datasets"

        def configured_path(names: tuple[str, ...], default: Path) -> Path:
            raw = next((os.getenv(name) for name in names if os.getenv(name)), None)
            candidate = Path(raw).expanduser() if raw else default
            if not candidate.is_absolute():
                candidate = project_root / candidate
            return candidate.resolve()

        models = configured_path(
            ("MODELS_DIR", "DEEPFAKE_MODELS_DIR", "XFAKE_MODELS_DIR"),
            data / "models",
        )
        logs = configured_path(("DEEPFAKE_LOGS_DIR", "XFAKE_LOGS_DIR"), data / "logs")
        results = configured_path(
            ("RESULTS_DIR", "DEEPFAKE_RESULTS_DIR", "XFAKE_RESULTS_DIR"),
            data / "results",
        )
        return cls(
            root=project_root,
            data=data,
            datasets=datasets,
            dataset_real=datasets / "real",
            dataset_fake=datasets / "fake",
            models=models,
            checkpoints=models / "checkpoints",
            results=results,
            benchmark_results=results / "benchmark",
            reporting_results=results / "reporting",
            logs=logs,
            cache=data / "cache",
            temp=data / "temp",
            uploads=data / "uploads",
            database=data / "app.db",
        )

    def directories(self) -> tuple[Path, ...]:
        return (
            self.dataset_real,
            self.dataset_fake,
            self.models,
            self.checkpoints,
            self.results,
            self.benchmark_results,
            self.reporting_results,
            self.logs,
            self.cache,
            self.temp,
            self.uploads,
        )


@dataclass(frozen=True)
class ResourceInventory:
    datasets: ResourceState
    models: ResourceState
    benchmarks: ResourceState
    reports: ResourceState
    dataset_real_files: int = 0
    dataset_fake_files: int = 0
    model_artifacts: int = 0


@dataclass(frozen=True)
class BootstrapReport:
    paths: OperationalPaths
    database_before: DatabaseState
    database_after: DatabaseState
    resources: ResourceInventory


def ensure_operational_directories(paths: OperationalPaths) -> None:
    for directory in paths.directories():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise BootstrapError(
                f"Sem permissão para criar o diretório operacional: {directory}"
            ) from exc
        if not directory.is_dir():
            raise BootstrapError(f"Caminho operacional não é diretório: {directory}")


def _files_with_suffix(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def discover_resources(paths: OperationalPaths) -> ResourceInventory:
    real = _files_with_suffix(paths.dataset_real, AUDIO_SUFFIXES)
    fake = _files_with_suffix(paths.dataset_fake, AUDIO_SUFFIXES)
    if real and fake:
        dataset_state = ResourceState.AVAILABLE
    elif real or fake:
        dataset_state = ResourceState.INCOMPLETE
    elif paths.datasets.exists():
        dataset_state = ResourceState.EMPTY
    else:
        dataset_state = ResourceState.MISSING

    models = _files_with_suffix(paths.models, MODEL_SUFFIXES)
    model_state = ResourceState.AVAILABLE if models else ResourceState.EMPTY
    benchmark_files = _files_with_suffix(paths.benchmark_results, {".json", ".csv"})
    report_files = _files_with_suffix(
        paths.reporting_results, {".md", ".html", ".pdf", ".tex"}
    )
    return ResourceInventory(
        datasets=dataset_state,
        models=model_state,
        benchmarks=(
            ResourceState.AVAILABLE if benchmark_files else ResourceState.EMPTY
        ),
        reports=ResourceState.AVAILABLE if report_files else ResourceState.EMPTY,
        dataset_real_files=len(real),
        dataset_fake_files=len(fake),
        model_artifacts=len(models),
    )


def inspect_database(path: Path, expected_schema: int = 2) -> DatabaseState:
    if not path.exists():
        return DatabaseState.MISSING
    if path.stat().st_size == 0:
        return DatabaseState.NEW
    try:
        with sqlite3.connect(path) as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                return DatabaseState.INVALID
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = connection.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
    except (sqlite3.DatabaseError, OSError):
        return DatabaseState.INVALID
    if not tables:
        return DatabaseState.NEW
    if version < expected_schema:
        return DatabaseState.MIGRATION_REQUIRED
    return DatabaseState.COMPATIBLE


def bootstrap_application(
    root: str | Path | None = None,
    *,
    initialize_database: bool = True,
) -> BootstrapReport:
    paths = OperationalPaths.resolve(root)
    configured_url = os.getenv("DATABASE_URL")
    if configured_url and configured_url.startswith("sqlite:///"):
        configured_path = configured_url.removeprefix("sqlite:///")
        if configured_path != ":memory:":
            configured_database = Path(configured_path)
            if not configured_database.is_absolute():
                configured_database = paths.root / configured_database
            paths = replace(paths, database=configured_database.resolve())
    before = inspect_database(paths.database)
    if before == DatabaseState.INVALID:
        raise BootstrapError(
            f"Banco SQLite inválido ou corrompido: {paths.database}. "
            "Restaure um backup antes de iniciar."
        )

    ensure_operational_directories(paths)
    after = before
    if initialize_database:
        expected_url = f"sqlite:///{paths.database.as_posix()}"
        if configured_url and configured_url != expected_url:
            logger.info("Usando DATABASE_URL explicitamente configurada.")
        else:
            os.environ.setdefault("DATABASE_URL", expected_url)
        from app.core.db.setup import init_db

        init_db(raise_on_error=True)
        after = inspect_database(paths.database)
        if after != DatabaseState.COMPATIBLE:
            raise BootstrapError(
                f"Schema SQLite não ficou compatível após inicialização: {after.value}"
            )

    resources = discover_resources(paths)
    for name in ("datasets", "models", "benchmarks", "reports"):
        state = getattr(resources, name)
        if state != ResourceState.AVAILABLE:
            logger.info("Recurso opcional %s: %s", name, state.value)
    return BootstrapReport(paths, before, after, resources)
