#!/usr/bin/env python3
"""Consolida configuração, resultados e estado do XFakeSong em data/app.db."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("consolidate_sqlite")


def _load_structured(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"value": payload}
    except (OSError, ValueError, ImportError) as exc:
        logger.warning("Ignorando %s: %s", path, exc)
        return None


def _module_constants(module: Any) -> dict[str, Any]:
    from app.core.db.experiment_store import json_safe

    constants = {}
    for name, value in vars(module).items():
        if not name.isupper() or name.startswith("_"):
            continue
        if callable(value) or isinstance(value, type):
            continue
        constants[name] = json_safe(value)
    return constants


def _backup_database(path: Path) -> Path | None:
    if not path.exists():
        return None
    base = path.with_name(f"{path.stem}.pre_consolidation{path.suffix}")
    backup = base
    index = 1
    while backup.exists():
        backup = path.with_name(f"{path.stem}.pre_consolidation.{index}{path.suffix}")
        index += 1
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolida dados persistidos no SQLite canônico."
    )
    parser.add_argument("--database", default="data/app.db")
    parser.add_argument("--results", default="data/results")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    database = (ROOT / args.database).resolve()
    if not str(database).startswith(str(ROOT.resolve())):
        parser.error("o banco deve ficar dentro do workspace")
    database.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{database.as_posix()}"

    from app.core.config.settings import get_config
    from app.core.db.experiment_store import (
        collect_system_snapshot,
        experiment_store,
        json_safe,
    )
    from app.domain.models.architectures.registry import architecture_registry
    from app.domain.models.training.hyperparameter_defaults import (
        get_recommended_hyperparameters,
    )
    from benchmarks import config as benchmark_config
    from benchmarks.config import (
        BenchmarkConfig,
        EXTENDED_MODEL_MANIFEST,
        MODEL_FAMILIES,
        OFFICIAL_TCC_MODEL_MANIFEST,
    )

    backup = None if args.no_backup else _backup_database(database)
    experiment_store.ensure_schema()
    snapshot_uid = experiment_store.record_system_snapshot("consolidation")

    experiment_store.set_configuration(
        "application",
        "system_config",
        json_safe(get_config()),
        category="system_configuration",
        source="app.core.config.settings:get_config",
    )
    experiment_store.persist_mapping(
        "environment",
        collect_system_snapshot().get("environment", {}),
        category="environment_variable",
        source="process_environment",
    )
    experiment_store.set_configuration(
        "benchmark",
        "defaults",
        BenchmarkConfig().to_dict(),
        category="benchmark_configuration",
        source="benchmarks.config:BenchmarkConfig",
    )
    experiment_store.set_configuration(
        "models",
        "official_manifest",
        OFFICIAL_TCC_MODEL_MANIFEST,
        category="model_manifest",
        scope="official",
        source="benchmarks.config",
    )
    experiment_store.set_configuration(
        "models",
        "extended_manifest",
        EXTENDED_MODEL_MANIFEST,
        category="model_manifest",
        scope="extended",
        source="benchmarks.config",
    )
    experiment_store.set_configuration(
        "models",
        "families",
        MODEL_FAMILIES,
        category="model_families",
        source="benchmarks.config",
    )
    experiment_store.persist_mapping(
        "constants",
        _module_constants(benchmark_config),
        category="code_constant",
        source="benchmarks.config",
    )

    for name, info in architecture_registry.get_all_architectures().items():
        experiment_store.set_configuration(
            "architecture_model_defaults",
            name,
            info.default_params,
            category="model_parameters",
            scope="default",
            source="app.domain.models.architectures.registry",
        )
        experiment_store.set_configuration(
            "training_hyperparameters",
            name,
            get_recommended_hyperparameters(name),
            category="training_hyperparameters",
            scope="default",
            source="app.domain.models.training.hyperparameter_defaults",
        )

    # Remove o namespace transitório usado pela primeira versão da migração.
    from app.core.db.session import SessionLocal
    from app.domain.models.experiment import ConfigurationEntry

    with SessionLocal() as db:
        db.query(ConfigurationEntry).filter(
            ConfigurationEntry.namespace == "architecture_defaults"
        ).delete(synchronize_session=False)
        db.commit()

    config_count = 0
    for path in sorted((ROOT / "configs").rglob("*")):
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        payload = _load_structured(path)
        if payload is None:
            continue
        experiment_store.set_configuration(
            "config_files",
            path.relative_to(ROOT).as_posix(),
            payload,
            category="configuration_file",
            source=str(path),
        )
        config_count += 1

    model_config_count = 0
    models_dir = ROOT / "app" / "models"
    if models_dir.exists():
        for path in sorted(models_dir.rglob("*_config.json")):
            payload = _load_structured(path)
            if payload is None:
                continue
            experiment_store.set_configuration(
                "model_sidecars",
                path.relative_to(models_dir).as_posix(),
                payload,
                category="model_configuration",
                scope=str(payload.get("architecture", "unknown")),
                source=str(path),
            )
            model_config_count += 1

    result_count = 0
    results_dir = (ROOT / args.results).resolve()
    if results_dir.exists():
        for path in sorted(results_dir.rglob("results.json")):
            payload = _load_structured(path)
            if not payload or not isinstance(payload.get("architectures"), dict):
                continue
            experiment_store.persist_benchmark_results(
                payload,
                output_dir=path.parent,
                source=str(path),
            )
            result_count += 1

    print(
        json.dumps(
            {
                "database": str(database),
                "backup": str(backup) if backup else None,
                "schema_version": 2,
                "system_snapshot_uid": snapshot_uid,
                "config_files": config_count,
                "model_configs": model_config_count,
                "benchmark_results": result_count,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
