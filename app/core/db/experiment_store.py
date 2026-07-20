"""Persistência canônica de experimentos, configuração e estado do sistema.

O SQLite armazena dados estruturados e consultáveis. JSON/CSV/figuras continuam
como projeções exportáveis e artefatos, nunca como a única fonte de verdade.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from sqlalchemy.orm.attributes import flag_modified

from app.core.db.session import SessionLocal

logger = logging.getLogger(__name__)

SECRET_TOKENS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "jwt",
)
ENV_PREFIXES = (
    "DEEPFAKE_",
    "XFAKE_",
    "GRADIO_",
    "TF_",
    "TORCH_",
    "CUDA_",
    "NVIDIA_",
    "OMP_",
    "MKL_",
    "PYTHON",
)


def json_safe(value: Any) -> Any:
    """Converte valores científicos/configuração para JSON determinístico."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def content_hash(value: Any) -> str:
    normalized = json.dumps(
        json_safe(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_secret(key: str) -> bool:
    compact = key.lower()
    return any(token in compact for token in SECRET_TOKENS)


def _wrap_value(value: Any, *, secret: bool = False) -> dict[str, Any]:
    return {"value": "<redacted>" if secret else json_safe(value)}


def collect_system_snapshot() -> dict[str, Any]:
    """Coleta ambiente reprodutível sem persistir segredos."""
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "process": {"pid": os.getpid(), "cwd": str(Path.cwd())},
        "environment": {},
    }
    for key, value in sorted(os.environ.items()):
        if key.startswith(ENV_PREFIXES):
            payload["environment"][key] = "<redacted>" if _is_secret(key) else value
    try:
        import psutil

        memory = psutil.virtual_memory()
        payload["hardware"] = {
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_total_bytes": int(memory.total),
            "memory_available_bytes": int(memory.available),
        }
    except ImportError:
        payload["hardware"] = {"cpu_logical": os.cpu_count()}
    try:
        import tensorflow as tf

        payload["accelerators"] = {
            "tensorflow": tf.__version__,
            "gpus": [
                getattr(device, "name", str(device))
                for device in tf.config.list_physical_devices("GPU")
            ],
        }
    except ImportError:
        payload["accelerators"] = {"tensorflow": None, "gpus": []}
    return payload


class ExperimentStore:
    """API transacional única para persistência científica e operacional."""

    def ensure_schema(self) -> None:
        from app.core.db.setup import init_db

        init_db(raise_on_error=True)

    def record_system_snapshot(self, purpose: str) -> str:
        from app.domain.models.experiment import SystemSnapshot

        payload = collect_system_snapshot()
        digest = content_hash(payload)
        snapshot_uid = f"sys_{digest[:24]}"
        with SessionLocal() as db:
            existing = (
                db.query(SystemSnapshot).filter_by(snapshot_uid=snapshot_uid).first()
            )
            if existing is None:
                db.add(
                    SystemSnapshot(
                        snapshot_uid=snapshot_uid,
                        purpose=purpose,
                        hostname=payload.get("hostname"),
                        platform=payload.get("platform"),
                        python_version=payload["python"]["version"],
                        payload=payload,
                        content_hash=digest,
                    )
                )
                db.commit()
        return snapshot_uid

    def set_configuration(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        category: str,
        scope: str = "global",
        source: str | None = None,
        secret: bool | None = None,
    ) -> None:
        from app.domain.models.experiment import ConfigurationEntry

        is_secret = _is_secret(key) if secret is None else secret
        wrapped = _wrap_value(value, secret=is_secret)
        digest = content_hash(wrapped)
        with SessionLocal() as db:
            entry = (
                db.query(ConfigurationEntry)
                .filter_by(namespace=namespace, key=key, scope=scope)
                .first()
            )
            if entry is None:
                entry = ConfigurationEntry(
                    namespace=namespace,
                    key=key,
                    scope=scope,
                    category=category,
                    value=wrapped,
                    value_type=type(value).__name__,
                    source=source,
                    content_hash=digest,
                    is_secret=is_secret,
                    is_active=True,
                )
                db.add(entry)
            else:
                entry.category = category
                entry.value = wrapped
                entry.value_type = type(value).__name__
                entry.source = source
                entry.content_hash = digest
                entry.is_secret = is_secret
                entry.is_active = True
                flag_modified(entry, "value")
            db.commit()

    def get_configuration(
        self, namespace: str, key: str, *, scope: str = "global"
    ) -> Any | None:
        from app.domain.models.experiment import ConfigurationEntry

        with SessionLocal() as db:
            entry = (
                db.query(ConfigurationEntry)
                .filter_by(
                    namespace=namespace,
                    key=key,
                    scope=scope,
                    is_active=True,
                )
                .first()
            )
            if entry is None or entry.is_secret:
                return None
            return (entry.value or {}).get("value")

    def persist_mapping(
        self,
        namespace: str,
        values: Mapping[str, Any],
        *,
        category: str,
        scope: str = "global",
        source: str | None = None,
    ) -> None:
        for key, value in values.items():
            self.set_configuration(
                namespace,
                str(key),
                value,
                category=category,
                scope=scope,
                source=source,
            )

    @staticmethod
    def _metric_rows(
        model_result: Mapping[str, Any],
    ) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
        conditions: dict[str, Mapping[str, Any]] = {}
        clean = model_result.get("clean")
        if isinstance(clean, Mapping):
            conditions["clean"] = clean
        for snr, metrics in (model_result.get("robustness") or {}).items():
            if isinstance(metrics, Mapping):
                conditions[f"awgn_{snr}db"] = metrics
        for codec, metrics in (model_result.get("codec_robustness") or {}).items():
            if isinstance(metrics, Mapping):
                conditions[f"codec_{codec}"] = metrics
        efficiency = model_result.get("efficiency")
        if isinstance(efficiency, Mapping):
            conditions["efficiency"] = efficiency

        for condition, metrics in conditions.items():
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    yield condition, str(name), float(value), dict(metrics)

    def persist_benchmark_results(
        self,
        results: Mapping[str, Any],
        *,
        output_dir: str | Path | None = None,
        run_uid: str | None = None,
        source: str | None = None,
    ) -> str:
        """Grava payload completo e projeções normalizadas de modelos/métricas."""
        from app.domain.models.experiment import (
            ArtifactRecord,
            ExperimentRun,
            MetricRecord,
            ModelRun,
        )

        payload = json_safe(results)
        config = payload.get("config") or {}
        dataset = payload.get("dataset") or {}
        environment = payload.get("environment") or {}
        config_digest = content_hash(config)
        identity = {
            "source": source,
            "output_dir": str(output_dir) if output_dir else None,
            "config_hash": config_digest,
            "dataset_fingerprint": dataset.get("test_split_sha256"),
            "seed": config.get("seed"),
        }
        run_uid = run_uid or f"run_{content_hash(identity)[:28]}"

        with SessionLocal() as db:
            run = db.query(ExperimentRun).filter_by(run_uid=run_uid).first()
            if run is None:
                run = ExperimentRun(run_uid=run_uid)
                db.add(run)
            else:
                db.query(MetricRecord).filter_by(experiment_id=run.id).delete()
                db.query(ArtifactRecord).filter_by(experiment_id=run.id).delete()
                db.query(ModelRun).filter_by(experiment_id=run.id).delete()
                db.flush()

            model_payloads = payload.get("architectures") or {}
            statuses = [
                str(item.get("status", "unknown"))
                for item in model_payloads.values()
                if isinstance(item, Mapping)
            ]
            run.run_type = "benchmark"
            run.status = (
                "ok"
                if statuses and all(status == "ok" for status in statuses)
                else "partial"
            )
            run.scope = str(config.get("experiment_scope", "official"))
            run.seed = config.get("seed")
            run.dataset_path = config.get("dataset_path")
            run.dataset_fingerprint = dataset.get("test_split_sha256")
            run.output_dir = str(output_dir) if output_dir else config.get("output_dir")
            run.config_hash = config_digest
            run.config = config
            run.environment = environment
            run.dataset = dataset
            run.result_payload = payload
            flag_modified(run, "config")
            flag_modified(run, "environment")
            flag_modified(run, "dataset")
            flag_modified(run, "result_payload")
            db.flush()

            for model_name, model_payload in model_payloads.items():
                if not isinstance(model_payload, Mapping):
                    continue
                training_config = model_payload.get("training_config") or {}
                runner_payload = model_payload.get("fit_strategy")
                runner = (
                    runner_payload
                    if runner_payload is None or isinstance(runner_payload, str)
                    else json.dumps(
                        json_safe(runner_payload),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                model_run = ModelRun(
                    experiment_id=run.id,
                    model_name=str(model_name),
                    family=training_config.get("model_family"),
                    runner=runner,
                    status=str(model_payload.get("status", "unknown")),
                    converged=model_payload.get("converged"),
                    epochs=model_payload.get("epochs"),
                    wall_time_s=model_payload.get("wall_time_s"),
                    parameters=model_payload.get("model_parameters") or {},
                    hyperparameters=training_config,
                    model_config=model_payload.get("final_training_metrics") or {},
                    input_contract=model_payload.get("input_contract") or {},
                    result_payload=json_safe(model_payload),
                )
                db.add(model_run)
                db.flush()
                for condition, name, value, details in self._metric_rows(model_payload):
                    threshold = details.get("decision_threshold")
                    db.add(
                        MetricRecord(
                            experiment_id=run.id,
                            model_run_id=model_run.id,
                            condition=condition,
                            metric_name=name,
                            value=value,
                            threshold=(
                                float(threshold)
                                if isinstance(threshold, (int, float))
                                else None
                            ),
                            sample_count=details.get("n"),
                            details=json_safe(details),
                        )
                    )
                artifact_path = model_payload.get("model_artifact")
                if artifact_path:
                    path = Path(str(artifact_path))
                    db.add(
                        ArtifactRecord(
                            experiment_id=run.id,
                            model_run_id=model_run.id,
                            artifact_type="model",
                            path=str(path),
                            sha256=None,
                            size_bytes=path.stat().st_size if path.exists() else None,
                            artifact_metadata={},
                        )
                    )
            db.commit()
        return run_uid

    def load_run_payload(self, run_uid: str) -> dict[str, Any] | None:
        from app.domain.models.experiment import ExperimentRun

        with SessionLocal() as db:
            run = db.query(ExperimentRun).filter_by(run_uid=run_uid).first()
            return dict(run.result_payload or {}) if run else None


experiment_store = ExperimentStore()
