"""Contratos da persistência científica consolidada."""

import json
from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import experiment_store as store_module
from app.core.db.experiment_store import ExperimentStore, content_hash, json_safe
from app.core.db.session import Base
from app.domain.models import (  # noqa: F401
    ArtifactRecord,
    ConfigurationEntry,
    ExperimentRun,
    MetricRecord,
    ModelRun,
    SystemSnapshot,
)


@dataclass
class ExampleConfig:
    epochs: int
    labels: tuple[str, ...]


def _isolated_store(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'experiments.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(store_module, "SessionLocal", factory)
    return ExperimentStore(), factory


def test_json_safe_and_hash_are_deterministic():
    left = {"b": ExampleConfig(3, ("real", "fake")), "a": 1}
    right = {"a": 1, "b": {"labels": ["real", "fake"], "epochs": 3}}
    assert json_safe(left) == json_safe(right)
    assert content_hash(left) == content_hash(right)


def test_configuration_redacts_secrets(tmp_path, monkeypatch):
    store, factory = _isolated_store(tmp_path, monkeypatch)
    store.set_configuration(
        "environment", "API_TOKEN", "never-store-this", category="system"
    )
    assert store.get_configuration("environment", "API_TOKEN") is None
    with factory() as db:
        row = db.query(ConfigurationEntry).one()
        assert row.value == {"value": "<redacted>"}
        assert row.is_secret is True


def test_benchmark_payload_is_normalized(tmp_path, monkeypatch):
    store, factory = _isolated_store(tmp_path, monkeypatch)
    payload = {
        "config": {"seed": 42, "experiment_scope": "official"},
        "dataset": {"test_split_sha256": "abc"},
        "environment": {"python": "3.11"},
        "architectures": {
            "svm": {
                "status": "ok",
                "fit_strategy": {
                    "kind": "single_fit",
                    "estimator": "sklearn",
                    "fit_samples": 20,
                },
                "training_config": {"model_family": "classical", "C": 1.0},
                "clean": {"accuracy": 0.9, "eer": 0.1, "n": 20},
            }
        },
    }
    run_uid = store.persist_benchmark_results(payload, output_dir=tmp_path)
    assert store.load_run_payload(run_uid) == payload
    with factory() as db:
        assert db.query(ExperimentRun).count() == 1
        model_run = db.query(ModelRun).one()
        assert (
            json.loads(model_run.runner)
            == payload["architectures"]["svm"]["fit_strategy"]
        )
        metrics = db.query(MetricRecord).all()
        assert {row.metric_name for row in metrics} >= {"accuracy", "eer", "n"}
