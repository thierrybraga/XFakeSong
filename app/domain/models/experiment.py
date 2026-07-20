"""Modelos canônicos para persistência de experimentos e configuração."""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.base_model import BaseModel


class ExperimentRun(BaseModel):
    __tablename__ = "experiment_runs"

    run_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    run_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    scope: Mapped[str] = mapped_column(String(32), default="official")
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(128), index=True)
    output_dir: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(128), index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset: Mapped[dict] = mapped_column(JSON, default=dict)
    result_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ModelRun(BaseModel):
    __tablename__ = "model_runs"

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(200), index=True)
    family: Mapped[str | None] = mapped_column(String(100), nullable=True)
    runner: Mapped[str | None] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    converged: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    wall_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    model_config: Mapped[dict] = mapped_column(JSON, default=dict)
    input_contract: Mapped[dict] = mapped_column(JSON, default=dict)
    result_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("ix_model_runs_experiment_model", "experiment_id", "model_name"),
    )


class MetricRecord(BaseModel):
    __tablename__ = "metric_records"

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True
    )
    model_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    condition: Mapped[str] = mapped_column(String(100), index=True)
    metric_name: Mapped[str] = mapped_column(String(100), index=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index(
            "ix_metric_lookup",
            "experiment_id",
            "model_run_id",
            "condition",
            "metric_name",
        ),
    )


class ConfigurationEntry(BaseModel):
    __tablename__ = "configuration_entries"

    namespace: Mapped[str] = mapped_column(String(100), index=True)
    key: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(50), index=True)
    scope: Mapped[str] = mapped_column(String(100), default="global", index=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    value_type: Mapped[str] = mapped_column(String(50))
    source: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index(
            "uq_configuration_identity",
            "namespace",
            "key",
            "scope",
            unique=True,
        ),
    )


class SystemSnapshot(BaseModel):
    __tablename__ = "system_snapshots"

    snapshot_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    purpose: Mapped[str] = mapped_column(String(100), index=True)
    hostname: Mapped[str | None] = mapped_column(String(255), nullable=True)
    platform: Mapped[str | None] = mapped_column(String(500), nullable=True)
    python_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)


class ArtifactRecord(BaseModel):
    __tablename__ = "artifact_records"

    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    model_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    artifact_type: Mapped[str] = mapped_column(String(100), index=True)
    path: Mapped[str] = mapped_column(String(1200), index=True)
    sha256: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
