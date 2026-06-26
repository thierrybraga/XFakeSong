#!/usr/bin/env python3
"""Materializa e valida artefatos treinados do benchmark na raiz do projeto.

O benchmark roda em Docker/WSL com bind mount do projeto em /app. Este script
gera manifestos locais a partir de app/models e results/<execucao>, deixando
claro quais modelos pre-treinados estao prontos para novas builds/inferencia.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "clean_benchmark_full_20260626"
DEFAULT_MODELS = ROOT / "app" / "models"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _model_files(models_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(models_dir.glob("bench_*")):
        if not path.is_file():
            continue
        stem = path.stem
        if stem.endswith("_config"):
            stem = stem[: -len("_config")]
        entry = files.setdefault(stem, {})
        key = "config" if path.suffix.lower() == ".json" else "artifact"
        entry[key] = {
            "path": _rel(path),
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                timespec="seconds"
            ),
        }
    return files


def _ensure_model_configs(summary: dict[str, Any], models_dir: Path) -> None:
    """Gera configs mínimos para artefatos treinados sem `_config.json`.

    Modelos sklearn salvos diretamente como `.pkl` não passam pelo mesmo
    `TrainingService.save_model` dos Keras. A inferência funciona com o pickle,
    mas o frontend/documentação esperam o par `<modelo> + _config.json`.
    """
    for item in summary.get("models", []):
        if item.get("status") != "ok":
            continue
        artifact = item.get("model_artifact") or ""
        artifact_name = Path(artifact).name if artifact else ""
        if not artifact_name:
            continue
        artifact_path = models_dir / artifact_name
        config_path = models_dir / f"{artifact_path.stem}_config.json"
        if config_path.exists() or not artifact_path.exists():
            continue
        model_name = str(item.get("model") or artifact_path.stem)
        payload = {
            "model_name": artifact_path.stem,
            "architecture": model_name,
            "model_type": "sklearn" if artifact_path.suffix == ".pkl" else "keras",
            "artifact_path": _rel(artifact_path),
            "source": "benchmark",
            "training_status": item.get("status"),
            "metrics": item.get("clean"),
            "efficiency": item.get("efficiency"),
            "output_dir": item.get("output_dir"),
        }
        config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _results_files(results_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not results_dir.exists():
        return payload
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        figures = sorted(model_dir.glob("figures/*.png"))
        payload[model_dir.name] = {
            "results_json": _rel(model_dir / "results.json")
            if (model_dir / "results.json").exists()
            else None,
            "summary_md": _rel(model_dir / "summary.md")
            if (model_dir / "summary.md").exists()
            else None,
            "run_log": _rel(model_dir / "run.log")
            if (model_dir / "run.log").exists()
            else None,
            "figures": [_rel(path) for path in figures],
            "figure_count": len(figures),
        }
    return payload


def materialize(results_dir: Path, models_dir: Path) -> dict[str, Any]:
    summary = _load_json(results_dir / "run_summary.json", {})
    _ensure_model_configs(summary, models_dir)
    by_stem = _model_files(models_dir)
    results = _results_files(results_dir)
    models = []

    for item in summary.get("models", []):
        artifact = item.get("model_artifact") or ""
        artifact_name = Path(artifact).name if artifact else ""
        stem = Path(artifact_name).stem if artifact_name else ""
        files = by_stem.get(stem, {})
        models.append(
            {
                "model": item.get("model"),
                "status": item.get("status"),
                "artifact": files.get("artifact"),
                "config": files.get("config"),
                "clean": item.get("clean"),
                "efficiency": item.get("efficiency"),
                "output_dir": item.get("output_dir"),
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "models_dir": _rel(models_dir),
        "results_dir": _rel(results_dir),
        "benchmark_status": summary.get("status"),
        "models": models,
        "local_model_files": by_stem,
        "result_artifacts": results,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "pretrained_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (results_dir / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera manifestos de modelos/resultados do benchmark materializados."
    )
    parser.add_argument("--results", default=str(DEFAULT_RESULTS.relative_to(ROOT)))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS.relative_to(ROOT)))
    args = parser.parse_args()

    results_dir = ROOT / args.results
    models_dir = ROOT / args.models_dir
    payload = materialize(results_dir=results_dir, models_dir=models_dir)
    ok_models = [
        item for item in payload["models"]
        if item.get("status") == "ok" and item.get("artifact")
    ]
    print(f"Modelos OK materializados: {len(ok_models)}")
    print(f"Manifesto modelos: {_rel(models_dir / 'pretrained_manifest.json')}")
    print(f"Manifesto resultados: {_rel(results_dir / 'artifact_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
