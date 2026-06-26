#!/usr/bin/env python3
"""Consolida modelos/resultados concluídos do benchmark em app/models.

O benchmark grava os artefatos principais em:
- app/models/bench_<modelo>.*      (usado pela inferência/Gradio)
- results/<execucao>/<modelo>/...  (métricas, figuras e relatórios)

Este script copia os modelos já concluídos para:
- app/models/benchmark_final/<modelo>/

Essa pasta é explicitamente incluída no Docker build pela regra de
`.dockerignore`, permitindo empacotar modelos pré-treinados sem novo treino.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("/app/"):
        return ROOT / text.removeprefix("/app/")
    path = Path(text)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def sync_completed(
    summary_path: Path,
    final_dir: Path,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    final_dir.mkdir(parents=True, exist_ok=True)
    synced: list[dict[str, Any]] = []

    for item in summary.get("models", []):
        if item.get("status") != "ok":
            continue

        model = str(item.get("model") or "")
        slug = _slug(model)
        target = final_dir / slug
        target.mkdir(parents=True, exist_ok=True)

        model_artifact = _project_path(item.get("model_artifact"))
        model_copied = False
        config_copied = False
        if model_artifact is not None:
            model_copied = _copy_file(model_artifact, target / model_artifact.name)
            config = model_artifact.with_name(f"{model_artifact.stem}_config.json")
            config_copied = _copy_file(config, target / config.name)

        output_dir = _project_path(item.get("output_dir"))
        results_copied = False
        if output_dir is not None:
            results_copied = _copy_tree(output_dir, target / "results")

        manifest = {
            "model": model,
            "slug": slug,
            "status": item.get("status"),
            "source_model_artifact": str(model_artifact) if model_artifact else None,
            "source_output_dir": str(output_dir) if output_dir else None,
            "model_copied": model_copied,
            "config_copied": config_copied,
            "results_copied": results_copied,
            "metrics": item.get("clean"),
            "efficiency": item.get("efficiency"),
        }
        (target / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        synced.append(manifest)

    index = {
        "summary_path": str(summary_path),
        "final_dir": str(final_dir),
        "synced_count": len(synced),
        "models": synced,
    }
    (final_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    root_manifest = final_dir.parent / "benchmark_final_manifest.json"
    root_manifest.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Benchmark Final Artifacts",
        "",
        f"- Origem: `{summary_path}`",
        f"- Modelos sincronizados: `{len(synced)}`",
        "",
        "| Modelo | Modelo | Config | Resultados | Accuracy | AUC | EER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in synced:
        metrics = item.get("metrics") or {}
        lines.append(
            f"| {item['model']} | {item['model_copied']} | "
            f"{item['config_copied']} | {item['results_copied']} | "
            f"{metrics.get('accuracy', '')} | {metrics.get('auc_roc', '')} | "
            f"{metrics.get('eer', '')} |"
        )
    (final_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="results/clean_benchmark_full_20260626/run_summary.json",
    )
    parser.add_argument("--final-dir", default="app/models/benchmark_final")
    args = parser.parse_args()

    summary = _project_path(args.summary)
    final_dir = _project_path(args.final_dir)
    if summary is None or not summary.exists():
        raise SystemExit(f"Resumo não encontrado: {summary}")
    assert final_dir is not None
    index = sync_completed(summary, final_dir)
    print(f"Sincronizados: {index['synced_count']}")
    print(f"Destino: {final_dir}")
    print(f"Manifesto: {final_dir.parent / 'benchmark_final_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
