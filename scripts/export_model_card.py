#!/usr/bin/env python3
"""Export a Markdown model card for the consolidated trained artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = Path(text)
    try:
        if path.is_absolute() and path.is_relative_to(ROOT):
            return path.relative_to(ROOT).as_posix()
    except Exception:
        pass
    return text.replace("\\", "/")


def _manifest_rows(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    items = manifest.get("models") or manifest.get("artifacts") or {}
    rows: list[tuple[str, str]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("model") or item.get("name") or item.get("slug") or "")
            artifact = str(
                item.get("source_model_artifact")
                or item.get("artifact")
                or item.get("path")
                or item.get("source_output_dir")
                or ""
            )
            if name:
                rows.append((name, _display_path(artifact)))
    elif isinstance(items, dict):
        for name, item in sorted(items.items()):
            artifact = item.get("path") if isinstance(item, dict) else item
            rows.append((str(name), _display_path(artifact)))
    return rows


def _metric_rows(results: Any) -> list[tuple[str, Any]]:
    if isinstance(results, list):
        return [
            (str(item.get("model") or item.get("key") or ""), item)
            for item in results
            if isinstance(item, dict)
        ]
    if isinstance(results, dict):
        archs = results.get("architectures") or results.get("models") or {}
        if isinstance(archs, list):
            return [
                (str(item.get("model") or item.get("key") or ""), item)
                for item in archs
                if isinstance(item, dict)
            ]
        if isinstance(archs, dict):
            return [(str(name), item) for name, item in sorted(archs.items())]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MODEL_CARD.md for app/models or a HF model repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", default="app/models")
    parser.add_argument("--results-json", default="results/tcc_consolidated/benchmark_summary.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    models_dir = _resolve(args.models_dir)
    results_json = _resolve(args.results_json)
    out = _resolve(args.out) if args.out else models_dir / "MODEL_CARD.md"
    manifest = _load_json(models_dir / "benchmark_final_manifest.json")
    results = _load_json(results_json)

    lines = [
        "# XFakeSong benchmark models",
        "",
        "Trained artifacts for the XFakeSong deepfake audio detection benchmark.",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Models directory: `{_display_path(models_dir)}`",
        f"- Results source: `{_display_path(results_json)}`",
        "",
        "## Intended use",
        "",
        "These models are intended for academic evaluation, local inference, Gradio/FastAPI demonstrations and reproducible benchmark analysis.",
        "",
        "## Dataset",
        "",
        "The default benchmark dataset is `app/datasets/benchmark_audio_raw_balanced_20k.npz`, with balanced real/fake classes when available.",
        "",
        "## Artifacts",
        "",
    ]

    manifest_rows = _manifest_rows(manifest)
    if manifest_rows:
        lines.extend(["| Model | Artifact |", "|---|---|"])
        for name, artifact in manifest_rows:
            lines.append(f"| {name} | `{artifact}` |")
    else:
        lines.append("- Manifest not found or empty.")

    metric_rows = _metric_rows(results)
    if metric_rows:
        lines.extend(["", "## Metrics", "", "| Model | Accuracy | AUC ROC | EER |", "|---|---:|---:|---:|"])
        for name, item in metric_rows:
            clean = item.get("clean") if isinstance(item, dict) else {}
            clean = clean or item
            lines.append(
                f"| {name} | {clean.get('accuracy', '')} | "
                f"{clean.get('auc_roc', clean.get('auc', ''))} | {clean.get('eer', '')} |"
            )

    lines.extend([
        "",
        "## Loading",
        "",
        "Use `scripts/sync_hf_models.py` to download these artifacts into `app/models`, then start the interface with `python main.py --gradio`.",
        "",
        "## Limitations",
        "",
        "Performance is tied to the benchmark dataset, preprocessing contract and library versions recorded with each training run. Validate on external audio before operational use.",
        "",
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
