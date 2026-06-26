#!/usr/bin/env python3
"""Validate benchmark model/result artifacts without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check trained model folders, manifests, metrics and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", default="app/models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings.")
    args = parser.parse_args()

    from benchmarks.config import ALL_TCC_ARCHITECTURES

    models_dir = _resolve(args.models_dir)
    results_dir = _resolve(args.results_dir)
    manifest = models_dir / "benchmark_final_manifest.json"
    benchmark_final = models_dir / "benchmark_final"

    warnings: list[str] = []
    ok: list[str] = []

    if manifest.exists():
        ok.append(str(manifest.relative_to(ROOT)))
    else:
        warnings.append(f"missing manifest: {manifest}")

    if benchmark_final.exists():
        ok.append(str(benchmark_final.relative_to(ROOT)))
    else:
        warnings.append(f"missing benchmark_final dir: {benchmark_final}")

    for arch in ALL_TCC_ARCHITECTURES:
        slug = _slug(arch)
        candidates = [
            benchmark_final / slug,
            benchmark_final / slug.replace("wavlm", "wavlm_original"),
            benchmark_final / slug.replace("hubert", "hubert_original"),
        ]
        if not any(path.exists() for path in candidates):
            warnings.append(f"missing model directory for {arch}: {candidates[0]}")

    result_files = sorted(results_dir.rglob("results.json")) if results_dir.exists() else []
    if result_files:
        ok.append(f"{len(result_files)} results.json file(s)")
    else:
        warnings.append(f"no results.json found under {results_dir}")

    figure_names = {
        "confusion_matrix.png",
        "roc.png",
        "score_distribution.png",
        "convergence.png",
    }
    figure_hits = 0
    for result_file in result_files:
        data = _load_json(result_file)
        archs = data.get("architectures") or {}
        for arch in archs:
            arch_dir = result_file.parent / "architectures" / _slug(arch)
            figure_hits += sum(1 for name in figure_names if (arch_dir / name).exists())
    if figure_hits:
        ok.append(f"{figure_hits} benchmark figure file(s)")
    else:
        warnings.append("no per-architecture benchmark figures found")

    report = {
        "status": "ok" if not warnings else "warning",
        "models_dir": str(models_dir),
        "results_dir": str(results_dir),
        "ok": ok,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.strict and warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
