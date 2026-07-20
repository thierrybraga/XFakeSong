#!/usr/bin/env python3
"""Valida artefatos de modelos/resultados do benchmark sem carregar pesos.

Confere presenca e esquema dos arquivos esperados por arquitetura
(``metrics.json``, ``results.csv/json``, ``predictions_clean.csv``,
``robustness.csv``, figuras) e a coerencia basica entre eles. Pensado para
rodar apos ``consolidate_results.py`` e antes de
``sync_completed_benchmark_artifacts.py``.

Uso:
    python scripts/reporting/validate_artifacts.py --results data/results/<run>
    python scripts/reporting/validate_artifacts.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config.paths import resolve_results_dir


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def _compact(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


# Formas alternativas (aliases) sob as quais um artefato bench_<...>.* pode
# aparecer para a mesma arquitetura. Duas convencoes de nome coexistem no
# projeto: o nome interno do registry (ex.: "SpectrogramTransformer",
# "Hybrid CNN-Transformer", "MultiscaleCNN") usado por ALL_TCC_ARCHITECTURES,
# e o nome de exibicao do TCC (ex.: "AST", "CCT", "Res2Net") usado por
# consolidate_results.py/sync_completed_benchmark_artifacts.py. O
# ModelLoader real (app/domain/services/detection/model_loader.py) descobre
# modelos via glob("benchmark_final/*/bench_*.*") e nao depende do nome da
# subpasta nem de qual das duas convencoes foi usada no arquivo -- so o
# validador precisa reconhecer ambas para nao gerar falso-positivo.
_ARCH_NAME_ALIASES: dict[str, set[str]] = {
    "hybridcnntransformer": {"hybridcnntransformer", "cct"},
    "spectrogramtransformer": {"spectrogramtransformer", "ast"},
    "multiscalecnn": {"multiscalecnn", "res2net"},
    "wavlmoriginal": {"wavlmoriginal", "wavlm"},
    "hubertoriginal": {"hubertoriginal", "hubert"},
}


def _accepted_compacts(arch: str) -> set[str]:
    compact = _compact(arch)
    return _ARCH_NAME_ALIASES.get(compact, {compact})


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
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings.")
    args = parser.parse_args()

    from benchmarks.config import ALL_TCC_ARCHITECTURES, SSL_DOCKER_ARCHITECTURES

    tcc_architectures = [*ALL_TCC_ARCHITECTURES, *SSL_DOCKER_ARCHITECTURES]

    models_dir = _resolve(args.models_dir)
    results_dir = (
        _resolve(args.results_dir)
        if args.results_dir
        else resolve_results_dir(ROOT)
    )
    manifest = models_dir / "registry.json"
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

    # Descobre os artefatos bench_*.* realmente presentes sob
    # benchmark_final/*/, do mesmo jeito que o ModelLoader de producao
    # (glob("benchmark_final/*/bench_*.*"), independente do nome da
    # subpasta). Evita falso-positivo quando o sync usou a convencao de
    # nome de exibicao do TCC (ex.: "ast/bench_spectrogramtransformer.keras")
    # em vez do nome interno do registry.
    found_compacts: set[str] = set()
    if benchmark_final.is_dir():
        for ext in ("*.keras", "*.h5", "*.pkl", "*.pt"):
            for artifact in benchmark_final.glob(f"*/bench_{ext}"):
                stem = artifact.stem[len("bench_") :] if artifact.stem.startswith("bench_") else artifact.stem
                found_compacts.add(_compact(stem))

    for arch in tcc_architectures:
        if not (_accepted_compacts(arch) & found_compacts):
            warnings.append(
                f"missing model artifact for {arch}: nenhum bench_*.* sob "
                f"{benchmark_final}/*/ corresponde a {sorted(_accepted_compacts(arch))}"
            )

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
