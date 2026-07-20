#!/usr/bin/env python3
"""Regenera os sidecars ``bench_<modelo>_config.json`` dos modelos promovidos.

Problema que corrige: os sidecars em ``data/models/benchmark_final/<arch>/``
continham contratos de runs de FUMAÇA (ex.: Conformer com 1 época e 4
amostras de validação — ``eer_threshold=0,209``), e é DELES que o
``model_loader``/``Predictor`` extraem o limiar de decisão, a temperatura e o
``input_contract`` usados na inferência do app.

Este script reconstrói cada sidecar a partir do run REAL do benchmark:

- ``eer_threshold``/``eer_value`` recalculados de
  ``data/results/predictions_clean.csv`` (ponto em que FPR = FNR);
- ``input_shape`` do ``data/results/metrics.json``;
- ``input_contract`` completo, incluindo o ``feature_frontend`` do benchmark
  (``benchmark_raw_v1``/``benchmark_logmel_v1``/``benchmark_tabular_v1``) que
  o ``FeaturePreparer`` usa para reproduzir EXATAMENTE o front-end do treino
  (``app/domain/features/benchmark_frontend.py``);
- proveniência (fonte + timestamp) para auditoria.

O sidecar antigo é preservado como ``*.stale.bak`` na primeira regeneração.

Uso:
    python scripts/reporting/rebuild_inference_contracts.py
    python scripts/reporting/rebuild_inference_contracts.py --dry-run
    python scripts/reporting/rebuild_inference_contracts.py --archs conformer svm
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._bootstrap import setup_logging  # noqa: E402

logger = logging.getLogger("contracts")

BENCH_FINAL = ROOT / "app" / "models" / "benchmark_final"

# arch dir -> (feature_frontend, display architecture, model_type)
ARCH_SPECS: dict[str, dict[str, str]] = {
    "aasist": {"frontend": "benchmark_raw_v1", "architecture": "AASIST",
               "model_type": "tensorflow"},
    "rawgat_st": {"frontend": "benchmark_raw_v1", "architecture": "RawGAT-ST",
                  "model_type": "tensorflow"},
    "rawnet2": {"frontend": "benchmark_raw_v1", "architecture": "RawNet2",
                "model_type": "tensorflow"},
    "conformer": {"frontend": "benchmark_logmel_v1", "architecture": "Conformer",
                  "model_type": "tensorflow"},
    "res2net": {"frontend": "benchmark_logmel_v1",
                "architecture": "MultiscaleCNN", "model_type": "tensorflow"},
    "cct": {"frontend": "benchmark_logmel_v1",
            "architecture": "Hybrid CNN-Transformer",
            "model_type": "tensorflow"},
    "ast": {"frontend": "benchmark_logmel_v1",
            "architecture": "SpectrogramTransformer",
            "model_type": "tensorflow"},
    "svm": {"frontend": "benchmark_tabular_v1", "architecture": "SVM",
            "model_type": "sklearn"},
    "random_forest": {"frontend": "benchmark_tabular_v1",
                      "architecture": "Random Forest",
                      "model_type": "sklearn"},
    # SSL originais (PyTorch): o runner dedicado grava calibração própria
    # (limiar de EER sob ruído) nos artefatos .pt/metrics.json — não são
    # regenerados aqui para não sobrescrever a calibração sob ruído.
}


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """(eer, threshold) no ponto em que FPR = FNR (classe positiva = spoof)."""
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fpr = fp / max(n_neg, 1)
    fnr = 1.0 - tp / max(n_pos, 1)
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(s[idx])


def load_predictions(arch_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    csv_path = arch_dir / "results" / "predictions_clean.csv"
    if not csv_path.exists():
        return None
    y_true, p_fake = [], []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            p_fake.append(float(row["p_fake"]))
    return np.asarray(y_true), np.asarray(p_fake)


def build_contract(arch_key: str, spec: dict, metrics: dict,
                   eer: float | None, threshold: float | None) -> dict:
    frontend = spec["frontend"]
    input_shape = metrics.get("input_shape")
    if not input_shape:
        input_shape = {
            "benchmark_raw_v1": [16000, 1],
            "benchmark_logmel_v1": [100, 80],
            "benchmark_tabular_v1": [63],
        }[frontend]
    input_shape = [int(v) for v in input_shape]

    contract: dict = {
        "architecture": spec["architecture"],
        "input_shape": input_shape,
        "sample_rate": 16000,
        "feature_frontend": frontend,
        "source_samples": 80000,  # janela canônica de 5 s do benchmark
        "normalization": "per_sample_zscore",
        "label_classes": [0, 1],
        "temperature": 1.0,
        "scaler_applied": False,
    }
    if frontend == "benchmark_raw_v1":
        contract.update({
            "type": "audio", "format": "raw", "input_type": "raw_audio",
            "target_sequence_length": int(input_shape[0]),
        })
    elif frontend == "benchmark_logmel_v1":
        contract.update({
            "type": "features", "format": "spectrogram",
            "input_type": "spectrogram",
            "time_steps": int(input_shape[0]),
            "feature_dim": int(input_shape[1]),
        })
    else:  # tabular
        contract.update({
            "type": "features", "format": "tabular", "input_type": "tabular",
            "feature_dim": int(input_shape[0]),
            # O artefato clássico é um Pipeline sklearn com o scaler DENTRO;
            # nenhum scaler externo deve ser aplicado pela inferência.
            "normalization": "pipeline_interno",
        })
    if threshold is not None:
        contract["eer_threshold"] = round(float(threshold), 6)
    if eer is not None:
        contract["eer_value"] = round(float(eer), 6)
    return contract


def rebuild(arch_key: str, dry_run: bool = False) -> bool:
    spec = ARCH_SPECS[arch_key]
    arch_dir = BENCH_FINAL / arch_key
    metrics_path = arch_dir / "results" / "metrics.json"
    if not metrics_path.exists():
        logger.warning("%s: metrics.json ausente — pulado", arch_key)
        return False

    models = sorted(arch_dir.glob("bench_*.keras")) + sorted(
        arch_dir.glob("bench_*.pkl")
    )
    models = [m for m in models if "_scaler" not in m.stem]
    if not models:
        logger.warning("%s: artefato bench_* ausente — pulado", arch_key)
        return False
    model_path = models[0]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    preds = load_predictions(arch_dir)
    eer = threshold = None
    if preds is not None:
        eer, threshold = compute_eer(*preds)

    contract = build_contract(arch_key, spec, metrics, eer, threshold)
    sidecar = {
        "architecture": spec["architecture"],
        "input_shape": contract["input_shape"],
        "num_classes": 2,
        "label_classes": [0, 1],
        "model_type": spec["model_type"],
        "input_contract": contract,
        "provenance": {
            "source": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
            "rebuilt_by": "scripts/reporting/rebuild_inference_contracts.py",
            "rebuilt_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "note": (
                "Sidecar regenerado do run real do benchmark; o anterior "
                "continha contrato de run de fumaça (ver *.stale.bak)."
            ),
        },
    }

    config_path = model_path.parent / f"{model_path.stem}_config.json"
    logger.info(
        "%s: %s  eer=%.4f thr=%.4f frontend=%s shape=%s",
        arch_key, config_path.name,
        eer if eer is not None else float("nan"),
        threshold if threshold is not None else float("nan"),
        contract["feature_frontend"], contract["input_shape"],
    )
    if dry_run:
        return True

    if config_path.exists():
        backup = config_path.with_suffix(".json.stale.bak")
        if not backup.exists():
            shutil.copy2(config_path, backup)
    config_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs", nargs="+", choices=sorted(ARCH_SPECS),
        default=sorted(ARCH_SPECS),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(name="contracts")
    done = sum(1 for arch in args.archs if rebuild(arch, dry_run=args.dry_run))
    logger.info("%d/%d sidecars %s", done, len(args.archs),
                "validados (dry-run)" if args.dry_run else "regenerados")
    return 0 if done == len(args.archs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
