#!/usr/bin/env python3
"""Generate benchmark reports for models that are already trained.

This is intentionally evaluation-only: it loads existing ``app/models/bench_*``
artifacts, evaluates them on the canonical benchmark test split, and writes the
same report/figure bundle produced by the full benchmark runner. Missing models
are recorded as pending instead of being trained.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from benchmarks.config import ALL_TCC_ARCHITECTURES, BenchmarkConfig
from benchmarks.data import BenchmarkData
from benchmarks.efficiency import count_params, file_size_mb, measure_latency_ms
from benchmarks.evaluate import evaluate_scores
from benchmarks.report import write_all
from benchmarks.runner import _env_snapshot


LOG = logging.getLogger("completed_benchmark_artifacts")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _artifact_path(models_dir: Path, arch: str) -> Path | None:
    stem = f"bench_{_slug(arch)}"
    for suffix in (".keras", ".h5", ".pkl", ".pt"):
        path = models_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _read_config(models_dir: Path, arch: str) -> dict[str, Any]:
    path = models_dir / f"bench_{_slug(arch)}_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Config invalido em %s: %s", path, exc)
        return {}


def _load_keras_model(path: Path):
    import tensorflow as tf

    from app.domain.services.detection.model_loader import (
        _load_custom_architecture_modules,
    )

    _load_custom_architecture_modules()
    custom_objects = dict(tf.keras.utils.get_custom_objects())
    return tf.keras.models.load_model(
        str(path),
        custom_objects=custom_objects,
        safe_mode=False,
        compile=False,
    )


def _predict_keras(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    pred = model.predict(np.asarray(X, dtype="float32"), batch_size=batch_size, verbose=0)
    pred = np.asarray(pred, dtype="float64")
    if pred.ndim > 1 and pred.shape[-1] > 1:
        return pred[:, 1]
    return pred.reshape(-1)


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64").ravel()
    if np.any(~np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)


def _stratified_test_indices(y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    idx = np.arange(len(y))
    _train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    _val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y[temp_idx],
        random_state=seed,
    )
    return np.asarray(test_idx, dtype=int)


def _test_view(data: BenchmarkData, indices: np.ndarray) -> BenchmarkData:
    groups = data.groups[indices] if data.groups is not None else None
    speakers = data.speakers[indices] if data.speakers is not None else None
    return BenchmarkData(
        X=np.asarray(data.X[indices], dtype="float32"),
        y=np.asarray(data.y[indices], dtype="int64"),
        name=data.name,
        metadata=dict(data.metadata or {}),
        groups=groups,
        speakers=speakers,
    )


def _evaluate_loaded_model(
    arch: str,
    artifact: Path,
    cfg: BenchmarkConfig,
    Xte: np.ndarray,
    yte: np.ndarray,
    models_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    config_payload = _read_config(models_dir, arch)
    metrics_payload = config_payload.get("metrics") or {}
    history = metrics_payload.get("history")
    training_config = (
        metrics_payload.get("training_config")
        or config_payload.get("training_config")
        or {}
    )

    if artifact.suffix == ".pkl":
        model = joblib.load(artifact)

        def predict_p_fake(X: np.ndarray) -> np.ndarray:
            X2 = np.asarray(X).reshape(len(X), -1)
            proba = model.predict_proba(X2)
            return proba[:, 1] if proba.ndim > 1 and proba.shape[1] > 1 else proba.ravel()

        def predict_fn(xb: np.ndarray):
            return model.predict_proba(np.asarray(xb).reshape(len(xb), -1))

        model_type = "classical"
        params = None
    else:
        model = _load_keras_model(artifact)

        def predict_p_fake(X: np.ndarray) -> np.ndarray:
            return _predict_keras(model, X, cfg.batch_size)

        def predict_fn(xb: np.ndarray):
            return model.predict(np.asarray(xb, dtype="float32"), verbose=0)

        model_type = "neural"
        params = count_params(model)

    pf_clean = _finite_scores(predict_p_fake(Xte))
    clean = evaluate_scores(yte, pf_clean)
    robustness: dict[str, Any] = {}
    for snr in cfg.snr_levels_db:
        Xn = BenchmarkData.add_awgn(Xte, snr, seed=cfg.seed)
        robustness[str(snr)] = evaluate_scores(
            yte,
            _finite_scores(predict_p_fake(Xn)),
        )

    latency = measure_latency_ms(
        predict_fn,
        Xte[0],
        runs=cfg.latency_runs,
    )
    converged = (
        np.isfinite(clean.get("auc_roc", float("nan")))
        and clean["auc_roc"] >= cfg.converge_auc_threshold
        and clean.get("accuracy", 0.0) >= cfg.converge_accuracy_threshold
    )
    epochs = None
    if isinstance(history, dict):
        for values in history.values():
            if isinstance(values, list):
                epochs = len(values)
                break
    payload = {
        "status": "ok",
        "type": model_type,
        "input_shape": list(np.asarray(Xte).shape[1:]),
        "converged": bool(converged),
        "convergence_criteria": {
            "auc_roc_min": cfg.converge_auc_threshold,
            "accuracy_min": cfg.converge_accuracy_threshold,
        },
        "clean": clean,
        "scores_clean": [round(float(v), 6) for v in pf_clean],
        "robustness": robustness,
        "efficiency": {
            "params": params,
            "size_mb": file_size_mb(artifact),
            "latency_ms": latency,
        },
        "history": history,
        "training_config": training_config,
        "model_parameters": config_payload.get("parameters") or {},
        "final_training_metrics": metrics_payload.get("final_metrics") or {},
        "fit_strategy": (
            training_config.get("fit_strategy")
            if isinstance(training_config, dict)
            else None
        ),
        "model_artifact": str(artifact),
        "training_artifacts_dir": str(Path(cfg.output_dir) / "architectures" / _slug(arch)),
        "epochs": epochs,
        "wall_time_s": round(time.time() - started, 1),
    }
    del model
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
    return payload


def generate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = BenchmarkConfig.full_tcc(
        dataset_path=str(args.dataset),
        output_dir=str(args.out),
        models_dir=str(args.models_dir),
        batch_size=args.batch_size,
        latency_runs=args.latency_runs,
        snr_levels_db=args.snr,
        run_api_probe=False,
        device_profile="cpu",
    )
    data = BenchmarkData.from_npz(str(args.dataset))
    test_idx = _stratified_test_indices(data.y, cfg.seed)
    raw_test = _test_view(data, test_idx)

    per_arch: dict[str, Any] = {}
    y_test_base = np.asarray(raw_test.y, dtype=int)
    prepared_shapes: dict[str, list[int]] = {}
    for arch in ALL_TCC_ARCHITECTURES:
        artifact = _artifact_path(args.models_dir, arch)
        if artifact is None:
            per_arch[arch] = {
                "status": "pending",
                "error": "artefato treinado ainda nao encontrado",
            }
            continue
        LOG.info("Avaliando %s (%s)", arch, artifact)
        try:
            prepared = raw_test.prepare_for_architecture(arch)
            Xte = prepared.X
            yte = prepared.y
            prepared_shapes[arch] = list(np.asarray(Xte).shape[1:])
            per_arch[arch] = _evaluate_loaded_model(
                arch=arch,
                artifact=artifact,
                cfg=cfg,
                Xte=Xte,
                yte=yte,
                models_dir=args.models_dir,
            )
            del prepared, Xte, yte
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("%s falhou", arch)
            per_arch[arch] = {
                "status": "error",
                "error": str(exc),
            }

    labels, counts = np.unique(y_test_base, return_counts=True)
    balance = {str(int(label)): int(count) for label, count in zip(labels, counts)}

    results = {
        "status": "partial_completed_models",
        "generated_from": "existing_model_artifacts",
        "environment": _env_snapshot(),
        "config": cfg.to_dict(),
        "dataset": {
            "name": data.name,
            "n_total": int(len(data.y)),
            "n_test": int(len(y_test_base)),
            "balance_test": balance,
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "prepared_shapes": prepared_shapes,
            "y_test": [int(v) for v in np.asarray(y_test_base).ravel()],
            "source": str(args.dataset),
            "metadata": data.metadata or {},
        },
        "architectures": per_arch,
    }
    write_all(results, str(args.out))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera resultados/figuras para modelos de benchmark ja treinados."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "app" / "datasets" / "benchmark_audio_raw_balanced_15k.npz",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "app" / "models",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "benchmark_completed_partial",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-runs", type=int, default=10)
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args.dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    args.models_dir = (
        args.models_dir if args.models_dir.is_absolute() else ROOT / args.models_dir
    )
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.mkdir(parents=True, exist_ok=True)

    results = generate(args)
    ok = sum(1 for item in results["architectures"].values() if item.get("status") == "ok")
    pending = [
        name
        for name, item in results["architectures"].items()
        if item.get("status") == "pending"
    ]
    errors = [
        name
        for name, item in results["architectures"].items()
        if item.get("status") == "error"
    ]
    print(f"Modelos avaliados: {ok}")
    print(f"Pendentes: {', '.join(pending) if pending else '-'}")
    print(f"Erros: {', '.join(errors) if errors else '-'}")
    print(f"Saida: {args.out}")
    print(f"Resultados: {args.out / 'results.json'}")
    print(f"Figuras: {args.out / 'figures'}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
