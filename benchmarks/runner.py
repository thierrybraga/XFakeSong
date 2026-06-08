"""Orquestrador do benchmark: treina, avalia, mede robustez e eficiência."""

from __future__ import annotations

import logging
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from benchmarks.config import BenchmarkConfig
from benchmarks.data import BenchmarkData
from benchmarks.efficiency import count_params, file_size_mb, measure_latency_ms
from benchmarks.evaluate import evaluate_scores

logger = logging.getLogger("benchmark")

# Modelos clássicos (sklearn) — não passam pelo TrainingService (Keras).
CLASSICAL_ARCH_SLUGS = {"svm", "randomforest"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _compact_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_classical_arch(name: str) -> bool:
    return _compact_slug(name) in CLASSICAL_ARCH_SLUGS


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64").ravel()
    if np.any(~np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)


def _env_snapshot() -> Dict[str, Any]:
    snap = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
    }
    try:
        import tensorflow as tf

        snap["tensorflow"] = tf.__version__
        snap["gpu"] = bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        snap["tensorflow"] = None
        snap["gpu"] = False
    try:
        from app.core.gpu import describe_gpu_setup

        snap["device"] = describe_gpu_setup()
    except Exception:
        snap["device"] = "?"
    return snap


def _run_neural(arch: str, cfg: BenchmarkConfig, splits, tmp: Path):
    """Treina via TrainingService e devolve callables de inferência."""
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.training_service import TrainingService

    Xtr, ytr, Xv, yv, _Xte, _yte = splits
    name = f"bench_{_slug(arch)}"
    npz = tmp / "ds.npz"
    np.savez(npz, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)

    svc = TrainingService(models_dir=str(tmp))
    train_config = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "model_name": name,
    }
    compact = _compact_slug(arch)
    if compact == "rawnet2":
        train_config.update(
            {
                "learning_rate": 1e-5,
                "use_augmentation": False,
                "use_mixed_precision": False,
            }
        )
    elif compact in {"aasist", "rawgatst"}:
        train_config.update({"learning_rate": 1e-4})

    res = svc.train_model(
        architecture=arch,
        dataset_path=str(npz),
        config=train_config,
    )
    if res.status != ProcessingStatus.SUCCESS:
        raise RuntimeError(f"train_model falhou: {res.errors}")
    train_data = (res.data.metrics or {}) if res.data else {}
    history = train_data.get("history")
    model = (res.metadata or {}).get("model")
    if model is None:
        raise RuntimeError("TrainingService não retornou o modelo treinado em memória")

    def predict_p_fake(X: np.ndarray) -> np.ndarray:
        pred = model.predict(np.asarray(X, dtype="float32"), verbose=0)
        pred = np.asarray(pred, dtype="float64")
        if pred.ndim > 1 and pred.shape[-1] > 1:
            return pred[:, 1]
        return pred.reshape(-1)

    def predict_fn(xb):  # latência: forward puro do modelo
        return model.predict(np.asarray(xb, dtype="float32"), verbose=0)

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        "params": count_params(model),
        "size_mb": file_size_mb(tmp / f"{name}.keras"),
        "history": history,
        "training_config": train_data.get("training_config") or train_config,
        "final_metrics": train_data.get("final_metrics") or {},
        "model_artifact": str(tmp / f"{name}.keras"),
    }


def _run_classical(arch: str, cfg: BenchmarkConfig, splits, tmp: Path):
    """Treina um modelo clássico (SVM/RF) diretamente (sklearn)."""
    Xtr, ytr, _Xv, _yv, _Xte, _yte = splits
    n_features = int(np.asarray(Xtr).reshape(len(Xtr), -1).shape[1])

    if "svm" in arch.lower():
        from app.domain.models.architectures.svm import create_svm_model as factory
    else:
        from app.domain.models.architectures.random_forest import (
            create_random_forest_model as factory,
        )
    model = factory(input_shape=(n_features,), num_classes=2)
    model.fit(np.asarray(Xtr).reshape(len(Xtr), -1), np.asarray(ytr).ravel())

    name = f"bench_{_slug(arch)}"
    path = tmp / f"{name}.pkl"
    try:
        model.save(str(path))
    except Exception:
        path = None

    def predict_p_fake(X: np.ndarray) -> np.ndarray:
        X2 = np.asarray(X).reshape(len(X), -1)
        proba = model.predict_proba(X2)
        return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()

    def predict_fn(xb):
        return model.predict_proba(np.asarray(xb).reshape(len(xb), -1))

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        "params": None,
        "size_mb": file_size_mb(path) if path else None,
        "history": None,
        "training_config": {
            "model_family": "classical",
            "batch_size": None,
            "epochs": None,
            "fit_samples": int(len(ytr)),
            "n_features": n_features,
        },
        "final_metrics": {},
        "model_artifact": str(path) if path else None,
    }


def _benchmark_one(arch: str, cfg: BenchmarkConfig, splits) -> Dict[str, Any]:
    """Roda uma arquitetura ponta-a-ponta; nunca propaga exceção."""
    _Xtr, _ytr, _Xv, _yv, Xte, yte = splits
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="bench_") as td:
            tmp = Path(td)
            is_classical = _is_classical_arch(arch)
            r = (_run_classical if is_classical else _run_neural)(
                arch, cfg, splits, tmp
            )
            predict_p_fake: Callable = r["predict_p_fake"]

            # --- Limpo ---
            pf_clean = _finite_scores(predict_p_fake(Xte))
            clean = evaluate_scores(yte, pf_clean)
            converged = (
                not np.isnan(clean.get("auc_roc", float("nan")))
                and clean["auc_roc"] >= cfg.converge_auc_threshold
                and clean.get("accuracy", 0.0) >= cfg.converge_accuracy_threshold
            )

            # --- Robustez AWGN ---
            robustness: Dict[str, Any] = {}
            for snr in cfg.snr_levels_db:
                Xn = BenchmarkData.add_awgn(Xte, snr, seed=cfg.seed)
                robustness[str(snr)] = evaluate_scores(
                    yte,
                    _finite_scores(predict_p_fake(Xn)),
                )

            # --- Eficiência ---
            latency = measure_latency_ms(
                r["predict_fn"], Xte[0], runs=cfg.latency_runs
            )

            return {
                "status": "ok",
                "type": "classical" if is_classical else "neural",
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
                    "params": r["params"],
                    "size_mb": r["size_mb"],
                    "latency_ms": latency,
                },
                "history": r["history"],
                "training_config": r.get("training_config") or {},
                "final_training_metrics": r.get("final_metrics") or {},
                "model_artifact": r.get("model_artifact"),
                "epochs": cfg.epochs,
                "wall_time_s": round(time.time() - t0, 1),
            }
    except Exception as e:  # noqa: BLE001 — isola falha por arquitetura
        logger.warning("Arquitetura %s falhou: %s", arch, e)
        return {
            "status": "error",
            "error": str(e),
            "wall_time_s": round(time.time() - t0, 1),
        }


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    """Executa o benchmark completo e grava os artefatos. Retorna o dict total."""
    import matplotlib

    matplotlib.use("Agg")

    # GPU/threads configurados como no app (idempotente)
    try:
        from app.core.gpu import setup_gpu

        setup_gpu(log_level=logging.WARNING)
    except Exception:
        pass

    if cfg.dataset_path:
        data = BenchmarkData.from_npz(cfg.dataset_path)
    else:
        data = BenchmarkData.synthetic(
            cfg.synthetic_n, cfg.synthetic_shape, cfg.seed
        )
    data.validate()
    base_splits = data.stratified_split(cfg.seed)
    n_test = len(base_splits[5])
    logger.info(
        "Dataset '%s': %d amostras | teste held-out: %d", data.name,
        len(data.y), n_test,
    )

    per_arch: Dict[str, Any] = {}
    for arch in cfg.architectures:
        logger.info("=== Benchmark: %s ===", arch)
        prepared = data.prepare_for_architecture(arch)
        splits = prepared.stratified_split(cfg.seed)
        per_arch[arch] = _benchmark_one(arch, cfg, splits)
        per_arch[arch]["input_preparation"] = prepared.metadata or {}

    results: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "environment": _env_snapshot(),
        "dataset": {
            "name": data.name,
            "n_total": int(len(data.y)),
            "n_test": int(n_test),
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "metadata": data.metadata or {},
            "source": (data.metadata or {}).get("source"),
            "balance_test": {
                "real": int((base_splits[5] == 0).sum()),
                "fake": int((base_splits[5] == 1).sum()),
            },
            "y_test": [int(v) for v in base_splits[5]],
        },
        "architectures": per_arch,
    }

    if cfg.run_api_probe:
        try:
            from benchmarks.api_probe import run_api_probe

            results["api"] = run_api_probe()
        except Exception as e:  # noqa: BLE001
            results["api"] = {"status": "error", "error": str(e)}

    from benchmarks.report import write_all

    write_all(results, cfg.output_dir)
    return results
