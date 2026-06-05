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
CLASSICAL_ARCHS = {"SVM", "RandomForest", "Random Forest"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


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
    """Treina via TrainingService, carrega via ModelLoader, devolve callables."""
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.detection.model_loader import ModelLoader
    from app.domain.services.detection.predictor import Predictor
    from app.domain.services.training_service import TrainingService

    Xtr, ytr, Xv, yv, _Xte, _yte = splits
    name = f"bench_{_slug(arch)}"
    npz = tmp / "ds.npz"
    np.savez(npz, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)

    svc = TrainingService(models_dir=str(tmp))
    res = svc.train_model(
        architecture=arch,
        dataset_path=str(npz),
        config={"epochs": cfg.epochs, "batch_size": cfg.batch_size,
                "model_name": name},
    )
    if res.status != ProcessingStatus.SUCCESS:
        raise RuntimeError(f"train_model falhou: {res.errors}")
    history = (res.data.metrics or {}).get("history") if res.data else None

    loader = ModelLoader(models_dir=str(tmp))
    loader.load_available_models()
    mi = loader.get_model(name)
    if mi is None:
        raise RuntimeError("modelo treinado não reconhecido pelo ModelLoader")

    predictor = Predictor()

    def predict_p_fake(X: np.ndarray) -> np.ndarray:
        r = predictor.predict_batch(mi, [x for x in np.asarray(X)])
        if r.status != ProcessingStatus.SUCCESS:
            raise RuntimeError(f"predict_batch falhou: {r.errors}")
        return np.array([float(d["p_fake"]) for d in r.data])

    def predict_fn(xb):  # latência: forward puro do modelo
        return mi.model.predict(np.asarray(xb, dtype="float32"), verbose=0)

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        "params": count_params(mi.model),
        "size_mb": file_size_mb(tmp / f"{name}.keras"),
        "history": history,
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
    }


def _benchmark_one(arch: str, cfg: BenchmarkConfig, splits) -> Dict[str, Any]:
    """Roda uma arquitetura ponta-a-ponta; nunca propaga exceção."""
    _Xtr, _ytr, _Xv, _yv, Xte, yte = splits
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="bench_") as td:
            tmp = Path(td)
            is_classical = arch in CLASSICAL_ARCHS
            r = (_run_classical if is_classical else _run_neural)(
                arch, cfg, splits, tmp
            )
            predict_p_fake: Callable = r["predict_p_fake"]

            # --- Limpo ---
            pf_clean = predict_p_fake(Xte)
            clean = evaluate_scores(yte, pf_clean)
            converged = (
                not np.isnan(clean.get("auc_roc", float("nan")))
                and clean["auc_roc"] >= cfg.converge_auc_threshold
            )

            # --- Robustez AWGN ---
            robustness: Dict[str, Any] = {}
            for snr in cfg.snr_levels_db:
                Xn = BenchmarkData.add_awgn(Xte, snr, seed=cfg.seed)
                robustness[str(snr)] = evaluate_scores(yte, predict_p_fake(Xn))

            # --- Eficiência ---
            latency = measure_latency_ms(
                r["predict_fn"], Xte[0], runs=cfg.latency_runs
            )

            return {
                "status": "ok",
                "type": "classical" if is_classical else "neural",
                "converged": bool(converged),
                "clean": clean,
                "scores_clean": [round(float(v), 6) for v in pf_clean],
                "robustness": robustness,
                "efficiency": {
                    "params": r["params"],
                    "size_mb": r["size_mb"],
                    "latency_ms": latency,
                },
                "history": r["history"],
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
    splits = data.stratified_split(cfg.seed)
    n_test = len(splits[5])
    logger.info(
        "Dataset '%s': %d amostras | teste held-out: %d", data.name,
        len(data.y), n_test,
    )

    per_arch: Dict[str, Any] = {}
    for arch in cfg.architectures:
        logger.info("=== Benchmark: %s ===", arch)
        per_arch[arch] = _benchmark_one(arch, cfg, splits)

    results: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "environment": _env_snapshot(),
        "dataset": {
            "name": data.name,
            "n_total": int(len(data.y)),
            "n_test": int(n_test),
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "balance_test": {
                "real": int((splits[5] == 0).sum()),
                "fake": int((splits[5] == 1).sum()),
            },
            "y_test": [int(v) for v in splits[5]],
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
