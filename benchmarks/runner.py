"""Orquestrador do benchmark: treina, avalia, mede robustez e eficiência."""

from __future__ import annotations

import logging
import os
import platform
import re
import csv
import json
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
from benchmarks.planning import (
    apply_plan_to_config,
    build_benchmark_plan,
    write_benchmark_plan,
)

logger = logging.getLogger("benchmark")

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modelos clássicos (sklearn) — não passam pelo TrainingService (Keras).
CLASSICAL_ARCH_SLUGS = {"svm", "randomforest"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _compact_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_classical_arch(name: str) -> bool:
    return _compact_slug(name) in CLASSICAL_ARCH_SLUGS


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _normalize_project_paths(cfg: BenchmarkConfig) -> None:
    """Ancora caminhos relativos na raiz do projeto, não no cwd do processo."""
    cfg.output_dir = str(_project_path(cfg.output_dir))
    cfg.models_dir = str(_project_path(cfg.models_dir))
    if cfg.dataset_path:
        cfg.dataset_path = str(_project_path(cfg.dataset_path))


def _architecture_dir(cfg: BenchmarkConfig, arch: str) -> Path:
    return _project_path(cfg.output_dir) / "architectures" / _slug(arch)


def _models_dir(cfg: BenchmarkConfig, arch: str) -> Path:
    explicit = (
        os.getenv("MODELS_DIR")
        or os.getenv("DEEPFAKE_MODELS_DIR")
        or os.getenv("XFAKE_MODELS_DIR")
    )
    if explicit:
        return _project_path(explicit)
    storage = os.getenv("XFAKE_STORAGE_DIR") or os.getenv("DEEPFAKE_STORAGE_DIR")
    if storage:
        return _project_path(storage) / "models"
    return _project_path(cfg.models_dir)


def _classical_search_space(arch: str, seed: int) -> tuple[dict[str, list], Any, str]:
    compact = _compact_slug(arch)
    if compact == "svm":
        from sklearn.svm import SVC

        return (
            {
                "svm__kernel": ["linear", "rbf"],
                "svm__C": [0.1, 1.0, 10.0],
                "svm__gamma": ["scale", "auto"],
            },
            SVC(probability=True, random_state=seed),
            "svm",
        )

    from sklearn.ensemble import RandomForestClassifier

    return (
        {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_leaf": [1, 2],
            "rf__max_features": ["sqrt", "log2"],
        },
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        "rf",
    )


def _run_classical_tuning(
    arch: str,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Otimiza SVM/RF e grava o histórico do grid search."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    values, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min()) if len(values) >= 2 else 0
    cv = min(3, min_class_count)
    grid, estimator, step_name = _classical_search_space(arch, seed)
    plan = {
        "enabled": True,
        "method": "GridSearchCV",
        "scoring": "roc_auc",
        "cv": cv,
        "param_grid": grid,
        "n_candidates": int(np.prod([len(v) for v in grid.values()])),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if cv < 2:
        plan.update(
            {
                "status": "skipped",
                "reason": "amostras insuficientes por classe para validação cruzada",
            }
        )
        (output_dir / "hyperparameter_tuning.json").write_text(
            json.dumps(_json_safe(plan), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return plan

    pipeline = Pipeline([("scaler", StandardScaler()), (step_name, estimator)])
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=False,
        return_train_score=True,
        verbose=0,
    )
    started = time.time()
    search.fit(X, y)
    elapsed = round(time.time() - started, 3)

    rows = []
    results = search.cv_results_
    for idx, params in enumerate(results["params"]):
        rows.append(
            {
                "rank": int(results["rank_test_score"][idx]),
                "mean_test_score": float(results["mean_test_score"][idx]),
                "std_test_score": float(results["std_test_score"][idx]),
                "mean_train_score": float(results.get("mean_train_score", [np.nan])[idx]),
                "params_json": json.dumps(_json_safe(params), ensure_ascii=False),
            }
        )
    rows.sort(key=lambda row: row["rank"])
    with (output_dir / "hyperparameter_tuning.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
                "params_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_model_params = {
        k.replace(f"{step_name}__", ""): v
        for k, v in search.best_params_.items()
        if k.startswith(f"{step_name}__")
    }
    plan.update(
        {
            "status": "ok",
            "elapsed_s": elapsed,
            "best_score": float(search.best_score_),
            "best_params": _json_safe(search.best_params_),
            "best_model_params": _json_safe(best_model_params),
            "top_candidates": rows[:5],
        }
    )
    (output_dir / "hyperparameter_tuning.json").write_text(
        json.dumps(_json_safe(plan), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return plan


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


def _stratified_test_labels(
    y: np.ndarray,
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> np.ndarray:
    """Retorna apenas os labels do teste, sem copiar o tensor X completo."""
    y = np.asarray(y)
    try:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(y))
        _train_idx, temp_idx = train_test_split(
            idx,
            test_size=val_frac + test_frac,
            stratify=y,
            random_state=seed,
        )
        rel_test = test_frac / (val_frac + test_frac)
        _val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            stratify=y[temp_idx],
            random_state=seed,
        )
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_test = max(1, int(len(idx) * test_frac))
        test_idx = idx[:n_test]
    return y[test_idx]


def _run_neural(
    arch: str,
    cfg: BenchmarkConfig,
    splits,
    tmp: Path,
    models_dir: Path,
):
    """Treina via TrainingService e devolve callables de inferência."""
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.training_service import TrainingService

    Xtr, ytr, Xv, yv, _Xte, _yte = splits
    name = f"bench_{_slug(arch)}"
    npz = tmp / "ds.npz"
    np.savez(npz, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)

    models_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = _architecture_dir(cfg, arch)
    svc = TrainingService(models_dir=str(models_dir))
    train_config = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "model_name": name,
        "verbose": 0,
        "progress_log_interval": 1,
        "progress_label": arch,
    }
    train_config.update(cfg.training_overrides.get(arch, {}))
    train_config["model_name"] = name
    train_config["verbose"] = 0
    compact = _compact_slug(arch)
    if compact == "rawnet2":
        # P2 — RawNet2 colapsava a ~50% sob ruído por treinar SEM augmentation.
        # Com o ruído agora calibrado por SNR (+ RawBoost/codec para raw-audio),
        # habilitar augmentation é o conserto direto da fragilidade a ruído.
        train_config.update(
            {
                "learning_rate": 1e-4,
                "use_augmentation": True,
                "use_mixed_precision": False,
            }
        )
    elif compact in {"aasist", "rawgatst"}:
        # AJUSTE (retune): antes forçava learning_rate=1e-4 e
        # use_augmentation=False aqui, sobrescrevendo os ajustes já feitos em
        # aasist.py/rawgat_st.py/registry.py (ver docs/RETREINO_AJUSTES.md) e
        # anulando o efeito do retreino de 2026-06-30. Agora só liga
        # augmentation e deixa o LR fluir de cfg.training_overrides
        # (benchmarks/planning.py, já sincronizado com os defaults tunados).
        train_config.update({"use_augmentation": True})
    elif compact == "efficientnetlstm":
        model_params = train_config.setdefault("parameters", {})
        if "dropout_rate" in train_config:
            model_params.setdefault("dropout_rate", float(train_config["dropout_rate"]))
        if "lstm_units" in train_config:
            model_params.setdefault("lstm_units", int(train_config["lstm_units"]))
        elif "hidden_units" in train_config:
            first_units = str(train_config["hidden_units"]).split("/", 1)[0]
            try:
                model_params.setdefault("lstm_units", int(first_units))
            except ValueError:
                pass
        if "pretrained" in train_config:
            model_params.setdefault("pretrained", bool(train_config["pretrained"]))
        train_config.update({"learning_rate": 1e-4})
    elif compact in {"hybridcnntransformer", "conformer", "spectrogramtransformer"}:
        train_config.update({"reduce_lr_on_plateau": False})
        if compact in {"conformer", "spectrogramtransformer"}:
            model_params = train_config.setdefault("parameters", {})
            for key in (
                "learning_rate",
                "weight_decay",
                "warmup_steps",
                "decay_steps",
                "alpha",
                "dropout_rate",
            ):
                if key in train_config:
                    model_params.setdefault(key, train_config[key])
            train_config.pop("learning_rate", None)
        if compact == "conformer":
            model_params = train_config.setdefault("parameters", {})
            for key in ("clipnorm", "label_smoothing"):
                if key in train_config:
                    model_params.setdefault(key, train_config[key])
        elif compact == "spectrogramtransformer":
            # P1 — retreino obrigatório. Liga augmentation (ruído SNR + SpecAug)
            # e força a restauração do MELHOR checkpoint (val_loss), atacando o
            # colapso val→teste por sobreajuste. Hiperparâmetros mais
            # regularizados vêm dos defaults da arquitetura (dropout 0.3,
            # weight_decay 1e-4, lr de pico 5e-5).
            train_config.update(
                {"use_augmentation": True, "checkpoint_best": True}
            )

    checkpoint_path = None
    if bool(train_config.pop("checkpoint_best", False)):
        checkpoint_dir = arch_dir / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_checkpoint.keras"
        train_config["checkpoint_path"] = str(checkpoint_path)

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

    model_path = models_dir / f"{name}.keras"
    reported_training_config = dict(train_data.get("training_config") or train_config)
    model_parameters = dict(train_config.get("parameters") or {})
    for key in (
        "epochs",
        "batch_size",
        "use_augmentation",
        "use_mixed_precision",
        "reduce_lr_on_plateau",
        "checkpoint_path",
    ):
        if key in train_config and reported_training_config.get(key) is None:
            reported_training_config[key] = train_config[key]
    if checkpoint_path is not None:
        reported_training_config["checkpoint_best"] = True
        reported_training_config["best_checkpoint_path"] = str(checkpoint_path)
    if model_parameters:
        if "learning_rate" in model_parameters:
            reported_training_config["learning_rate"] = model_parameters["learning_rate"]
        reported_training_config["model_parameters"] = model_parameters

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        "params": count_params(model),
        "size_mb": file_size_mb(model_path),
        "history": history,
        "training_config": reported_training_config,
        "model_parameters": model_parameters,
        "final_metrics": train_data.get("final_metrics") or {},
        "model_artifact": str(model_path),
    }


def _run_classical(
    arch: str,
    cfg: BenchmarkConfig,
    splits,
    tmp: Path,
    models_dir: Path,
):
    """Treina um modelo clássico (SVM/RF) diretamente (sklearn)."""
    Xtr, ytr, Xv, yv, _Xte, _yte = splits
    n_features = int(np.asarray(Xtr).reshape(len(Xtr), -1).shape[1])

    if "svm" in arch.lower():
        from app.domain.models.architectures.svm import create_svm_model as factory
    else:
        from app.domain.models.architectures.random_forest import (
            create_random_forest_model as factory,
        )
    if os.getenv("XFAKE_BENCHMARK_VERBOSE", "0") != "1":
        for logger_name in (
            "app.domain.models.architectures.svm",
            "app.domain.models.architectures.random_forest",
            "app.domain.models.architectures.classical_ml_helpers",
        ):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.disabled = True
    name = f"bench_{_slug(arch)}"
    models_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = _architecture_dir(cfg, arch)

    X_train_2d = np.asarray(Xtr).reshape(len(Xtr), -1)
    y_train = np.asarray(ytr).ravel()
    X_fit_2d = X_train_2d
    y_fit = y_train
    if len(yv):
        X_fit_2d = np.concatenate(
            [X_train_2d, np.asarray(Xv).reshape(len(Xv), -1)],
            axis=0,
        )
        y_fit = np.concatenate([y_train, np.asarray(yv).ravel()], axis=0)

    # P2 — augmentation ruidoso (SVM/RF): anexa cópias do treino com AWGN nos
    # mesmos SNRs avaliados. Feito no espaço de feature em que a robustez é
    # medida (BenchmarkData.add_awgn opera sobre o vetor de features tabular),
    # então o treino passa a ver a degradação que antes só aparecia no teste.
    aug_snrs = []
    if getattr(cfg, "classical_noise_augmentation", True):
        aug_snrs = list(cfg.snr_levels_db)
    if aug_snrs:
        extra_X = [BenchmarkData.add_awgn(X_fit_2d, snr, seed=cfg.seed + i)
                   for i, snr in enumerate(aug_snrs)]
        X_fit_2d = np.concatenate([X_fit_2d, *extra_X], axis=0)
        y_fit = np.concatenate([y_fit] * (1 + len(aug_snrs)), axis=0)
        logging.getLogger("benchmark").info(
            "[%s] augmentation clássico: +%d cópias ruidosas (SNRs=%s) → %d amostras",
            arch, len(aug_snrs), aug_snrs, len(y_fit),
        )

    tuning = {"enabled": False, "status": "disabled"}
    model_kwargs: dict[str, Any] = {}
    if cfg.optimize_hyperparameters:
        tuning = _run_classical_tuning(
            arch=arch,
            X=X_train_2d,
            y=y_train,
            output_dir=arch_dir,
            seed=cfg.seed,
        )
        if tuning.get("status") == "ok":
            model_kwargs.update(tuning.get("best_model_params") or {})

    model = factory(input_shape=(n_features,), num_classes=2, **model_kwargs)
    model.fit(X_fit_2d, y_fit)

    path = models_dir / f"{name}.pkl"
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

    fit_strategy = {
        "kind": "single_fit",
        "estimator": "sklearn",
        "fit_samples": int(len(y_fit)),
        "n_features": n_features,
    }
    if tuning.get("enabled"):
        fit_strategy.update(
            {
                "kind": "grid_search_cv_then_refit",
                "cv": tuning.get("cv"),
                "n_candidates": tuning.get("n_candidates"),
                "n_fits": (
                    int(tuning.get("cv", 0)) * int(tuning.get("n_candidates", 0))
                    if tuning.get("cv") and tuning.get("n_candidates")
                    else None
                ),
                "final_refit": True,
                "scoring": tuning.get("scoring"),
            }
        )
        if fit_strategy.get("n_fits"):
            fit_strategy["total_fit_calls_estimate"] = int(fit_strategy["n_fits"]) + 1

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        "params": None,
        "size_mb": file_size_mb(path) if path else None,
        "history": None,
        "epochs": None,
        "fit_strategy": fit_strategy,
        "training_config": {
            "model_family": "classical",
            "batch_size": None,
            "epochs": None,
            "fit_strategy": fit_strategy,
            "fit_samples": int(len(y_fit)),
            "n_features": n_features,
            "hyperparameter_tuning": tuning,
            "best_hyperparameters": tuning.get("best_model_params") or model_kwargs,
        },
        "final_metrics": {
            "fit_samples": int(len(y_fit)),
            "n_features": n_features,
            "classes": [int(v) for v in np.unique(y_fit).tolist()],
            "hyperparameter_tuning_status": tuning.get("status"),
            "hyperparameter_tuning_best_score": tuning.get("best_score"),
            "hyperparameter_tuning_best_params": tuning.get("best_model_params"),
        },
        "model_artifact": str(path) if path else None,
        "hyperparameter_tuning": tuning,
    }


def _benchmark_one(arch: str, cfg: BenchmarkConfig, splits) -> Dict[str, Any]:
    """Roda uma arquitetura ponta-a-ponta; nunca propaga exceção."""
    _Xtr, _ytr, _Xv, _yv, Xte, yte = splits
    t0 = time.time()
    models_dir = _models_dir(cfg, arch)
    try:
        with tempfile.TemporaryDirectory(prefix="bench_") as td:
            tmp = Path(td)
            is_classical = _is_classical_arch(arch)
            r = (_run_classical if is_classical else _run_neural)(
                arch, cfg, splits, tmp, models_dir
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

            training_config = dict(r.get("training_config") or {})
            history = r.get("history")
            effective_epochs = r.get("epochs")
            if history:
                for values in history.values():
                    if isinstance(values, (list, tuple)):
                        effective_epochs = len(values)
                        training_config.setdefault(
                            "epochs_budget", training_config.get("epochs")
                        )
                        training_config["epochs_executed"] = effective_epochs
                        break
            if effective_epochs is None:
                effective_epochs = training_config.get("epochs")
            if effective_epochs is None and not is_classical:
                effective_epochs = cfg.epochs

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
                "history": history,
                "training_config": training_config,
                "model_parameters": r.get("model_parameters") or {},
                "final_training_metrics": r.get("final_metrics") or {},
                "fit_strategy": r.get("fit_strategy"),
                "model_artifact": r.get("model_artifact"),
                "training_artifacts_dir": str(_architecture_dir(cfg, arch)),
                "epochs": int(effective_epochs) if effective_epochs is not None else None,
                "wall_time_s": round(time.time() - t0, 1),
            }
    except Exception as e:  # noqa: BLE001 — isola falha por arquitetura
        logger.warning("Arquitetura %s falhou: %s", arch, e)
        return {
            "status": "error",
            "error": str(e),
            "wall_time_s": round(time.time() - t0, 1),
        }


def _load_and_validate_data(cfg: BenchmarkConfig) -> BenchmarkData:
    if cfg.dataset_path:
        data = BenchmarkData.from_npz(cfg.dataset_path)
    else:
        data = BenchmarkData.synthetic(
            cfg.synthetic_n, cfg.synthetic_shape, cfg.seed
        )
    data.validate()
    return data


def plan_benchmark(cfg: BenchmarkConfig, write: bool = True) -> Dict[str, Any]:
    """Valida dataset/configuração e grava o plano antes de treinar."""
    _normalize_project_paths(cfg)
    data = _load_and_validate_data(cfg)
    plan = build_benchmark_plan(cfg, data)
    if write:
        write_benchmark_plan(plan, cfg.output_dir)
    return plan


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    """Executa o benchmark completo e grava os artefatos. Retorna o dict total."""
    import matplotlib

    matplotlib.use("Agg")
    _normalize_project_paths(cfg)

    # GPU/threads configurados como no app (idempotente)
    try:
        from app.core.gpu import setup_gpu

        setup_gpu(log_level=logging.WARNING)
    except Exception:
        pass

    data = _load_and_validate_data(cfg)
    plan = build_benchmark_plan(cfg, data)
    write_benchmark_plan(plan, cfg.output_dir)
    apply_plan_to_config(cfg, plan)

    # Rótulos do teste held-out para o resumo do dataset. Sob split por
    # grupo/cross-generator, reproduz a MESMA partição (label-only, sem copiar
    # o X grande) para que balance_test/y_test reflitam o protocolo real.
    _protocol_split = (
        cfg.group_split or cfg.holdout_generator
        or cfg.speaker_split or cfg.holdout_speaker
    )
    if _protocol_split and (data.groups is not None or data.speakers is not None):
        label_view = BenchmarkData(
            X=np.zeros((len(data.y), 1), dtype="float32"),
            y=data.y, name=data.name, groups=data.groups, speakers=data.speakers,
        )
        _, _, _, _, _, y_test_base = label_view.stratified_split(
            cfg.seed,
            group_split=cfg.group_split,
            holdout_generator=cfg.holdout_generator,
            speaker_split=cfg.speaker_split,
            holdout_speaker=cfg.holdout_speaker,
        )
    else:
        y_test_base = _stratified_test_labels(data.y, cfg.seed)
    n_test = len(y_test_base)
    logger.info(
        "Dataset '%s': %d amostras | teste held-out: %d", data.name,
        len(data.y), n_test,
    )

    per_arch: Dict[str, Any] = {}
    for arch in cfg.architectures:
        logger.info("=== Benchmark: %s ===", arch)
        prepared = data.prepare_for_architecture(arch)
        splits = prepared.stratified_split(
            cfg.seed,
            group_split=cfg.group_split,
            holdout_generator=cfg.holdout_generator,
            speaker_split=cfg.speaker_split,
            holdout_speaker=cfg.holdout_speaker,
        )
        per_arch[arch] = _benchmark_one(arch, cfg, splits)
        per_arch[arch]["input_preparation"] = prepared.metadata or {}

    results: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "preflight": plan,
        "environment": _env_snapshot(),
        "dataset": {
            "name": data.name,
            "n_total": int(len(data.y)),
            "n_test": int(n_test),
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "metadata": data.metadata or {},
            "source": (data.metadata or {}).get("source"),
            "balance_test": {
                "real": int((y_test_base == 0).sum()),
                "fake": int((y_test_base == 1).sum()),
            },
            "y_test": [int(v) for v in y_test_base],
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
