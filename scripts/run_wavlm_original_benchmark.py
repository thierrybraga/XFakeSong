#!/usr/bin/env python3
"""Benchmark de SSL original (PyTorch/Hugging Face), sem fallback TensorFlow.

Este runner cobre WavLM e HuBERT reais via PyTorch. O backbone Hugging Face e
carregado congelado por padrao, e a cabeca de classificacao e treinada sobre
embeddings SSL reais.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logger = logging.getLogger("ssl_original_benchmark")


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


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64").ravel()
    if np.any(~np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _fit_length(flat: np.ndarray, target_len: int) -> np.ndarray:
    if flat.shape[1] == target_len:
        return flat
    if flat.shape[1] > target_len:
        start = max(0, (flat.shape[1] - target_len) // 2)
        return flat[:, start : start + target_len]
    repeats = int(np.ceil(target_len / max(1, flat.shape[1])))
    return np.tile(flat, (1, repeats))[:, :target_len]


class _LoadedData:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str,
        metadata: dict[str, Any],
        original_shape: list[int],
    ):
        self.X = X
        self.y = y
        self.name = name
        self.metadata = metadata
        self.original_shape = original_shape


def _load_dataset(path: str, seed: int):
    from sklearn.model_selection import train_test_split

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {path}")
    raw = np.load(p, allow_pickle=False)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xk, yk in (("X_train", "y_train"), ("X_val", "y_val"),
                   ("X_test", "y_test"), ("X", "y")):
        if xk in raw and yk in raw:
            xs.append(np.asarray(raw[xk], dtype="float32"))
            ys.append(np.asarray(raw[yk]))
    if not xs:
        raise ValueError(
            f"{path}: esperado X_train/y_train ou X/y; chaves={list(raw.keys())}"
        )

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if y.ndim > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    y = y.ravel().astype("int64")
    original_shape = list(X.shape[1:])
    flat = X.reshape(len(X), -1).astype("float32")
    X = _fit_length(flat, 16000)[..., np.newaxis].astype("float32")
    labels = set(np.unique(y).astype(int).tolist())
    if labels != {0, 1}:
        raise ValueError(f"Labels esperados {{0,1}}, encontrados {sorted(labels)}")

    metadata: dict[str, Any] = {"source": str(p), "npz_path": str(p)}
    if "metadata_json" in raw:
        try:
            meta_raw = raw["metadata_json"]
            if hasattr(meta_raw, "item"):
                meta_raw = meta_raw.item()
            metadata.update(json.loads(str(meta_raw)))
        except Exception:
            metadata["metadata_parse_error"] = True

    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed
    )
    data = _LoadedData(X, y, p.stem, metadata, original_shape)
    return data, (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def _normalize_wave_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    flat = x.reshape(len(x), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    flat = (flat - mean) / np.maximum(std, 1e-6)
    return np.clip(flat, -5.0, 5.0).astype("float32")


def _add_awgn_raw(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype="float32")
    flat = X.reshape(len(X), -1)
    sig_power = np.mean(flat ** 2, axis=1, keepdims=True)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    noise_std = np.sqrt(sig_power / max(snr_lin, 1e-12))
    noise = rng.standard_normal(flat.shape).astype("float32") * noise_std
    return (flat + noise).reshape(X.shape).astype("float32")


def _count_torch_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _path_size_mb(path: Path) -> float:
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / (1024 * 1024), 2)


def _write_predictions(path: Path, y_true: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "y_true", "p_fake"])
        writer.writeheader()
        for idx, (yt, pf) in enumerate(zip(y_true, scores)):
            writer.writerow(
                {"idx": idx, "y_true": int(yt), "p_fake": float(pf)}
            )


def _write_robustness(path: Path, robustness: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["snr_db", "accuracy", "precision", "recall", "f1", "auc_roc", "eer"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for snr, metrics in robustness.items():
            row = {"snr_db": snr}
            row.update({k: metrics.get(k) for k in fields if k != "snr_db"})
            writer.writerow(row)


def _write_model_card(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['display_name']}",
        "",
        f"Modelo treinado com backbone Hugging Face `{payload['model_class']}` real.",
        "",
        f"- Backbone: `{payload['model_name']}`",
        f"- Backbone congelado: `{payload['freeze_backbone']}`",
        f"- Epocas da cabeca: `{payload['epochs']}`",
        f"- Batch embeddings: `{payload['feature_batch_size']}`",
        f"- Batch treino: `{payload['train_batch_size']}`",
        f"- Artefato: `{payload['artifact']}`",
        f"- Backbone local: `{payload['backbone_artifact']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_scores(
    y_true: np.ndarray, p_fake: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).ravel().astype(int)
    p_fake = _finite_scores(p_fake)
    y_pred = (p_fake >= threshold).astype(int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "nonfinite_scores": int((~np.isfinite(np.asarray(p_fake))).sum()),
    }
    if n_pos > 0 and n_neg > 0:
        out["auc_roc"] = float(roc_auc_score(y_true, p_fake))
        fpr, tpr, thresholds = roc_curve(y_true, p_fake)
        fnr = 1.0 - tpr
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[idx] + fnr[idx]) / 2.0)
        out["eer_threshold"] = float(thresholds[idx])
        p_target = 0.01
        risks = p_target * fnr + (1.0 - p_target) * fpr
        out["min_tdcf"] = float(np.nanmin(risks) / max(p_target, 1e-12))
    else:
        out["auc_roc"] = float("nan")
        out["eer"] = float("nan")
        out["eer_threshold"] = float("nan")
        out["min_tdcf"] = float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="app/datasets/benchmark_audio_raw_balanced_20k.npz",
        help="Dataset .npz balanceado com audio bruto.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Diretorio de saida.",
    )
    parser.add_argument(
        "--architecture",
        choices=["wavlm", "hubert"],
        default="wavlm",
        help="Backbone SSL original a executar.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    parser.add_argument("--latency-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Congela o backbone original e treina apenas a cabeca.",
    )
    args = parser.parse_args()
    arch_meta = {
        "wavlm": {
            "display": "WavLM Original",
            "compact": "wavlm_original",
            "model_class": "WavLMModel",
            "default_model": "microsoft/wavlm-base",
            "default_out": "results/benchmark_wavlm_original_gpu_100e",
            "artifact": "bench_wavlm_original.pt",
        },
        "hubert": {
            "display": "HuBERT Original",
            "compact": "hubert_original",
            "model_class": "HubertModel",
            "default_model": "facebook/hubert-base-ls960",
            "default_out": "results/benchmark_hubert_original_gpu_100e",
            "artifact": "bench_hubert_original.pt",
        },
    }[args.architecture]
    if args.model_name is None:
        args.model_name = arch_meta["default_model"]
    if args.out is None:
        args.out = arch_meta["default_out"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _set_seed(args.seed)

    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import HubertModel, WavLMModel

    from benchmarks.report import write_all
    BackboneModel = WavLMModel if args.architecture == "wavlm" else HubertModel

    out = Path(args.out)
    arch_out = out / "architectures" / arch_meta["compact"]
    models_dir = arch_out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device PyTorch: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Carregando dataset: %s", args.dataset)
    data, splits = _load_dataset(args.dataset, args.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    X_train = _normalize_wave_batch(X_train)
    X_val = _normalize_wave_batch(X_val)
    X_test = _normalize_wave_batch(X_test)

    logger.info("Carregando %s: %s", arch_meta["display"], args.model_name)
    backbone = BackboneModel.from_pretrained(args.model_name).to(device)
    backbone.train(not args.freeze_backbone)
    for param in backbone.parameters():
        param.requires_grad = not args.freeze_backbone

    hidden_size = int(backbone.config.hidden_size)
    classifier = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(256, 2),
    ).to(device)

    def embed(X: np.ndarray, label: str) -> np.ndarray:
        backbone.eval()
        Xn = _normalize_wave_batch(X)
        ds = TensorDataset(torch.from_numpy(Xn))
        loader = DataLoader(ds, batch_size=args.feature_batch_size, shuffle=False)
        chunks = []
        started = time.time()
        with torch.no_grad():
            for step, (xb,) in enumerate(loader, start=1):
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast(
                    "cuda", enabled=(device.type == "cuda"), dtype=torch.float16
                ):
                    outp = backbone(xb)
                    pooled = outp.last_hidden_state.mean(dim=1)
                chunks.append(pooled.detach().float().cpu().numpy())
                if step == 1 or step % 50 == 0 or step == len(loader):
                    logger.info(
                        "Embeddings %s: %d/%d batches", label, step, len(loader)
                    )
        logger.info(
            "Embeddings %s concluidos em %.1fs", label, time.time() - started
        )
        return np.concatenate(chunks, axis=0).astype("float32")

    Z_train = embed(X_train, "train")
    Z_val = embed(X_val, "val")
    Z_test = embed(X_test, "test")

    train_ds = TensorDataset(
        torch.from_numpy(Z_train), torch.from_numpy(y_train.astype("int64"))
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=False
    )
    val_x = torch.from_numpy(Z_val).to(device)
    val_y = torch.from_numpy(y_val.astype("int64")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    logger.info("Treinando cabeca %s por %d epocas", arch_meta["display"], args.epochs)
    started_train = time.time()
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        losses = []
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().detach().cpu())
            total += int(yb.numel())

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_x)
            val_loss = float(criterion(val_logits, val_y).detach().cpu())
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == val_y).float().mean().detach().cpu())

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = float(correct / max(total, 1))
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoca %03d/%03d - loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    train_time_s = round(time.time() - started_train, 3)

    def predict_scores_from_embeddings(Z: np.ndarray) -> np.ndarray:
        classifier.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Z.astype("float32"))),
            batch_size=args.train_batch_size,
            shuffle=False,
        )
        scores = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = classifier(xb.to(device, non_blocking=True))
                scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        return _finite_scores(np.concatenate(scores))

    scores_clean = predict_scores_from_embeddings(Z_test)
    clean = _evaluate_scores(y_test, scores_clean)

    robustness: dict[str, dict[str, float]] = {}
    for snr in args.snr:
        noisy = _add_awgn_raw(X_test, snr, seed=args.seed + int(snr))
        Z_noisy = embed(noisy, f"snr_{snr}")
        robustness[str(snr)] = _evaluate_scores(
            y_test, predict_scores_from_embeddings(Z_noisy)
        )

    test_tensor = torch.from_numpy(Z_test).to(device)
    y_test_tensor = torch.from_numpy(y_test.astype("int64")).to(device)
    with torch.no_grad():
        logits_test = classifier(test_tensor)
        y_pred = logits_test.argmax(dim=1).detach().cpu().numpy()
        test_loss = float(criterion(logits_test, y_test_tensor).detach().cpu())

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [float(v) for v in cm.ravel()]
    final_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores_clean)),
        "pr_auc": float(average_precision_score(y_test, scores_clean)),
        "eer": clean.get("eer"),
        "eer_threshold": clean.get("eer_threshold"),
        "min_tdcf": clean.get("min_tdcf"),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "specificity": float(tn / max(tn + fp, 1.0)),
        "sensitivity": float(tp / max(tp + fn, 1.0)),
        "false_positive_rate": float(fp / max(fp + tn, 1.0)),
        "false_negative_rate": float(fn / max(fn + tp, 1.0)),
        "total_samples": float(len(y_test)),
        "test_loss": test_loss,
    }

    artifact = models_dir / arch_meta["artifact"]
    backbone_dir = models_dir / f"{args.architecture}_backbone"
    backbone.save_pretrained(backbone_dir, safe_serialization=False)
    torch.save(
        {
            "model_name": args.model_name,
            "architecture": arch_meta["display"],
            "model_class": arch_meta["model_class"],
            "backbone_config": backbone.config.to_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "freeze_backbone": args.freeze_backbone,
            "hidden_size": hidden_size,
            "labels": {"real": 0, "fake": 1},
            "input_shape": [16000, 1],
            "history": history,
            "clean_metrics": clean,
            "training_config": vars(args),
        },
        artifact,
    )
    config_payload = {
        "architecture": arch_meta["display"].replace(" ", ""),
        "display_name": arch_meta["display"],
        "model_class": arch_meta["model_class"],
        "model_name": args.model_name,
        "artifact": str(artifact),
        "backbone_artifact": str(backbone_dir),
        "input_shape": [16000, 1],
        "freeze_backbone": args.freeze_backbone,
        "epochs": args.epochs,
        "feature_batch_size": args.feature_batch_size,
        "train_batch_size": args.train_batch_size,
    }
    (models_dir / f"bench_{arch_meta['compact']}_config.json").write_text(
        json.dumps(_json_safe(config_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_model_card(models_dir / "README.md", config_payload)

    latency_ms = None
    if args.latency_runs > 0:
        x_latency = torch.from_numpy(X_test[:1]).to(device)
        backbone.eval()
        classifier.eval()
        for _ in range(2):
            with torch.no_grad():
                pooled = backbone(x_latency).last_hidden_state.mean(dim=1)
                _ = classifier(pooled)
        times = []
        for _ in range(args.latency_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                pooled = backbone(x_latency).last_hidden_state.mean(dim=1)
                _ = classifier(pooled)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        latency_ms = round(float(np.median(times)), 2)

    size_mb = round(_path_size_mb(artifact) + _path_size_mb(backbone_dir), 2)
    params = _count_torch_params(backbone) + _count_torch_params(classifier)
    converged = bool(
        clean.get("auc_roc", 0.0) >= 0.70 and clean.get("accuracy", 0.0) >= 0.55
    )

    results = {
        "config": {
            "preset_name": f"single:{arch_meta['display'].replace(' ', '')}",
            "architectures": [arch_meta["display"]],
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "snr_levels_db": args.snr,
            "seed": args.seed,
            "device_profile": "gpu" if device.type == "cuda" else "cpu",
            "model_name": args.model_name,
            "runner": "scripts/run_wavlm_original_benchmark.py",
        },
        "preflight": {
            "status": "ok",
            "requires": ["torch", "transformers", arch_meta["model_class"]],
            "fallback_used": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "dataset": {
            "name": data.name,
            "source": args.dataset,
            "n_total": int(len(data.y)),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "input_shape": data.original_shape,
            "prepared_shape": [16000, 1],
            "balance_test": {
                "real": int((y_test == 0).sum()),
                "fake": int((y_test == 1).sum()),
            },
            "y_test": y_test.astype(int).tolist(),
            "metadata": _json_safe(data.metadata or {}),
        },
        "architectures": {
            arch_meta["display"]: {
                "status": "ok",
                "type": "neural",
                "input_shape": [16000, 1],
                "converged": converged,
                "convergence_criteria": {
                    "accuracy_min": 0.55,
                    "auc_roc_min": 0.70,
                },
                "clean": clean,
                "scores_clean": scores_clean.tolist(),
                "robustness": robustness,
                "efficiency": {
                    "params": params,
                    "size_mb": size_mb,
                    "latency_ms": latency_ms,
                },
                "history": history,
                "training_config": {
                    "model_family": "pytorch_transformers",
                    "model_name": args.model_name,
                    "architecture": arch_meta["display"],
                    "model_class": arch_meta["model_class"],
                    "freeze_backbone": args.freeze_backbone,
                    "epochs": args.epochs,
                    "feature_batch_size": args.feature_batch_size,
                    "train_batch_size": args.train_batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "fallback_used": False,
                },
                "model_parameters": {
                    "backbone_params": _count_torch_params(backbone),
                    "classifier_params": _count_torch_params(classifier),
                    "hidden_size": hidden_size,
                },
                "final_training_metrics": final_metrics,
                "fit_strategy": {
                    "kind": "frozen_backbone_embedding_then_classifier_fit",
                    "backbone": args.model_name,
                    "backbone_trainable": not args.freeze_backbone,
                    "train_time_s": train_time_s,
                },
                "model_artifact": str(artifact),
                "backbone_artifact": str(backbone_dir),
                "training_artifacts_dir": str(arch_out),
                "epochs": args.epochs,
                "wall_time_s": None,
                "input_preparation": {
                    "input_type": "raw_audio",
                    "original_shape": data.original_shape,
                    "prepared_shape": [16000, 1],
                    "sample_rate": 16000,
                    "crop_strategy": "center",
                },
            }
        },
    }

    _write_predictions(arch_out / "predictions_clean.csv", y_test, scores_clean)
    _write_robustness(arch_out / "robustness.csv", robustness)
    arch_result = results["architectures"][arch_meta["display"]]
    (arch_out / "metrics.json").write_text(
        json.dumps(_json_safe(arch_result), indent=2),
        encoding="utf-8",
    )

    write_all(_json_safe(results), str(out))
    logger.info("Benchmark %s finalizado em: %s", arch_meta["display"], out)
    print(json.dumps(_json_safe({
        "output_dir": str(out),
        "model_artifact": str(artifact),
        "accuracy": clean.get("accuracy"),
        "auc_roc": clean.get("auc_roc"),
        "eer": clean.get("eer"),
        "latency_ms": latency_ms,
        "fallback_used": False,
    }), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
