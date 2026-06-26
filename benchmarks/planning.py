"""Preflight e plano executável do benchmark."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict

import numpy as np

from benchmarks.config import BenchmarkConfig


CLASSICAL_ARCHES = {"svm", "randomforest"}
HEAVY_ARCHES = {
    "wavlm",
    "hubert",
    "rawnet2",
    "aasist",
    "rawgatst",
    "spectrogramtransformer",
    "efficientnetlstm",
}

NEURAL_BENCHMARK_HPARAMS: Dict[str, Dict[str, Any]] = {
    "wavlm": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "architecture_default",
        "use_augmentation": False,
        "use_mixed_precision": False,
        "recommended_epochs": 100,
        "notes": "WavLM é PyTorch-only; neste pipeline TF usa fallback CNN 1D treinado do zero, com LR 1e-4 e recorte central 1s.",
    },
    "hubert": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 8,
        "learning_rate": 1e-5,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "architecture_default",
        "use_augmentation": False,
        "use_mixed_precision": False,
        "recommended_epochs": 100,
        "notes": "HuBERT usa from_pt=True quando o backbone real existe; fallback simplificado deve ser reportado.",
    },
    "rawnet2": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.3,
        "l2_reg_strength": 1e-4,
        "optimizer": "Adam",
        "scheduler": "architecture_default",
        "use_augmentation": False,
        "use_mixed_precision": False,
        "early_stopping": False,
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "Raw waveform + Sinc/GRU: recorte central 1s, mixed precision desligado; LR segue o compile da arquitetura.",
    },
    "sonicsleuth": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 100,
        "dropout_rate": 0.3,
        "l2_reg_strength": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "recommended_epochs": 100,
    },
    "aasist": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "attention_heads": 12,
        "hidden_units": 512,
        "optimizer": "AdamW",
        "scheduler": "architecture_default",
        "recommended_epochs": 100,
        "notes": "Raw waveform + Sinc/GAT/HS-GAL; recorte central 1s para conter custo quadrático do GAT temporal. Mantém compile-respect: loss/margem da própria arquitetura e LR estável.",
    },
    "rawgatst": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "architecture_default",
        "use_augmentation": False,
        "use_mixed_precision": False,
        "recommended_epochs": 100,
        "notes": "Raw waveform + SincNet/GAT; segue a factory atual com entrada de áudio bruto e recorte central do benchmark.",
    },
    "conformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.3,
        "l2_reg_strength": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1500,
        "clipnorm": 1.0,
        "label_smoothing": 0.05,
        "attention_heads": 8,
        "hidden_units": 256,
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "Compile-respect: LR/weight_decay/clipnorm são passados ao construtor do Conformer.",
    },
    "hybridcnntransformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "base_filters": 64,
        "num_residual_blocks": 3,
        "num_transformer_layers": 2,
        "attention_heads": 8,
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
    },
    "spectrogramtransformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 8,
        "learning_rate": 2e-5,
        "epochs": 100,
        "dropout_rate": 0.25,
        "l2_reg_strength": 5e-5,
        "weight_decay": 5e-5,
        "warmup_steps": 3000,
        "decay_steps": 100000,
        "alpha": 1e-6,
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "use_augmentation": False,
        "early_stopping": True,
        "early_stopping_patience": 20,
        "checkpoint_best": True,
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "AST conforme a implementação do artigo: arquitetura inalterada, LR/weight_decay menores, warmup maior e checkpoint obrigatório para evitar salvar o estado final colapsado.",
    },
    "efficientnetlstm": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,
        "dropout_rate": 0.4,
        "l2_reg_strength": 2e-4,
        "lstm_units": 128,
        "hidden_units": "128/64",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "pretrained": True,
        "recommended_epochs": 100,
        "notes": "EfficientNetB0 + Delta features + BiLSTM; usa ImageNet quando pesos estão disponíveis e fallback offline com pesos aleatórios. LR segue o compile da arquitetura.",
    },
    "multiscalecnn": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 64,
        "learning_rate": 2e-3,
        "epochs": 100,
        "dropout_rate": 0.5,
        "l2_reg_strength": 5e-4,
        "hidden_units": "128/256",
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "recommended_epochs": 100,
    },
    "ensemble": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        "learning_rate": 5e-5,
        "epochs": 100,
        "dropout_rate": 0.3,
        "l2_reg_strength": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "architecture_default",
        "use_augmentation": False,
        "use_mixed_precision": False,
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "Ensemble real opera em áudio bruto: branches Mel/LFCC/CQT/MFCC e fusão multimodal; recorte central 1s para custo controlado.",
    },
}


def _compact(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _base_recommended_hparams(arch: str) -> Dict[str, Any]:
    compact = _compact(arch)
    if compact in CLASSICAL_ARCHES:
        return {
            "model_family": "classical",
            "fit": "sklearn",
            "feature_scaling": True,
            "training_budget": "GridSearchCV + final refit",
            "epochs": None,
        }

    params = dict(
        NEURAL_BENCHMARK_HPARAMS.get(
            compact,
            {
                "model_family": "neural",
                "input_domain": "spectrogram",
                "batch_size": 32,
                "learning_rate": 1e-3,
                "epochs": 100,
                "dropout_rate": 0.3,
                "l2_reg_strength": 1e-4,
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                "recommended_epochs": 100,
            },
        )
    )
    return params


def _fit_to_device(params: Dict[str, Any], arch: str, device: Dict[str, Any]) -> Dict[str, Any]:
    tuned = dict(params)
    compact = _compact(arch)
    if compact in CLASSICAL_ARCHES:
        return tuned

    batch = int(tuned.get("batch_size", 32))
    if device.get("resolved_profile") == "cpu":
        if compact in {"wavlm", "hubert", "rawnet2", "aasist", "rawgatst"}:
            cap = 4
        elif compact in HEAVY_ARCHES:
            cap = 8
        else:
            cap = 16
        tuned["batch_size"] = min(batch, cap)
        tuned["device_adjustment"] = "cpu_batch_cap"
        tuned["use_mixed_precision"] = False
    else:
        if compact == "hubert":
            # HuBERT real/fallback é pesado, mas batch=1 torna o benchmark
            # impraticável no dataset completo. Mantém um padrão conservador
            # para GPUs de 12 GB e permite reduzir via ambiente se necessário.
            cap = int(os.getenv("XFAKE_HUBERT_GPU_BATCH_CAP", "8"))
        elif compact == "wavlm":
            cap = 8
        elif compact == "rawnet2":
            cap = 16
        elif compact == "rawgatst":
            cap = 8
        elif compact == "ensemble":
            cap = 16
        elif compact in {"aasist", "spectrogramtransformer", "efficientnetlstm"}:
            cap = 16
        else:
            cap = 32
        tuned["batch_size"] = min(batch, cap)
        tuned["device_adjustment"] = (
            "gpu_ssl_safe_cap" if compact in {"hubert"}
            else "gpu_wavlm_fallback_cap" if compact == "wavlm"
            else "gpu_vram_safe_cap"
        )
        if compact not in {"wavlm", "hubert", "rawnet2", "ensemble"}:
            tuned.setdefault("use_mixed_precision", True)
        else:
            tuned["use_mixed_precision"] = False

    return tuned


def _device_snapshot(profile: str) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "requested_profile": profile,
        "resolved_profile": "cpu",
        "platform": f"{platform.system()} {platform.release()}",
        "gpu_available": False,
        "gpu_names": [],
    }
    if profile == "cpu":
        snap["resolved_profile"] = "cpu"
        return snap

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        snap["gpu_available"] = bool(gpus)
        snap["gpu_names"] = [getattr(gpu, "name", str(gpu)) for gpu in gpus]
        if profile == "gpu":
            snap["resolved_profile"] = "gpu" if gpus else "cpu"
            if not gpus:
                snap["gpu_request_unavailable"] = True
        elif profile == "auto":
            snap["resolved_profile"] = "gpu" if gpus else "cpu"
    except Exception as exc:
        snap["tensorflow_probe_error"] = str(exc)
    return snap


def _merge_effective_hparams(
    cfg: BenchmarkConfig,
    arch: str,
    device: Dict[str, Any],
) -> Dict[str, Any]:
    compact = _compact(arch)
    if cfg.optimize_hyperparameters:
        params = _base_recommended_hparams(arch)
    elif compact in CLASSICAL_ARCHES:
        params = {
            "model_family": "classical",
            "fit": "sklearn",
            "feature_scaling": True,
            "training_budget": "single fit",
            "epochs": None,
        }
    else:
        params = {"epochs": cfg.epochs, "batch_size": cfg.batch_size}

    params = _fit_to_device(params, arch, device)
    if compact not in CLASSICAL_ARCHES:
        params["epochs"] = int(cfg.epochs)
        params.setdefault("batch_size", int(cfg.batch_size))
        params.setdefault("learning_rate", 1e-3)
        params.setdefault("early_stopping", True)
        params.setdefault("lr_scheduler", "architecture_default")
        params["epochs_source"] = "benchmark_cli"

    params.update(cfg.training_overrides.get(arch, {}))
    return params


def build_benchmark_plan(cfg: BenchmarkConfig, data: Any | None = None) -> Dict[str, Any]:
    """Cria o plano de execução antes do treino."""
    device = _device_snapshot(cfg.device_profile)
    dataset = {}
    if data is not None:
        y = np.asarray(data.y)
        dataset = {
            "name": getattr(data, "name", None),
            "n_total": int(len(y)),
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "balance": {
                "real": int((y == 0).sum()),
                "fake": int((y == 1).sum()),
            },
            "metadata": getattr(data, "metadata", {}) or {},
        }

    architectures = {}
    for arch in cfg.architectures:
        compact = _compact(arch)
        architectures[arch] = {
            "type": "classical" if compact in CLASSICAL_ARCHES else "neural",
            "training_config": _merge_effective_hparams(cfg, arch, device),
        }

    return {
        "preset": cfg.preset_name,
        "device": device,
        "dataset": dataset,
        "benchmark": {
            "architectures": list(cfg.architectures),
            "snr_levels_db": list(cfg.snr_levels_db),
            "latency_runs": int(cfg.latency_runs),
            "run_api_probe": bool(cfg.run_api_probe),
            "optimize_hyperparameters": bool(cfg.optimize_hyperparameters),
            "convergence": {
                "auc_roc_min": float(cfg.converge_auc_threshold),
                "accuracy_min": float(cfg.converge_accuracy_threshold),
            },
        },
        "architectures": architectures,
    }


def apply_plan_to_config(cfg: BenchmarkConfig, plan: Dict[str, Any]) -> BenchmarkConfig:
    cfg.training_overrides = {
        arch: dict(info.get("training_config") or {})
        for arch, info in (plan.get("architectures") or {}).items()
    }
    return cfg


def write_benchmark_plan(plan: Dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Plano de Benchmark",
        "",
        f"- Preset: `{plan.get('preset')}`",
        f"- Perfil de dispositivo: `{plan.get('device', {}).get('resolved_profile')}`",
        f"- Dataset: `{plan.get('dataset', {}).get('name')}`",
        f"- Amostras: `{plan.get('dataset', {}).get('n_total')}`",
        f"- SNRs: `{plan.get('benchmark', {}).get('snr_levels_db')}`",
        f"- API probe: `{plan.get('benchmark', {}).get('run_api_probe')}`",
        "",
        "## Hiperparâmetros Efetivos",
        "",
        "| Arquitetura | Tipo | Treino | Batch | LR | Ajuste |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arch, info in (plan.get("architectures") or {}).items():
        hp = info.get("training_config") or {}
        lines.append(
            f"| {arch} | {info.get('type')} | "
            f"{hp.get('training_budget') or hp.get('epochs', '-')} | "
            f"{hp.get('batch_size', '-')} | {hp.get('learning_rate', '-')} | "
            f"{hp.get('device_adjustment', '-')} |"
        )
    (out / "benchmark_plan.md").write_text("\n".join(lines), encoding="utf-8")
