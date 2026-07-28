"""Preflight e plano executável do benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict

import numpy as np

from benchmarks.config import BenchmarkConfig

CLASSICAL_ARCHES = {"svm", "randomforest"}
HEAVY_ARCHES = {
    "rawnet2",
    "aasist",
    "rawgatst",
    "spectrogramtransformer",
}
ARCH_ALIASES = {
    "cct": "hybridcnntransformer",
    "ast": "spectrogramtransformer",
    "audiospectrogramtransformer": "spectrogramtransformer",
    "res2net": "multiscalecnn",
    "randomforest": "randomforest",
}

# Hiperparâmetros recomendados por arquitetura para o benchmark (aplicados
# quando cfg.optimize_hyperparameters=True; ver _merge_effective_hparams).
# Este é 1 dos 3 locais de hparams por arquitetura — mantenha em sincronia com:
#   - app/domain/models/architectures/registry.py::default_params (regularização);
#   - o create_model(...) de cada architectures/<nome>.py (LR/optimizer/loss).
# Chaves sobrepostas (dropout_rate, l2_reg_strength) podem divergir — revise as 3
# fontes ao ajustar um modelo.
NEURAL_BENCHMARK_HPARAMS: Dict[str, Dict[str, Any]] = {
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
        "notes": "Raw waveform + Sinc/GRU: recorte central 1s, mixed precision desligado; LR segue o compile da arquitetura. 2026-07-14: topologia corrigida p/ paridade com o paper — MaxPool(3) após cada bloco residual (GRU passa a ver ~7 passos, não ~590) e FMS mul+add.",
    },
    "aasist": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 24,
        # AJUSTE (retune): LR 1e-4->3e-4 e l2 1e-4->2e-4, em sincronia com
        # aasist.py::create_model e registry.py::default_params (ver
        # docs/evaluation/retraining-adjustments.md). Augmentation ligado — subajuste + colapso
        # de recall sob ruído (0.29 @10dB) no diagnóstico original.
        # CORREÇÃO 2026-07-15: o valor estava revertido para 1e-4/1e-4 (drift
        # silencioso — o comentário acima já documentava 3e-4/2e-4 como a
        # decisão vigente). Restaurado para bater com o que o comentário e o
        # docs/evaluation/retraining-adjustments.md sempre descreveram.
        "learning_rate": 3e-4,
        "min_learning_rate": 5e-6,
        "decay_steps": 100000,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 2e-4,
        "classifier_head": "cross_entropy",
        # (attention_heads/hidden_units removidos: o create_model do AASIST
        # não os aceita — eram filtrados pela assinatura, config morto.)
        "optimizer": "AdamW",
        "scheduler": "CosineDecay",
        "use_augmentation": True,
        "use_mixed_precision": True,
        "recommended_epochs": 100,
        "notes": (
            "Sinc 2D + GAT S/T + master/HS-GAL/MGO; janela canônica 48.000 "
            "(3 s @ 16 kHz), crop aleatório no treino e multicrop na avaliação."
        ),
    },
    "rawgatst": {
        "model_family": "neural",
        "input_domain": "raw_audio",
        "batch_size": 16,
        # AJUSTE (retune): LR 1e-4->5e-5, dropout 0.2->0.35 e l2 1e-4->1e-3,
        # em sincronia com rawgat_st.py::create_model e
        # registry.py::default_params (ver docs/evaluation/retraining-adjustments.md).
        # Augmentation ligado — pior modelo do recorte, overfit/divergência
        # após a época 4 no diagnóstico original.
        "learning_rate": 5e-5,
        "min_learning_rate": 5e-6,
        "decay_steps": 100000,
        "epochs": 100,
        "dropout_rate": 0.35,
        "l2_reg_strength": 1e-3,
        "optimizer": "AdamW",
        "scheduler": "CosineDecay",
        "use_augmentation": True,
        # A/B retunado em 2026-07-17: o híbrido oscilou até loss=6.75,
        # enquanto o controle float32 ficou em loss=0.63/0.69/0.64 e
        # val_loss=0.58/0.58/0.58. Mantém o pipeline confirmatório em
        # float32; a implementação híbrida permanece disponível para estudo.
        "use_mixed_precision": False,
        "recommended_epochs": 100,
        "notes": (
            "Dois encoders 2D + GAT S/T + produto + terceiro GAT; janela "
            "canônica 48.000 (3 s @ 16 kHz), crop aleatório no treino e "
            "multicrop na avaliação."
        ),
    },
    "conformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,
        # AJUSTE 2026-07-27 (consolidação do Conformer): dropout_rate 0.3->0.1.
        # O valor 0.3 nunca chegou ao encoder — a variante sobrescrevia o
        # dropout por módulo (ff=0.2/attn=0.1/conv=0.1) e o ConvSubsampling
        # tinha 0.1 hardcoded, de modo que 0.3 só afetava a cabeça de
        # classificação. Agora `dropout_rate` vale para o encoder inteiro,
        # então usamos o P_drop=0.1 do paper (Gulati et al.) — mantém o
        # comportamento efetivo anterior e passa a ser um knob real.
        "dropout_rate": 0.1,
        "l2_reg_strength": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1500,
        "clipnorm": 1.0,
        "label_smoothing": 0.05,
        # (attention_heads/hidden_units REMOVIDOS: o runner só promove
        # lr/weight_decay/warmup/decay/alpha/dropout/clipnorm/label_smoothing
        # para `parameters`, então essas duas chaves NUNCA chegavam ao
        # construtor — o modelo sempre rodou com num_heads=4/d_model=256 do
        # Conformer-M. Eram config morto, como as já removidas do AASIST.)
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": (
            "Conformer-M do paper (16 blocos, d_model=256, 4 cabeças, "
            "d_ff=1024, kernel 31, P_drop=0.1) — configuração ÚNICA desde "
            "2026-07-27. Compile-respect: LR/weight_decay/clipnorm/dropout são "
            "passados ao construtor."
        ),
    },
    "hybridcnntransformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 32,
        # AJUSTE 2026-07-14: LR 1e-3->3e-4 e schedule agora passado ao
        # construtor do CCT (antes o compile era hardcoded lr=1e-3 e
        # decay_steps=50000 — com batch 32 sao ~657 passos/epoca x 100 =
        # 65700 passos reais, o LR zerava (alpha) na epoca ~76; mesmo
        # mismatch de decay_steps ja diagnosticado no AST). Colapso sob
        # ruido (AUC 0.46 @10dB) tratado com pico de LR menor + retreino
        # com a copia ruidosa do protocolo.
        "learning_rate": 3e-4,
        "epochs": 100,
        "dropout_rate": 0.2,
        "l2_reg_strength": 1e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1500,
        "decay_steps": 65700,
        "alpha": 1e-7,
        "clipnorm": 1.0,
        # (base_filters/num_residual_blocks/num_transformer_layers/
        # attention_heads removidos: o builder CCT usa projection_dim/
        # num_heads/transformer_layers/conv_channels do registry — as chaves
        # antigas nunca chegavam ao modelo, eram config morto.)
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "CCT compile-respect: LR/warmup/decay/weight_decay/clipnorm passados ao construtor (2026-07-14).",
    },
    "spectrogramtransformer": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 8,
        # AJUSTE 2026-07-14: LR 2e-5->1e-5 e weight_decay 5e-5->1e-5. Mesmo
        # com decay_steps corrigido o treino degradava lentamente ate chute
        # aleatorio (EER final ~51%); 87M params do zero pedem passo menor.
        # Acompanha a mudanca estrutural p/ blocos pre-LN (paper) em
        # spectrogram_transformer.py — em sincronia com registry.py.
        "learning_rate": 1e-5,
        "epochs": 100,
        "dropout_rate": 0.25,
        "l2_reg_strength": 1e-5,
        "weight_decay": 1e-5,
        "warmup_steps": 3000,
        # decay_steps cobre o total real de passos (100 epocas x 2625
        # passos/epoca, ja considerando o 1 copy de ruido do
        # train_noise_copies=1 que dobra 10500->21000 amostras). O valor
        # antigo (100000) fazia o LR zerar (alpha=1e-6) por volta da epoca
        # 38 e o treino degradava ate accuracy=chute aleatorio dali ate a
        # epoca 100 (diagnosticado em 2026-07-13).
        "decay_steps": 262500,
        "alpha": 1e-6,
        "clipnorm": 1.0,
        # AJUSTE 2026-07-27: pesos AudioSet ligados. O AST do artigo PARTE de
        # inicialização pré-treinada (ImageNet→AudioSet); treinar 85M params do
        # zero sobre este dataset é o regime que degradava até chute aleatório
        # (EER ~51%). A transferência lê o checkpoint PyTorch e escreve nas
        # camadas Keras (app/domain/models/architectures/ast_pretrained.py) —
        # validada contra o PyTorch bloco a bloco (max|dif| ~1e-5).
        # Exige rede na 1ª execução (~350 MB, cacheado depois) e FALHA ALTO se
        # indisponível — nunca degrada em silêncio para treino do zero.
        "pretrained": True,
        "optimizer": "AdamW",
        "scheduler": "WarmupCosineDecay",
        "use_augmentation": False,
        "early_stopping": True,
        "early_stopping_patience": 20,
        "checkpoint_best": True,
        "reduce_lr_on_plateau": False,
        "recommended_epochs": 100,
        "notes": "AST pre-LN (paper) com cabeça linear sobre o CLS, entrada 300x128 (128 mel, hop 10 ms, janela 25 ms) e pesos AudioSet transferidos do checkpoint PyTorch; LR de pico 1e-5 e weight_decay 1e-5, warmup 3000, clipnorm=1.0 e checkpoint obrigatório com restauração guardada (validada no val).",
    },
    "multiscalecnn": {
        "model_family": "neural",
        "input_domain": "spectrogram",
        "batch_size": 64,
        # AJUSTE 2026-07-14: LR 2e-3->1e-3 (alinha com o compile do builder)
        # e regularização EFETIVA contra o overfit train 100% / val 64,5%:
        # o dropout_rate/l2_reg_strength antigos deste plano eram config
        # morto (nunca chegavam ao create_model). Agora o dropout 0.5 flui
        # pelo registry (default_params) e o weight_decay real vem do AdamW
        # do builder (multiscale_cnn.py, default 1e-2 acoplado ao LR).
        "learning_rate": 1e-3,
        "epochs": 100,
        "dropout_rate": 0.5,
        "weight_decay": 1e-2,
        # (l2_reg_strength/hidden_units removidos: nenhum caminho os
        # consumia para esta arquitetura — config morto.)
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "recommended_epochs": 100,
        "notes": "Res2Net-50 com AdamW (weight decay real) + dropout 0.5 e checkpoint com restauração guardada (2026-07-14).",
    },
}


def _compact(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _canonical_arch_key(arch: str) -> str:
    compact = _compact(arch)
    return ARCH_ALIASES.get(compact, compact)


def _base_recommended_hparams(arch: str) -> Dict[str, Any]:
    compact = _canonical_arch_key(arch)
    if compact in CLASSICAL_ARCHES:
        return {
            "model_family": "classical",
            "fit": "sklearn",
            "feature_scaling": True,
            "training_budget": "GridSearchCV + final refit",
            "epochs": None,
        }

    if compact not in NEURAL_BENCHMARK_HPARAMS:
        raise ValueError(
            f"Arquitetura sem hiperparâmetros de plano: {arch}. "
            "O escopo OFICIAL cobre RawNet2, AASIST, RawGAT-ST, Conformer, "
            "CCT/Hybrid CNN-Transformer, AST/SpectrogramTransformer, "
            "Res2Net/MultiscaleCNN, SVM e RandomForest (WavLM/HuBERT Original "
            "rodam pelo runner PyTorch dedicado). Sonic Sleuth, "
            "EfficientNet-LSTM e Ensemble pertencem ao escopo ESTENDIDO: use "
            "`--experiment-scope extended` (que já desliga "
            "optimize_hyperparameters) ou `--no-optimize-hparams`."
        )

    params = dict(NEURAL_BENCHMARK_HPARAMS[compact])
    return params


def _fit_to_device(
    params: Dict[str, Any], arch: str, device: Dict[str, Any]
) -> Dict[str, Any]:
    tuned = dict(params)
    compact = _canonical_arch_key(arch)
    if compact in CLASSICAL_ARCHES:
        return tuned

    # Os caps abaixo foram calibrados com a janela antiga de 64.600 amostras
    # (~4,04 s). A janela canonica passou a 48.000 (3 s), 26% menor, entao eles
    # seguem SEGUROS — uma entrada menor so consome menos memoria. Conservadores
    # de proposito: subi-los e otimizacao, e otimizacao sem medir na GPU alvo
    # troca tempo de execucao por risco de OOM no meio de um run de horas.
    batch = int(tuned.get("batch_size", 32))
    if device.get("resolved_profile") == "cpu":
        if compact in {"rawnet2", "aasist", "rawgatst"}:
            cap = 4
        elif compact in HEAVY_ARCHES:
            cap = 8
        else:
            cap = 16
        tuned["batch_size"] = min(batch, cap)
        tuned["device_adjustment"] = "cpu_batch_cap"
        tuned["use_mixed_precision"] = False
    else:
        if compact == "rawnet2":
            cap = 16
        elif compact == "aasist":
            cap = 24
        elif compact == "rawgatst":
            cap = 16
        elif compact == "spectrogramtransformer":
            cap = 16
        else:
            cap = 32
        tuned["batch_size"] = min(batch, cap)
        tuned["device_adjustment"] = "gpu_vram_safe_cap"
        if compact == "aasist":
            # Sinc e logits permanecem float32 nas próprias camadas; o encoder
            # 2D/GAT usa Tensor Cores com loss scaling automático do Keras.
            tuned["use_mixed_precision"] = True
        elif compact != "rawnet2":
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

    overrides = cfg.training_overrides.get(arch, {})
    params.update(overrides)
    if compact not in CLASSICAL_ARCHES:
        # Controles do protocolo sobrescrevem apenas aspectos de comparabilidade.
        params["epochs"] = int(cfg.epochs)
        params["epochs_source"] = "standardized_benchmark_budget"
        # `early_stopping` derivado do orçamento fixo, MAS um override explícito
        # do chamador vence: a linha incondicional anterior descartava em
        # silêncio o `--no-early-stopping` sempre que fixed_epoch_budget=False.
        if "early_stopping" not in overrides:
            params["early_stopping"] = not bool(cfg.fixed_epoch_budget)
        params["select_best_checkpoint"] = bool(cfg.select_best_checkpoint)
        params["validation_condition"] = "clean"
        params["calibrate_under_noise"] = False
        params["decision_threshold"] = float(cfg.decision_threshold)
    return params


def build_benchmark_plan(
    cfg: BenchmarkConfig, data: Any | None = None
) -> Dict[str, Any]:
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
            "standardized_controls": {
                "epochs": int(cfg.epochs),
                "fixed_epoch_budget": bool(cfg.fixed_epoch_budget),
                "select_best_checkpoint": bool(cfg.select_best_checkpoint),
                "validation_condition": "clean",
                "decision_threshold": float(cfg.decision_threshold),
                "metric_threshold_policy": cfg.metric_threshold_policy,
                "experiment_scope": cfg.experiment_scope,
                "preserve_predefined_splits": bool(cfg.preserve_predefined_splits),
                "fail_on_split_overlap": bool(cfg.fail_on_split_overlap),
                "waveform_awgn_before_frontend": True,
            },
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
    normalized = json.dumps(
        plan, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    effective = {
        "schema": "xfakesong-effective-training-config-v1",
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "plan": plan,
    }
    (out / "effective_training_config.json").write_text(
        json.dumps(effective, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
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
