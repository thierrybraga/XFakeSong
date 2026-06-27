"""Ablação de fine-tuning do WavLM (P2).

O WavLM ficou abaixo do HuBERT (86% vs 92%) e colapsou a ~51% em 10 dB. Esta
ablação varre o learning rate (e, opcionalmente, épocas) do fine-tuning para
explicar/corrigir o subdesempenho, comparando contra o HuBERT como baseline.

Reusa o harness de benchmark (treino + robustez AWGN + eficiência). Para cada
LR, roda WavLM no mesmo dataset/seed e tabula acurácia limpa, AUC e robustez
por SNR. O HuBERT é avaliado uma vez como referência.

Sem GPU em Windows nativo (TF>=2.11) → rode sob WSL2. Exemplos:

    python scripts/ablate_wavlm_finetune.py \
        --dataset app/datasets/benchmark_audio_raw_balanced_15k.npz \
        --lrs 1e-5 3e-5 1e-4 --epochs 30 --out results/ablation_wavlm

    # Verificação rápida do harness (sintético, sem áudio):
    python scripts/ablate_wavlm_finetune.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_one(architecture, dataset, epochs, snrs, seed, out_dir, overrides,
             quick):
    from benchmarks.config import BenchmarkConfig
    from benchmarks.runner import run_benchmark

    if quick:
        cfg = BenchmarkConfig.quick(architectures=[architecture])
    else:
        cfg = BenchmarkConfig(
            architectures=[architecture],
            dataset_path=dataset,
            epochs=epochs,
            snr_levels_db=snrs,
            optimize_hyperparameters=False,
            run_api_probe=False,
        )
    cfg.seed = seed
    cfg.output_dir = str(out_dir)
    if overrides:
        cfg.training_overrides = {architecture: overrides}
    results = run_benchmark(cfg)
    return results["architectures"].get(architecture, {})


def _row(label, arch_result):
    clean = arch_result.get("clean", {}) or {}
    rob = arch_result.get("robustness", {}) or {}
    row = {
        "config": label,
        "status": arch_result.get("status"),
        "clean_accuracy": round(float(clean.get("accuracy", float("nan"))), 4),
        "clean_auc": round(float(clean.get("auc_roc", float("nan"))), 4),
        "epochs_executed": (arch_result.get("training_config") or {}).get(
            "epochs_executed"
        ),
    }
    for snr, metrics in sorted(rob.items(), key=lambda kv: -int(kv[0])):
        row[f"acc_{snr}dB"] = round(float((metrics or {}).get("accuracy",
                                    float("nan"))), 4)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Ablação de fine-tuning do WavLM")
    p.add_argument("--dataset", help=".npz com X_train/y_train (raw audio)")
    p.add_argument("--lrs", nargs="+", type=float,
                   default=[1e-5, 3e-5, 1e-4],
                   help="learning rates de fine-tuning a comparar")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/ablation_wavlm")
    p.add_argument("--no-hubert", action="store_true",
                   help="não roda o baseline HuBERT")
    p.add_argument("--quick", action="store_true",
                   help="preset sintético de verificação (1 época, sem áudio)")
    args = p.parse_args()

    if not args.quick and not args.dataset:
        p.error("--dataset é obrigatório (ou use --quick)")

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("=== Ablação WavLM (fine-tuning) ===")
    for lr in args.lrs:
        label = f"WavLM lr={lr:g}"
        print(f"-> {label} (epochs={args.epochs})")
        res = _run_one(
            "WavLM", args.dataset, args.epochs, args.snr, args.seed,
            out_dir / f"wavlm_lr_{lr:g}",
            overrides={"learning_rate": lr}, quick=args.quick,
        )
        rows.append(_row(label, res))

    if not args.no_hubert:
        print("-> HuBERT (baseline)")
        res = _run_one(
            "HuBERT", args.dataset, args.epochs, args.snr, args.seed,
            out_dir / "hubert_baseline", overrides=None, quick=args.quick,
        )
        rows.append(_row("HuBERT baseline", res))

    # Persiste CSV + JSON
    fields = sorted({k for r in rows for k in r.keys()},
                    key=lambda k: (k != "config", k))
    with (out_dir / "ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Resumo ===")
    for r in rows:
        print(" | ".join(f"{k}={r[k]}" for k in fields if k in r))
    print(f"\nArtefatos: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
