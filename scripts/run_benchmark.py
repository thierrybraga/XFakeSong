#!/usr/bin/env python3
"""CLI do sistema de benchmark/teste do XFakeSong (para o TCC).

Treina/avalia arquiteturas pelo PIPELINE real e (opcionalmente) testa a API,
gerando tabelas LaTeX + figuras + JSON/CSV prontos para a monografia.

Exemplos:
    # Verificação rápida (sintético, 1 época) — valida o harness:
    python scripts/run_benchmark.py --quick

    # Execução do TCC (arquiteturas-chave + SVM/RF + API), dataset real .npz:
    python scripts/run_benchmark.py --full --dataset app/datasets/brspeech_df.npz

    # Sob medida:
    python scripts/run_benchmark.py --archs MultiscaleCNN Ensemble SVM \
        --dataset data.npz --epochs 20 --snr 30 20 10 --api --out results/bench
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="preset rápido sintético (1 época) para verificar o harness")
    p.add_argument("--full", action="store_true",
                   help="preset do TCC (arquiteturas-chave + SVM/RF + API)")
    p.add_argument("--archs", nargs="+", metavar="NOME",
                   help="arquiteturas a avaliar (nomes do registry)")
    p.add_argument("--dataset", metavar="NPZ",
                   help=".npz com X_train/y_train (senão usa dataset sintético)")
    p.add_argument("--epochs", type=int, help="épocas de treino por arquitetura")
    p.add_argument("--snr", nargs="+", type=int, metavar="DB",
                   help="níveis de SNR (dB) do teste de robustez")
    p.add_argument("--out", metavar="DIR", help="pasta de saída dos artefatos")
    p.add_argument("--api", action="store_true",
                   help="também roda o teste de sistema da API (TestClient)")
    p.add_argument("--no-api", action="store_true",
                   help="desativa o teste da API mesmo em presets que o habilitam")
    p.add_argument("--batch-size", type=int, help="batch size de treino")
    p.add_argument("--latency-runs", type=int,
                   help="número de medições de latência por arquitetura")
    p.add_argument("--synthetic-n", type=int,
                   help="número de amostras do dataset sintético")
    p.add_argument("--synthetic-shape", nargs="+", type=int, metavar="DIM",
                   help="shape por amostra do dataset sintético, ex: 128 80")
    p.add_argument("--converge-auc", type=float,
                   help="AUC mínima para marcar convergência")
    p.add_argument("--converge-accuracy", type=float,
                   help="acurácia mínima no threshold 0.5 para convergência")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from benchmarks import BenchmarkConfig, run_benchmark

    if args.full:
        cfg = BenchmarkConfig.full_tcc()
    elif args.quick:
        cfg = BenchmarkConfig.quick()
    else:
        cfg = BenchmarkConfig()

    if args.archs:
        cfg.architectures = args.archs
    if args.dataset:
        cfg.dataset_path = args.dataset
    if args.epochs:
        cfg.epochs = args.epochs
    if args.snr:
        cfg.snr_levels_db = args.snr
    if args.out:
        cfg.output_dir = args.out
    if args.api:
        cfg.run_api_probe = True
    if args.no_api:
        cfg.run_api_probe = False
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.latency_runs:
        cfg.latency_runs = args.latency_runs
    if args.synthetic_n:
        cfg.synthetic_n = args.synthetic_n
    if args.synthetic_shape:
        cfg.synthetic_shape = tuple(args.synthetic_shape)
    if args.converge_auc is not None:
        cfg.converge_auc_threshold = args.converge_auc
    if args.converge_accuracy is not None:
        cfg.converge_accuracy_threshold = args.converge_accuracy
    cfg.seed = args.seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n=== Benchmark XFakeSong ===")
    print(f"Arquiteturas : {', '.join(cfg.architectures)}")
    print(f"Dataset      : {cfg.dataset_path or '(sintético)'}")
    print(f"Épocas       : {cfg.epochs} | SNRs: {cfg.snr_levels_db} | "
          f"API: {cfg.run_api_probe}")
    print(f"Saída        : {cfg.output_dir}\n")

    results = run_benchmark(cfg)

    # Resumo no terminal
    print("\n" + "=" * 72)
    print(f"{'Arquitetura':<24}{'Conv':<6}{'Acur':>8}{'EER':>8}"
          f"{'AUC':>8}{'Lat(ms)':>10}")
    print("-" * 72)
    for name, r in results["architectures"].items():
        if r.get("status") != "ok":
            print(f"{name:<24}{'ERRO':<6}  {str(r.get('error',''))[:40]}")
            continue
        c = r["clean"]
        conv = "sim" if r.get("converged") else "não"
        acc = f"{c.get('accuracy', 0) * 100:.2f}"
        eer = c.get("eer")
        eer = f"{eer * 100:.2f}" if eer == eer else "---"
        auc = c.get("auc_roc")
        auc = f"{auc:.3f}" if auc == auc else "---"
        lat = r["efficiency"].get("latency_ms")
        print(f"{name:<24}{conv:<6}{acc:>8}{eer:>8}{auc:>8}{str(lat):>10}")
    print("=" * 72)
    if "api" in results:
        api = results["api"]
        print(f"API: {api.get('status')} — {api.get('n_2xx')}/"
              f"{api.get('n_probed')} endpoints 2xx, "
              f"mediana {api.get('latency_ms_median')} ms "
              f"({api.get('n_routes')} rotas registradas)")
    out = Path(cfg.output_dir).resolve()
    print(f"\nArtefatos: {out}")
    print("  • results.json / results.csv / summary.md")
    print("  • tables/*.tex  (resultados, eficiência, robustez)")
    print("  • figures/*.png (roc, robustez, eficiência, convergência)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
