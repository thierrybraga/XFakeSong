#!/usr/bin/env python3
"""CLI do sistema de benchmark/teste do XFakeSong (para o TCC).

Treina/avalia arquiteturas pelo PIPELINE real e (opcionalmente) testa a API,
gerando tabelas LaTeX + figuras + JSON/CSV prontos para a monografia.

Exemplos:
    # Verificação rápida (sintético, 1 época) — valida o harness:
    python scripts/benchmark/run_benchmark.py --quick

    # Execução do TCC (9 modelos do artigo + API), dataset real .npz:
    python scripts/benchmark/run_benchmark.py --full --dataset data/datasets/brspeech_df.npz

    # Sob medida:
    python scripts/benchmark/run_benchmark.py --archs RawNet2 AASIST SVM RandomForest \
        --dataset data.npz --epochs 20 --snr 30 20 10 --api --out data/results/bench

    # Modelo individual:
    python scripts/benchmark/run_benchmark.py --model AASIST \
        --dataset data.npz --epochs 20 --out data/results/bench_aasist
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_ROOT = Path(__file__).resolve().parents[2]
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
                   help="preset do TCC (9 modelos documentados no artigo + API)")
    p.add_argument("--neural", action="store_true",
                   help="preset neural do TCC (7 arquiteturas neurais do artigo, sem SVM/RF)")
    p.add_argument("--preset", choices=[
        "custom", "quick", "full_tcc", "neural_tcc", "rawnet2_100e"
    ],
                   help="preset explícito; equivalente a --quick ou --full")
    p.add_argument("--plan-only", action="store_true",
                   help="valida dataset/configuração, grava benchmark_plan.* e para antes do treino")
    p.add_argument("--archs", nargs="+", metavar="NOME",
                   help="arquiteturas a avaliar (nomes do registry)")
    p.add_argument("--model", metavar="NOME",
                   help="atalho para avaliar uma única arquitetura/modelo")
    p.add_argument("--dataset", metavar="NPZ",
                   help=".npz com X_train/y_train (senão usa dataset sintético)")
    p.add_argument("--epochs", type=int, help="épocas de treino por arquitetura")
    p.add_argument("--snr", nargs="+", type=int, metavar="DB",
                   help="níveis de SNR (dB) do teste de robustez")
    p.add_argument(
        "--waveform-train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="gera augmentation AWGN na forma de onda antes do frontend",
    )
    p.add_argument(
        "--train-aug-snr", nargs="+", type=int, metavar="DB",
        help="níveis de SNR distribuídos nas cópias ruidosas de treino",
    )
    p.add_argument(
        "--train-noise-copies", type=int,
        help="número de cópias ruidosas balanceadas por amostra (default: 1)",
    )
    p.add_argument(
        "--waveform-noise-batch-size", type=int,
        help="lote da geração AWGN em forma de onda (não altera hiperparâmetros do modelo)",
    )
    p.add_argument(
        "--allow-feature-space-awgn", action="store_true",
        help="permite fallback legado para NPZ sem forma de onda (não usar no artigo)",
    )
    p.add_argument(
        "--architecture-specific-augmentation",
        action="store_true",
        help="habilita aumento otimizado por arquitetura; usar apenas em ablação",
    )
    p.add_argument("--out", metavar="DIR", help="pasta de saída dos artefatos")
    p.add_argument(
        "--models-dir",
        metavar="DIR",
        help="pasta onde salvar modelos treinados (default: data/models)",
    )
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
    p.add_argument(
        "--experiment-scope",
        choices=["official", "extended"],
        default="official",
    )
    p.add_argument("--device-profile", choices=["auto", "cpu", "gpu"],
                   default=None,
                   help="perfil usado para cap de batch e mixed precision")
    p.add_argument("--no-optimize-hparams", action="store_true",
                   help="não aplica hiperparâmetros recomendados por arquitetura")
    p.add_argument("--no-early-stopping", action="store_true",
                   help="força o treino a executar todas as épocas solicitadas")
    p.add_argument("--verbose", action="store_true",
                   help="exibe logs INFO detalhados durante o benchmark")
    p.add_argument("--group-split", action="store_true",
                   help="P0: split disjunto por fonte/gerador (anti-vazamento)")
    p.add_argument("--cross-generator", metavar="GERADOR", default=None,
                   help="P0.4: segura este gerador fora do treino e o usa como "
                        "teste (ex.: fkvoice). Reteste cross-generator.")
    p.add_argument("--speaker-split", action="store_true",
                   help="tier large: split disjunto por falante (usuários não vistos)")
    p.add_argument("--unseen-speaker", metavar="FALANTE", default=None,
                   help="tier large: segura este falante fora do treino e o usa "
                        "como teste (protocolo de usuário não visto)")
    p.add_argument(
        "--fail-on-source-shortcut", action="store_true",
        help="recusa dataset em que a fonte prediz o rotulo acima de 55%%",
    )
    p.add_argument(
        "--source-shortcut-limit", type=float, default=None,
        help="sobrepoe o limite do oraculo fonte-rotulo (default: 0.55; "
             "use para datasets com confundimento fonte-classe documentado)",
    )
    p.add_argument("--codec-eval", nargs="+", default=None,
                   metavar="CODEC", choices=["mp3", "opus"],
                   help="robustez a codec com perdas (round-trip ffmpeg na "
                        "forma de onda, mesmo ponto do AWGN). Ex.: mp3 opus")
    p.add_argument("--bootstrap-ci", type=int, default=None,
                   help="nº de reamostragens do IC 95%% de bootstrap "
                        "(default: 1000; 0 desliga)")
    p.add_argument("--n-seeds", type=int, default=None,
                   help="repetições por arquitetura com sementes de TREINO "
                        "distintas; métricas viram média ± desvio (default: 1). "
                        "O split e o ruído de avaliação NÃO mudam. Custo: "
                        "multiplica o tempo de treino por N")
    args = p.parse_args()

    from benchmarks import (
        ALL_TCC_ARCHITECTURES,
        EXTENDED_MODEL_MANIFEST,
        BenchmarkConfig,
        plan_benchmark,
        run_benchmark,
    )

    selected_preset = args.preset
    if args.full:
        selected_preset = "full_tcc"
    elif args.neural:
        selected_preset = "neural_tcc"
    elif args.quick:
        selected_preset = "quick"

    if selected_preset == "full_tcc":
        cfg = BenchmarkConfig.full_tcc()
    elif selected_preset == "neural_tcc":
        cfg = BenchmarkConfig.neural_tcc()
    elif selected_preset == "rawnet2_100e":
        cfg = BenchmarkConfig.rawnet2_100e()
    elif selected_preset == "quick":
        cfg = BenchmarkConfig.quick()
    else:
        cfg = BenchmarkConfig()

    if args.model and args.archs:
        p.error("Use --model para um único modelo ou --archs para vários, não ambos.")
    if args.model:
        cfg.architectures = [args.model]
        cfg.preset_name = f"single:{args.model}"
        compact_model = "".join(ch for ch in args.model.lower() if ch.isalnum())
        if compact_model == "rawnet2" and args.epochs is None:
            cfg.epochs = 100
            cfg.batch_size = min(int(cfg.batch_size), 16)
            if cfg.device_profile == "auto":
                cfg.device_profile = "gpu"
    elif args.archs:
        cfg.architectures = args.archs
    if args.dataset:
        cfg.dataset_path = args.dataset
    if args.epochs:
        cfg.epochs = args.epochs
    if args.snr:
        cfg.snr_levels_db = args.snr
    if args.waveform_train_augmentation is not None:
        cfg.waveform_noise_augmentation = args.waveform_train_augmentation
    if args.train_aug_snr:
        cfg.train_aug_snr_db = args.train_aug_snr
    if args.architecture_specific_augmentation:
        cfg.architecture_specific_augmentation = True
    if args.waveform_noise_batch_size is not None:
        if args.waveform_noise_batch_size <= 0:
            p.error("--waveform-noise-batch-size deve ser > 0")
        cfg.waveform_noise_batch_size = args.waveform_noise_batch_size
    if args.train_noise_copies is not None:
        if args.train_noise_copies < 0:
            p.error("--train-noise-copies deve ser >= 0")
        cfg.train_noise_copies = args.train_noise_copies
    if args.allow_feature_space_awgn:
        cfg.strict_waveform_awgn = False
    if args.out:
        cfg.output_dir = args.out
    if args.models_dir:
        cfg.models_dir = args.models_dir
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
    cfg.experiment_scope = args.experiment_scope
    if args.group_split:
        cfg.group_split = True
    if args.cross_generator:
        cfg.holdout_generator = args.cross_generator
    if args.speaker_split:
        cfg.speaker_split = True
    if args.unseen_speaker:
        cfg.holdout_speaker = args.unseen_speaker
    if args.device_profile:
        cfg.device_profile = args.device_profile
    if args.fail_on_source_shortcut:
        cfg.fail_on_source_shortcut = True
    if args.source_shortcut_limit is not None:
        cfg.source_oracle_threshold = args.source_shortcut_limit
    if args.codec_eval:
        cfg.codec_eval = list(args.codec_eval)
    if args.bootstrap_ci is not None:
        if args.bootstrap_ci < 0:
            p.error("--bootstrap-ci deve ser >= 0")
        cfg.bootstrap_ci_samples = args.bootstrap_ci
    if args.n_seeds is not None:
        if args.n_seeds < 1:
            p.error("--n-seeds deve ser >= 1")
        cfg.n_seeds = args.n_seeds
    if args.no_optimize_hparams:
        cfg.optimize_hyperparameters = False
    official = set(ALL_TCC_ARCHITECTURES)
    extended = {item["benchmark_name"] for item in EXTENDED_MODEL_MANIFEST}
    allowed = extended if cfg.experiment_scope == "extended" else official
    invalid = [arch for arch in cfg.architectures if arch not in allowed]
    if invalid:
        p.error(
            f"arquiteturas fora do escopo {cfg.experiment_scope}: {invalid}"
        )
    if cfg.experiment_scope == "extended":
        cfg.optimize_hyperparameters = False
    if args.no_early_stopping:
        for arch in cfg.architectures:
            cfg.training_overrides.setdefault(arch, {})["early_stopping"] = False

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ["XFAKE_BENCHMARK_VERBOSE"] = "1" if args.verbose else "0"
    if not args.verbose:
        for noisy_logger in (
            "SVM",
            "RandomForest",
            "app",
            "app.domain",
            "app.domain.models",
            "app.domain.models.architectures",
            "app.domain.services",
            "benchmark",
        ):
            logger = logging.getLogger(noisy_logger)
            logger.setLevel(logging.WARNING)
            logger.disabled = True

    print("\n=== Benchmark XFakeSong ===")
    print(f"Arquiteturas : {', '.join(cfg.architectures)}")
    print(f"Dataset      : {cfg.dataset_path or '(sintetico)'}")
    print(f"Preset       : {cfg.preset_name} | HParams: "
          f"{'otimizados' if cfg.optimize_hyperparameters else 'globais'} | "
          f"Device: {cfg.device_profile}")
    classical = {
        "".join(ch for ch in arch.lower() if ch.isalnum())
        for arch in cfg.architectures
    } <= {"svm", "randomforest"}
    if classical:
        print("Treino       : GridSearchCV + fit final | "
              f"SNRs: {cfg.snr_levels_db} | API: {cfg.run_api_probe}")
    else:
        print(f"Epocas       : {cfg.epochs} | SNRs: {cfg.snr_levels_db} | "
              f"API: {cfg.run_api_probe}")
    print(f"Saida        : {cfg.output_dir}")
    print(f"Modelos      : {cfg.models_dir}\n")

    if args.plan_only:
        plan = plan_benchmark(cfg, write=True)
        out = Path(cfg.output_dir).resolve()
        print("Preflight concluido; treino nao iniciado.")
        print(f"Plano JSON: {out / 'benchmark_plan.json'}")
        print(f"Plano MD  : {out / 'benchmark_plan.md'}")
        print(f"Arquiteturas planejadas: {len(plan.get('architectures', {}))}")
        return 0

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
        conv = "sim" if r.get("converged") else "nao"
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
        print(f"API: {api.get('status')} - {api.get('n_2xx')}/"
              f"{api.get('n_probed')} endpoints 2xx, "
              f"mediana {api.get('latency_ms_median')} ms "
              f"({api.get('n_routes')} rotas registradas)")
    out = Path(cfg.output_dir).resolve()
    print(f"\nArtefatos: {out}")
    print("  - results.json / results.csv / summary.md")
    print("  - tables/*.tex  (resultados, eficiencia, robustez)")
    print("  - figures/*.png (roc, robustez, eficiencia, convergencia)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
