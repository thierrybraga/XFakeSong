#!/usr/bin/env python
"""Retreino fatorial do RawGAT-ST: isola dropout e regularização L2.

MOTIVO
------
O artefato publicado do RawGAT-ST sobreajusta: treino ≈ 99,8% contra validação
≈ 88% já a partir da época 10, com o melhor `val_loss` na época 17 de 100. O
ajuste diagnosticado (2026-08-06) dobrou o dropout (0,35 -> 0,50) E triplicou o
L2 (1e-3 -> 3e-3) **ao mesmo tempo**, e nunca foi executado.

Mudar dois fatores de uma vez é exatamente o que produziu, no AST, uma causa de
colapso não atribuída por semanas -- só desfeita por uma ablação de 12 h. Este
script evita repetir o erro: cada braço muda UM fator.

DESENHO
-------
  publicado : dropout 0,35 + L2 1e-3   -> 87,55% limpo (referência, já existe)
  braço (d) : dropout 0,50 + L2 1e-3   -> contribuição do dropout
  braço (l) : dropout 0,35 + L2 3e-3   -> contribuição do L2
  braço (dl): dropout 0,50 + L2 3e-3   -> o combinado

RESULTADO (2026-08-17) — três das quatro células medidas, ambos os braços com
`decay_steps` já em 152.100:

  publicado : pico val_acc 0,8997 (época 88)
  braço (d) : val_acc 0,5000 EXATO da época 1 à 25, com TREINO em 95,4%
  braço (l) : pico val_acc 0,8984 (época 40), oscilando entre 0,53 e 0,90

Dropout 0,50 é o fator letal — o modelo memoriza o ajuste e não generaliza
nada. L2 3e-3 isolado custa 0,13 p.p. de teto e agrava a oscilação. A célula
(dl) não precisa rodar: o fatorial já a prevê pelo dropout. O código voltou a
0,35/1e-3 nas três fontes; o que sobreviveu das correções foi `decay_steps`
completo e a seleção por `val_eer`.

CUSTO: ~27 h por braço em RTX 3060 (medido: 26,85 h no run publicado).

USO
---
    python scripts/benchmark/run_rawgat_retune.py --plan-only
    python scripts/benchmark/run_rawgat_retune.py --arm d --out data/results/rawgat_arm_d
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

#: Valores do artefato publicado (o "antes" do fatorial).
PUBLICADO = {"dropout_rate": 0.35, "l2_reg_strength": 0.001}

BRACOS = {
    "d": {
        "descricao": "dropout 0,50 + L2 1e-3 (isola o dropout)",
        "overrides": {"dropout_rate": 0.50, "l2_reg_strength": 0.001},
    },
    "l": {
        "descricao": "dropout 0,35 + L2 3e-3 (isola a regularização L2)",
        "overrides": {"dropout_rate": 0.35, "l2_reg_strength": 0.003},
    },
    # Já NÃO é o que está no código: o código voltou a 0,35/1e-3 em
    # 2026-08-17. Mantido como célula do fatorial, não como configuração
    # candidata — o braço (d) mostra que o dropout 0,50 a condena.
    "dl": {
        "descricao": "dropout 0,50 + L2 3e-3 (combinado; previsto falhar)",
        "overrides": {"dropout_rate": 0.50, "l2_reg_strength": 0.003},
    },
}


def _plano(arm: str) -> dict:
    from benchmarks import planning
    from app.domain.models.architectures.registry import ArchitectureRegistry

    base = dict(planning.NEURAL_BENCHMARK_HPARAMS["rawgatst"])
    reg = dict(ArchitectureRegistry().get_architecture("RawGAT-ST").default_params)
    cfg = BRACOS[arm]
    efetivo = {**reg, **base, **cfg["overrides"]}
    return {
        "arm": arm,
        "descricao": cfg["descricao"],
        "overrides": cfg["overrides"],
        "publicado": PUBLICADO,
        "efetivo": {k: efetivo[k] for k in sorted(
            ("dropout_rate", "l2_reg_strength", "decay_steps", "learning_rate",
             "global_clipnorm", "batch_size", "optimizer", "scheduler"))
            if k in efetivo},
        "referencia_publicada": {"accuracy": 0.8755, "eer": 0.1187,
                                 "min_tdcf": 0.3149, "best_epoch": 17},
        "custo_estimado_h": 26.85,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(BRACOS))
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--dataset", default="data/datasets/benchmark_dataset_15k.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--checkpoint-monitor", default="val_eer",
                    choices=["val_eer", "val_loss"],
                    help="métrica de seleção do checkpoint (padrão: val_eer)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.plan_only:
        for arm in sorted(BRACOS):
            p = _plano(arm)
            print(f"\n=== braço {arm}: {p['descricao']} ===")
            for k, v in p["efetivo"].items():
                marca = "  <-- alterado" if k in p["overrides"] else ""
                print(f"   {k:<20} {v}{marca}")
        print(f"\npublicado (referência): dropout {PUBLICADO['dropout_rate']}, "
              f"L2 {PUBLICADO['l2_reg_strength']} -> acurácia 87,55%, EER 11,87%")
        print("\nPara executar um braço (~27 h em RTX 3060):")
        print("  python scripts/benchmark/run_rawgat_retune.py --arm d "
              "--out data/results/rawgat_arm_d")
        return 0

    if not args.arm:
        ap.error("informe --arm {d,l,dl} ou use --plan-only")

    plano = _plano(args.arm)
    destino = Path(args.out or f"data/results/rawgat_retune_{args.arm}")
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "retune_plan.json").write_text(
        json.dumps(plano, indent=1, ensure_ascii=False), encoding="utf-8")

    from benchmarks.config import BenchmarkConfig
    from benchmarks.runner import run_benchmark
    from benchmarks import planning
    from app.domain.models.architectures.registry import ArchitectureRegistry

    # As sobrescritas entram nos DOIS lugares que alimentam o create_model:
    # o plano do benchmark e os defaults do registry.
    planning.NEURAL_BENCHMARK_HPARAMS["rawgatst"].update(plano["overrides"])
    ArchitectureRegistry().get_architecture("RawGAT-ST").default_params.update(
        plano["overrides"])

    cfg = BenchmarkConfig()
    cfg.architectures = ["RawGAT-ST"]
    cfg.preset_name = f"rawgat_retune:{args.arm}"
    cfg.dataset_path = args.dataset
    cfg.output_dir = str(destino)
    cfg.epochs = args.epochs
    cfg.optimize_hyperparameters = True
    # Seleção por EER de validação, não por perda.
    #
    # O run abortado em 2026-08-17 mostrou por quê: com L2=3e-3 o mínimo de
    # `val_loss` caiu na ÉPOCA 1 (0,6923 ≈ ln 2, modelo desinformativo) e
    # nenhuma das 50 épocas seguintes o bateu — o artefato final seria um
    # modelo de 53% de acurácia. O EER mede ordenação, é imune a esse piso, e
    # é a métrica primária pela qual o protocolo avalia.
    cfg.checkpoint_monitor = str(args.checkpoint_monitor)

    logger.info("braço %s (%s) — estimado em %.1f h",
                args.arm, plano["descricao"], plano["custo_estimado_h"])
    resultado = run_benchmark(cfg)
    (destino / "retune_result.json").write_text(
        json.dumps(resultado, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    logger.info("concluído: %s", destino / "retune_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
