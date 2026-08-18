#!/usr/bin/env python
"""Ablação fatorial do AST: isola pré-treinamento e resolução de entrada.

MOTIVO
------
No recorte oficial o AST é o único modelo que acumula DUAS vantagens não
controladas sobre as demais redes espectrais: parte de pesos AudioSet
transferidos e consome a grade 300x128, contra 100x80 de Conformer/CCT/Res2Net.
A comparação publicada, portanto, confunde arquitetura, resolução e
pré-treinamento.

Pior: a configuração histórica que colapsou (EER ~51%) combinava grade 100x80 E
treino do zero, e o commit af416a1 (2026-07-27) corrigiu as duas ao mesmo tempo
-- a causa nunca foi atribuída.

DESENHO
-------
Dois braços, cada um variando UM fator em relação ao publicado:

  publicado : 300x128 + pretrained=True   -> 99,71% (referência, já executado)
  braço (a) : 300x128 + pretrained=False  -> contribuição do pré-treinamento
  braço (b) : 100x80  + pretrained=True   -> contribuição da resolução

Cada braço é uma execução completa de 100 épocas (~11 h em RTX 3060). Use
--plan-only para validar a configuração sem treinar.

USO
---
    python scripts/benchmark/run_ast_ablation.py --plan-only
    python scripts/benchmark/run_ast_ablation.py --arm a --out data/results/ast_abl_a
    python scripts/benchmark/run_ast_ablation.py --arm b --out data/results/ast_abl_b
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

#: Sobrescritas por braço. Cada uma altera UM fator sobre o plano publicado.
BRACOS = {
    "a": {
        "descricao": "300x128 + pretrained=False (isola o pré-treinamento)",
        "hparams": {"pretrained": False},
        "input_requirements": {},
    },
    "b": {
        "descricao": "100x80 + pretrained=True (isola a resolução)",
        "hparams": {},
        # n_fft volta ao default derivado do salto (1024 p/ hop 480), como nas
        # demais redes espectrais; ver benchmark_frontend.resolve_n_fft.
        "input_requirements": {"min_sequence_length": 100, "feature_dim": 80,
                               "n_fft": None},
    },
}


def _plano(arm: str) -> dict:
    """Monta o plano efetivo do braço, sem treinar."""
    from app.domain.models.architectures.registry import ArchitectureRegistry
    from benchmarks import planning

    info = ArchitectureRegistry().get_architecture("SpectrogramTransformer")
    base_req = dict(getattr(info, "input_requirements", {}) or {})
    base_hp = dict(planning.NEURAL_BENCHMARK_HPARAMS.get("spectrogramtransformer", {}))

    cfg = BRACOS[arm]
    req = {**base_req, **cfg["input_requirements"]}
    req = {k: v for k, v in req.items() if v is not None}
    hp = {**base_hp, **cfg["hparams"]}
    return {
        "arm": arm,
        "descricao": cfg["descricao"],
        "input_requirements": req,
        "hyperparameters": hp,
        "referencia_publicada": {
            "input_requirements": base_req,
            "pretrained": base_hp.get("pretrained"),
            "accuracy": 0.9971,
            "eer": 0.0014,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(BRACOS), help="braço da ablação")
    ap.add_argument("--plan-only", action="store_true",
                    help="valida e imprime o plano dos dois braços, sem treinar")
    ap.add_argument("--dataset", default="data/datasets/benchmark_dataset_15k.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.plan_only:
        for arm in sorted(BRACOS):
            p = _plano(arm)
            print(f"\n=== braço {arm}: {p['descricao']} ===")
            print(json.dumps(p["input_requirements"], indent=1, ensure_ascii=False))
            print(f"  pretrained = {p['hyperparameters'].get('pretrained')}")
            ref = p["referencia_publicada"]
            print(f"  referência publicada: {ref['input_requirements'].get('min_sequence_length')}"
                  f"x{ref['input_requirements'].get('feature_dim')}, "
                  f"pretrained={ref['pretrained']}, acc={ref['accuracy']:.4f}")
        print("\nPlano válido. Para executar um braço (~11 h em RTX 3060):")
        print("  python scripts/benchmark/run_ast_ablation.py --arm a "
              "--out data/results/ast_abl_a")
        return 0

    if not args.arm:
        ap.error("informe --arm {a,b} ou use --plan-only")

    plano = _plano(args.arm)
    destino = Path(args.out or f"data/results/ast_ablation_{args.arm}")
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "ablation_plan.json").write_text(
        json.dumps(plano, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("plano gravado em %s", destino / "ablation_plan.json")

    # A execução reusa o runner oficial (mesmo caminho de run_benchmark.py),
    # aplicando as sobrescritas do braço ANTES de chamá-lo.
    from benchmarks.config import BenchmarkConfig
    from benchmarks.runner import run_benchmark
    from benchmarks import planning
    from app.domain.models.architectures.registry import ArchitectureRegistry

    cfg = BenchmarkConfig()
    cfg.architectures = ["SpectrogramTransformer"]
    cfg.preset_name = f"ast_ablation:{args.arm}"
    cfg.dataset_path = args.dataset
    cfg.output_dir = str(destino)
    cfg.epochs = args.epochs
    cfg.optimize_hyperparameters = True

    # Sobrescritas do braço: hiperparâmetros (pretrained) e contrato de entrada.
    hp = planning.NEURAL_BENCHMARK_HPARAMS.get("spectrogramtransformer")
    if hp is not None and plano["hyperparameters"]:
        hp.update({k: v for k, v in BRACOS[args.arm]["hparams"].items()})
    req_over = BRACOS[args.arm]["input_requirements"]
    if req_over:
        info = ArchitectureRegistry().get_architecture("SpectrogramTransformer")
        for chave, valor in req_over.items():
            if valor is None:
                info.input_requirements.pop(chave, None)
            else:
                info.input_requirements[chave] = valor

    logger.info("iniciando braço %s (%s) — isto leva horas",
                args.arm, plano["descricao"])
    resultado = run_benchmark(cfg)
    (destino / "ablation_result.json").write_text(
        json.dumps(resultado, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    logger.info("concluído: %s", destino / "ablation_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
