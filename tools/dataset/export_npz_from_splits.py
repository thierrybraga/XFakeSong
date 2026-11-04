#!/usr/bin/env python3
"""Exporta um .npz do fluxo genérico por tiers a partir de splits existentes.

Este exportador de janela de 5 s não gera o dataset de 3 s do artigo. Para
reconstruir o recorte CETUC–XTTS-v2, use `export_paired_npz.py`.

Uso típico depois de `tools/dataset/build_dataset.py --tier medium`:

    python tools/dataset/export_npz_from_splits.py \
        --out data/datasets/benchmark_audio_raw_balanced_15k.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_tcc_pipeline import export_npz_from_splits


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta um NPZ do fluxo genérico por tiers; não gera o artefato "
            "de 3 s do artigo."
        )
    )
    parser.add_argument(
        "--splits-dir",
        default="data/datasets/splits",
        help="Diretório com train/val/test já preparados.",
    )
    parser.add_argument(
        "--out",
        default="data/datasets/benchmark_audio_raw_balanced_15k.npz",
        help="Arquivo .npz de saída.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=5.0,
        help="Janela do fluxo genérico (5 s por padrão; o artigo usa 3 s).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limite opcional por classe em cada split (debug/smoke).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    out_npz = Path(args.out)
    if not splits_dir.is_absolute():
        splits_dir = ROOT / splits_dir
    if not out_npz.is_absolute():
        out_npz = ROOT / out_npz

    for split in ("train", "val", "test"):
        split_dir = splits_dir / split
        if not split_dir.exists():
            parser.error(f"Split ausente: {split_dir}")

    export_npz_from_splits(
        splits_dir=splits_dir,
        out_npz=out_npz,
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print(f"NPZ pronto: {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
