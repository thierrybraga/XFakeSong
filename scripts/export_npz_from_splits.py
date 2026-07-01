#!/usr/bin/env python3
"""Exporta um .npz de áudio bruto a partir de app/datasets/splits.

Uso típico depois de `scripts/build_dataset.py --tier medium`:

    python scripts/export_npz_from_splits.py \
        --out app/datasets/benchmark_audio_raw_balanced_15k.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_tcc_pipeline import export_npz_from_splits


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporta dataset .npz a partir dos splits existentes."
    )
    parser.add_argument(
        "--splits-dir",
        default="app/datasets/splits",
        help="Diretório com train/val/test já preparados.",
    )
    parser.add_argument(
        "--out",
        default="app/datasets/benchmark_audio_raw_balanced_15k.npz",
        help="Arquivo .npz de saída.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limite opcional por classe em cada split (debug/smoke).",
    )
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
    )
    print(f"NPZ pronto: {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
