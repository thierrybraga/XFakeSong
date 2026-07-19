#!/usr/bin/env python3
"""Sela um NOVO teste intocado antes de qualquer treinamento ou inspeção."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.run_models_sequential import _inspect_npz, _sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <dataset>.test-lock.json",
    )
    parser.add_argument(
        "--declare-untouched",
        action="store_true",
        help="declara que o teste foi criado agora e não orientou decisões anteriores",
    )
    args = parser.parse_args()
    if not args.declare_untouched:
        parser.error(
            "o selo só pode ser criado com --declare-untouched após gerar um NOVO "
            "teste que não tenha sido inspecionado"
        )

    dataset = args.dataset.resolve()
    if not dataset.exists():
        parser.error(f"dataset não encontrado: {dataset}")
    inspection = _inspect_npz(dataset)
    if not inspection["predefined_splits"]:
        parser.error("o dataset precisa conter train/val/test predefinidos")
    out = (
        args.out.resolve()
        if args.out
        else dataset.with_suffix(dataset.suffix + ".test-lock.json")
    )
    if out.exists():
        parser.error(
            f"selo já existe e não será sobrescrito: {out}. Gere outro dataset/versione-o."
        )

    payload = {
        "protocol_version": "xfakesong-test-lock-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset),
        "dataset_size_bytes": dataset.stat().st_size,
        "dataset_sha256": _sha256_file(dataset),
        "test_archive_identity_sha256": inspection["test_archive_identity_sha256"],
        "test_archive_identity_method": inspection["test_archive_identity_method"],
        "split_counts": inspection["split_counts"],
        "declared_untouched": True,
        "created_before_training": True,
        "declaration": (
            "A partição de teste foi definida antes do treinamento confirmatório e "
            "não foi usada para selecionar arquitetura, hiperparâmetros ou correções."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Teste selado: {out}")
    print(f"Dataset SHA-256: {payload['dataset_sha256']}")
    print(f"Teste identity: {payload['test_archive_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())