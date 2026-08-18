#!/usr/bin/env python3
"""Auditoria independente de sobreposicao entre treino/validacao/teste.

Verifica, diretamente nos arrays do .npz (o que os modelos realmente
consomem, nao os WAVs de origem), quatro dimensoes de vazamento entre
splits:

  1. audio  - hash SHA-256 do array de forma de onda (bytes exatos).
  2. path   - `sample_paths` (arquivo de origem).
  3. falante- `speaker_ids` restrito a `speaker_known=True` (falantes sem
              identidade explicita nao contam como "o mesmo falante" so
              por compartilharem o rotulo de fonte de fallback).
  4. conteudo - `utterance_ids`/`text_ids`, quando presentes.

Formaliza (e substitui) a checagem ad-hoc que antes so existia como
`data/datasets/splits/independent_hash_audit.json` sem script associado no
repositorio - esse arquivo nao podia ser regerado nem auditado de novo.

Uso:
    python scripts/dataset/audit_split_overlap.py \
        --dataset data/datasets/benchmark_audio_raw_balanced_15k_academic_v2.npz \
        --out data/datasets/splits/independent_hash_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _audio_hashes(X: np.ndarray) -> list[str]:
    return [
        hashlib.sha256(np.ascontiguousarray(X[i]).tobytes()).hexdigest()
        for i in range(len(X))
    ]


def _overlap(a: set, b: set) -> list:
    return sorted(str(x) for x in (a & b))[:20]


def audit(dataset_path: Path) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as z:
        n_train, n_val, n_test = len(z["y_train"]), len(z["y_val"]), len(z["y_test"])
        X_all = np.concatenate([z["X_train"], z["X_val"], z["X_test"]], axis=0)
        sample_paths = z["sample_paths"] if "sample_paths" in z else None
        speaker_ids = z["speaker_ids"] if "speaker_ids" in z else None
        speaker_known = z["speaker_known"] if "speaker_known" in z else None
        utterance_ids = z["utterance_ids"] if "utterance_ids" in z else None
        text_ids = z["text_ids"] if "text_ids" in z else None

    idx = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_train + n_val + n_test),
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "counts": {name: (sl.stop - sl.start) for name, sl in idx.items()},
    }

    audio_hashes = {name: _audio_hashes(X_all[sl]) for name, sl in idx.items()}
    report["audio_sha256"] = {
        "unique_per_split": {name: len(set(h)) for name, h in audio_hashes.items()},
        "overlap": {
            f"{a}_{b}": len(set(audio_hashes[a]) & set(audio_hashes[b]))
            for a, b in pairs
        },
    }

    if sample_paths is not None:
        paths = {name: set(sample_paths[sl].tolist()) for name, sl in idx.items()}
        report["sample_paths"] = {
            "overlap": {f"{a}_{b}": len(paths[a] & paths[b]) for a, b in pairs},
        }

    if speaker_ids is not None and speaker_known is not None:
        known_speakers = {
            name: set(speaker_ids[sl][speaker_known[sl]].tolist())
            for name, sl in idx.items()
        }
        report["known_speakers"] = {
            "unique_per_split": {name: len(s) for name, s in known_speakers.items()},
            "overlap": {
                f"{a}_{b}": len(known_speakers[a] & known_speakers[b])
                for a, b in pairs
            },
        }

    for field_name, field in (("utterance_ids", utterance_ids), ("text_ids", text_ids)):
        if field is None:
            continue
        values = {name: set(field[sl].tolist()) for name, sl in idx.items()}
        report[field_name] = {
            "overlap": {f"{a}_{b}": len(values[a] & values[b]) for a, b in pairs},
        }

    failures = []
    for section in ("audio_sha256", "sample_paths", "known_speakers", "utterance_ids", "text_ids"):
        if section not in report:
            continue
        for pair_key, count in report[section]["overlap"].items():
            if count > 0:
                failures.append(f"{section}.{pair_key}={count}")
    report["passed"] = not failures
    report["failures"] = failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="grava o relatorio JSON aqui (default: nao grava, so imprime)",
    )
    parser.add_argument(
        "--fail-on-overlap", action="store_true",
        help="retorna codigo de saida != 0 se qualquer sobreposicao for encontrada",
    )
    args = parser.parse_args()

    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    if not dataset_path.exists():
        parser.error(f"dataset nao encontrado: {dataset_path}")

    report = audit(dataset_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRelatorio salvo: {args.out}")

    if args.fail_on_overlap and not report["passed"]:
        print(f"\nFALHA: sobreposicao entre splits detectada: {report['failures']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
