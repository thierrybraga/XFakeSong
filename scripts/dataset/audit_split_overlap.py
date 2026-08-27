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
    # Hashea SPLIT A SPLIT, sem `X_all`.
    #
    # O `np.concatenate([X_train, X_val, X_test])` materializava os tres splits
    # E a copia concatenada: ~2x o dataset descomprimido (~16 GB no completo),
    # so para computar hashes que nunca precisaram das tres particoes juntas.
    # Numa estacao de 32 GB isso tornava a auditoria do dataset canonico
    # impraticavel — e auditoria que nao roda nao audita.
    with np.load(dataset_path, allow_pickle=True) as z:
        n_train, n_val, n_test = len(z["y_train"]), len(z["y_val"]), len(z["y_test"])
        audio_hashes = {
            "train": _audio_hashes(z["X_train"]),
            "val": _audio_hashes(z["X_val"]),
            "test": _audio_hashes(z["X_test"]),
        }
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

    # VEREDITO.
    #
    # As dimensoes sao EXIGIDAS, nao opcionais. Antes, cada bloco so entrava no
    # relatorio se a chave existisse no .npz, e o veredito olhava apenas os
    # blocos PRESENTES — entao um dataset sem `speaker_ids`/`utterance_ids`/
    # `text_ids` era declarado `passed: true` sem que a disjuncao de locutor,
    # enunciado ou texto fosse testada uma vez sequer. Comparacao de conjuntos
    # vazios passa trivialmente; ausencia de prova nao e prova de ausencia.
    #
    # Mesmo raciocinio para a cobertura de locutor: `speaker_known` todo falso
    # produz conjuntos vazios em todos os splits, interseccao zero e um
    # "sem sobreposicao" que nao mediu nada.
    exigidas = ("audio_sha256", "sample_paths", "known_speakers",
                "utterance_ids", "text_ids")
    failures = []
    for section in exigidas:
        if section not in report:
            failures.append(f"{section}.ausente")
            continue
        for pair_key, count in report[section]["overlap"].items():
            if count > 0:
                failures.append(f"{section}.{pair_key}={count}")
    cobertura = (report.get("known_speakers") or {}).get("unique_per_split") or {}
    if cobertura and not any(int(v) > 0 for v in cobertura.values()):
        failures.append("known_speakers.cobertura_zero")
    report["passed"] = not failures
    report["failures"] = failures
    report["required_dimensions"] = list(exigidas)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="grava o relatorio JSON aqui (default: nao grava, so imprime)",
    )
    # DEFAULT INVERTIDO. Era `store_true` (opt-in): rodado como o README
    # documenta, o script detectava sobreposicao, gravava `"passed": false` no
    # JSON e saia com codigo 0 — em CI ou num script de pipeline o vazamento
    # passava como sucesso. Uma auditoria que nao reprova nao e guarda-corpo.
    parser.add_argument(
        "--fail-on-overlap", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "retorna codigo de saida != 0 se qualquer sobreposicao for "
            "encontrada (padrao: ligado; use --no-fail-on-overlap para uso "
            "exploratorio)"
        ),
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
