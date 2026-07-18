#!/usr/bin/env python3
"""Auditoria de SHORTCUT de fonte (rigor acadêmico, 2026-07-14).

Contexto: no dataset canônico, três das quatro fontes são PURAS de classe
(mlspt/ttsport só real; fkvoice só fake) — apenas brspeech tem pares
real↔clone. Um detector pode então atingir alta acurácia identificando a
ASSINATURA DE FONTE (canal, microfone, corpus) em vez de artefatos de
síntese. Este script quantifica o confounder:

1. Treina um classificador raso (RandomForest) para prever a FONTE a
   partir do MESMO vetor tabular de 63 descritores usado pelos clássicos
   do benchmark (validação cruzada estratificada).
2. Reporta acurácia por fonte e a acurácia real→fake implicada pela
   regra trivial "fonte → classe majoritária da fonte".

Interpretação: se a fonte é prevista com acurácia ≫ acaso, o atalho é
mensurável e deve ser reportado como ameaça à validade no artigo (e
mitigado com o reteste cross-generator / speaker-disjoint).

Uso:
    python scripts/dataset/audit_source_shortcut.py \
        --dataset data/datasets/benchmark_audio_raw_balanced_15k.npz \
        --max-per-source 500 --out results/04_helpers/source_shortcut.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/datasets/benchmark_audio_raw_balanced_15k.npz",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=500,
        help="subamostra por fonte (63 descritores por clipe são custosos)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="results/04_helpers/source_shortcut.json"
    )
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    from app.domain.features.benchmark_frontend import tabular_features_batch
    from benchmarks.data import BenchmarkData

    data = BenchmarkData.from_npz(str(ROOT / args.dataset))
    if data.groups is None:
        print("Dataset sem proveniência de fonte (groups) — nada a auditar.")
        return 2

    groups = np.asarray(data.groups).astype(str)
    y_class = np.asarray(data.y)
    rng = np.random.default_rng(args.seed)

    # Subamostra balanceada por fonte
    idx_parts = []
    for src in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == src)
        if len(idx) > args.max_per_source:
            idx = rng.choice(idx, args.max_per_source, replace=False)
        idx_parts.append(idx)
    idx_all = np.concatenate(idx_parts)
    rng.shuffle(idx_all)

    print(f"Extraindo 63 descritores de {len(idx_all)} clipes...")
    X = tabular_features_batch(
        np.asarray(data.X[idx_all], dtype="float32").reshape(len(idx_all), -1)
    )
    y_src = groups[idx_all]
    y_cls = y_class[idx_all]

    clf = RandomForestClassifier(
        n_estimators=200, random_state=args.seed, n_jobs=-1
    )
    pred_src = cross_val_predict(clf, X, y_src, cv=5, n_jobs=-1)

    acc_src = float((pred_src == y_src).mean())
    n_sources = len(set(y_src.tolist()))
    chance = max(Counter(y_src.tolist()).values()) / len(y_src)

    # Regra trivial: fonte prevista → classe majoritária daquela fonte.
    src_to_class = {
        src: int(round(float(y_cls[y_src == src].mean())))
        for src in set(y_src.tolist())
    }
    pred_cls_via_src = np.array([src_to_class[s] for s in pred_src])
    acc_cls_via_src = float((pred_cls_via_src == y_cls).mean())

    per_source = {
        src: float((pred_src[y_src == src] == src).mean())
        for src in sorted(set(y_src.tolist()))
    }

    report = {
        "dataset": str(args.dataset),
        "n_samples_audited": int(len(idx_all)),
        "n_sources": n_sources,
        "source_accuracy_cv5": acc_src,
        "source_chance_level": float(chance),
        "per_source_accuracy": per_source,
        "class_accuracy_via_source_rule": acc_cls_via_src,
        "source_to_majority_class": src_to_class,
        "interpretation": (
            "source_accuracy_cv5 >> chance indica assinatura de fonte "
            "detectável nos mesmos features usados pelos classificadores; "
            "class_accuracy_via_source_rule é o teto de acurácia real/fake "
            "atingível SEM detectar síntese (só identificando o corpus). "
            "Reportar como ameaça à validade; mitigar com cross-generator "
            "e speaker-disjoint."
        ),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRelatório: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
