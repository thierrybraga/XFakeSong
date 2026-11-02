#!/usr/bin/env python3
"""Mede a latência do FRONT-END de cada família, em CPU, amostra única.

A tabela de latência publicada mede apenas a passagem direta do modelo
(``component: model_forward_only``). Isso favorece sistematicamente os
classificadores clássicos: o SVM decide em ~1 ms sobre um vetor de 183
descritores que alguém precisou extrair, enquanto as redes de áudio bruto
recebem a forma de onda praticamente sem preparo. Sem medir o front-end, a
comparação de custo de inferência não corresponde ao que roda em produção.

Protocolo idêntico ao da medição de forward: lote unitário, 2 execuções de
aquecimento, 30 medidas, mediana.

Saída: JSON em ``data/results/paper/consolidated/latencia_frontend.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.features import benchmark_frontend as bf  # noqa: E402

WARMUP_RUNS = 2
MEASURED_RUNS = 30
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 48000


def _median_ms(fn: Callable[[], object]) -> dict[str, float]:
    for _ in range(WARMUP_RUNS):
        fn()
    samples: list[float] = []
    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "mean_ms": round(statistics.fmean(samples), 2),
        "std_ms": round(statistics.pstdev(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/results/paper/consolidated/latencia_frontend.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    y = rng.standard_normal(WINDOW_SAMPLES).astype("float32") * 0.05

    frontends: dict[str, tuple[str, Callable[[], object]]] = {
        "raw": (
            "WavLM Original, HuBERT Original, RawNet2, AASIST, RawGAT-ST",
            lambda: bf.prepare_single(y, bf.FRONTEND_RAW),
        ),
        "logmel_100x80": (
            "Conformer, CCT, Res2Net",
            lambda: bf.prepare_single(
                y, bf.FRONTEND_LOGMEL, feature_dim=80, time_steps=100
            ),
        ),
        "logmel_300x128": (
            "AST",
            lambda: bf.prepare_single(
                y,
                bf.FRONTEND_LOGMEL,
                feature_dim=128,
                time_steps=300,
                n_fft=400,
            ),
        ),
        "tabular_v2_183": (
            "SVM, Random Forest",
            lambda: bf.prepare_single(y, bf.FRONTEND_TABULAR_V2),
        ),
    }

    resultados: dict[str, dict[str, object]] = {}
    for key, (consumidores, fn) in frontends.items():
        shape = np.asarray(fn()).shape
        stats = _median_ms(fn)
        stats.update(
            {
                "consumidores": consumidores,
                "output_shape": list(shape),
                "measured_runs": MEASURED_RUNS,
                "warmup_runs": WARMUP_RUNS,
                "device": "cpu",
                "batch_size": 1,
                "component": "frontend_only",
            }
        )
        resultados[key] = stats
        print(f"{key:16s} {stats['median_ms']:8.2f} ms  shape={shape}")

    payload = {
        "protocolo": {
            "device": "cpu",
            "batch_size": 1,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "component": "frontend_only",
            "sample_rate": SAMPLE_RATE,
            "window_samples": WINDOW_SAMPLES,
            "nota": (
                "Custo de preparo da entrada, excluído da tabela de latência "
                "publicada (model_forward_only). Somar ao forward do mesmo "
                "dispositivo para obter o custo de inferência ponta a ponta "
                "do modelo."
            ),
        },
        "resultados": resultados,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\ngravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
