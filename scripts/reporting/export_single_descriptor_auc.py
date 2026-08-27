#!/usr/bin/env python3
"""Teto de um detector trivial: AUC de descritor único na janela de teste.

O protocolo audita vazamento de partição (locutor, frase, texto, hash) e de
metadado (oráculo majoritário). Nenhum dos dois alcança a classe de atalho que
mora no PRÓPRIO SINAL: assimetrias de empacotamento entre as duas fontes
(nível, banda, taxa de amostragem) separariam as classes sem que artefato de
síntese algum fosse detectado.

Este script mede, sobre a janela de 3 s que o modelo efetivamente recebe, a AUC
que um detector de um único descritor escalar alcançaria. O maior valor é o
teto trivial do corpus — a régua contra a qual as AUC dos onze modelos devem
ser lidas.

Saída: JSON em ``data/results/paper/consolidated/descritor_unico_teste.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EPS = 1e-12
SAMPLE_RATE = 16000


def _db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.asarray(values, dtype="float64") + EPS)


def _relative_band_db(
    power: np.ndarray, freqs: np.ndarray, low: float, high: float
) -> np.ndarray:
    """Energia da banda relativa à energia total, em dB (invariante a nível)."""
    mask = (freqs >= low) & (freqs < high)
    total = power.sum(axis=1) + EPS
    return 10.0 * np.log10(power[:, mask].sum(axis=1) / total + EPS)


def descriptors(X: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
    rms_db = _db(np.sqrt(np.mean(flat**2, axis=1)))
    peak_db = _db(np.max(np.abs(flat), axis=1))
    window = np.hanning(flat.shape[1]).astype("float32")
    power = np.abs(np.fft.rfft(flat * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(flat.shape[1], 1.0 / SAMPLE_RATE)
    total = power.sum(axis=1) + EPS
    return {
        # Invariante a escala: é o que sobra depois da normalização de RMS.
        "crest_db": peak_db - rms_db,
        "band_6k_7k_rel_db": _relative_band_db(power, freqs, 6000, 7000),
        "zcr": np.mean(
            np.signbit(flat[:, 1:]) != np.signbit(flat[:, :-1]), axis=1
        ),
        # Banda onde apareceria a assinatura de reamostragem 24 -> 16 kHz.
        "band_7k_8k_rel_db": _relative_band_db(power, freqs, 7000, 8000),
        "spectral_centroid_hz": (power * freqs).sum(axis=1) / total,
        # Descritor de EMPACOTAMENTO: precisa ficar no acaso.
        "rms_db": rms_db,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/datasets/benchmark_dataset_15k.npz",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/results/paper/consolidated/descritor_unico_teste.json",
    )
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    data = np.load(args.dataset, allow_pickle=True)
    X = data[f"X_{args.split}"]
    y = np.asarray(data[f"y_{args.split}"]).ravel().astype(int)

    resultados: dict[str, dict[str, float]] = {}
    for name, values in descriptors(X).items():
        auc = roc_auc_score(y, values)
        # AUC de descritor único é orientada: 1 - auc é o mesmo detector com o
        # sinal invertido. O teto é max(auc, 1 - auc).
        auc = max(float(auc), 1.0 - float(auc))
        resultados[name] = {
            "auc": round(auc, 4),
            "bonafide_mediana": round(float(np.median(values[y == 0])), 4),
            "spoof_mediana": round(float(np.median(values[y == 1])), 4),
        }
        print(f"{name:22s} AUC={auc:.4f}")

    teto = max(resultados.items(), key=lambda kv: kv[1]["auc"])
    payload = {
        "protocolo": {
            "dataset": str(args.dataset.name),
            "split": args.split,
            "n": int(len(y)),
            "janela": "3 s (48.000 amostras) — a mesma que o modelo recebe",
            "nota": (
                "AUC de detector de descritor único. O maior valor é o teto de "
                "um detector trivial: um modelo próximo dele não aprendeu mais "
                "que uma estatística escalar."
            ),
        },
        "teto_trivial": {"descritor": teto[0], "auc": teto[1]["auc"]},
        "resultados": resultados,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nteto trivial: {teto[0]} (AUC {teto[1]['auc']})\ngravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
