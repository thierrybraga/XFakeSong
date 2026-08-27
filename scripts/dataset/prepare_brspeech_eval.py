#!/usr/bin/env python3
"""Prepara o BRSpeech-DF como conjunto de teste EXTERNO, em formato canonico.

POR QUE. O corpus proprio mede um unico gerador (XTTS-v2). O BRSpeech-DF traz
CINCO sintetizadores zero-shot em portugues -- Fish-Speech, XTTS, F5-TTS,
YourTTS e ToucanTTS -- com bonafide proprio (62 locutores do LibriVox via
CML-TTS). Avaliar nele responde a limitacao dominante do trabalho: a
generalizacao para geradores nao vistos.

NAO E PARA TREINAR. Este NPZ e conjunto de TESTE. Misturar as duas
procedencias no treino recriaria o confundimento que o projeto acabou de
remover: dois pipelines de gravacao e reamostragem distintos, e o detector
aprende a distinguir CORPUS em vez de sintese.

A ASSIMETRIA DELE. Os cinco sintetizadores saem em taxas diferentes --
Fish-Speech 48 kHz, XTTS/F5-TTS/ToucanTTS 24 kHz, YourTTS 16 kHz -- e o
dataset padronizou tudo em 24 kHz. Isso deixa o YourTTS com teto espectral em
8 kHz enquanto os demais chegam a 12, ou seja, uma assinatura de reamostragem
POR GERADOR, do mesmo tipo que este projeto encontrou no proprio corpus.

A correcao canonica (`apply_band_correction`: passa-baixas 7,5 kHz, remocao de
DC, renormalizacao de RMS) e aplicada aqui, o que resolve os dois lados: leva
tudo para 16 kHz efetivos e apaga a diferenca entre os geradores. Sem ela, uma
avaliacao neste corpus mediria o artefato DELES.

Uso:
    python scripts/dataset/prepare_brspeech_eval.py --per-generator 300
    python scripts/dataset/prepare_brspeech_eval.py --plan
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("brspeech_eval")

REPO = "AKCIT-Deepfake/BRSpeech-DF"
SR_ALVO = 16000
JANELA = 48000  # 3 s, a janela canonica do protocolo
DESTINO = ROOT / "data" / "datasets" / "brspeech_eval.npz"


def _canonicaliza(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Mono, 16 kHz, janela central de 3 s. Devolve None se curto demais."""
    import soxr

    x = np.asarray(audio, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR_ALVO:
        x = soxr.resample(x, sr, SR_ALVO, quality="HQ").astype("float32")
    if len(x) < JANELA:
        # DESCARTA em vez de repetir: o tiling cria periodicidade artificial, e
        # num conjunto de TESTE isso seria uma pista que nao existe em producao.
        return None
    ini = (len(x) - JANELA) // 2
    return x[ini : ini + JANELA]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--per-generator", type=int, default=300,
                    help="amostras spoof por sintetizador (e o mesmo total de "
                         "bonafide, para manter 1:1)")
    ap.add_argument("--plan", action="store_true",
                    help="inspeciona o dataset e imprime a composicao, sem baixar tudo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DESTINO))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "falta a biblioteca `datasets`. Instale com:\n"
            "    pip install datasets soxr"
        )
        return 2

    logger.info("abrindo %s em streaming (o dataset tem 983 h)", REPO)
    ds = load_dataset(REPO, split="train", streaming=True)

    if args.plan:
        vistos: dict[str, int] = {}
        taxas: dict[str, set] = {}
        for i, ex in enumerate(ds):
            g = str(ex.get("model") or ex.get("generator") or ex.get("label"))
            vistos[g] = vistos.get(g, 0) + 1
            a = ex.get("audio") or {}
            if a.get("sampling_rate"):
                taxas.setdefault(g, set()).add(int(a["sampling_rate"]))
            if i >= 2000:
                break
        logger.info("\ncomposicao nas primeiras 2000 amostras:")
        for g, n in sorted(vistos.items(), key=lambda x: -x[1]):
            logger.info("   %-28s %5d   taxas=%s", g, n, sorted(taxas.get(g, [])))
        logger.info("\ncampos disponiveis: %s", sorted(ex.keys()))
        return 0

    rng = np.random.default_rng(args.seed)
    por_gerador: dict[str, list] = {}
    bonafide: list = []

    for ex in ds:
        rotulo = ex.get("label")
        gerador = str(ex.get("model") or ex.get("generator") or "desconhecido")
        a = ex.get("audio") or {}
        if a.get("array") is None:
            continue
        janela = _canonicaliza(np.asarray(a["array"]), int(a["sampling_rate"]))
        if janela is None:
            continue

        e_falso = rotulo in (1, "1", "fake", "spoof")
        if e_falso:
            alvo = por_gerador.setdefault(gerador, [])
            if len(alvo) < args.per_generator:
                alvo.append(janela)
        elif len(bonafide) < args.per_generator * 5:
            bonafide.append(janela)

        completo = (
            len(por_gerador) >= 5
            and all(len(v) >= args.per_generator for v in por_gerador.values())
            and len(bonafide) >= sum(len(v) for v in por_gerador.values())
        )
        if completo:
            break

    spoof = [w for v in por_gerador.values() for w in v]
    n = min(len(spoof), len(bonafide))
    if n == 0:
        logger.error("nada coletado — confira os nomes de campo com --plan")
        return 1
    idx_b = rng.permutation(len(bonafide))[:n]

    X = np.concatenate([np.array(bonafide)[idx_b], np.array(spoof)[:n]])
    y = np.concatenate([np.zeros(n, dtype="int64"), np.ones(n, dtype="int64")])
    geradores = np.array(
        ["bonafide"] * n
        + [g for g, v in por_gerador.items() for _ in v][:n]
    )

    from app.domain.features.benchmark_frontend import apply_band_correction

    X = apply_band_correction(X)
    logger.info("correcao de formato aplicada (banda + DC + RMS)")

    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino,
        X_test=X[:, :, None].astype("float32"),
        y_test=y,
        generator_ids=geradores,
        metadata_json=json.dumps(
            {
                "fonte": REPO,
                "papel": "conjunto de teste EXTERNO — nao usar em treino",
                "band_correction": "aplicada no preparo",
                "por_gerador": {g: len(v) for g, v in por_gerador.items()},
                "n": int(len(y)),
            },
            ensure_ascii=False,
        ),
    )
    logger.info("\n%d amostras (%d bonafide, %d spoof) -> %s",
                len(y), n, n, destino)
    for g, v in sorted(por_gerador.items()):
        logger.info("   %-28s %d", g, len(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
