#!/usr/bin/env python3
"""Avalia artefatos promovidos num corpus EXTERNO, sem retreinar.

POR QUE ISTO EXISTE. O protocolo do trabalho mede um unico gerador (XTTS-v2)
sobre um unico corpus (CETUC), e declara a generalizacao entre geradores como a
limitacao dominante. Ate agora nao havia como medi-la: `--codec-eval` e os
demais eixos de avaliacao vivem dentro de `_benchmark_one`, que TREINA antes.
Este script separa avaliacao de treino.

O QUE ELE NAO FAZ. Nao treina, nao promove e nao toca no `data/models`. Le o
contrato de inferencia de cada artefato promovido, prepara o corpus externo
EXATAMENTE como aquele contrato pede, e reporta.

CORRECAO DE FORMATO. O corpus externo passa pela mesma correcao aplicada ao
proprio (`apply_band_correction`: passa-baixas 7,5 kHz, remocao de DC,
renormalizacao de RMS). Sem isso a medida seria contaminada pelo artefato do
OUTRO corpus -- e o BRSpeech-DF tem um: os cinco sintetizadores dele saem em
taxas diferentes (Fish-Speech 48 kHz, XTTS/F5-TTS/ToucanTTS 24 kHz, YourTTS
16 kHz) e foram padronizados em 24 kHz, o que deixa o YourTTS com teto em 8 kHz
enquanto os outros vao a 12.

Uso:
    python scripts/benchmark/eval_cross_generator.py \\
        --dataset data/datasets/brspeech_eval.npz \\
        --out data/results/cross_generator
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("cross_generator")


def _carrega_contrato(pasta: Path) -> tuple[Path, dict] | None:
    """Devolve (caminho do modelo, contrato) do artefato promovido."""
    cfgs = sorted(pasta.glob("*_config.json"))
    if not cfgs:
        return None
    cfg = json.loads(cfgs[0].read_text(encoding="utf-8"))
    contrato = cfg.get("input_contract") or {}
    contrato.setdefault("architecture", cfg.get("architecture"))
    contrato.setdefault("input_shape", cfg.get("input_shape"))

    for padrao in ("*.keras", "*.pkl", "*.pt"):
        alvos = [p for p in sorted(pasta.glob(padrao))
                 if not p.name.endswith("_config.json")]
        if alvos:
            return alvos[0], contrato
    return None


def _prepara(X: np.ndarray, contrato: dict) -> np.ndarray:
    """Aplica o front-end que o CONTRATO declara, não o que a arquitetura sugere.

    O contrato é a fonte: foi ele que registrou, no momento da promoção, qual
    vetor o artefato entende. Um modelo tabular v1 (63 colunas) e um v2 (183)
    coexistem no repositório, e resolver pelo nome da arquitetura entregaria o
    vetor errado a um deles.
    """
    from app.domain.features import benchmark_frontend as bf

    frontend = contrato.get("feature_frontend")
    alvo = int(contrato.get("target_sequence_length") or 48000)

    if frontend == bf.FRONTEND_RAW:
        return bf.raw_audio_batch(X, target_len=alvo, crop_strategy='center')
    if frontend == bf.FRONTEND_LOGMEL:
        forma = contrato.get("input_shape") or [100, 80]
        return bf.log_mel_batch(
            X, time_steps=int(forma[0]), feature_dim=int(forma[1]),
            n_fft=contrato.get("n_fft"),
        )
    if frontend == bf.FRONTEND_TABULAR_V2:
        return bf.tabular_features_v2_batch(X)
    if frontend == bf.FRONTEND_TABULAR:
        return bf.tabular_features_batch(X)
    raise ValueError(
        f"front-end nao reconhecido no contrato: {frontend!r}. O artefato "
        "precisa declarar `feature_frontend` -- resolver por heuristica "
        "entregaria o vetor errado em silencio."
    )


def _pontua(modelo_path: Path, X: np.ndarray, contrato: dict) -> np.ndarray:
    """Pontuação de FAKE por amostra, em [0, 1]."""
    if modelo_path.suffix == ".pkl":
        import joblib

        est = joblib.load(modelo_path)
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(X))[:, 1]
        margem = np.asarray(est.decision_function(X)).ravel()
        return 1.0 / (1.0 + np.exp(-margem))

    import tensorflow as tf

    # As camadas customizadas (SincConv, GAT, AMSoftmax, AudioFeatureNormalization)
    # sao registradas por decorador no import do modulo. Sem esta linha o
    # `load_model` levanta "Could not locate class 'AudioFeatureNormalization'"
    # em TODA arquitetura Keras do escopo -- o script falharia nos nove.
    import app.domain.models.architectures.layers  # noqa: F401

    modelo = tf.keras.models.load_model(modelo_path, compile=False)
    bruto = np.asarray(modelo.predict(X, verbose=0, batch_size=32))
    if bruto.ndim == 2 and bruto.shape[1] == 2:
        # LOGITS crus em varias arquiteturas (AASIST/AMSoftmax). O contrato diz.
        if contrato.get("output_is_logits"):
            e = np.exp(bruto - bruto.max(axis=1, keepdims=True))
            return (e / e.sum(axis=1, keepdims=True))[:, 1]
        return bruto[:, 1]
    return bruto.ravel()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="NPZ externo com X_test/y_test em forma de onda")
    ap.add_argument("--artifacts-dir", default="data/models/benchmark_final")
    ap.add_argument("--out", default="data/results/cross_generator")
    ap.add_argument("--band-correction-hz", type=float, default=7500.0,
                    help="0 desliga; o default aplica a MESMA correção do "
                         "corpus próprio, sem a qual a medida carrega o "
                         "artefato de formato do corpus externo")
    ap.add_argument("--models", nargs="+", default=None,
                    help="subconjunto de pastas de artefato a avaliar")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from benchmarks.evaluate import evaluate_scores

    d = np.load(args.dataset, allow_pickle=True)
    X = np.asarray(d["X_test"] if "X_test" in d.files else d["X"])
    y = np.asarray(d["y_test"] if "y_test" in d.files else d["y"]).ravel()
    logger.info("corpus externo: %d amostras (%d bonafide, %d spoof)",
                len(y), int((y == 0).sum()), int((y == 1).sum()))

    if args.band_correction_hz and args.band_correction_hz > 0:
        from app.domain.features.benchmark_frontend import apply_band_correction

        X = apply_band_correction(X, float(args.band_correction_hz))
        logger.info("correção de formato aplicada ao corpus externo")

    raiz = ROOT / args.artifacts_dir
    pastas = sorted(p for p in raiz.iterdir() if p.is_dir())
    if args.models:
        alvo = {m.lower() for m in args.models}
        pastas = [p for p in pastas if p.name.lower() in alvo]

    saida = ROOT / args.out
    saida.mkdir(parents=True, exist_ok=True)
    resultados: dict[str, Any] = {}

    for pasta in pastas:
        item = _carrega_contrato(pasta)
        if item is None:
            logger.warning("  %-24s sem artefato/contrato — pulado", pasta.name)
            continue
        modelo_path, contrato = item
        try:
            Xp = _prepara(X, contrato)
            p = _pontua(modelo_path, Xp, contrato)
            m = evaluate_scores(y, np.clip(p, 0.0, 1.0), n_bootstrap=1000)
            resultados[pasta.name] = {
                "arquitetura": contrato.get("architecture"),
                "feature_frontend": contrato.get("feature_frontend"),
                "modelo": modelo_path.name,
                "metricas": m,
            }
            logger.info("  %-24s acc=%.4f  EER=%.4f  AUC=%.4f",
                        pasta.name, m["accuracy"], m["eer"], m["auc_roc"])
        except Exception as exc:  # noqa: BLE001
            # Registra o MOTIVO. Um `status: error` mudo já custou caro neste
            # projeto (estudo de sementes do Conformer, 2026-08-18).
            logger.warning("  %-24s FALHOU: %s", pasta.name, exc)
            resultados[pasta.name] = {"erro": str(exc)}

    destino = saida / "cross_generator_results.json"
    destino.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "band_correction_hz": args.band_correction_hz,
                "n": int(len(y)),
                "resultados": resultados,
            },
            indent=1, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    logger.info("\nresultados em %s", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
