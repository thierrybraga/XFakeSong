#!/usr/bin/env python
"""Latência dos 11 modelos em UM ÚNICO dispositivo, com protocolo idêntico.

MOTIVO
------
As latências publicadas foram medidas em três ambientes distintos (Keras/GPU,
PyTorch/GPU, scikit-learn/CPU), e por isso os artefatos declaram
`cross_runtime_comparable: false`. Comparar 15,77 ms de PyTorch com 44,46 ms de
Keras separa *runtime* e dispositivo, não arquitetura.

Este script remove essa confusão medindo tudo em **CPU** -- o único denominador
comum, já que SVM e RandomForest (scikit-learn) não têm caminho de GPU.

PROTOCOLO (idêntico ao publicado, para o método ser comparável)
--------------------------------------------------------------
  * lote 1 (amostra única)
  * 2 execuções de aquecimento, 30 medidas
  * mediana como estatística principal; p95, média e desvio também gravados
  * component = model_forward_only: NÃO inclui extração de características
    nem pós-processamento, como no artefato original

A entrada de cada modelo segue o seu próprio contrato (áudio bruto 48000,
log-Mel 100x80 ou 300x128, vetor tabular 183) -- é o que "forward" significa
para cada família.

USO
---
    python scripts/benchmark/latency_single_device.py --out data/results/paper/consolidated/latencia_cpu.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

BASE = RAIZ / "data" / "models" / "benchmark_final"

#: (rótulo do TCC, pasta, tipo de runtime, forma da entrada sem o lote)
MODELOS = [
    ("Random Forest", "randomforest", "sklearn", (183,)),
    ("SVM", "svm", "sklearn", (183,)),
    ("CCT", "hybrid_cnn_transformer", "keras", (100, 80)),
    ("AST", "spectrogramtransformer", "keras", (300, 128)),
    ("Res2Net", "multiscalecnn", "keras", (100, 80)),
    ("Conformer", "conformer", "keras", (100, 80)),
    ("RawNet2", "rawnet2", "keras", (48000, 1)),
    ("AASIST", "aasist", "keras", (48000, 1)),
    ("RawGAT-ST", "rawgat_st", "keras", (48000, 1)),
    ("WavLM Original", "wavlm_original", "torch", (48000,)),
    ("HuBERT Original", "hubert_original", "torch", (48000,)),
]

WARMUP = 2
RUNS = 30


def _stats(ms: list[float]) -> dict:
    ms_ord = sorted(ms)
    return {
        "median_ms": round(statistics.median(ms), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
        "std_ms": round(statistics.pstdev(ms), 2) if len(ms) > 1 else 0.0,
        "p95_ms": round(ms_ord[max(0, int(0.95 * len(ms_ord)) - 1)], 2),
        "min_ms": round(min(ms), 2),
        "measured_runs": len(ms),
        "warmup_runs": WARMUP,
    }


def _cronometra(fn, entrada) -> dict:
    for _ in range(WARMUP):
        fn(entrada)
    ms = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(entrada)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(ms)


def _mede_sklearn(pasta: str, forma) -> dict:
    import joblib
    obj = joblib.load(next((BASE / pasta).glob("*.pkl")))
    modelo = obj.get("model", obj) if isinstance(obj, dict) else obj
    x = np.random.randn(1, *forma).astype("float64")
    return _cronometra(lambda e: modelo.predict_proba(e), x)


def _registra_camadas() -> None:
    """Importa os módulos cujas camadas usam @register_keras_serializable.

    Sem isto o `load_model` falha com "Could not locate class 'X'": o
    decorador só popula o registro do Keras quando o módulo é importado, e
    carregar um `.keras` não importa nada por conta própria.
    """
    import app.domain.models.architectures.layers  # noqa: F401
    for mod in ("conformer", "rawnet2", "aasist", "rawgat_st",
                "spectrogram_transformer", "hybrid_cnn_transformer",
                "multiscale_cnn"):
        try:
            __import__(f"app.domain.models.architectures.{mod}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("módulo %s não importado: %s", mod, exc)


def _mede_keras(pasta: str, forma) -> dict:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # força CPU
    from tensorflow import keras
    _registra_camadas()
    caminho = next((BASE / pasta).glob("*.keras"))
    modelo = keras.models.load_model(caminho, compile=False, safe_mode=False)
    x = np.random.randn(1, *forma).astype("float32")
    fn = getattr(modelo, "predict_on_batch", None) or modelo
    return _cronometra(lambda e: fn(e), x)


def _mede_torch(pasta: str, forma) -> dict:
    import torch
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) // 2))
    ckpt = torch.load(next((BASE / pasta).glob("*.pt")), map_location="cpu",
                      weights_only=False)
    from transformers import AutoModel
    sub = "wavlm_backbone" if "wavlm" in pasta else "hubert_backbone"
    # O backbone congelado não é copiado para benchmark_final/: vive no run
    # consolidado. Aceita as duas localizações para o script não depender de
    # qual delas existe na máquina.
    candidatos = [
        BASE / pasta / sub,
        RAIZ / "data" / "results" / "clean_benchmark_15k" / pasta /
        "architectures" / pasta / "models" / sub,
    ]
    origem = next((c for c in candidatos if (c / "config.json").exists()), None)
    if origem is None:
        raise FileNotFoundError(
            f"backbone {sub} não encontrado em: "
            + " | ".join(str(c) for c in candidatos))
    backbone = AutoModel.from_pretrained(str(origem)).eval()
    estado = ckpt.get("classifier_state_dict") if isinstance(ckpt, dict) else None
    cfg = (ckpt.get("embedding_config") or {}) if isinstance(ckpt, dict) else {}
    from app.domain.models.inference.ssl_head import build_ssl_classifier
    cabeca = build_ssl_classifier(cfg, 0.0).eval()
    if estado:
        cabeca.load_state_dict(estado)
    x = torch.randn(1, *forma)

    @torch.no_grad()
    def passo(e):
        saida = backbone(e, output_hidden_states=True)
        from app.domain.models.inference.ssl_head import pool_hidden_states
        return cabeca(pool_hidden_states(saida, cfg))

    return _cronometra(passo, x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/results/paper/consolidated/latencia_cpu.json")
    ap.add_argument("--only", nargs="*", help="mede apenas estes rótulos")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import platform
    resultados, falhas = {}, {}
    for rotulo, pasta, runtime, forma in MODELOS:
        if args.only and rotulo not in args.only:
            continue
        try:
            fn = {"sklearn": _mede_sklearn, "keras": _mede_keras,
                  "torch": _mede_torch}[runtime]
            r = fn(pasta, forma)
            r.update({"runtime": runtime, "device": "cpu", "batch_size": 1,
                      "component": "model_forward_only",
                      "input_shape": list(forma)})
            resultados[rotulo] = r
            logger.info("%-18s %8.2f ms (p95 %.2f)", rotulo, r["median_ms"],
                        r["p95_ms"])
        except Exception as exc:  # noqa: BLE001
            falhas[rotulo] = f"{type(exc).__name__}: {exc}"
            logger.warning("%-18s FALHOU: %s", rotulo, exc)

    saida = {
        "protocolo": {
            "device": "cpu", "batch_size": 1, "warmup_runs": WARMUP,
            "measured_runs": RUNS, "component": "model_forward_only",
            "cross_runtime_comparable": True,
            "nota": "Todos os 11 modelos no MESMO dispositivo (CPU). Substitui "
                    "as latências publicadas, medidas em 3 runtimes distintos.",
            "cpu": platform.processor() or platform.machine(),
        },
        "resultados": resultados,
        "falhas": falhas,
    }
    destino = Path(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=1, ensure_ascii=False),
                       encoding="utf-8", newline="\n")
    logger.info("\ngravado: %s  (%d ok, %d falhas)", destino, len(resultados),
                len(falhas))
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
