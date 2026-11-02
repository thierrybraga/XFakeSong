#!/usr/bin/env python3
"""Regenera os sidecars ``bench_<modelo>_config.json`` dos modelos promovidos.

Problema que corrige: os sidecars em ``data/models/benchmark_final/<arch>/``
continham contratos de runs de FUMAÇA (ex.: Conformer com 1 época e 4
amostras de validação — ``eer_threshold=0,209``), e é DELES que o
``model_loader``/``Predictor`` extraem o limiar de decisão, a temperatura e o
``input_contract`` usados na inferência do app.

Este script reconstrói cada sidecar a partir do run REAL do benchmark:

- ``eer_threshold``/``eer_value`` recalculados de
  ``data/results/predictions_clean.csv`` (ponto em que FPR = FNR);
- ``input_shape`` do ``data/results/metrics.json``;
- ``input_contract`` completo, incluindo o ``feature_frontend`` do benchmark
  (``benchmark_raw_v1``/``benchmark_logmel_v1``/``benchmark_tabular_v1``) que
  o ``FeaturePreparer`` usa para reproduzir EXATAMENTE o front-end do treino
  (``app/domain/features/benchmark_frontend.py``);
- proveniência (fonte + timestamp) para auditoria.

O sidecar antigo é preservado como ``*.stale.bak`` na primeira regeneração.

``--stamp-band-correction`` é um reparo ESTREITO, para os sidecars anteriores a
2026-11-01 (desde então o run dos clássicos já grava o campo) que o app
carrega de ``data/models/``: só acrescenta ao ``input_contract`` a política de
correção de banda que o run registrou em
``data/results/benchmark/<slug>/results.json``
(``architectures.<arch>.input_preparation.band_correction``). Limiar,
temperatura e o resto do contrato ficam intactos — a regeneração completa
recalcularia o limiar de ``predictions_clean.csv``, que é a partição de TESTE.
Só carimba quando o sha256 do artefato bate com o
``model_artifact_fingerprint`` do mesmo run. Motivo (2026-10-31): SVM e
RandomForest foram treinados com a correção, mas seus sidecars não a
declaravam; a inferência pulava o passa-baixas e o SVM acertava 4% do áudio
real. O sidecar anterior fica em ``*.json.pre-band.bak``; a escrita troca o
arquivo, sem alterar os hardlinks da saída do run.

Uso:
    python tools/reporting/rebuild_inference_contracts.py
    python tools/reporting/rebuild_inference_contracts.py --dry-run
    python tools/reporting/rebuild_inference_contracts.py --archs conformer svm
    python tools/reporting/rebuild_inference_contracts.py \\
        --stamp-band-correction svm randomforest --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.file_storage import (  # noqa: E402
    atomic_write_text,
    copy_file_atomic,
)
from tools._bootstrap import setup_logging  # noqa: E402

logger = logging.getLogger("contracts")

BENCH_FINAL = ROOT / "data" / "models" / "benchmark_final"
MODELS_DIR = ROOT / "data" / "models"
RESULTS_BENCHMARK = ROOT / "data" / "results" / "benchmark"

# arch dir -> (feature_frontend, display architecture, model_type)
ARCH_SPECS: dict[str, dict[str, str]] = {
    "aasist": {"frontend": "benchmark_raw_v1", "architecture": "AASIST",
               "model_type": "tensorflow"},
    "rawgat_st": {"frontend": "benchmark_raw_v1", "architecture": "RawGAT-ST",
                  "model_type": "tensorflow"},
    "rawnet2": {"frontend": "benchmark_raw_v1", "architecture": "RawNet2",
                "model_type": "tensorflow"},
    "conformer": {"frontend": "benchmark_logmel_v1", "architecture": "Conformer",
                  "model_type": "tensorflow"},
    "res2net": {"frontend": "benchmark_logmel_v1",
                "architecture": "MultiscaleCNN", "model_type": "tensorflow"},
    "cct": {"frontend": "benchmark_logmel_v1",
            "architecture": "Hybrid CNN-Transformer",
            "model_type": "tensorflow"},
    "ast": {"frontend": "benchmark_logmel_v1",
            "architecture": "SpectrogramTransformer",
            "model_type": "tensorflow"},
    # v2 (183 colunas), nao v1 (63). Os artefatos promovidos declaram
    # `benchmark_tabular_v2` (confira data/models/benchmark_final/svm/
    # bench_svm_config.json). Com o v1 aqui, rodar este reparo gravava um
    # contrato pedindo 63 descritores para um modelo de 183: a guarda de
    # `feature_preparer` recusa o vetor e SVM/RandomForest param de inferir.
    "svm": {"frontend": "benchmark_tabular_v2", "architecture": "SVM",
            "model_type": "sklearn"},
    "random_forest": {"frontend": "benchmark_tabular_v2",
                      "architecture": "Random Forest",
                      "model_type": "sklearn"},
    # Escopo estendido (2026-07-28). Faltavam aqui, então nunca recebiam
    # `feature_frontend` e iam para produção com o front-end errado.
    "sonic_sleuth": {"frontend": "benchmark_logmel_v1",
                     "architecture": "Sonic Sleuth",
                     "model_type": "tensorflow"},
    "efficientnet_lstm": {"frontend": "benchmark_logmel_v1",
                          "architecture": "EfficientNet-LSTM",
                          "model_type": "tensorflow"},
    "ensemble": {"frontend": "benchmark_raw_v1", "architecture": "Ensemble",
                 "model_type": "tensorflow"},
    "wavlm": {"frontend": "benchmark_raw_v1", "architecture": "WavLM",
              "model_type": "tensorflow"},
    "hubert": {"frontend": "benchmark_raw_v1", "architecture": "HuBERT",
               "model_type": "tensorflow"},
    # SSL originais (PyTorch): o runner dedicado grava calibração própria
    # (limiar de EER sob ruído) nos artefatos .pt/metrics.json — não são
    # regenerados aqui para não sobrescrever a calibração sob ruído.
}


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """(eer, threshold) no ponto em que FPR = FNR (classe positiva = spoof)."""
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fpr = fp / max(n_neg, 1)
    fnr = 1.0 - tp / max(n_pos, 1)
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(s[idx])


def load_predictions(arch_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    csv_path = arch_dir / "results" / "predictions_clean.csv"
    if not csv_path.exists():
        return None
    y_true, p_fake = [], []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            p_fake.append(float(row["p_fake"]))
    return np.asarray(y_true), np.asarray(p_fake)


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def architecture_record(results: dict, architecture: str) -> dict | None:
    """Bloco de ``results.json`` daquela arquitetura.

    Casa o nome sem caixa, espaço nem hífen: o ``ARCH_SPECS`` diz
    ``"Random Forest"`` e o runner grava ``"RandomForest"``.
    """
    alvo = _compact(architecture)
    for nome, bloco in (results.get("architectures") or {}).items():
        if _compact(nome) == alvo and isinstance(bloco, dict):
            return bloco
    return None


def band_correction_from_results(results: dict, architecture: str) -> dict | None:
    """Política de banda que o run APLICOU à arquitetura, ou ``None``.

    Lê ``input_preparation.band_correction`` — o que o runner fez com as
    partições daquela entrada —, nunca o ``config.band_correction_hz`` global,
    que é só o que foi pedido.
    """
    bloco = architecture_record(results, architecture) or {}
    banda = (bloco.get("input_preparation") or {}).get("band_correction")
    if isinstance(banda, dict) and banda.get("cutoff_hz"):
        return dict(banda)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for bloco in iter(lambda: handle.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def stamp_band_correction(
    slug: str,
    *,
    dry_run: bool = False,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_BENCHMARK,
) -> bool:
    """Acrescenta ao sidecar promovido a política de banda registrada no run.

    Só esse campo muda. Recusa quando o artefato não é o que o run produziu
    (sha256 diferente do ``model_artifact_fingerprint``), para não atribuir a
    um modelo a política de outro treino.
    """
    sidecar_path = models_dir / f"bench_{slug}_config.json"
    modelos = [
        models_dir / f"bench_{slug}{sufixo}"
        for sufixo in (".pkl", ".keras")
        if (models_dir / f"bench_{slug}{sufixo}").exists()
    ]
    results_path = results_dir / slug / "results.json"
    faltando = [
        str(p) for p in (sidecar_path, results_path) if not p.exists()
    ] + ([] if modelos else [f"bench_{slug}.(pkl|keras)"])
    if faltando:
        logger.warning("%s: ausente(s): %s — pulado", slug, ", ".join(faltando))
        return False

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contrato = dict(sidecar.get("input_contract") or {})
    if contrato.get("band_correction"):
        logger.info("%s: o contrato já declara band_correction", slug)
        return True

    results = json.loads(results_path.read_text(encoding="utf-8"))
    arquitetura = sidecar.get("architecture") or contrato.get("architecture") or slug
    bloco = architecture_record(results, arquitetura)
    if bloco is None:
        logger.error("%s: %s não aparece em %s", slug, arquitetura, results_path)
        return False
    esperado = (bloco.get("model_artifact_fingerprint") or {}).get("sha256")
    atual = _sha256(modelos[0])
    if esperado != atual:
        logger.error(
            "%s: %s não é o artefato desse run (sha256 %s, run registrou %s)",
            slug, modelos[0].name, atual[:12], str(esperado)[:12],
        )
        return False
    banda = band_correction_from_results(results, arquitetura)
    if banda is None:
        logger.warning("%s: o run não registrou correção de banda", slug)
        return False

    contrato["band_correction"] = banda
    sidecar["input_contract"] = contrato
    proveniencia = dict(sidecar.get("provenance") or {})
    try:
        origem = results_path.relative_to(ROOT).as_posix()
    except ValueError:
        origem = results_path.as_posix()
    proveniencia["band_correction"] = {
        "source": origem,
        "field": f"architectures.{arquitetura}.input_preparation.band_correction",
        "model_sha256": atual,
        "stamped_by": "tools/reporting/rebuild_inference_contracts.py",
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar["provenance"] = proveniencia
    logger.info(
        "%s: band_correction %s (sha256 do artefato confere)", slug, banda
    )
    if dry_run:
        return True

    backup = sidecar_path.with_suffix(".json.pre-band.bak")
    if not backup.exists():
        copy_file_atomic(sidecar_path, backup)
    atomic_write_text(
        sidecar_path, json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n"
    )
    return True


def build_contract(arch_key: str, spec: dict, metrics: dict,
                   eer: float | None, threshold: float | None,
                   temperature: float = 1.0) -> dict:
    frontend = spec["frontend"]
    input_shape = metrics.get("input_shape")
    if not input_shape:
        input_shape = {
            # 48000 = janela canonica de 3 s @ 16 kHz (o 16000 anterior era de
            # uma convencao antiga e produzia contrato quebrado em silencio).
            "benchmark_raw_v1": [48000, 1],
            "benchmark_logmel_v1": [100, 80],
            "benchmark_tabular_v1": [63],
            # v2 = os 63 do v1 + 120 descritores LFCC (20 estaticos, delta e
            # delta-delta, com media e desvio de cada bloco).
            "benchmark_tabular_v2": [183],
        }[frontend]
    input_shape = [int(v) for v in input_shape]

    contract: dict = {
        "architecture": spec["architecture"],
        "input_shape": input_shape,
        "sample_rate": 16000,
        "feature_frontend": frontend,
        "source_samples": 48000,  # janela canônica de 3 s do benchmark
        "normalization": "per_sample_zscore",
        "label_classes": [0, 1],
        # A temperatura calibrada e parte da ESCALA em que o eer_threshold foi
        # derivado: o `model_loader` le este campo e o `Predictor` a aplica.
        # Fixa-la em 1,0 fazia a producao aplicar o limiar numa escala diferente
        # da avaliada — o mesmo defeito ja corrigido no runner do benchmark.
        "temperature": float(temperature),
        "scaler_applied": False,
    }
    if frontend == "benchmark_raw_v1":
        contract.update({
            "type": "audio", "format": "raw", "input_type": "raw_audio",
            "target_sequence_length": int(input_shape[0]),
        })
    elif frontend == "benchmark_logmel_v1":
        contract.update({
            "type": "features", "format": "spectrogram",
            "input_type": "spectrogram",
            "time_steps": int(input_shape[0]),
            "feature_dim": int(input_shape[1]),
        })
    else:  # tabular
        contract.update({
            "type": "features", "format": "tabular", "input_type": "tabular",
            "feature_dim": int(input_shape[0]),
            # O artefato clássico é um Pipeline sklearn com o scaler DENTRO;
            # nenhum scaler externo deve ser aplicado pela inferência.
            "normalization": "pipeline_interno",
        })
    if threshold is not None:
        contract["eer_threshold"] = round(float(threshold), 6)
    if eer is not None:
        contract["eer_value"] = round(float(eer), 6)
    return contract


def rebuild(arch_key: str, dry_run: bool = False) -> bool:
    spec = ARCH_SPECS[arch_key]
    arch_dir = BENCH_FINAL / arch_key
    metrics_path = arch_dir / "results" / "metrics.json"
    if not metrics_path.exists():
        logger.warning("%s: metrics.json ausente — pulado", arch_key)
        return False

    models = sorted(arch_dir.glob("bench_*.keras")) + sorted(
        arch_dir.glob("bench_*.pkl")
    )
    models = [m for m in models if "_scaler" not in m.stem]
    if not models:
        logger.warning("%s: artefato bench_* ausente — pulado", arch_key)
        return False
    model_path = models[0]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    preds = load_predictions(arch_dir)
    eer = threshold = None
    if preds is not None:
        eer, threshold = compute_eer(*preds)

    # Preserva o que o treino gravou e este reparo NAO sabe derivar.
    #
    # `build_contract` monta um dicionario LITERAL a partir do frontend e do
    # metrics.json. Tudo que so o runner conhece — a politica de correcao de
    # banda, a estrategia de crop resolvida na avaliacao, a janela de STFT do
    # AST, a convencao de logits, o limiar de OOD — nao tem como ser
    # reconstruido aqui. Ate 2026-08-20 so a `temperature` era resgatada, e
    # rodar o reparo APAGAVA o resto em silencio: o AST voltava a inferir com a
    # janela derivada em vez dos 25 ms do artigo, e as arquiteturas raw-audio
    # perdiam o multicrop que o benchmark usou para medir.
    previous_contract: dict = {}
    sidecar_path = arch_dir / f"{model_path.stem}_config.json"
    if sidecar_path.exists():
        try:
            previous = json.loads(sidecar_path.read_text(encoding="utf-8"))
            previous_contract = dict(previous.get("input_contract") or {})
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            previous_contract = {}
    try:
        temperature = float(previous_contract.get("temperature", 1.0))
    except (TypeError, ValueError):
        temperature = 1.0

    contract = build_contract(arch_key, spec, metrics, eer, threshold, temperature)

    #: Campos que so o treino/benchmark produz. Se existirem no sidecar
    #: anterior, sobrevivem ao reparo.
    PRESERVADOS = (
        "band_correction",
        "crop_strategy",
        "eval_num_crops",
        "eval_score_aggregation",
        "n_fft",
        "hop_length",
        "output_is_logits",
        "ood_threshold",
        "feature_frontend_reason",
        "target_sequence_length",
    )
    resgatados = []
    for chave in PRESERVADOS:
        if chave in previous_contract and chave not in contract:
            contract[chave] = previous_contract[chave]
            resgatados.append(chave)
    # Sidecar anterior sem a política de banda: o run a registra no
    # results.json (é o caso dos clássicos), então ela é DERIVADA, não perdida.
    results_path = arch_dir / "results" / "results.json"
    if "band_correction" not in contract and results_path.exists():
        banda = band_correction_from_results(
            json.loads(results_path.read_text(encoding="utf-8")),
            spec["architecture"],
        )
        if banda:
            contract["band_correction"] = banda
            resgatados.append("band_correction (results.json)")
    if resgatados:
        logger.info(
            "%s: campos preservados do sidecar anterior: %s",
            arch_key,
            ", ".join(resgatados),
        )
    sidecar = {
        "architecture": spec["architecture"],
        "input_shape": contract["input_shape"],
        "num_classes": 2,
        "label_classes": [0, 1],
        "model_type": spec["model_type"],
        "input_contract": contract,
        "provenance": {
            "source": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
            "rebuilt_by": "tools/reporting/rebuild_inference_contracts.py",
            "rebuilt_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "note": (
                "Sidecar regenerado do run real do benchmark; o anterior "
                "continha contrato de run de fumaça (ver *.stale.bak)."
            ),
        },
    }

    config_path = model_path.parent / f"{model_path.stem}_config.json"
    logger.info(
        "%s: %s  eer=%.4f thr=%.4f frontend=%s shape=%s",
        arch_key, config_path.name,
        eer if eer is not None else float("nan"),
        threshold if threshold is not None else float("nan"),
        contract["feature_frontend"], contract["input_shape"],
    )
    if dry_run:
        return True

    if config_path.exists():
        backup = config_path.with_suffix(".json.stale.bak")
        if not backup.exists():
            copy_file_atomic(config_path, backup)
    atomic_write_text(
        config_path,
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs", nargs="+", choices=sorted(ARCH_SPECS),
        default=sorted(ARCH_SPECS),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stamp-band-correction",
        nargs="+",
        metavar="SLUG",
        help=(
            "Só acrescenta a política de banda do run aos sidecars de "
            "data/models/bench_<SLUG>_config.json (ex.: svm randomforest)."
        ),
    )
    args = parser.parse_args(argv)

    setup_logging(name="contracts")
    if args.stamp_band_correction:
        done = sum(
            1
            for slug in args.stamp_band_correction
            if stamp_band_correction(slug, dry_run=args.dry_run)
        )
        logger.info(
            "%d/%d sidecars %s", done, len(args.stamp_band_correction),
            "conferidos (dry-run)" if args.dry_run else "com a política de banda",
        )
        return 0 if done == len(args.stamp_band_correction) else 1

    done = sum(1 for arch in args.archs if rebuild(arch, dry_run=args.dry_run))
    logger.info("%d/%d sidecars %s", done, len(args.archs),
                "validados (dry-run)" if args.dry_run else "regenerados")
    return 0 if done == len(args.archs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
