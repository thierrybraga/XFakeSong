#!/usr/bin/env python3
"""Recalibra a temperatura dos modelos neurais promovidos, sem treinar.

Refaz a calibração na validação, sobre o logit real do modelo (ver
``app/domain/services/detection/temperature_scaling.py``), para quem tem no sidecar uma
temperatura da grade antiga. Regrava temperatura, limiar e limiar de OOD, e declara
``temperature_on_logits``.

O limiar é transportado, não reestimado: o mesmo log-odds na nova escala. Assim nenhum
veredito da validação muda, e a ferramenta recusa o modelo se algum mudar. EER, AUC e t-DCF
não mudam. O que muda é ``p_fake``, a pontuação e a confiança mostradas.

Os ``results.json`` do run continuam descrevendo a calibração antiga, inclusive a acurácia
no limiar fixo 0,5 e o ECE: quem precisar dos números novos publica um run novo.

Simula por padrão. ``--apply`` guarda o sidecar em ``*.json.pre-temperature.bak``.

Uso:
    python tools/reporting/recalibrate_temperature.py
    python tools/reporting/recalibrate_temperature.py --models bench_aasist --report saida.json
    python tools/reporting/recalibrate_temperature.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.services.detection.operating_point import (  # noqa: E402
    verdict_confidence,
)
from app.domain.services.detection.predictor import (  # noqa: E402
    compute_energy_score,
)
from app.domain.services.detection.temperature_scaling import (  # noqa: E402
    calibrated_probabilities,
    fit_temperature,
    logit_output_model,
    logits_from_probabilities,
    transported_threshold,
)
from app.utils.file_storage import (  # noqa: E402
    atomic_write_text,
    copy_file_atomic,
)
from tools._bootstrap import setup_logging  # noqa: E402

logger = logging.getLogger("recalibrate_temperature")

#: Mesmo quantil de `TrainingConfig.ood_quantile`, que gerou os limiares atuais.
OOD_QUANTILE = 0.95
SATURACAO = float(np.log((1.0 - 1e-7) / 1e-7))


def _p_fake(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities)
    return p[:, 1] if p.ndim > 1 and p.shape[-1] > 1 else p.ravel()


def _nll(y: np.ndarray, p_fake: np.ndarray) -> float:
    p = np.clip(p_fake, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def card_calibration_error(
    y: np.ndarray, p_fake: np.ndarray, threshold: float, bins: int = 10
) -> float:
    """Distância entre a confiança mostrada no cartão e o acerto real, por faixa."""
    confianca = np.array([verdict_confidence(p, threshold) for p in p_fake])
    acertos = ((p_fake >= threshold).astype(int) == y).astype(float)
    cortes = np.linspace(0.5, 1.0, bins + 1)
    erro = 0.0
    for a, b in zip(cortes[:-1], cortes[1:]):
        faixa = (confianca >= a) & (confianca <= b if b == 1.0 else confianca < b)
        if faixa.any():
            erro += faixa.mean() * abs(acertos[faixa].mean() - confianca[faixa].mean())
    return float(erro)


def recalibrate(
    logits: np.ndarray, labels: np.ndarray, contract: Dict[str, Any]
) -> Dict[str, Any]:
    """Calibração nova e o que ela muda, sem gravar nada."""
    y = np.asarray(labels).astype(int).ravel()
    t_antes = float(contract.get("temperature", 1.0))
    limiar_antes = float(contract["eer_threshold"])
    antes = calibrated_probabilities(
        logits_from_probabilities(calibrated_probabilities(logits)), t_antes
    )
    t_depois = fit_temperature(logits, y)
    limiar_depois = transported_threshold(limiar_antes, t_antes, t_depois)
    depois = calibrated_probabilities(logits, t_depois)
    p_antes, p_depois = _p_fake(antes), _p_fake(depois)
    z = np.asarray(logits, dtype=np.float64)
    log_odds = z[:, 1] - z[:, 0] if z.ndim > 1 and z.shape[-1] > 1 else z.ravel()
    return {
        "temperature": t_depois,
        "eer_threshold": limiar_depois,
        "ood_threshold": float(
            np.quantile(
                compute_energy_score(depois, temperature=t_depois), 1.0 - OOD_QUANTILE
            )
        ),
        "previous": {
            "temperature": t_antes,
            "eer_threshold": limiar_antes,
            "ood_threshold": contract.get("ood_threshold"),
        },
        "changed_verdicts": int(
            ((p_antes >= limiar_antes) != (p_depois >= limiar_depois)).sum()
        ),
        "saturated_fraction": float((np.abs(log_odds) > SATURACAO).mean()),
        "nll": {"before": _nll(y, p_antes), "after": _nll(y, p_depois)},
        "card_calibration_error": {
            "before": card_calibration_error(y, p_antes, limiar_antes),
            "after": card_calibration_error(y, p_depois, limiar_depois),
        },
    }


def write_sidecar(sidecar_path: Path, resultado: Dict[str, Any], dataset: Path) -> None:
    """Grava o contrato novo, guardando o anterior na primeira vez."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contrato = dict(sidecar.get("input_contract") or {})
    contrato.update(
        temperature=resultado["temperature"],
        eer_threshold=resultado["eer_threshold"],
        ood_threshold=resultado["ood_threshold"],
        temperature_on_logits=True,
    )
    sidecar["input_contract"] = contrato
    proveniencia = dict(sidecar.get("provenance") or {})
    proveniencia["temperature_recalibration"] = {
        "previous": resultado["previous"],
        "dataset": dataset.name,
        "partition": "validation",
        "recalibrated_by": "tools/reporting/recalibrate_temperature.py",
        "recalibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar["provenance"] = proveniencia
    backup = sidecar_path.with_suffix(".json.pre-temperature.bak")
    if not backup.exists():
        copy_file_atomic(sidecar_path, backup)
    atomic_write_text(
        sidecar_path, json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n"
    )


def run(
    models_dir: Path,
    dataset: Path,
    names: Optional[Sequence[str]] = None,
    apply: bool = False,
    batch_size: int = 16,
) -> Dict[str, Any]:
    from app.domain.services.detection_service import DetectionService
    from benchmarks.validation_scores import load_validation, prepared_input

    validacao = load_validation(dataset)
    servico = DetectionService(models_dir=str(models_dir))
    relatorio: Dict[str, Any] = {}
    for nome in names or servico.get_available_models():
        info = servico.model_loader.get_model(nome)
        contrato = info.input_contract if isinstance(info.input_contract, dict) else {}
        if info.model_type != "tensorflow":
            relatorio[nome] = {"skipped": "sem temperatura: modelo clássico"}
        elif contrato.get("temperature_on_logits"):
            relatorio[nome] = {"skipped": "já calibrado sobre logits"}
        elif contrato.get("eer_threshold") is None:
            relatorio[nome] = {"skipped": "contrato sem limiar para transportar"}
        else:
            logit_model = logit_output_model(info.model)
            if logit_model is None:
                relatorio[nome] = {"skipped": "sem acesso ao logit"}
                continue
            logits = logit_model.predict(
                prepared_input(validacao, info), batch_size=batch_size, verbose=0
            )
            resultado = recalibrate(logits, validacao.labels, contrato)
            relatorio[nome] = resultado
            logger.info(
                "%s: T %.3f -> %.3f | limiar %.4f -> %.4f | saturados %.1f%% | "
                "vereditos alterados %d | NLL %.4f -> %.4f | erro do cartão %.4f -> %.4f",
                nome,
                resultado["previous"]["temperature"],
                resultado["temperature"],
                resultado["previous"]["eer_threshold"],
                resultado["eer_threshold"],
                resultado["saturated_fraction"] * 100,
                resultado["changed_verdicts"],
                resultado["nll"]["before"],
                resultado["nll"]["after"],
                resultado["card_calibration_error"]["before"],
                resultado["card_calibration_error"]["after"],
            )
            if resultado["changed_verdicts"]:
                logger.error("%s: recusado, a validação mudaria de veredito", nome)
            elif apply:
                write_sidecar(models_dir / f"{nome}_config.json", resultado, dataset)
    return relatorio


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=ROOT / "data" / "models")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "datasets" / "benchmark_dataset_15k.npz",
    )
    parser.add_argument("--models", nargs="+", help="nomes como bench_aasist")
    parser.add_argument("--apply", action="store_true", help="grava os sidecars")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", type=Path, help="grava o relatório em JSON")
    args = parser.parse_args(argv)

    setup_logging(name="recalibrate_temperature")
    relatorio = run(
        args.models_dir, args.dataset, args.models, args.apply, args.batch_size
    )
    if args.report:
        atomic_write_text(
            args.report, json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n"
        )
    recusados = [
        nome for nome, r in relatorio.items() if r.get("changed_verdicts")
    ]
    if not args.apply:
        logger.info("Simulação: nenhum sidecar foi alterado. Use --apply para gravar.")
    return 1 if recusados else 0


if __name__ == "__main__":
    raise SystemExit(main())
