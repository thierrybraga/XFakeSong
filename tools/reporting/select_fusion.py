#!/usr/bin/env python3
"""Mede a fusão entre modelos, pela pontuação centrada, só com a validação.

Usa os modelos de ``--models-dir`` pelo mesmo caminho da inferência, com a calibração que o
sidecar declara. Limiares e conjunto saem de dez locutores, e a medida vem do décimo primeiro
(ver ``benchmarks/fusion.py``). Não abre a partição de teste nem altera nada no modelo.

Os SSL não estão entre os modelos do app. Para incluí-los, aponte a pasta do artefato e a do
backbone. Os da geração atual ficam no run:

    --ssl data/results/benchmark/wavlm_original/architectures/wavlm_original/models=<backbone>

O rótulo vem da última pasta (aqui, ``bench_models``), então um segundo ``--ssl`` de pasta com
o mesmo nome substitui o primeiro.

Uso:
    python tools/reporting/select_fusion.py --report fusao.json
    python tools/reporting/select_fusion.py --models bench_aasist bench_hybrid_cnn_transformer \\
        --snr 10 5 --set bench_aasist,bench_hybrid_cnn_transformer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.contracts.base import ProcessingStatus  # noqa: E402
from app.utils.file_storage import atomic_write_text  # noqa: E402
from tools._bootstrap import setup_logging  # noqa: E402

logger = logging.getLogger("select_fusion")


def production_p_fake(
    predictor: Any, info: Any, prepared: np.ndarray, batch_size: int = 16
) -> np.ndarray:
    """P(fake) como a inferência calcula, em lotes."""
    saida: List[float] = []
    for inicio in range(0, len(prepared), batch_size):
        lote = predictor.predict_batch(info, list(prepared[inicio : inicio + batch_size]))
        if lote.status != ProcessingStatus.SUCCESS:
            raise RuntimeError(f"{info.name}: {lote.errors}")
        saida.extend(float(linha["p_fake"]) for linha in lote.data)
    return np.asarray(saida)


def ssl_p_fake(artifact_dir: Path, backbone_dir: str, waveforms: np.ndarray) -> np.ndarray:
    """P(fake) de um SSL original, pelo wrapper de inferência."""
    from app.domain.services.detection.model_loader import TorchSSLOriginalModel

    artefato = next(artifact_dir.glob("bench_*.pt"))
    metadados = json.loads(
        (artifact_dir / f"{artefato.stem}_config.json").read_text(encoding="utf-8")
    )
    metadados["backbone_artifact"] = backbone_dir
    saida = TorchSSLOriginalModel(artefato, metadados).predict(waveforms, batch_size=8)
    return np.asarray(saida, dtype=np.float64)[:, 1]


def collect_scores(
    models_dir: Path,
    dataset: Path,
    names: Optional[Sequence[str]],
    snr_levels: Sequence[int],
    ssl: Sequence[str] = (),
) -> Dict[str, Any]:
    """P(fake) por modelo e condição. Os SSL entram só na condição limpa."""
    from app.domain.services.detection.predictor import Predictor
    from app.domain.services.detection_service import DetectionService
    from benchmarks.validation_scores import load_validation, prepared_input

    validacao = load_validation(dataset)
    if validacao.speakers is None:
        raise ValueError(f"{dataset} não tem speaker_ids alinhados à validação")
    servico = DetectionService(models_dir=str(models_dir))
    predictor = Predictor()
    limpo: Dict[str, np.ndarray] = {}
    ruidoso: Dict[str, Dict[str, np.ndarray]] = {f"{s} dB": {} for s in snr_levels}
    for nome in names or servico.get_available_models():
        info = servico.model_loader.get_model(nome)
        limpo[nome] = production_p_fake(predictor, info, prepared_input(validacao, info))
        logger.info("%s: validação limpa medida", nome)
        for snr in snr_levels:
            ruidoso[f"{snr} dB"][nome] = production_p_fake(
                predictor, info, prepared_input(validacao, info, snr_db=snr)
            )
            logger.info("%s: %s dB medido", nome, snr)
    for item in ssl:
        pasta, _, backbone = item.partition("=")
        caminho = Path(pasta)
        limpo[f"bench_{caminho.name}"] = ssl_p_fake(
            caminho, backbone, validacao.corrected(7500.0)
        )
        logger.info("%s: validação limpa medida", caminho.name)
    return {
        "labels": validacao.labels,
        "speakers": validacao.speakers,
        "clean": limpo,
        "noisy": ruidoso,
    }


def format_table(linhas: Sequence[Dict[str, Any]]) -> str:
    saida = ["condição | regra | EER % | acurácia em 0,5 %"]
    for linha in linhas:
        saida.append(
            f"{linha['condition']:>6} | {linha['rule']:<70} | "
            f"{linha['eer'] * 100:6.2f} | {linha['accuracy_at_center'] * 100:6.2f}"
        )
    return "\n".join(saida)


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
    parser.add_argument("--snr", nargs="*", type=int, default=[], help="ex.: 10 5")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="fixed_sets",
        help="conjunto fixado antes da medida, separado por vírgula; pode repetir",
    )
    parser.add_argument(
        "--ssl",
        action="append",
        default=[],
        metavar="ARTEFATO=BACKBONE",
        help="pasta do artefato SSL e pasta do backbone; pode repetir",
    )
    parser.add_argument("--report", type=Path, help="grava o relatório em JSON")
    args = parser.parse_args(argv)

    setup_logging(name="select_fusion")
    from benchmarks.fusion import compare_fusions

    dados = collect_scores(
        args.models_dir, args.dataset, args.models, args.snr, args.ssl
    )
    linhas = compare_fusions(
        dados["clean"],
        dados["labels"],
        dados["speakers"],
        dados["noisy"],
        fixed_sets=[conjunto.split(",") for conjunto in args.fixed_sets],
    )
    logger.info("\n%s", format_table(linhas))
    if args.report:
        atomic_write_text(
            args.report, json.dumps(linhas, indent=2, ensure_ascii=False) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
