#!/usr/bin/env python3
"""
Pré-processamento e validação de datasets de áudio para o SpeechChecker.

Funcionalidades:
  - Verifica integridade dos WAVs (corrompidos, silenciosos)
  - Reamostra para 16kHz mono
  - Normaliza amplitude
  - Remove amostras fora do range de duração (1-30s)
  - Remove duplicatas por hash
  - Gera relatório detalhado
  - Cria splits train/val/test estratificados

Uso:
  python tools/dataset/preprocess_dataset.py --validate          # Apenas validar
  python tools/dataset/preprocess_dataset.py --normalize         # Normalizar todos
  python tools/dataset/preprocess_dataset.py --create-splits     # Criar train/val/test
  python tools/dataset/preprocess_dataset.py --full              # Tudo acima
  python tools/dataset/preprocess_dataset.py --create-zip        # Gerar ZIP para upload na UI

CLI fina: a lógica vive em ``app/domain/services/dataset/preprocessing.py``,
a mesma que a aba de datasets do Gradio usa.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.services.dataset.preprocessing import (  # noqa: E402
    audit_near_duplicates,
    create_splits,
    create_training_zip,
    normalize_all,
    remove_duplicates,
    validate_dataset,
)

logger = logging.getLogger("DatasetPreprocessor")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-processamento de dataset de audio")
    parser.add_argument("--validate", action="store_true", help="Validar integridade")
    parser.add_argument("--normalize", action="store_true", help="Normalizar todos os WAVs")
    parser.add_argument("--force-normalize", action="store_true",
                        help="Re-normaliza tudo, ignorando o cache (.normalized_manifest.json)")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remover duplicatas")
    parser.add_argument(
        "--audit-near-duplicates",
        action="store_true",
        help="Auditar quase-duplicatas por fingerprint log-Mel",
    )
    parser.add_argument("--create-splits", action="store_true", help="Criar splits train/val/test")
    parser.add_argument("--create-zip", action="store_true", help="Criar ZIP para upload na UI")
    parser.add_argument("--full", action="store_true", help="Executar pipeline completo")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Proporção de treino (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Proporção de validação (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Proporção de teste (default: 0.15)",
    )
    parser.add_argument(
        "--speaker-disjoint", action="store_true",
        help=(
            "Split disjunto por falante (usuários não vistos). Com cobertura "
            "parcial de speaker_id (ex.: uma fonte pareada sem metadados de "
            "locutor), usa automaticamente o modo híbrido: disjunto por "
            "falante na fatia identificada + rebalanceamento de classe na "
            "fatia sem identidade, preservando o tamanho total do dataset."
        ),
    )
    parser.add_argument(
        "--expected-per-class",
        type=int,
        help="Falha se a camada processada nao tiver exatamente este total por classe",
    )
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        metavar="CLASSE:PREFIXO=TOTAL",
        help="Quota aplicada apos validacao/dedup; ex.: real:brspeech=3750",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not any([args.validate, args.normalize, args.remove_duplicates,
                args.audit_near_duplicates, args.create_splits, args.create_zip, args.full]):
        parser.print_help()
        return

    # Validar ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.001:
        logger.error(f"Soma dos ratios deve ser 1.0 (atual: {total:.3f})")
        sys.exit(1)

    if args.full or args.validate:
        validate_dataset()

    if args.full or args.normalize:
        normalize_all(force=args.force_normalize)

    if args.full or args.remove_duplicates:
        remove_duplicates()

    if args.full or args.audit_near_duplicates:
        audit_near_duplicates()

    source_quotas: dict[str, dict[str, int]] = {}
    for spec in args.source_quota:
        try:
            class_prefix, raw_total = spec.split("=", 1)
            class_name, prefix = class_prefix.split(":", 1)
            class_name = class_name.strip().lower()
            prefix = prefix.strip().lower()
            total = int(raw_total)
        except (TypeError, ValueError):
            parser.error(f"quota invalida: {spec!r}")
        if class_name not in {"real", "fake"} or not prefix or total <= 0:
            parser.error(f"quota invalida: {spec!r}")
        source_quotas.setdefault(class_name, {})[prefix] = total

    if source_quotas and args.expected_per_class:
        for class_name in ("real", "fake"):
            if sum(source_quotas.get(class_name, {}).values()) != args.expected_per_class:
                parser.error(
                    f"quotas de {class_name} devem somar "
                    f"--expected-per-class={args.expected_per_class}"
                )

    if args.full or args.create_splits:
        create_splits(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            speaker_disjoint=args.speaker_disjoint,
            expected_per_class=args.expected_per_class,
            source_quotas=source_quotas or None,
        )

    if args.create_zip:
        create_training_zip()

    logger.info("\nPipeline concluido!")


if __name__ == "__main__":
    main()
