#!/usr/bin/env python3
"""
build_dataset.py — Orquestrador da Fase 1 (Dataset Real e Balanceamento).

Composicao definida:
  REAL  (2.000 amostras):
    - 1.000 x BRSpeech-DF bonafide/   (fala real PT-BR, padrão ASVspoof)
    - 1.000 x CommonVoice v17 PT + FLEURS PT-BR (diversidade de falantes)

  FAKE  (2.000 amostras):
    - 1.000 x BRSpeech-DF spoof/      (múltiplos geradores TTS/VC)
    - 1.000 x Fake Voices XTTS        (gerador independente — cross-generator)

  TOTAL : 4.000 amostras balanceadas 1:1
  SPLIT : 70% treino / 15% validacao / 15% teste (estratificado)

Uso:
  python scripts/build_dataset.py                        # executa tudo
  python scripts/build_dataset.py --target 1000          # 1000 por classe (debug)
  python scripts/build_dataset.py --skip-download        # apenas preprocessa
  python scripts/build_dataset.py --only-splits          # apenas recria os splits
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BuildDataset")

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"

# ---------------------------------------------------------------------------
# Composicao do dataset (valores padrao — ajustavel via --target)
# ---------------------------------------------------------------------------
COMPOSITION = {
    "target_per_class": 2000,
    "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    "sources": {
        "real": [
            {
                "name": "BRSpeech-DF bonafide",
                "hf_repo": "AKCIT-Deepfake/BRSpeech-DF",
                "type": "real",
                "target": 1000,
                "file_prefix": "brspeech",
                "script": "download_pt_datasets_v2.py",
                "args": ["--brspeech"],
            },
            {
                "name": "CommonVoice v17 PT + FLEURS PT-BR",
                "hf_repo": "mozilla-foundation/common_voice_17_0 + google/fleurs",
                "type": "real",
                "target": 1000,
                "file_prefix": "cetuc|fleurs",
                "script": "download_portuguese_datasets.py",
                "args": ["--cetuc"],
            },
        ],
        "fake": [
            {
                "name": "BRSpeech-DF spoof",
                "hf_repo": "AKCIT-Deepfake/BRSpeech-DF",
                "type": "fake",
                "target": 1000,
                "file_prefix": "brspeech",
                "script": "download_pt_datasets_v2.py",
                "args": ["--brspeech"],
            },
            {
                "name": "Fake Voices XTTS (unfake/fake_voices)",
                "hf_repo": "unfake/fake_voices",
                "type": "fake",
                "target": 1000,
                "file_prefix": "fkvoice|fakevoice",
                "script": "download_pt_datasets_v2.py",
                "args": ["--fake-voices"],
            },
        ],
    },
}


def run(cmd: list, description: str) -> int:
    """Executa um subprocesso e loga o resultado."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXECUTANDO: {description}")
    logger.info(f"  Comando: {' '.join(str(c) for c in cmd)}")
    logger.info(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        logger.error(f"FALHOU (codigo {result.returncode}): {description}")
    else:
        logger.info(f"OK: {description}")
    return result.returncode


def count_wavs(directory: Path, prefixes: list = None) -> int:
    """Conta WAVs no diretorio, opcionalmente filtrando por prefixo."""
    if not directory.exists():
        return 0
    if not prefixes:
        return len(list(directory.glob("*.wav")))
    total = 0
    for prefix in prefixes:
        total += len(list(directory.glob(f"{prefix}_*.wav")))
    return total


def print_status():
    """Imprime situacao atual do dataset."""
    real_total = count_wavs(REAL_DIR)
    fake_total = count_wavs(FAKE_DIR)

    brspeech_real = count_wavs(REAL_DIR, ["brspeech"])
    cv_real = count_wavs(REAL_DIR, ["cetuc", "cv", "fleurs"])
    brspeech_fake = count_wavs(FAKE_DIR, ["brspeech"])
    xtts_fake = count_wavs(FAKE_DIR, ["fkvoice", "fakevoice"])

    logger.info("\n" + "=" * 60)
    logger.info("STATUS ATUAL DO DATASET")
    logger.info("=" * 60)
    logger.info(f"  REAL  total : {real_total:>5d}")
    logger.info(f"    BRSpeech bonafide : {brspeech_real:>5d}  (meta: 1000)")
    logger.info(f"    CommonVoice/FLEURS: {cv_real:>5d}  (meta: 1000)")
    logger.info(f"  FAKE  total : {fake_total:>5d}")
    logger.info(f"    BRSpeech spoof    : {brspeech_fake:>5d}  (meta: 1000)")
    logger.info(f"    Fake Voices XTTS  : {xtts_fake:>5d}  (meta: 1000)")
    logger.info(f"  TOTAL       : {real_total + fake_total:>5d}  (meta: 4000)")
    logger.info("=" * 60)

    return real_total, fake_total


def step_download(target_per_class: int, skip_real_cv: bool = False):
    """Fase de download dos 4 fontes."""
    half = target_per_class // 2

    # --- BRSpeech-DF (real bonafide + fake spoof juntos, max_samples = total das duas classes)
    brspeech_real = count_wavs(REAL_DIR, ["brspeech"])
    brspeech_fake = count_wavs(FAKE_DIR, ["brspeech"])
    brspeech_needed = max(0, half - brspeech_real) + max(0, half - brspeech_fake)

    if brspeech_needed > 0:
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_pt_datasets_v2.py"),
             "--brspeech", "--max-samples", str(half * 2)],
            f"BRSpeech-DF bonafide + spoof ({half} por classe)",
        )
    else:
        logger.info(f"BRSpeech-DF ja tem {brspeech_real} real + {brspeech_fake} fake. Pulando.")

    # --- CommonVoice PT + FLEURS (apenas real)
    if not skip_real_cv:
        cv_real = count_wavs(REAL_DIR, ["cetuc", "cv", "fleurs"])
        if cv_real < half:
            run(
                [sys.executable, str(SCRIPTS_DIR / "download_portuguese_datasets.py"),
                 "--cetuc", "--max-samples", str(half)],
                f"CommonVoice PT + FLEURS PT-BR (meta: {half} reais)",
            )
        else:
            logger.info(f"CommonVoice/FLEURS ja tem {cv_real} amostras reais. Pulando.")

    # --- Fake Voices XTTS
    xtts_fake = count_wavs(FAKE_DIR, ["fkvoice", "fakevoice"])
    if xtts_fake < half:
        # max_speakers * max_per_speaker ~= half
        max_speakers = max(10, half // 50)
        run(
            [sys.executable, str(SCRIPTS_DIR / "download_pt_datasets_v2.py"),
             "--fake-voices", "--max-speakers", str(max_speakers)],
            f"Fake Voices XTTS (meta: {half} fakes, {max_speakers} falantes)",
        )
    else:
        logger.info(f"Fake Voices XTTS ja tem {xtts_fake} amostras. Pulando.")


def step_balance(target_per_class: int):
    """
    Garante balanceamento 1:1 entre classes.
    Se uma classe tiver mais que target_per_class, remove o excesso
    priorizando manter diversidade (remove os mais recentes por nome).
    """
    logger.info("\n" + "=" * 60)
    logger.info("BALANCEAMENTO 1:1")
    logger.info("=" * 60)

    real_files = sorted(REAL_DIR.glob("*.wav"))
    fake_files = sorted(FAKE_DIR.glob("*.wav"))

    real_count = len(real_files)
    fake_count = len(fake_files)

    logger.info(f"  Antes: {real_count} real + {fake_count} fake")

    # Calcular alvo: minimo entre target e o que existe, garantindo igualdade
    effective_target = min(target_per_class, real_count, fake_count)

    if effective_target < 100:
        logger.warning(
            f"  ATENCAO: Apenas {effective_target} amostras por classe disponíveis. "
            "Execute os downloads primeiro."
        )
        return real_count, fake_count

    # Remover excesso de REAL
    if real_count > effective_target:
        excess = real_count - effective_target
        # Remover do final (manter diversidade de fontes no inicio)
        to_remove = real_files[effective_target:]
        for f in to_remove:
            f.unlink()
        logger.info(f"  Real: removidos {excess} arquivos excedentes")

    # Remover excesso de FAKE
    if fake_count > effective_target:
        excess = fake_count - effective_target
        to_remove = fake_files[effective_target:]
        for f in to_remove:
            f.unlink()
        logger.info(f"  Fake: removidos {excess} arquivos excedentes")

    real_final = len(list(REAL_DIR.glob("*.wav")))
    fake_final = len(list(FAKE_DIR.glob("*.wav")))
    ratio = real_final / max(fake_final, 1)

    logger.info(f"  Depois: {real_final} real + {fake_final} fake (ratio {ratio:.3f})")

    if abs(ratio - 1.0) > 0.05:
        logger.warning("  AVISO: Ratio fora de 5% de 1.0. Verifique os downloads.")
    else:
        logger.info("  Balanceamento OK (ratio dentro de 5% de 1:1)")

    return real_final, fake_final


def step_preprocess(train_ratio: float, val_ratio: float, test_ratio: float):
    """Roda o pipeline de pre-processamento com os ratios corretos."""
    run(
        [sys.executable, str(SCRIPTS_DIR / "preprocess_dataset.py"),
         "--full",
         "--train-ratio", str(train_ratio),
         "--val-ratio",   str(val_ratio),
         "--test-ratio",  str(test_ratio)],
        f"Pre-processamento + splits {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}",
    )


def save_dataset_config(target_per_class: int, train_r: float, val_r: float, test_r: float):
    """Salva dataset_config.json com a estrategia documentada."""
    real_total = count_wavs(REAL_DIR)
    fake_total = count_wavs(FAKE_DIR)

    brspeech_real = count_wavs(REAL_DIR, ["brspeech"])
    cv_real = count_wavs(REAL_DIR, ["cetuc", "cv", "fleurs"])
    brspeech_fake = count_wavs(FAKE_DIR, ["brspeech"])
    xtts_fake = count_wavs(FAKE_DIR, ["fkvoice", "fakevoice"])

    config = {
        "version": "1.0",
        "description": "Dataset PT-BR para deteccao de deepfake de audio — TCC UFSJ 2026",
        "balancing_strategy": "1:1 (real:fake), corte por classe no target_per_class",
        "target_per_class": target_per_class,
        "total_samples": real_total + fake_total,
        "real_samples": real_total,
        "fake_samples": fake_total,
        "ratio_real_fake": round(real_total / max(fake_total, 1), 4),
        "split": {
            "train": train_r,
            "val": val_r,
            "test": test_r,
            "method": "StratifiedShuffleSplit (random_state=42)",
        },
        "sources": {
            "real": {
                "BRSpeech-DF bonafide": {
                    "repo": "AKCIT-Deepfake/BRSpeech-DF",
                    "license": "CC-BY-NC 4.0",
                    "count": brspeech_real,
                    "target": target_per_class // 2,
                },
                "CommonVoice_v17_PT_FLEURS": {
                    "repo": "mozilla-foundation/common_voice_17_0 + google/fleurs",
                    "license": "CC0 / CC-BY-4.0",
                    "count": cv_real,
                    "target": target_per_class // 2,
                },
            },
            "fake": {
                "BRSpeech-DF spoof": {
                    "repo": "AKCIT-Deepfake/BRSpeech-DF",
                    "license": "CC-BY-NC 4.0",
                    "generators": "multiplos TTS/VC PT-BR",
                    "count": brspeech_fake,
                    "target": target_per_class // 2,
                },
                "Fake_Voices_XTTS": {
                    "repo": "unfake/fake_voices",
                    "license": "MIT",
                    "generators": "XTTS v2 (Coqui TTS)",
                    "count": xtts_fake,
                    "target": target_per_class // 2,
                },
            },
        },
        "audio_format": {
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "encoding": "PCM_16",
            "duration_range_sec": [1.0, 30.0],
        },
        "preprocessing": {
            "vad": "Silero VAD (torch.hub)",
            "agc": "Peak normalization 0.95 headroom",
            "resampling": "librosa (kaiser_best)",
            "duplicate_removal": "MD5 hash",
        },
    }

    config_path = DATASETS_DIR / "dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"\nConfig salvo em: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador de dataset PT-BR para deepfake detection (Fase 1 TCC)"
    )
    parser.add_argument(
        "--target", type=int, default=2000,
        help="Numero de amostras por classe (default: 2000, total = target*2)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Proporcao de treino (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Proporcao de validacao (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Proporcao de teste (default: 0.15)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Pular downloads (usar apenas o que ja existe)",
    )
    parser.add_argument(
        "--skip-real-cv", action="store_true",
        help="Pular download de CommonVoice/FLEURS (apenas BRSpeech + FakeVoices)",
    )
    parser.add_argument(
        "--only-splits", action="store_true",
        help="Apenas recriar os splits (skip download + balance)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Apenas mostrar status atual do dataset",
    )

    args = parser.parse_args()

    # Validar ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Soma dos ratios deve ser 1.0 (atual: {total_ratio:.3f})")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("BUILD DATASET — FASE 1 TCC (UFSJ 2026)")
    logger.info("=" * 60)
    logger.info(f"  Target por classe : {args.target}")
    logger.info(f"  Total esperado    : {args.target * 2}")
    logger.info(f"  Split             : {int(args.train_ratio*100)}/{int(args.val_ratio*100)}/{int(args.test_ratio*100)}")
    logger.info(f"  Composicao real   : {args.target//2} BRSpeech-bonafide + {args.target//2} CommonVoice/FLEURS")
    logger.info(f"  Composicao fake   : {args.target//2} BRSpeech-spoof + {args.target//2} Fake Voices XTTS")

    if args.status:
        print_status()
        return

    # --- Etapa 1: Downloads
    if not args.skip_download and not args.only_splits:
        logger.info("\n>>> ETAPA 1: DOWNLOADS")
        step_download(args.target, skip_real_cv=args.skip_real_cv)
    else:
        logger.info("\n>>> ETAPA 1: DOWNLOADS (pulado)")

    # --- Etapa 2: Balanceamento
    if not args.only_splits:
        logger.info("\n>>> ETAPA 2: BALANCEAMENTO")
        step_balance(args.target)

    # --- Etapa 3: Pre-processamento + Splits
    logger.info("\n>>> ETAPA 3: PRE-PROCESSAMENTO + SPLITS")
    step_preprocess(args.train_ratio, args.val_ratio, args.test_ratio)

    # --- Etapa 4: Salvar config
    logger.info("\n>>> ETAPA 4: SALVAR CONFIG")
    save_dataset_config(args.target, args.train_ratio, args.val_ratio, args.test_ratio)

    # --- Status final
    real_total, fake_total = print_status()

    logger.info("\n" + "=" * 60)
    logger.info("BUILD CONCLUIDO")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {real_total + fake_total} amostras ({real_total} real + {fake_total} fake)")
    logger.info(f"  Splits : app/datasets/splits/train/ | val/ | test/")
    logger.info(f"  Config : app/datasets/dataset_config.json")
    logger.info("\nProximo passo (Fase 2):")
    logger.info("  python scripts/train_advanced.py --model conformer --epochs 100")


if __name__ == "__main__":
    main()
