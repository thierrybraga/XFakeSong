#!/usr/bin/env python3
"""
Pré-processamento e validação de datasets de áudio para o XFakeSong.

Funcionalidades:
  - Verifica integridade dos WAVs (corrompidos, silenciosos)
  - Reamostra para 16kHz mono
  - Normaliza amplitude
  - Remove amostras fora do range de duração (1-30s)
  - Remove duplicatas por hash
  - Gera relatório detalhado
  - Cria splits train/val/test estratificados

Uso:
  python scripts/preprocess_dataset.py --validate          # Apenas validar
  python scripts/preprocess_dataset.py --normalize         # Normalizar todos
  python scripts/preprocess_dataset.py --create-splits     # Criar train/val/test
  python scripts/preprocess_dataset.py --full              # Tudo acima
  python scripts/preprocess_dataset.py --create-zip        # Gerar ZIP para upload na UI
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DatasetPreprocessor")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
SPLITS_DIR = DATASETS_DIR / "splits"

TARGET_SR = 16_000
MIN_DURATION = 1.0
MAX_DURATION = 30.0
SILENCE_THRESHOLD_DB = -50  # abaixo disso = silêncio


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------
def validate_dataset():
    """Valida integridade de todos os arquivos WAV."""
    logger.info("=" * 60)
    logger.info("VALIDACAO DO DATASET")
    logger.info("=" * 60)

    issues = {"corrupted": [], "too_short": [], "too_long": [], "silent": [],
              "wrong_sr": [], "stereo": [], "nan_inf": []}
    stats = {"real": {"count": 0, "total_duration": 0.0, "durations": []},
             "fake": {"count": 0, "total_duration": 0.0, "durations": []}}

    for label, directory in [("real", REAL_DIR), ("fake", FAKE_DIR)]:
        wav_files = sorted(directory.glob("*.wav"))
        logger.info(f"\nValidando {len(wav_files)} arquivos em {directory.name}/...")

        for wav_path in wav_files:
            try:
                info = sf.info(str(wav_path))
                duration = info.duration
                sr = info.samplerate
                channels = info.channels

                if sr != TARGET_SR:
                    issues["wrong_sr"].append((str(wav_path), sr))

                if channels > 1:
                    issues["stereo"].append(str(wav_path))

                if duration < MIN_DURATION:
                    issues["too_short"].append((str(wav_path), duration))
                    continue

                if duration > MAX_DURATION:
                    issues["too_long"].append((str(wav_path), duration))
                    continue

                # Carregar amostra para checagens de conteúdo
                y, _ = librosa.load(str(wav_path), sr=TARGET_SR, duration=5.0)

                # BUG FIX: detectar NaN/Inf — arquivos corrompidos que passariam
                # silenciosamente (NaN < threshold é sempre False → contado válido!)
                # e depois causariam loss:nan no treino.
                if not np.all(np.isfinite(y)):
                    issues["nan_inf"].append(str(wav_path))
                    continue

                # Checar silêncio
                rms = np.sqrt(np.mean(y**2))
                if rms < 10 ** (SILENCE_THRESHOLD_DB / 20):
                    issues["silent"].append(str(wav_path))
                    continue

                stats[label]["count"] += 1
                stats[label]["total_duration"] += duration
                stats[label]["durations"].append(duration)

            except Exception as e:
                issues["corrupted"].append((str(wav_path), str(e)))

    # Relatório
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADO DA VALIDACAO")
    logger.info("=" * 60)

    for label in ["real", "fake"]:
        s = stats[label]
        if s["durations"]:
            logger.info(f"\n  [{label.upper()}]")
            logger.info(f"    Arquivos validos : {s['count']}")
            logger.info(f"    Duracao total    : {s['total_duration']/3600:.1f}h")
            logger.info(f"    Duracao media    : {np.mean(s['durations']):.1f}s")
            logger.info(f"    Duracao min/max  : {np.min(s['durations']):.1f}s / {np.max(s['durations']):.1f}s")
        else:
            logger.warning(f"\n  [{label.upper()}] Nenhum arquivo valido encontrado!")

    total_issues = sum(len(v) for v in issues.values())
    if total_issues > 0:
        logger.warning(f"\n  Problemas encontrados: {total_issues}")
        if issues["corrupted"]:
            logger.warning(f"    Corrompidos    : {len(issues['corrupted'])}")
        if issues["nan_inf"]:
            logger.warning(f"    NaN/Inf (graves): {len(issues['nan_inf'])} — REMOVA antes de treinar!")
        if issues["too_short"]:
            logger.warning(f"    Muito curtos   : {len(issues['too_short'])}")
        if issues["too_long"]:
            logger.warning(f"    Muito longos   : {len(issues['too_long'])}")
        if issues["silent"]:
            logger.warning(f"    Silenciosos    : {len(issues['silent'])}")
        if issues["wrong_sr"]:
            logger.warning(f"    Sample rate != 16kHz: {len(issues['wrong_sr'])}")
        if issues["stereo"]:
            logger.warning(f"    Stereo (nao mono)   : {len(issues['stereo'])}")
    else:
        logger.info("\n  Nenhum problema encontrado!")

    return stats, issues


# ---------------------------------------------------------------------------
# Normalização
# ---------------------------------------------------------------------------
def normalize_all():
    """Normaliza todos os WAVs: reamostra para 16kHz mono, normaliza amplitude."""
    logger.info("=" * 60)
    logger.info("NORMALIZACAO DO DATASET")
    logger.info("=" * 60)

    fixed = 0
    removed = 0

    for label, directory in [("real", REAL_DIR), ("fake", FAKE_DIR)]:
        wav_files = sorted(directory.glob("*.wav"))
        logger.info(f"\nNormalizando {len(wav_files)} arquivos em {directory.name}/...")

        for wav_path in wav_files:
            try:
                y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

                # BUG FIX: sanitizar NaN/Inf ANTES de qualquer cálculo numérico.
                # Sem isso: np.max(np.abs(NaN))=NaN → y/NaN=NaN → arquivo salvo
                # como NaN → wizard de treino lê e gera loss:nan na 1ª época.
                if not np.all(np.isfinite(y)):
                    finite = np.isfinite(y)
                    if finite.sum() < y.size * 0.5:
                        logger.warning(f"  Removendo (NaN/Inf >50%): {wav_path.name}")
                        wav_path.unlink()
                        removed += 1
                        continue
                    # Substitui NaN/Inf locais por silêncio, preserva o resto
                    y = np.where(finite, y, 0.0).astype(np.float32)

                duration = len(y) / TARGET_SR
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    wav_path.unlink()
                    removed += 1
                    continue

                # Checar silêncio
                rms = np.sqrt(np.mean(y**2))
                if not np.isfinite(rms) or rms < 10 ** (SILENCE_THRESHOLD_DB / 20):
                    wav_path.unlink()
                    removed += 1
                    continue

                # Normalizar amplitude (peak já garantido finito acima)
                peak = float(np.max(np.abs(y)))
                if peak > 1e-6:
                    y = (y / peak * 0.95).astype(np.float32)
                else:
                    wav_path.unlink()
                    removed += 1
                    continue

                # Validação final antes de gravar (defesa em profundidade)
                if not np.all(np.isfinite(y)):
                    logger.warning(f"  Removendo (NaN pós-normalização): {wav_path.name}")
                    wav_path.unlink()
                    removed += 1
                    continue

                sf.write(str(wav_path), y, TARGET_SR, subtype="PCM_16")
                fixed += 1

            except Exception as e:
                logger.warning(f"  Removendo arquivo corrompido: {wav_path.name} ({e})")
                try:
                    wav_path.unlink()
                except OSError:
                    pass
                removed += 1

    logger.info(f"\nNormalizacao completa: {fixed} normalizados, {removed} removidos")


# ---------------------------------------------------------------------------
# Remover duplicatas
# ---------------------------------------------------------------------------
def remove_duplicates():
    """Remove duplicatas baseado em hash MD5 do conteúdo."""
    logger.info("Verificando duplicatas...")
    hashes = {}
    removed = 0

    for directory in [REAL_DIR, FAKE_DIR]:
        for wav_path in sorted(directory.glob("*.wav")):
            h = hashlib.md5(wav_path.read_bytes()).hexdigest()
            if h in hashes:
                logger.info(f"  Duplicata: {wav_path.name} == {hashes[h]}")
                wav_path.unlink()
                removed += 1
            else:
                hashes[h] = wav_path.name

    logger.info(f"Duplicatas removidas: {removed}")
    return removed


# ---------------------------------------------------------------------------
# Criar splits train/val/test
# ---------------------------------------------------------------------------
def create_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Cria splits estratificados train/val/test."""
    logger.info("=" * 60)
    logger.info("CRIANDO SPLITS TRAIN/VAL/TEST")
    logger.info("=" * 60)

    # Coletar todos os arquivos
    files = []
    labels = []

    for wav_path in sorted(REAL_DIR.glob("*.wav")):
        files.append(wav_path)
        labels.append(0)  # real

    for wav_path in sorted(FAKE_DIR.glob("*.wav")):
        files.append(wav_path)
        labels.append(1)  # fake

    files = np.array(files)
    labels = np.array(labels)

    logger.info(f"Total: {len(files)} ({sum(labels == 0)} real + {sum(labels == 1)} fake)")

    if len(files) < 10:
        logger.error("Muito poucos arquivos para criar splits!")
        return

    # Primeiro split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=42)
    train_idx, temp_idx = next(sss1.split(files, labels))

    # Segundo split: val vs test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=42)
    val_idx_local, test_idx_local = next(sss2.split(files[temp_idx], labels[temp_idx]))
    val_idx = temp_idx[val_idx_local]
    test_idx = temp_idx[test_idx_local]

    # Criar diretórios e copiar
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_real = SPLITS_DIR / split_name / "real"
        split_fake = SPLITS_DIR / split_name / "fake"
        split_real.mkdir(parents=True, exist_ok=True)
        split_fake.mkdir(parents=True, exist_ok=True)

        # Limpar existentes
        for f in split_real.glob("*.wav"):
            f.unlink()
        for f in split_fake.glob("*.wav"):
            f.unlink()

        for idx in indices:
            src = files[idx]
            if labels[idx] == 0:
                dst = split_real / src.name
            else:
                dst = split_fake / src.name
            shutil.copy2(str(src), str(dst))

        real_count = len(list(split_real.glob("*.wav")))
        fake_count = len(list(split_fake.glob("*.wav")))
        logger.info(f"  {split_name}: {real_count} real + {fake_count} fake = {real_count + fake_count}")

    # Salvar metadata
    metadata = {
        "total_files": len(files),
        "total_real": int(sum(labels == 0)),
        "total_fake": int(sum(labels == 1)),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "target_sr": TARGET_SR,
    }

    with open(SPLITS_DIR / "splits_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Splits salvos em {SPLITS_DIR}")
    logger.info(f"Metadata salvo em {SPLITS_DIR / 'splits_metadata.json'}")


# ---------------------------------------------------------------------------
# Criar ZIP para upload via UI
# ---------------------------------------------------------------------------
def create_training_zip(output_name: str = "dataset_pt_deepfake.zip"):
    """Cria ZIP com estrutura real/ + fake/ para upload na UI do XFakeSong."""
    logger.info("=" * 60)
    logger.info("CRIANDO ZIP PARA UPLOAD")
    logger.info("=" * 60)

    output_path = DATASETS_DIR / output_name

    real_files = sorted(REAL_DIR.glob("*.wav"))
    fake_files = sorted(FAKE_DIR.glob("*.wav"))

    logger.info(f"Empacotando {len(real_files)} real + {len(fake_files)} fake...")

    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for wav_path in real_files:
            zf.write(str(wav_path), f"real/{wav_path.name}")
        for wav_path in fake_files:
            zf.write(str(wav_path), f"fake/{wav_path.name}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ZIP criado: {output_path} ({size_mb:.1f} MB)")
    logger.info(f"Estrutura: real/ ({len(real_files)} arquivos) + fake/ ({len(fake_files)} arquivos)")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-processamento de dataset de audio")
    parser.add_argument("--validate", action="store_true", help="Validar integridade")
    parser.add_argument("--normalize", action="store_true", help="Normalizar todos os WAVs")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remover duplicatas")
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

    args = parser.parse_args()

    if not any([args.validate, args.normalize, args.remove_duplicates,
                args.create_splits, args.create_zip, args.full]):
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
        normalize_all()

    if args.full or args.remove_duplicates:
        remove_duplicates()

    if args.full or args.create_splits:
        create_splits(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    if args.create_zip:
        create_training_zip()

    logger.info("\nPipeline concluido!")


if __name__ == "__main__":
    main()
