#!/usr/bin/env python3
"""
Download otimizado de datasets PT-BR para deepfake detection.
Compativel com datasets>=4.x (sem scripts legados).

Datasets:
  1. BRSpeech-DF (parquet) - bonafide + spoof -> real/ + fake/
  2. Fake Voices (ZIPs de audio por falante) -> fake/
  3. FLEURS PT-BR (parquet) -> real/

Uso:
  python scripts/download_pt_datasets_v2.py --all --max-samples 1000
  python scripts/download_pt_datasets_v2.py --brspeech --max-samples 500
  python scripts/download_pt_datasets_v2.py --fake-voices --max-speakers 10
  python scripts/download_pt_datasets_v2.py --fleurs --max-samples 500
"""

import argparse
import io
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PTDatasetDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
RAW_DIR = DATASETS_DIR / "raw"

TARGET_SR = 16_000
MIN_DURATION = 1.0
MAX_DURATION = 30.0


def setup_dirs():
    for d in [REAL_DIR, FAKE_DIR, RAW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def process_audio(audio_array, sample_rate):
    """Reamostra, normaliza e valida audio. Retorna (array, True) ou (None, False)."""
    if sample_rate != TARGET_SR:
        audio_array = librosa.resample(
            audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
        )

    duration = len(audio_array) / TARGET_SR
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None, False

    peak = np.max(np.abs(audio_array))
    if peak < 1e-6:  # silencio
        return None, False
    audio_array = audio_array / peak * 0.95

    return audio_array, True


def count_wavs(directory, prefix=""):
    if prefix:
        return len(list(directory.glob(f"{prefix}_*.wav")))
    return len(list(directory.glob("*.wav")))


# ---------------------------------------------------------------------------
# 1. BRSpeech-DF (Parquet - streaming)
# ---------------------------------------------------------------------------
def download_brspeech(max_samples=1000):
    """
    Baixa BRSpeech-DF via streaming de parquet.
    bonafide/ -> real, spoof/ -> fake.
    """
    logger.info("=" * 60)
    logger.info("BRSPEECH-DF (bonafide + spoof)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        logger.error("pip install datasets")
        return

    samples_per_class = max_samples // 2

    # Carregar dataset completo (bonafide + spoof juntos) via streaming
    real_count = count_wavs(REAL_DIR, "brspeech")
    fake_count = count_wavs(FAKE_DIR, "brspeech")

    if real_count >= samples_per_class and fake_count >= samples_per_class:
        logger.info(f"Ja existem {real_count} real + {fake_count} fake BRSpeech. Pulando.")
        return

    logger.info(f"Baixando BRSpeech-DF... ja tem {real_count} real + {fake_count} fake")

    # Tentar carregar cada split separadamente
    for data_dir, label_name, is_real in [("bonafide", "real", True), ("spoof", "fake", False)]:
        target_count = real_count if is_real else fake_count
        target_dir = REAL_DIR if is_real else FAKE_DIR
        prefix = "brspeech"

        if target_count >= samples_per_class:
            continue

        logger.info(f"  Processando {data_dir}...")
        try:
            ds = load_dataset(
                "AKCIT-Deepfake/BRSpeech-DF",
                data_dir=data_dir,
                split="train",
                streaming=True,
            )
            # Desabilita decodificacao automatica (evita dependencia de torchcodec/torch)
            ds = ds.cast_column("audio", Audio(decode=False))
            for item in ds:
                if target_count >= samples_per_class:
                    break
                if "audio" not in item:
                    continue

                audio_entry = item["audio"]
                audio_bytes = audio_entry.get("bytes") or audio_entry.get("path")
                if not audio_bytes:
                    continue
                try:
                    if isinstance(audio_bytes, (bytes, bytearray)):
                        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                    else:
                        audio_data, sr = sf.read(audio_bytes)
                except Exception:
                    continue
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)

                audio, ok = process_audio(audio_data.astype(np.float32), sr)
                if not ok:
                    continue

                out = target_dir / f"{prefix}_{target_count:05d}.wav"
                sf.write(str(out), audio, TARGET_SR, subtype="PCM_16")
                target_count += 1

                if target_count % 50 == 0:
                    logger.info(f"    {data_dir}: {target_count}/{samples_per_class}")

            if is_real:
                real_count = target_count
            else:
                fake_count = target_count

        except Exception as e:
            logger.warning(f"  Erro com data_dir={data_dir}: {e}")
            logger.info(f"  Tentando via download direto de parquet...")
            target_count = _download_brspeech_parquet(
                data_dir, target_dir, prefix, target_count, samples_per_class
            )
            if is_real:
                real_count = target_count
            else:
                fake_count = target_count

    logger.info(f"BRSpeech-DF: {real_count} real + {fake_count} fake")


def _download_brspeech_parquet(data_dir, target_dir, prefix, current_count, max_count):
    """Fallback: baixa parquet files diretamente e extrai audio com pandas."""
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    api = HfApi()
    all_files = api.list_repo_files("AKCIT-Deepfake/BRSpeech-DF", repo_type="dataset")
    parquet_files = [f for f in all_files if f.startswith(f"{data_dir}/train") and f.endswith(".parquet")]

    logger.info(f"    Encontrados {len(parquet_files)} parquet files para {data_dir}")

    count = current_count
    for pf in parquet_files:
        if count >= max_count:
            break

        try:
            local_path = hf_hub_download(
                "AKCIT-Deepfake/BRSpeech-DF",
                pf,
                repo_type="dataset",
                cache_dir=str(RAW_DIR / "brspeech_cache"),
            )

            df = pd.read_parquet(local_path)

            for _, row in df.iterrows():
                if count >= max_count:
                    break

                audio_info = row.get("audio", {})
                if isinstance(audio_info, dict) and "bytes" in audio_info:
                    audio_bytes = audio_info["bytes"]
                    try:
                        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1)

                        audio, ok = process_audio(audio_data.astype(np.float32), sr)
                        if not ok:
                            continue

                        out = target_dir / f"{prefix}_{count:05d}.wav"
                        sf.write(str(out), audio, TARGET_SR, subtype="PCM_16")
                        count += 1

                        if count % 50 == 0:
                            logger.info(f"    {data_dir} (parquet): {count}/{max_count}")
                    except Exception:
                        continue

        except Exception as e:
            logger.warning(f"    Erro no parquet {pf}: {e}")
            continue

    return count


# ---------------------------------------------------------------------------
# 2. Fake Voices (ZIPs por falante)
# ---------------------------------------------------------------------------
def download_fake_voices(max_speakers=20, max_per_speaker=100):
    """
    Baixa Fake Voices (unfake/fake_voices) - ZIPs por falante.
    Cada ZIP contem WAVs gerados por XTTS.
    """
    logger.info("=" * 60)
    logger.info("FAKE VOICES (XTTS-generated)")
    logger.info("=" * 60)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files("unfake/fake_voices", repo_type="dataset")
    zip_files = [f for f in files if f.endswith(".zip")]

    logger.info(f"Encontrados {len(zip_files)} ZIPs de falantes")

    total_count = count_wavs(FAKE_DIR, "fkvoice")
    speakers_done = 0

    for zip_name in zip_files:
        if speakers_done >= max_speakers:
            break

        speaker = zip_name.split("/")[-1].replace(".zip", "")
        logger.info(f"  Processando falante: {speaker} ({speakers_done+1}/{max_speakers})")

        try:
            local_path = hf_hub_download(
                "unfake/fake_voices",
                zip_name,
                repo_type="dataset",
                cache_dir=str(RAW_DIR / "fake_voices_cache"),
            )

            speaker_count = 0
            with zipfile.ZipFile(local_path, "r") as zf:
                wav_names = [n for n in zf.namelist() if n.lower().endswith(".wav")]

                for wav_name in wav_names:
                    if speaker_count >= max_per_speaker:
                        break

                    try:
                        with zf.open(wav_name) as audio_file:
                            audio_data, sr = sf.read(io.BytesIO(audio_file.read()))

                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1)

                        audio, ok = process_audio(audio_data.astype(np.float32), sr)
                        if not ok:
                            continue

                        out = FAKE_DIR / f"fkvoice_{total_count:05d}.wav"
                        sf.write(str(out), audio, TARGET_SR, subtype="PCM_16")
                        total_count += 1
                        speaker_count += 1

                    except Exception as e:
                        logger.debug(f"    Erro em {wav_name}: {e}")
                        continue

            speakers_done += 1
            logger.info(f"    {speaker_count} amostras do falante {speaker}")

        except Exception as e:
            logger.error(f"  Erro ao processar {speaker}: {e}")

    logger.info(f"Fake Voices: {total_count} amostras fake totais")


# ---------------------------------------------------------------------------
# 3. FLEURS PT-BR (parquet)
# ---------------------------------------------------------------------------
def download_fleurs(max_samples=500):
    """Baixa FLEURS PT-BR como fonte de fala real."""
    logger.info("=" * 60)
    logger.info("FLEURS PT-BR (fala real)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset, Audio
    except ImportError:
        logger.error("pip install datasets")
        return

    existing = count_wavs(REAL_DIR, "fleurs")
    if existing >= max_samples:
        logger.info(f"Ja existem {existing} amostras FLEURS. Pulando.")
        return

    count = existing
    try:
        # FLEURS em parquet nao usa script
        ds = load_dataset(
            "google/fleurs",
            "pt_br",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
    except Exception:
        # Fallback: baixar parquet diretamente
        logger.info("Tentando via parquet direto...")
        try:
            ds = load_dataset(
                "google/fleurs",
                "pt_br",
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.error(f"FLEURS indisponivel: {e}")
            logger.info("Use --brspeech para obter amostras reais do BRSpeech-DF bonafide/")
            return

    # Desabilita decodificacao automatica (evita dependencia de torchcodec/torch)
    ds = ds.cast_column("audio", Audio(decode=False))

    for item in ds:
        if count >= max_samples:
            break
        if "audio" not in item:
            continue

        audio_entry = item["audio"]
        audio_bytes = audio_entry.get("bytes") or audio_entry.get("path")
        if not audio_bytes:
            continue
        try:
            if isinstance(audio_bytes, (bytes, bytearray)):
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            else:
                audio_data, sr = sf.read(audio_bytes)
        except Exception:
            continue
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        audio, ok = process_audio(audio_data.astype(np.float32), sr)
        if not ok:
            continue

        out = REAL_DIR / f"fleurs_{count:05d}.wav"
        sf.write(str(out), audio, TARGET_SR, subtype="PCM_16")
        count += 1

        if count % 50 == 0:
            logger.info(f"  FLEURS: {count}/{max_samples}")

    logger.info(f"FLEURS: {count} amostras reais")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------
def print_report():
    real_count = len(list(REAL_DIR.glob("*.wav")))
    fake_count = len(list(FAKE_DIR.glob("*.wav")))

    # Amostrar duracoes
    def sample_duration(directory, n=50):
        total = 0.0
        files = list(directory.glob("*.wav"))[:n]
        for f in files:
            try:
                info = sf.info(str(f))
                total += info.duration
            except Exception:
                pass
        if len(files) > 0:
            avg = total / len(files)
            return avg * len(list(directory.glob("*.wav")))
        return 0.0

    real_h = sample_duration(REAL_DIR) / 3600
    fake_h = sample_duration(FAKE_DIR) / 3600

    logger.info("")
    logger.info("=" * 60)
    logger.info("RELATORIO FINAL")
    logger.info("=" * 60)
    logger.info(f"  Real : {real_count:>6d} arquivos (~{real_h:.1f}h)")
    logger.info(f"  Fake : {fake_count:>6d} arquivos (~{fake_h:.1f}h)")
    logger.info(f"  Total: {real_count + fake_count:>6d}")
    logger.info(f"  Ratio: {real_count / max(fake_count, 1):.2f}")
    logger.info(f"  Dir  : {DATASETS_DIR}")
    logger.info("=" * 60)

    if real_count == 0 or fake_count == 0:
        logger.warning("Uma das classes esta vazia!")
    else:
        logger.info("Dataset pronto para treinamento!")
        logger.info("Proximo passo: python scripts/preprocess_dataset.py --full")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download datasets PT-BR para deepfake detection")
    parser.add_argument("--all", action="store_true", help="Baixar BRSpeech-DF + Fake Voices (recomendado)")
    parser.add_argument("--brspeech", action="store_true", help="BRSpeech-DF (real + fake)")
    parser.add_argument("--fake-voices", action="store_true", help="Fake Voices (fake)")
    parser.add_argument("--fleurs", action="store_true", help="FLEURS PT-BR (real)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max amostras por classe (default: 1000)")
    parser.add_argument("--max-speakers", type=int, default=20, help="Max falantes do Fake Voices (default: 20)")

    args = parser.parse_args()

    if not any([args.all, args.brspeech, args.fake_voices, args.fleurs]):
        parser.print_help()
        print("\nRecomendado:")
        print("  python scripts/download_pt_datasets_v2.py --all --max-samples 1000")
        return

    setup_dirs()

    if args.all or args.brspeech:
        download_brspeech(args.max_samples)

    if args.all or args.fake_voices:
        download_fake_voices(args.max_speakers)

    if args.fleurs:
        download_fleurs(args.max_samples)

    print_report()


if __name__ == "__main__":
    main()
