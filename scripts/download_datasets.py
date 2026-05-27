#!/usr/bin/env python3
"""
download_datasets.py — Download de datasets PT-BR para detecção de deepfake de áudio.

Substitui download_portuguese_datasets.py e download_pt_datasets_v2.py.

Datasets suportados:
  --brspeech      BRSpeech-DF (AKCIT-Deepfake/BRSpeech-DF): bonafide→real, spoof→fake
  --cetuc         CETUC/CommonVoice v17 PT-BR / FLEURS / OpenSLR (real)
  --fake-voices   Fake Voices XTTS (unfake/fake_voices): ZIPs por falante (fake)
  --fleurs        FLEURS PT-BR Google (real)
  --mlaad-pt      MLAAD v9 subconjunto PT (fake, CC-BY-NC 4.0)
  --all           brspeech + cetuc + fake-voices (recomendado para build_dataset.py)

Uso:
  python scripts/download_datasets.py --all --max-samples 2000
  python scripts/download_datasets.py --brspeech --max-samples 1000
  python scripts/download_datasets.py --cetuc --max-samples 1000
  python scripts/download_datasets.py --fake-voices --max-speakers 20
  python scripts/download_datasets.py --fleurs --max-samples 500
  python scripts/download_datasets.py --mlaad-pt --max-samples 500
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DownloadDatasets")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
RAW_DIR = DATASETS_DIR / "raw"

TARGET_SR = 16_000
MIN_DURATION = 1.0
MAX_DURATION = 30.0
DEFAULT_MAX_SAMPLES = 5_000


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def setup_dirs() -> None:
    for d in [REAL_DIR, FAKE_DIR, RAW_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretórios prontos em {DATASETS_DIR}")


def count_wavs(directory: Path, prefix: str = "") -> int:
    if not directory.exists():
        return 0
    if prefix:
        return len(list(directory.glob(f"{prefix}_*.wav")))
    return len(list(directory.glob("*.wav")))


def process_audio(audio_array: np.ndarray, sample_rate: int) -> tuple[np.ndarray | None, bool]:
    """Reamostra, valida duração, checa silêncio e normaliza.

    Returns (normalized_array, True) ou (None, False).
    """
    if sample_rate != TARGET_SR:
        audio_array = librosa.resample(
            audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
        )
    duration = len(audio_array) / TARGET_SR
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None, False
    peak = np.max(np.abs(audio_array))
    if peak < 1e-6:
        return None, False
    return (audio_array / peak * 0.95).astype(np.float32), True


def print_report() -> None:
    real_count = count_wavs(REAL_DIR)
    fake_count = count_wavs(FAKE_DIR)

    def sample_hours(directory: Path, n: int = 50) -> float:
        files = list(directory.glob("*.wav"))[:n]
        total = sum(sf.info(str(f)).duration for f in files if f.exists())
        avg = total / max(len(files), 1)
        return avg * count_wavs(directory) / 3600

    logger.info("")
    logger.info("=" * 60)
    logger.info("RELATÓRIO FINAL DO DATASET")
    logger.info("=" * 60)
    logger.info(f"  Real : {real_count:>6d}  (~{sample_hours(REAL_DIR):.1f}h)")
    logger.info(f"  Fake : {fake_count:>6d}  (~{sample_hours(FAKE_DIR):.1f}h)")
    logger.info(f"  Total: {real_count + fake_count:>6d}")
    logger.info(f"  Ratio: {real_count / max(fake_count, 1):.2f}")
    logger.info(f"  Dir  : {DATASETS_DIR}")
    logger.info("=" * 60)

    if real_count == 0 or fake_count == 0:
        logger.warning("ATENÇÃO: uma das classes está vazia!")
    elif abs(real_count - fake_count) / max(real_count, fake_count) > 0.5:
        logger.warning("ATENÇÃO: desbalanceamento significativo — use class_weight='balanced' no treino.")
    else:
        logger.info("Dataset pronto. Próximo passo: python scripts/preprocess_dataset.py --full")


# ---------------------------------------------------------------------------
# 1. BRSpeech-DF — bonafide (real) + spoof (fake) via parquet streaming
# ---------------------------------------------------------------------------

def download_brspeech(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """BRSpeech-DF (AKCIT-Deepfake/BRSpeech-DF): bonafide→real, spoof→fake."""
    logger.info("=" * 60)
    logger.info("BRSPEECH-DF (bonafide + spoof)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    samples_per_class = max_samples // 2
    real_count = count_wavs(REAL_DIR, "brspeech")
    fake_count = count_wavs(FAKE_DIR, "brspeech")

    if real_count >= samples_per_class and fake_count >= samples_per_class:
        logger.info(f"Já existem {real_count} real + {fake_count} fake BRSpeech. Pulando.")
        return

    logger.info(f"Iniciando (já tem {real_count} real + {fake_count} fake)...")

    for data_dir, target_dir, prefix, is_real in [
        ("bonafide", REAL_DIR, "brspeech", True),
        ("spoof",    FAKE_DIR, "brspeech", False),
    ]:
        current = real_count if is_real else fake_count
        if current >= samples_per_class:
            continue

        logger.info(f"  {data_dir}...")
        try:
            ds = load_dataset(
                "AKCIT-Deepfake/BRSpeech-DF",
                data_dir=data_dir,
                split="train",
                streaming=True,
            )
            for item in ds:
                if current >= samples_per_class:
                    break
                if "audio" not in item:
                    continue
                audio, ok = process_audio(item["audio"]["array"], item["audio"]["sampling_rate"])
                if not ok:
                    continue
                sf.write(str(target_dir / f"{prefix}_{current:05d}.wav"), audio, TARGET_SR, subtype="PCM_16")
                current += 1
                if current % 50 == 0:
                    logger.info(f"    {data_dir}: {current}/{samples_per_class}")

        except Exception as e:
            logger.warning(f"  Streaming falhou ({e}), tentando parquet direto...")
            current = _download_brspeech_parquet(data_dir, target_dir, prefix, current, samples_per_class)

        if is_real:
            real_count = current
        else:
            fake_count = current

    logger.info(f"BRSpeech-DF: {real_count} real + {fake_count} fake")


def _download_brspeech_parquet(
    data_dir: str, target_dir: Path, prefix: str,
    current_count: int, max_count: int,
) -> int:
    """Fallback: download parquet direto via HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import pandas as pd
    except ImportError:
        logger.error("Instale: pip install huggingface_hub pandas")
        return current_count

    api = HfApi()
    all_files = api.list_repo_files("AKCIT-Deepfake/BRSpeech-DF", repo_type="dataset")
    parquet_files = [f for f in all_files if f.startswith(f"{data_dir}/train") and f.endswith(".parquet")]
    logger.info(f"    {len(parquet_files)} parquet files para {data_dir}")

    count = current_count
    cache_dir = str(RAW_DIR / "brspeech_cache")
    for pf in parquet_files:
        if count >= max_count:
            break
        try:
            local_path = hf_hub_download("AKCIT-Deepfake/BRSpeech-DF", pf,
                                          repo_type="dataset", cache_dir=cache_dir)
            df = pd.read_parquet(local_path)
            for _, row in df.iterrows():
                if count >= max_count:
                    break
                audio_info = row.get("audio", {})
                if isinstance(audio_info, dict) and "bytes" in audio_info:
                    try:
                        raw_data, sr = sf.read(io.BytesIO(audio_info["bytes"]))
                        if raw_data.ndim > 1:
                            raw_data = raw_data.mean(axis=1)
                        audio, ok = process_audio(raw_data.astype(np.float32), sr)
                        if not ok:
                            continue
                        sf.write(str(target_dir / f"{prefix}_{count:05d}.wav"), audio, TARGET_SR, subtype="PCM_16")
                        count += 1
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"    Erro no parquet {pf}: {e}")
    return count


# ---------------------------------------------------------------------------
# 2. CETUC / CommonVoice / FLEURS — fala real PT-BR
# ---------------------------------------------------------------------------

def download_cetuc(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """CETUC Corpus / CommonVoice v17 PT / FLEURS (fallback em cascata → OpenSLR)."""
    logger.info("=" * 60)
    logger.info("CETUC / CommonVoice PT-BR (real)")
    logger.info("=" * 60)

    existing = count_wavs(REAL_DIR, "cetuc")
    if existing >= max_samples:
        logger.info(f"Já existem {existing} amostras CETUC. Pulando.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return _download_cetuc_openslr(max_samples)

    dataset = None
    for repo, config in [
        ("mozilla-foundation/common_voice_17_0", "pt"),
        ("google/fleurs", "pt_br"),
    ]:
        try:
            dataset = load_dataset(repo, config, split="train", streaming=True)
            logger.info(f"Usando {repo} ({config})...")
            break
        except Exception as e:
            logger.info(f"  {repo} indisponível: {e}")

    if dataset is None:
        return _download_cetuc_openslr(max_samples)

    count = existing
    for item in dataset:
        if count >= max_samples:
            break
        if "audio" not in item:
            continue
        audio, ok = process_audio(item["audio"]["array"], item["audio"]["sampling_rate"])
        if not ok:
            continue
        sf.write(str(REAL_DIR / f"cetuc_{count:05d}.wav"), audio, TARGET_SR, subtype="PCM_16")
        count += 1
        if count % 100 == 0:
            logger.info(f"  cetuc: {count}/{max_samples}")

    logger.info(f"CETUC/CommonVoice: {count} amostras reais")


def _download_cetuc_openslr(max_samples: int) -> None:
    """Fallback: CETUC via OpenSLR 132."""
    import tarfile

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("Instale: pip install requests tqdm")
        return

    url = "https://www.openslr.org/resources/132/cetuc.tar.gz"
    cetuc_raw = RAW_DIR / "cetuc"
    cetuc_raw.mkdir(parents=True, exist_ok=True)
    dest = cetuc_raw / "cetuc.tar.gz"

    if not dest.exists():
        logger.info(f"Baixando {url}...")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        except Exception as e:
            logger.error(f"Erro ao baixar CETUC: {e}")
            return

    logger.info("Extraindo CETUC...")
    with tarfile.open(str(dest), "r:gz") as tar:
        tar.extractall(str(cetuc_raw))

    wav_files = list(cetuc_raw.rglob("*.wav"))
    logger.info(f"Encontrados {len(wav_files)} WAVs no CETUC")
    count = count_wavs(REAL_DIR, "cetuc")
    for wav_path in wav_files:
        if count >= max_samples:
            break
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        audio, ok = process_audio(y, TARGET_SR)
        if ok:
            sf.write(str(REAL_DIR / f"cetuc_{count:05d}.wav"), audio, TARGET_SR, subtype="PCM_16")
            count += 1

    logger.info(f"CETUC OpenSLR: {count} amostras reais")


# ---------------------------------------------------------------------------
# 3. Fake Voices XTTS — ZIPs por falante (mais robusto que streaming)
# ---------------------------------------------------------------------------

def download_fake_voices(max_speakers: int = 20, max_per_speaker: int = 100) -> None:
    """Fake Voices (unfake/fake_voices) — ZIPs por falante."""
    logger.info("=" * 60)
    logger.info("FAKE VOICES (XTTS-generated)")
    logger.info("=" * 60)

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        logger.error("Instale: pip install huggingface_hub")
        return

    api = HfApi()
    all_files = api.list_repo_files("unfake/fake_voices", repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]
    logger.info(f"{len(zip_files)} ZIPs encontrados")

    total_count = count_wavs(FAKE_DIR, "fkvoice")
    speakers_done = 0

    for zip_name in zip_files:
        if speakers_done >= max_speakers:
            break
        speaker = zip_name.split("/")[-1].replace(".zip", "")
        logger.info(f"  Falante {speakers_done + 1}/{max_speakers}: {speaker}")
        try:
            local_path = hf_hub_download(
                "unfake/fake_voices", zip_name, repo_type="dataset",
                cache_dir=str(RAW_DIR / "fake_voices_cache"),
            )
            speaker_count = 0
            with zipfile.ZipFile(local_path, "r") as zf:
                wav_names = [n for n in zf.namelist() if n.lower().endswith(".wav")]
                for wav_name in wav_names:
                    if speaker_count >= max_per_speaker:
                        break
                    try:
                        with zf.open(wav_name) as af:
                            raw_data, sr = sf.read(io.BytesIO(af.read()))
                        if raw_data.ndim > 1:
                            raw_data = raw_data.mean(axis=1)
                        audio, ok = process_audio(raw_data.astype(np.float32), sr)
                        if not ok:
                            continue
                        sf.write(
                            str(FAKE_DIR / f"fkvoice_{total_count:05d}.wav"),
                            audio, TARGET_SR, subtype="PCM_16",
                        )
                        total_count += 1
                        speaker_count += 1
                    except Exception as e:
                        logger.debug(f"    Erro em {wav_name}: {e}")
            logger.info(f"    {speaker_count} amostras do falante {speaker}")
            speakers_done += 1
        except Exception as e:
            logger.error(f"  Erro ao processar {speaker}: {e}")

    logger.info(f"Fake Voices: {total_count} amostras fake totais")


# ---------------------------------------------------------------------------
# 4. FLEURS PT-BR — fala real do Google
# ---------------------------------------------------------------------------

def download_fleurs(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """FLEURS PT-BR (google/fleurs, pt_br) — fala real."""
    logger.info("=" * 60)
    logger.info("FLEURS PT-BR (real)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    existing = count_wavs(REAL_DIR, "fleurs")
    if existing >= max_samples:
        logger.info(f"Já existem {existing} amostras FLEURS. Pulando.")
        return

    try:
        ds = load_dataset("google/fleurs", "pt_br", split="train", streaming=True)
    except Exception as e:
        logger.error(f"FLEURS indisponível: {e}")
        return

    count = existing
    for item in ds:
        if count >= max_samples:
            break
        if "audio" not in item:
            continue
        audio, ok = process_audio(item["audio"]["array"], item["audio"]["sampling_rate"])
        if not ok:
            continue
        sf.write(str(REAL_DIR / f"fleurs_{count:05d}.wav"), audio, TARGET_SR, subtype="PCM_16")
        count += 1
        if count % 50 == 0:
            logger.info(f"  FLEURS: {count}/{max_samples}")

    logger.info(f"FLEURS: {count} amostras reais")


# ---------------------------------------------------------------------------
# 5. MLAAD v9 — subconjunto PT (fake only, CC-BY-NC 4.0)
# ---------------------------------------------------------------------------

def download_mlaad_pt(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """MLAAD v9 (OU-CSAIL/MLAAD) — subconjunto PT-BR, apenas fakes."""
    logger.info("=" * 60)
    logger.info("MLAAD v9 PT-BR (fake)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    existing = count_wavs(FAKE_DIR, "mlaad")
    if existing >= max_samples:
        logger.info(f"Já existem {existing} amostras MLAAD. Pulando.")
        return

    try:
        ds = load_dataset("OU-CSAIL/MLAAD", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        logger.error(f"MLAAD indisponível: {e}")
        return

    count = existing
    skipped = 0
    for item in ds:
        if count >= max_samples:
            break
        lang = item.get("language", item.get("lang", ""))
        if isinstance(lang, str):
            lang_lower = lang.lower()
            if "pt" not in lang_lower and "portuguese" not in lang_lower and "brasil" not in lang_lower:
                skipped += 1
                if skipped % 5000 == 0:
                    logger.info(f"  Pulando amostras não-PT ({skipped} até agora)...")
                continue
        if "audio" not in item:
            continue
        audio, ok = process_audio(item["audio"]["array"], item["audio"]["sampling_rate"])
        if not ok:
            continue
        sf.write(str(FAKE_DIR / f"mlaad_{count:05d}.wav"), audio, TARGET_SR, subtype="PCM_16")
        count += 1
        if count % 100 == 0:
            logger.info(f"  MLAAD PT: {count}/{max_samples}")

    logger.info(f"MLAAD: {count} amostras fake PT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true",
                        help="brspeech + cetuc + fake-voices (recomendado)")
    parser.add_argument("--brspeech", action="store_true",
                        help="BRSpeech-DF (real + fake)")
    parser.add_argument("--cetuc", action="store_true",
                        help="CETUC/CommonVoice PT-BR (real)")
    parser.add_argument("--fake-voices", action="store_true",
                        help="Fake Voices XTTS via ZIPs (fake)")
    parser.add_argument("--fleurs", action="store_true",
                        help="FLEURS PT-BR do Google (real)")
    parser.add_argument("--mlaad-pt", action="store_true",
                        help="MLAAD v9 subconjunto PT (fake, CC-BY-NC 4.0)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"Máximo de amostras por classe (padrão: {DEFAULT_MAX_SAMPLES})")
    parser.add_argument("--max-speakers", type=int, default=20,
                        help="Máximo de falantes do Fake Voices (padrão: 20)")

    args = parser.parse_args()

    if not any([args.all, args.brspeech, args.cetuc, args.fake_voices, args.fleurs, args.mlaad_pt]):
        parser.print_help()
        print("\nExemplo recomendado:")
        print("  python scripts/download_datasets.py --all --max-samples 2000")
        return

    setup_dirs()
    logger.info(f"Max amostras por classe: {args.max_samples}")

    if args.all or args.brspeech:
        download_brspeech(args.max_samples)

    if args.all or args.cetuc:
        download_cetuc(args.max_samples)

    if args.all or args.fake_voices:
        download_fake_voices(max_speakers=args.max_speakers)

    if args.fleurs:
        download_fleurs(args.max_samples)

    if args.mlaad_pt:
        download_mlaad_pt(args.max_samples)

    print_report()
    logger.info("Download concluído!")


if __name__ == "__main__":
    main()
