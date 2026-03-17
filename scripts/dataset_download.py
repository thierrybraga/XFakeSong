#!/usr/bin/env python3
"""
Download de datasets de audio para o XFakeSong.

NOTA: Para datasets em Portugues focados em deepfake detection, use:
  python scripts/download_portuguese_datasets.py --all

Este script baixa datasets genericos (Common Voice, FLEURS) como complemento.
Para o pipeline completo de preparacao, use:
  python scripts/preprocess_dataset.py --full
"""

import os
import logging
from pathlib import Path
import shutil
import tarfile
import zipfile
import requests
from tqdm import tqdm
from datasets import load_dataset
import soundfile as sf
import numpy as np
import librosa

# Configuracao de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuracoes de Diretorios
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATASETS_DIR = APP_DIR / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"

TARGET_SR = 16_000

# Configuracao dos Datasets
DATASETS_CONFIG = {
    "common_voice_pt": {
        "name": "mozilla-foundation/common_voice_11_0",
        "subset": "pt",
        "split": "train",
        "type": "real",
        "max_samples": 500
    },
    "fleurs_pt": {
        "name": "google/fleurs",
        "subset": "pt_br",
        "split": "train",
        "type": "real",
        "max_samples": 200
    },
}

def setup_directories():
    """Cria a estrutura de diretorios necessaria."""
    dirs = [DATASETS_DIR, REAL_DIR, FAKE_DIR, DATASETS_DIR / "features",
            DATASETS_DIR / "raw", DATASETS_DIR / "splits"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretorios configurados em: {DATASETS_DIR}")

def export_audio_from_hf_dataset(dataset_name, subset, split, output_dir, max_samples=None, prefix="hf"):
    """Baixa e exporta audio de um dataset do Hugging Face com normalizacao."""
    try:
        logger.info(f"Carregando dataset {dataset_name} ({subset})...")
        dataset = load_dataset(dataset_name, subset, split=split, streaming=True)

        count = 0
        for i, item in enumerate(dataset):
            if max_samples and count >= max_samples:
                break

            if "audio" in item:
                audio_data = item["audio"]["array"].astype(np.float32)
                sample_rate = item["audio"]["sampling_rate"]

                # Reamostrar para 16kHz se necessario
                if sample_rate != TARGET_SR:
                    audio_data = librosa.resample(
                        audio_data, orig_sr=sample_rate, target_sr=TARGET_SR
                    )

                # Filtrar por duracao
                duration = len(audio_data) / TARGET_SR
                if duration < 1.0 or duration > 30.0:
                    continue

                # Normalizar amplitude
                peak = np.max(np.abs(audio_data))
                if peak > 0:
                    audio_data = audio_data / peak * 0.95

                filename = f"{prefix}_{count:05d}.wav"
                filepath = output_dir / filename

                sf.write(str(filepath), audio_data, TARGET_SR, subtype="PCM_16")
                count += 1

                if count % 100 == 0:
                    logger.info(f"  {count}/{max_samples} exportados...")

        logger.info(f"Exportados {count} arquivos de {dataset_name} para {output_dir}")
        return count
    except Exception as e:
        logger.error(f"Erro ao processar dataset {dataset_name}: {e}")
        return 0

def download_file(url, dest_path):
    """Baixa um arquivo com barra de progresso."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    setup_directories()

    total_real = 0

    # Common Voice PT
    logger.info("Processando Common Voice PT...")
    count = export_audio_from_hf_dataset(
        DATASETS_CONFIG["common_voice_pt"]["name"],
        DATASETS_CONFIG["common_voice_pt"]["subset"],
        DATASETS_CONFIG["common_voice_pt"]["split"],
        REAL_DIR,
        DATASETS_CONFIG["common_voice_pt"]["max_samples"],
        prefix="cv"
    )
    total_real += count

    # FLEURS PT-BR
    logger.info("Processando FLEURS PT-BR...")
    count = export_audio_from_hf_dataset(
        DATASETS_CONFIG["fleurs_pt"]["name"],
        DATASETS_CONFIG["fleurs_pt"]["subset"],
        DATASETS_CONFIG["fleurs_pt"]["split"],
        REAL_DIR,
        DATASETS_CONFIG["fleurs_pt"]["max_samples"],
        prefix="fleurs"
    )
    total_real += count

    logger.info(f"Total de audios reais: {total_real}")

    # Verificar audios fake
    fake_count = len(list(FAKE_DIR.glob("*.wav")))
    logger.info(f"Total de audios fake existentes: {fake_count}")

    if fake_count == 0:
        logger.warning(
            "Nenhum audio fake encontrado. Execute:\n"
            "  python scripts/download_portuguese_datasets.py --fake-voices\n"
            "  ou: python scripts/setup_fake_dataset.py"
        )

    # Relatorio
    real_total = len(list(REAL_DIR.glob("*.wav")))
    fake_total = len(list(FAKE_DIR.glob("*.wav")))
    logger.info(f"\nDataset final: {real_total} real + {fake_total} fake = {real_total + fake_total} total")

if __name__ == "__main__":
    main()
