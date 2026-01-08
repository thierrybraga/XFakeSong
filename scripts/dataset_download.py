
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

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações de Diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATASETS_DIR = APP_DIR / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"

# Configuração dos Datasets
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
    "fake_voice_dataset": {
        "url": "https://example.com/fake_dataset.tar.gz",  # Placeholder URL
        "type": "fake"
    }
}

def setup_directories():
    """Cria a estrutura de diretórios necessária."""
    dirs = [DATASETS_DIR, REAL_DIR, FAKE_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretórios configurados em: {DATASETS_DIR}")

def export_audio_from_hf_dataset(dataset_name, subset, split, output_dir, max_samples=None, prefix="hf"):
    """Baixa e exporta áudio de um dataset do Hugging Face."""
    try:
        logger.info(f"Carregando dataset {dataset_name} ({subset})...")
        # Removido trust_remote_code=True pois não é suportado em versões recentes para este dataset ou não é necessário
        dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        
        count = 0
        for i, item in enumerate(dataset):
            if max_samples and count >= max_samples:
                break
                
            if "audio" in item:
                audio_data = item["audio"]["array"]
                sample_rate = item["audio"]["sampling_rate"]
                
                filename = f"{prefix}_{count:04d}.wav"
                filepath = output_dir / filename
                
                # Salvar como WAV
                sf.write(filepath, audio_data, sample_rate)
                count += 1
                
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

def process_direct_dataset(url, output_dir, type_name):
    """Baixa e extrai um dataset direto de URL (exemplo)."""
    # Implementação de exemplo para download direto
    pass

def main():
    setup_directories()
    
    # Processar datasets reais do Hugging Face
    total_real = 0
    
    # Common Voice
    logger.info("Processando Common Voice...")
    count = export_audio_from_hf_dataset(
        DATASETS_CONFIG["common_voice_pt"]["name"],
        DATASETS_CONFIG["common_voice_pt"]["subset"],
        DATASETS_CONFIG["common_voice_pt"]["split"],
        REAL_DIR,
        DATASETS_CONFIG["common_voice_pt"]["max_samples"],
        prefix="cv"
    )
    total_real += count
    
    # FLEURS
    logger.info("Processando FLEURS...")
    count = export_audio_from_hf_dataset(
        DATASETS_CONFIG["fleurs_pt"]["name"],
        DATASETS_CONFIG["fleurs_pt"]["subset"],
        DATASETS_CONFIG["fleurs_pt"]["split"],
        REAL_DIR,
        DATASETS_CONFIG["fleurs_pt"]["max_samples"],
        prefix="fleurs"
    )
    total_real += count
    
    logger.info(f"Total de áudios reais: {total_real}")
    
    # Verificar áudios fake (já devem estar lá via setup_fake_dataset.py ou manual)
    fake_count = len(list(FAKE_DIR.glob("*.wav")))
    logger.info(f"Total de áudios fake existentes: {fake_count}")
    
    if fake_count == 0:
        logger.warning("Nenhum áudio fake encontrado. Execute scripts/setup_fake_dataset.py para gerar dados sintéticos.")

if __name__ == "__main__":
    main()
