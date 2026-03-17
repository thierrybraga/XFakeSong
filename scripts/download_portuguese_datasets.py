#!/usr/bin/env python3
"""
Download e preparação de datasets em Português para detecção de deepfake de áudio.

Datasets suportados:
  1. CETUC Corpus      – ~145h de fala real, 100 falantes, 16kHz WAV (FalaBrasil)
  2. Fake Voices       – ~140h de fala fake (XTTS), 101 falantes, MIT (Hugging Face: unfake/fake_voices)
  3. BRSpeech-DF       – 459K arquivos, 243GB, 24kHz (Hugging Face: AKCIT-Deepfake/BRSpeech-DF) — subset
  4. MLAAD v9          – 678h multilíngue incl. pt-BR, CC-BY-NC 4.0 (Hugging Face: OU-CSAIL/MLAAD)
  5. FakeBrAccent      – ~1500 arquivos, 5 sotaques BR (Hugging Face)

Uso:
  python scripts/download_portuguese_datasets.py --all              # Baixa todos (recomendado: cetuc + fake_voices)
  python scripts/download_portuguese_datasets.py --cetuc            # Apenas CETUC (real)
  python scripts/download_portuguese_datasets.py --fake-voices      # Apenas Fake Voices (fake)
  python scripts/download_portuguese_datasets.py --brspeech-subset  # Subset do BRSpeech-DF
  python scripts/download_portuguese_datasets.py --mlaad-pt         # Apenas pt-BR do MLAAD
  python scripts/download_portuguese_datasets.py --max-samples 1000 # Limitar amostras por classe
"""

import os
import sys
import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import librosa

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DatasetDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "app" / "datasets"
REAL_DIR = DATASETS_DIR / "real"
FAKE_DIR = DATASETS_DIR / "fake"
RAW_DIR = DATASETS_DIR / "raw"  # downloads originais antes de processar

TARGET_SR = 16_000
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
DEFAULT_MAX_SAMPLES = 5000  # por classe, para não explodir disco


def setup_dirs():
    """Cria estrutura de diretórios."""
    for d in [REAL_DIR, FAKE_DIR, RAW_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diretorios prontos em {DATASETS_DIR}")


# ---------------------------------------------------------------------------
# Utilidades de áudio
# ---------------------------------------------------------------------------
def normalize_audio(audio_path: Path, output_path: Path, target_sr: int = TARGET_SR) -> bool:
    """
    Carrega um arquivo de áudio, reamostra para target_sr,
    normaliza amplitude e salva como WAV 16-bit.
    Retorna True se bem-sucedido.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

        duration = len(y) / target_sr
        if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
            return False

        # Normalizar amplitude para [-1, 1]
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.95  # headroom

        sf.write(str(output_path), y, target_sr, subtype="PCM_16")
        return True
    except Exception as e:
        logger.debug(f"Falha ao processar {audio_path}: {e}")
        return False


def count_existing(directory: Path, prefix: str = "") -> int:
    """Conta WAVs existentes com dado prefixo."""
    if prefix:
        return len(list(directory.glob(f"{prefix}_*.wav")))
    return len(list(directory.glob("*.wav")))


# ---------------------------------------------------------------------------
# 1. CETUC Corpus (real speech)
# ---------------------------------------------------------------------------
def download_cetuc(max_samples: int = DEFAULT_MAX_SAMPLES):
    """
    Baixa o CETUC Corpus do FalaBrasil (Universidade Federal do Pará).
    ~145h de fala real de 100 falantes brasileiros, já em 16kHz WAV.

    Fonte: https://github.com/falabrasil/speech-datasets
    Download direto: https://www.openslr.org/resources/132/
    """
    logger.info("=" * 60)
    logger.info("Baixando CETUC Corpus (fala real PT-BR)...")
    logger.info("=" * 60)

    existing = count_existing(REAL_DIR, "cetuc")
    if existing >= max_samples:
        logger.info(f"Ja existem {existing} amostras CETUC. Pulando.")
        return existing

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Pacote 'datasets' nao encontrado. Instale: pip install datasets")
        return 0

    # Estratégia: tentar múltiplas fontes de fala real PT-BR
    # 1. Common Voice 17 (sem trust_remote_code)
    # 2. FLEURS PT-BR (Google, acesso público)
    # 3. CETUC via OpenSLR (download direto)
    dataset = None

    # Tentativa 1: Common Voice 17
    logger.info("Tentando Common Voice 17 PT...")
    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "pt",
            split="train",
            streaming=True,
        )
    except Exception as e1:
        logger.info(f"Common Voice 17 indisponivel: {e1}")

        # Tentativa 2: FLEURS PT-BR (acesso público, sem autenticação)
        logger.info("Tentando FLEURS PT-BR...")
        try:
            dataset = load_dataset(
                "google/fleurs",
                "pt_br",
                split="train",
                streaming=True,
            )
        except Exception as e2:
            logger.info(f"FLEURS indisponivel: {e2}")

            # Tentativa 3: OpenSLR
            logger.info("Tentando CETUC via OpenSLR...")
            return _download_cetuc_openslr(max_samples)

    if dataset is None:
        logger.info("Nenhum dataset HF disponivel, tentando OpenSLR...")
        return _download_cetuc_openslr(max_samples)

    count = existing
    for item in dataset:
        if count >= max_samples:
            break

        if "audio" not in item:
            continue

        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]

        # Reamostrar se necessário
        if sample_rate != TARGET_SR:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
            )

        duration = len(audio_array) / TARGET_SR
        if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
            continue

        # Normalizar
        peak = np.max(np.abs(audio_array))
        if peak > 0:
            audio_array = audio_array / peak * 0.95

        output_path = REAL_DIR / f"cetuc_{count:05d}.wav"
        sf.write(str(output_path), audio_array, TARGET_SR, subtype="PCM_16")
        count += 1

        if count % 100 == 0:
            logger.info(f"  Processadas {count}/{max_samples} amostras reais...")

    logger.info(f"CETUC/CommonVoice: {count} amostras reais salvas em {REAL_DIR}")
    return count


def _download_cetuc_openslr(max_samples: int) -> int:
    """Fallback: baixa CETUC via OpenSLR."""
    import requests
    from tqdm import tqdm

    # CETUC está disponível em OpenSLR 132
    urls = [
        "https://www.openslr.org/resources/132/cetuc.tar.gz",
    ]

    cetuc_raw = RAW_DIR / "cetuc"
    cetuc_raw.mkdir(parents=True, exist_ok=True)

    for url in urls:
        filename = url.split("/")[-1]
        dest = cetuc_raw / filename

        if dest.exists():
            logger.info(f"Arquivo ja existe: {dest}")
        else:
            logger.info(f"Baixando {url}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                with open(dest, "wb") as f, tqdm(
                    total=total, unit="iB", unit_scale=True, desc=filename
                ) as bar:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            except Exception as e:
                logger.error(f"Erro ao baixar {url}: {e}")
                return 0

        # Extrair
        if filename.endswith(".tar.gz"):
            import tarfile

            logger.info(f"Extraindo {filename}...")
            with tarfile.open(str(dest), "r:gz") as tar:
                tar.extractall(str(cetuc_raw))

    # Processar WAVs extraídos
    wav_files = list(cetuc_raw.rglob("*.wav"))
    logger.info(f"Encontrados {len(wav_files)} arquivos WAV no CETUC")

    count = count_existing(REAL_DIR, "cetuc")
    for wav_path in wav_files:
        if count >= max_samples:
            break
        output_path = REAL_DIR / f"cetuc_{count:05d}.wav"
        if normalize_audio(wav_path, output_path):
            count += 1

        if count % 100 == 0:
            logger.info(f"  Processadas {count}/{max_samples} amostras reais...")

    logger.info(f"CETUC: {count} amostras reais salvas")
    return count


# ---------------------------------------------------------------------------
# 2. Fake Voices (fake speech via XTTS)
# ---------------------------------------------------------------------------
def download_fake_voices(max_samples: int = DEFAULT_MAX_SAMPLES):
    """
    Baixa o dataset Fake Voices do Hugging Face (unfake/fake_voices).
    ~140h de fala sintética gerada com XTTS, 101 falantes, licença MIT.
    """
    logger.info("=" * 60)
    logger.info("Baixando Fake Voices (fala sintetica PT-BR)...")
    logger.info("=" * 60)

    existing = count_existing(FAKE_DIR, "fakevoice")
    if existing >= max_samples:
        logger.info(f"Ja existem {existing} amostras Fake Voices. Pulando.")
        return existing

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Pacote 'datasets' nao encontrado. Instale: pip install datasets")
        return 0

    try:
        logger.info("Carregando unfake/fake_voices do Hugging Face (streaming)...")
        dataset = load_dataset(
            "unfake/fake_voices",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Erro ao carregar fake_voices: {e}")
        return 0

    count = existing
    for item in dataset:
        if count >= max_samples:
            break

        if "audio" not in item:
            continue

        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]

        # Reamostrar para 16kHz
        if sample_rate != TARGET_SR:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
            )

        duration = len(audio_array) / TARGET_SR
        if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
            continue

        # Normalizar
        peak = np.max(np.abs(audio_array))
        if peak > 0:
            audio_array = audio_array / peak * 0.95

        output_path = FAKE_DIR / f"fakevoice_{count:05d}.wav"
        sf.write(str(output_path), audio_array, TARGET_SR, subtype="PCM_16")
        count += 1

        if count % 100 == 0:
            logger.info(f"  Processadas {count}/{max_samples} amostras fake...")

    logger.info(f"Fake Voices: {count} amostras fake salvas em {FAKE_DIR}")
    return count


# ---------------------------------------------------------------------------
# 3. BRSpeech-DF (subset)
# ---------------------------------------------------------------------------
def download_brspeech_subset(max_samples: int = DEFAULT_MAX_SAMPLES):
    """
    Baixa um subset do BRSpeech-DF (AKCIT-Deepfake/BRSpeech-DF).
    Dataset completo: 459K arquivos, 243GB, 24kHz.
    Baixa apenas um subset gerenciável para treinamento.
    Estrutura original: bonafide/ (real) + spoof/ (fake)
    """
    logger.info("=" * 60)
    logger.info("Baixando BRSpeech-DF subset...")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Pacote 'datasets' nao encontrado. Instale: pip install datasets")
        return 0, 0

    # Calcular quantos por classe
    samples_per_class = max_samples // 2

    existing_real = count_existing(REAL_DIR, "brspeech")
    existing_fake = count_existing(FAKE_DIR, "brspeech")

    if existing_real >= samples_per_class and existing_fake >= samples_per_class:
        logger.info(f"Ja existem {existing_real} real + {existing_fake} fake BRSpeech. Pulando.")
        return existing_real, existing_fake

    try:
        logger.info("Carregando AKCIT-Deepfake/BRSpeech-DF (streaming)...")
        dataset = load_dataset(
            "AKCIT-Deepfake/BRSpeech-DF",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Erro ao carregar BRSpeech-DF: {e}")
        return 0, 0

    real_count = existing_real
    fake_count = existing_fake

    for item in dataset:
        if real_count >= samples_per_class and fake_count >= samples_per_class:
            break

        if "audio" not in item:
            continue

        # Determinar label: bonafide → real, spoof → fake
        label = item.get("label", item.get("class", ""))
        if isinstance(label, int):
            # 0=bonafide, 1=spoof (convenção ASVspoof)
            is_real = label == 0
        elif isinstance(label, str):
            label_lower = label.lower()
            is_real = "bonafide" in label_lower or "real" in label_lower
        else:
            continue

        if is_real and real_count >= samples_per_class:
            continue
        if not is_real and fake_count >= samples_per_class:
            continue

        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]

        # Reamostrar de 24kHz para 16kHz
        if sample_rate != TARGET_SR:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
            )

        duration = len(audio_array) / TARGET_SR
        if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
            continue

        # Normalizar
        peak = np.max(np.abs(audio_array))
        if peak > 0:
            audio_array = audio_array / peak * 0.95

        if is_real:
            output_path = REAL_DIR / f"brspeech_{real_count:05d}.wav"
            sf.write(str(output_path), audio_array, TARGET_SR, subtype="PCM_16")
            real_count += 1
        else:
            output_path = FAKE_DIR / f"brspeech_{fake_count:05d}.wav"
            sf.write(str(output_path), audio_array, TARGET_SR, subtype="PCM_16")
            fake_count += 1

        total = real_count + fake_count
        if total % 200 == 0:
            logger.info(f"  BRSpeech: {real_count} real + {fake_count} fake processadas...")

    logger.info(f"BRSpeech-DF: {real_count} real + {fake_count} fake salvas")
    return real_count, fake_count


# ---------------------------------------------------------------------------
# 4. MLAAD v9 (Portuguese subset — fake only)
# ---------------------------------------------------------------------------
def download_mlaad_pt(max_samples: int = DEFAULT_MAX_SAMPLES):
    """
    Baixa o subset em Português do MLAAD v9 (OU-CSAIL/MLAAD).
    678h multilíngue, 51 línguas. Apenas fakes (gerados por diversos TTS).
    Licença: CC-BY-NC 4.0.
    """
    logger.info("=" * 60)
    logger.info("Baixando MLAAD v9 PT-BR subset (fake)...")
    logger.info("=" * 60)

    existing = count_existing(FAKE_DIR, "mlaad")
    if existing >= max_samples:
        logger.info(f"Ja existem {existing} amostras MLAAD. Pulando.")
        return existing

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Pacote 'datasets' nao encontrado. Instale: pip install datasets")
        return 0

    try:
        logger.info("Carregando OU-CSAIL/MLAAD (streaming)...")
        dataset = load_dataset(
            "OU-CSAIL/MLAAD",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Erro ao carregar MLAAD: {e}")
        return 0

    count = existing
    skipped = 0

    for item in dataset:
        if count >= max_samples:
            break

        # Filtrar apenas português
        lang = item.get("language", item.get("lang", ""))
        if isinstance(lang, str):
            lang_lower = lang.lower()
            if "pt" not in lang_lower and "portuguese" not in lang_lower and "brasil" not in lang_lower:
                skipped += 1
                if skipped % 5000 == 0:
                    logger.info(f"  Pulando amostras nao-PT ({skipped} ate agora)...")
                continue

        if "audio" not in item:
            continue

        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]

        if sample_rate != TARGET_SR:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
            )

        duration = len(audio_array) / TARGET_SR
        if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
            continue

        peak = np.max(np.abs(audio_array))
        if peak > 0:
            audio_array = audio_array / peak * 0.95

        output_path = FAKE_DIR / f"mlaad_{count:05d}.wav"
        sf.write(str(output_path), audio_array, TARGET_SR, subtype="PCM_16")
        count += 1

        if count % 100 == 0:
            logger.info(f"  MLAAD PT: {count}/{max_samples} amostras fake...")

    logger.info(f"MLAAD: {count} amostras fake PT salvas em {FAKE_DIR}")
    return count


# ---------------------------------------------------------------------------
# Relatório final
# ---------------------------------------------------------------------------
def print_report():
    """Imprime estatisticas do dataset."""
    real_count = len(list(REAL_DIR.glob("*.wav")))
    fake_count = len(list(FAKE_DIR.glob("*.wav")))

    # Calcular duração total estimada
    real_duration = 0.0
    for f in list(REAL_DIR.glob("*.wav"))[:100]:  # amostrar 100
        try:
            info = sf.info(str(f))
            real_duration += info.duration
        except Exception:
            pass
    if real_count > 100:
        real_duration = real_duration / 100 * real_count

    fake_duration = 0.0
    for f in list(FAKE_DIR.glob("*.wav"))[:100]:
        try:
            info = sf.info(str(f))
            fake_duration += info.duration
        except Exception:
            pass
    if fake_count > 100:
        fake_duration = fake_duration / 100 * fake_count

    logger.info("")
    logger.info("=" * 60)
    logger.info("RELATORIO FINAL DO DATASET")
    logger.info("=" * 60)
    logger.info(f"  Amostras reais : {real_count:>6d}  (~{real_duration/3600:.1f}h)")
    logger.info(f"  Amostras fake  : {fake_count:>6d}  (~{fake_duration/3600:.1f}h)")
    logger.info(f"  Total          : {real_count + fake_count:>6d}")
    logger.info(f"  Ratio real/fake: {real_count/(fake_count or 1):.2f}")
    logger.info(f"  Diretorio real : {REAL_DIR}")
    logger.info(f"  Diretorio fake : {FAKE_DIR}")
    logger.info("=" * 60)

    if real_count == 0 or fake_count == 0:
        logger.warning("ATENCAO: Uma das classes esta vazia! Treinamento nao sera possivel.")
    elif abs(real_count - fake_count) / max(real_count, fake_count) > 0.5:
        logger.warning(
            "ATENCAO: Desbalanceamento significativo entre classes. "
            "O sistema usa class_weight='balanced' para compensar."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download de datasets em Portugues para deteccao de deepfake de audio"
    )
    parser.add_argument("--all", action="store_true", help="Baixar todos os datasets recomendados (CETUC + Fake Voices)")
    parser.add_argument("--cetuc", action="store_true", help="Baixar CETUC/Common Voice PT (real)")
    parser.add_argument("--fake-voices", action="store_true", help="Baixar Fake Voices (fake)")
    parser.add_argument("--brspeech-subset", action="store_true", help="Baixar subset do BRSpeech-DF (real + fake)")
    parser.add_argument("--mlaad-pt", action="store_true", help="Baixar MLAAD PT (fake)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximo de amostras por classe (default: {DEFAULT_MAX_SAMPLES})",
    )

    args = parser.parse_args()

    # Se nenhum flag, mostrar ajuda
    if not any([args.all, args.cetuc, args.fake_voices, args.brspeech_subset, args.mlaad_pt]):
        parser.print_help()
        print("\nExemplo recomendado:")
        print("  python scripts/download_portuguese_datasets.py --all --max-samples 2000")
        return

    setup_dirs()
    logger.info(f"Max amostras por classe: {args.max_samples}")

    if args.all or args.cetuc:
        download_cetuc(args.max_samples)

    if args.all or args.fake_voices:
        download_fake_voices(args.max_samples)

    if args.brspeech_subset:
        download_brspeech_subset(args.max_samples)

    if args.mlaad_pt:
        download_mlaad_pt(args.max_samples)

    print_report()
    logger.info("Download concluido!")


if __name__ == "__main__":
    main()
