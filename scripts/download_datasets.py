#!/usr/bin/env python3
"""
download_datasets.py — Download de datasets para detecção de deepfake de áudio.

Substitui download_portuguese_datasets.py e download_pt_datasets_v2.py.

Datasets PT-BR (foco local):
  --brspeech      BRSpeech-DF (AKCIT-Deepfake/BRSpeech-DF): bonafide→real, spoof→fake
  --cetuc         CETUC/CommonVoice v17 PT-BR / FLEURS / OpenSLR (real)
  --fake-voices   Fake Voices XTTS (unfake/fake_voices): ZIPs por falante (fake)
  --fleurs        FLEURS PT-BR Google (real)
  --mlaad-pt      MLAAD v9 subconjunto PT (fake, CC-BY-NC 4.0)
  --common-voice-pt  Mozilla Common Voice v17 PT (real, CC0)

Datasets internacionais (referência anti-spoofing):
  --asvspoof2019  ASVspoof 2019 LA — benchmark padrão (real + fake, ODC-BY)
  --wavefake      WaveFake — 6 vocoders TTS (fake only, MIT)
  --in-the-wild   In-the-Wild — 58 celebridades (real + deepfake, CC 4.0)
  --asvspoof5     ASVspoof 5 (2024) — 20+ tipos de ataque (real + fake)

Atalhos:
  --all           brspeech + cetuc + fake-voices (recomendado para build_dataset.py)
  --all-intl      asvspoof2019 + wavefake + in-the-wild (datasets internacionais)

Uso:
  python scripts/download_datasets.py --all --max-samples 2000
  python scripts/download_datasets.py --brspeech --max-samples 1000
  python scripts/download_datasets.py --cetuc --max-samples 1000
  python scripts/download_datasets.py --fake-voices --max-speakers 20
  python scripts/download_datasets.py --fleurs --max-samples 500
  python scripts/download_datasets.py --mlaad-pt --max-samples 500
  python scripts/download_datasets.py --asvspoof2019 --max-samples 2000
  python scripts/download_datasets.py --wavefake --max-samples 2000
  python scripts/download_datasets.py --in-the-wild --max-samples 1000
  python scripts/download_datasets.py --asvspoof5 --max-samples 2000
  python scripts/download_datasets.py --all-intl --max-samples 2000
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
# Diagnóstico de erros acionável (BUG.Datasets.2)
# ---------------------------------------------------------------------------

def check_dependencies(required: list[str]) -> list[str]:
    """Verifica pacotes e retorna os que faltam (com nome pip).

    Args:
        required: nomes de import (ex: ["datasets", "huggingface_hub"]).
    Returns:
        Lista de comandos pip para instalar o que falta (vazia se tudo OK).
    """
    import importlib.util
    _pip_names = {
        "datasets": "datasets>=4.0",
        "huggingface_hub": "huggingface_hub",
        "pandas": "pandas",
        "requests": "requests",
        "tqdm": "tqdm",
    }
    missing = []
    for mod in required:
        if importlib.util.find_spec(mod) is None:
            missing.append(_pip_names.get(mod, mod))
    return missing


def log_missing_deps(required: list[str], dataset_name: str) -> bool:
    """Loga uma mensagem acionável se faltarem dependências. Retorna True se OK."""
    missing = check_dependencies(required)
    if missing:
        logger.error(
            f"[{dataset_name}] Dependências faltando. Para corrigir, execute:\n"
            f"    pip install {' '.join(missing)}\n"
            f"  (ou instale tudo de uma vez: pip install -r requirements.txt)"
        )
        return False
    return True


def diagnose_hf_error(exc: Exception, dataset_name: str, repo: str = "") -> str:
    """Analisa exceção do HuggingFace e devolve mensagem ACIONÁVEL.

    Distingue: autenticação, dataset gated (aceitar termos), rede, repo
    inexistente — em vez do genérico 'indisponível: {e}'.
    """
    msg = str(exc).lower()
    repo_hint = f" ({repo})" if repo else ""

    # Autenticação / token ausente
    if any(k in msg for k in ("401", "unauthorized", "authentication", "token",
                              "must be authenticated", "login")):
        return (
            f"[{dataset_name}]{repo_hint} requer AUTENTICAÇÃO no HuggingFace.\n"
            "  1. Crie um token: https://huggingface.co/settings/tokens\n"
            "  2. Autentique: huggingface-cli login\n"
            "  3. Tente novamente."
        )
    # Dataset gated (precisa aceitar termos)
    if any(k in msg for k in ("403", "forbidden", "gated", "access to model",
                              "accept", "agree", "terms")):
        url = f"https://huggingface.co/datasets/{repo}" if repo else "a página do dataset"
        return (
            f"[{dataset_name}]{repo_hint} é GATED — você precisa aceitar os termos.\n"
            f"  1. Acesse: {url}\n"
            "  2. Clique em 'Agree and access repository'\n"
            "  3. Autentique: huggingface-cli login\n"
            "  4. Tente novamente."
        )
    # Rede / timeout
    if any(k in msg for k in ("connection", "timeout", "timed out", "max retries",
                              "network", "temporarily unavailable", "503", "502")):
        return (
            f"[{dataset_name}]{repo_hint}: falha de REDE.\n"
            "  • Verifique sua conexão com a internet.\n"
            "  • HuggingFace pode estar instável — tente novamente em alguns minutos.\n"
            "  • Atrás de proxy? Configure HF_ENDPOINT ou HTTPS_PROXY."
        )
    # Repo inexistente
    if any(k in msg for k in ("404", "not found", "doesn't exist", "repository not found",
                              "couldn't find")):
        return (
            f"[{dataset_name}]{repo_hint}: repositório NÃO encontrado.\n"
            "  • O nome do dataset pode ter mudado no HuggingFace.\n"
            "  • Verifique se o repo ainda existe e está público."
        )
    # Genérico (último recurso) — mas ainda mostra o erro real
    return f"[{dataset_name}]{repo_hint} indisponível: {exc}"


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


def next_index(directory: Path, prefix: str) -> int:
    """Retorna o próximo índice livre para o prefixo (evita sobrescrever arquivos).

    BUG FIX: count_wavs() retorna a CONTAGEM, não o índice máximo.
    Se o usuário deletar `brspeech_00050.wav` no meio, count=N mas o
    próximo arquivo `brspeech_{N:05d}` colide com `brspeech_{N}.wav` existente.
    """
    if not directory.exists():
        return 0
    max_idx = -1
    for p in directory.glob(f"{prefix}_*.wav"):
        try:
            idx = int(p.stem.split("_")[-1])
            if idx > max_idx:
                max_idx = idx
        except (ValueError, IndexError):
            continue
    return max_idx + 1


def safe_write_wav(target_path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> bool:
    """Salva WAV com validação final de NaN/Inf — última linha de defesa.

    Returns True se gravado com sucesso; False se o áudio era inválido.
    """
    if audio is None or audio.size == 0:
        return False
    if not np.all(np.isfinite(audio)):
        logger.warning(f"  safe_write: rejeitado por NaN/Inf — {target_path.name}")
        return False
    if float(np.max(np.abs(audio))) < 1e-6:
        return False
    try:
        sf.write(str(target_path), audio.astype(np.float32), sr, subtype="PCM_16")
        return True
    except Exception as e:
        logger.warning(f"  safe_write: erro ao gravar {target_path.name}: {e}")
        return False


def cast_no_decode(ds):
    """Desativa o decode automático de áudio do HuggingFace.

    BUG FIX: a partir de `datasets` >= 4.0, o decode de áudio em streaming exige
    o pacote `torchcodec` (que depende de torch + FFmpeg e é problemático no
    Windows). Sem ele, iterar o dataset lança:
        "To support decoding audio data, please install 'torchcodec'."
    e TODOS os downloads via streaming falham.

    Solução: desativamos o decode (Audio(decode=False)) e decodificamos os bytes
    manualmente com soundfile (já uma dependência) em `extract_audio()`.
    """
    try:
        from datasets import Audio
        return ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        logger.debug(f"  cast decode=False indisponível ({e}); usando decode padrão")
        return ds


def extract_audio(item) -> tuple:
    """Extrai (array, sample_rate) de um item HuggingFace, tolerante a decode on/off.

    Com decode desativado (cast_no_decode), `item['audio']` traz bytes/path em vez
    de um array já decodificado — evitando a dependência de torchcodec.
    Retorna (None, None) se não for possível extrair.
    """
    audio = item.get("audio") if isinstance(item, dict) else None
    if not isinstance(audio, dict):
        return None, None

    # 1. decode=False → bytes embutidos (caminho mais comum em streaming)
    raw = audio.get("bytes")
    if raw:
        try:
            data, sr = sf.read(io.BytesIO(raw))
            return data, sr
        except Exception as e:
            logger.debug(f"  extract_audio bytes falhou: {e}")

    # 2. caminho local em disco
    path = audio.get("path")
    if path and Path(str(path)).exists():
        try:
            data, sr = sf.read(str(path))
            return data, sr
        except Exception:
            try:
                data, sr = librosa.load(str(path), sr=None, mono=False)
                return data, sr
            except Exception as e:
                logger.debug(f"  extract_audio path falhou: {e}")

    # 3. decode=True já funcionou (array presente) — fallback
    arr = audio.get("array")
    if arr is not None:
        return arr, audio.get("sampling_rate", TARGET_SR)

    return None, None


def process_audio(audio_array, sample_rate: int) -> tuple[np.ndarray | None, bool]:
    """Reamostra, valida duração, checa silêncio e normaliza.

    Sanitiza NaN/Inf, força mono, converte para float32 [-1, 1].
    Returns (normalized_array, True) ou (None, False).

    BUG FIX: Sem essa sanitização, áudios corrompidos com NaN passam direto
    pelo pipeline e são salvos no disco. O wizard de treino lia esses WAVs
    e gerava `loss: nan` na 1ª época.
    """
    # 1. Coerção para numpy float32 (HF às vezes retorna list/torch.Tensor)
    try:
        audio_array = np.asarray(audio_array, dtype=np.float32)
    except (ValueError, TypeError) as e:
        logger.debug(f"  process_audio: conversão falhou ({e})")
        return None, False

    if audio_array.size == 0:
        return None, False

    # 2. Stereo → mono (HF streaming pode retornar (T, 2) ou (2, T))
    if audio_array.ndim > 1:
        # Se a primeira dim é menor (ex: 2), é channels-first → transpor antes
        if audio_array.shape[0] < audio_array.shape[-1] and audio_array.shape[0] <= 8:
            audio_array = audio_array.T
        audio_array = audio_array.mean(axis=-1)

    # Garante 1D
    audio_array = audio_array.squeeze()
    if audio_array.ndim != 1:
        return None, False

    # 3. Sanitiza NaN/Inf ANTES de qualquer cálculo numérico
    if not np.all(np.isfinite(audio_array)):
        finite_mask = np.isfinite(audio_array)
        if finite_mask.sum() < audio_array.size * 0.5:
            # Mais da metade do áudio é NaN/Inf → arquivo corrompido demais
            return None, False
        # Substitui NaN/Inf por 0 (silêncio local) — preserva o resto válido
        audio_array = np.where(finite_mask, audio_array, 0.0).astype(np.float32)

    # 4. Reamostragem (apenas se necessário)
    if sample_rate != TARGET_SR:
        try:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=TARGET_SR
            )
        except Exception as e:
            logger.debug(f"  resample falhou ({sample_rate}→{TARGET_SR}): {e}")
            return None, False
        # Resample pode introduzir NaN se o sinal tiver descontinuidades extremas
        if not np.all(np.isfinite(audio_array)):
            return None, False

    # 5. Validação de duração
    duration = len(audio_array) / TARGET_SR
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None, False

    # 6. Sanity check: rejeita áudios com magnitude absurda (>10x range esperado).
    # Indica conversão errada de int16/int32 sem dividir por 32768/2147483648.
    abs_max = float(np.max(np.abs(audio_array)))
    if abs_max > 100.0:
        logger.debug(f"  process_audio: magnitude absurda ({abs_max:.1f}), pulando")
        return None, False

    # 7. Checagem de silêncio (peak baixo demais)
    if abs_max < 1e-6:
        return None, False

    # 8. Normalização: peak → 0.95 (evita clipping no PCM 16-bit)
    return (audio_array / abs_max * 0.95).astype(np.float32), True


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

        # BUG FIX: use next_index para evitar sobrescrever arquivos quando
        # o usuário deletou itens no meio (count_wavs() < max_index existente).
        idx = next_index(target_dir, prefix)

        logger.info(f"  {data_dir}...")
        try:
            ds = load_dataset(
                "AKCIT-Deepfake/BRSpeech-DF",
                data_dir=data_dir,
                split="train",
                streaming=True,
            )
            ds = cast_no_decode(ds)  # evita exigir torchcodec
            for item in ds:
                if current >= samples_per_class:
                    break
                if "audio" not in item:
                    continue
                _arr, _sr = extract_audio(item)
                if _arr is None:
                    continue
                audio, ok = process_audio(_arr, _sr)
                if not ok:
                    continue
                if safe_write_wav(target_dir / f"{prefix}_{idx:05d}.wav", audio):
                    current += 1
                    idx += 1
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
    idx = next_index(target_dir, prefix)
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
                        # process_audio agora trata stereo internamente
                        audio, ok = process_audio(raw_data, sr)
                        if not ok:
                            continue
                        if safe_write_wav(target_dir / f"{prefix}_{idx:05d}.wav", audio):
                            count += 1
                            idx += 1
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

    dataset = cast_no_decode(dataset)  # evita exigir torchcodec
    count = existing
    idx = next_index(REAL_DIR, "cetuc")
    for item in dataset:
        if count >= max_samples:
            break
        if "audio" not in item:
            continue
        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue
        if safe_write_wav(REAL_DIR / f"cetuc_{idx:05d}.wav", audio):
            count += 1
            idx += 1
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
    idx = next_index(REAL_DIR, "cetuc")
    for wav_path in wav_files:
        if count >= max_samples:
            break
        try:
            y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            logger.debug(f"  librosa.load falhou em {wav_path.name}: {e}")
            continue
        audio, ok = process_audio(y, TARGET_SR)
        if ok and safe_write_wav(REAL_DIR / f"cetuc_{idx:05d}.wav", audio):
            count += 1
            idx += 1

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
    idx = next_index(FAKE_DIR, "fkvoice")
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
                        # process_audio trata stereo→mono internamente
                        audio, ok = process_audio(raw_data, sr)
                        if not ok:
                            continue
                        if safe_write_wav(FAKE_DIR / f"fkvoice_{idx:05d}.wav", audio):
                            total_count += 1
                            idx += 1
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
        ds = cast_no_decode(ds)  # evita exigir torchcodec
    except Exception as e:
        logger.error(diagnose_hf_error(e, "FLEURS", "google/fleurs"))
        return

    count = existing
    idx = next_index(REAL_DIR, "fleurs")
    for item in ds:
        if count >= max_samples:
            break
        if "audio" not in item:
            continue
        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue
        if safe_write_wav(REAL_DIR / f"fleurs_{idx:05d}.wav", audio):
            count += 1
            idx += 1
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
        ds = cast_no_decode(ds)  # evita exigir torchcodec
    except Exception as e:
        logger.error(diagnose_hf_error(e, "MLAAD", "OU-CSAIL/MLAAD"))
        return

    count = existing
    skipped = 0
    seen = 0
    # BUG FIX: hard cap para evitar spin infinito. MLAAD tem ~160K amostras
    # multilíngue; se PT for raro/inexistente, sem o cap iteraríamos todos.
    _MAX_SEEN = max(50_000, max_samples * 50)
    next_idx = next_index(FAKE_DIR, "mlaad")
    if count == 0:
        next_idx = max(next_idx, 0)

    for item in ds:
        if count >= max_samples:
            break
        seen += 1
        if seen > _MAX_SEEN:
            logger.warning(
                f"  MLAAD: limite de iterações ({_MAX_SEEN}) atingido sem completar. "
                f"Encontrados {count} PT em {seen} amostras. "
                "MLAAD pode ter poucos exemplos PT — tente outro idioma ou WaveFake."
            )
            break

        lang = item.get("language", item.get("lang", ""))
        if isinstance(lang, str):
            lang_lower = lang.lower()
            if "pt" not in lang_lower and "portuguese" not in lang_lower and "brasil" not in lang_lower:
                skipped += 1
                if skipped % 5000 == 0:
                    logger.info(f"  Pulando amostras não-PT ({skipped} de {seen} vistas)...")
                continue
        if "audio" not in item:
            continue
        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue
        if safe_write_wav(FAKE_DIR / f"mlaad_{next_idx:05d}.wav", audio):
            count += 1
            next_idx += 1
            if count % 100 == 0:
                logger.info(f"  MLAAD PT: {count}/{max_samples}")

    logger.info(f"MLAAD: {count} amostras fake PT (de {seen} amostras vistas)")


# ---------------------------------------------------------------------------
# 6. ASVspoof 2019 LA — benchmark padrão anti-spoofing (real + fake, ODC-BY)
# ---------------------------------------------------------------------------

def download_asvspoof2019(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """ASVspoof 2019 LA (LanceaKing/asvspoof2019) — bonafide→real, spoof→fake.

    Licença: ODC-BY 1.0
    HuggingFace: https://huggingface.co/datasets/LanceaKing/asvspoof2019
    """
    logger.info("=" * 60)
    logger.info("ASVSPOOF 2019 LA (bonafide + spoof)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    samples_per_class = max_samples // 2
    real_count = count_wavs(REAL_DIR, "asv2019")
    fake_count = count_wavs(FAKE_DIR, "asv2019")

    if real_count >= samples_per_class and fake_count >= samples_per_class:
        logger.info(f"Já existem {real_count} real + {fake_count} fake ASVspoof2019. Pulando.")
        return

    logger.info(f"Iniciando (já tem {real_count} real + {fake_count} fake)...")

    # Tentar config LA; sem config como fallback
    ds = None
    last_exc = None
    for kwargs in [
        {"name": "LA", "split": "train", "streaming": True},
        {"split": "train", "streaming": True},
    ]:
        try:
            ds = load_dataset("LanceaKing/asvspoof2019", **kwargs)
            logger.info(f"  Dataset carregado com {kwargs}")
            break
        except Exception as e:
            last_exc = e
            logger.info(f"  Tentativa falhou ({e}), tentando próxima configuração...")

    if ds is None:
        logger.error(diagnose_hf_error(
            last_exc or Exception("indisponível"),
            "ASVspoof2019", "LanceaKing/asvspoof2019"))
        return

    ds = cast_no_decode(ds)  # evita exigir torchcodec
    real_c, fake_c = real_count, fake_count
    real_idx = next_index(REAL_DIR, "asv2019")
    fake_idx = next_index(FAKE_DIR, "asv2019")
    seen = 0
    _MAX_SEEN = max(50_000, max_samples * 30)  # cap anti-spin
    _BONAFIDE_VALUES = {0, "0", "bonafide", "genuine", "real"}
    _SPOOF_VALUES    = {1, "1", "spoof", "fake", "synthetic", "attack"}

    for item in ds:
        if real_c >= samples_per_class and fake_c >= samples_per_class:
            break
        seen += 1
        if seen > _MAX_SEEN:
            logger.warning(f"  ASVspoof2019: cap de iterações ({_MAX_SEEN}) atingido.")
            break
        if "audio" not in item:
            continue

        raw_label = item.get("label", item.get("class", item.get("speaker_type", None)))
        if raw_label is None:
            continue

        label_str = str(raw_label).lower()
        is_real = raw_label in _BONAFIDE_VALUES or label_str in _BONAFIDE_VALUES
        is_fake = raw_label in _SPOOF_VALUES    or label_str in _SPOOF_VALUES

        if not is_real and not is_fake:
            continue
        if is_real and real_c >= samples_per_class:
            continue
        if is_fake and fake_c >= samples_per_class:
            continue

        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue

        if is_real:
            if safe_write_wav(REAL_DIR / f"asv2019_{real_idx:05d}.wav", audio):
                real_c += 1
                real_idx += 1
                if real_c % 100 == 0:
                    logger.info(f"  real: {real_c}/{samples_per_class}")
        else:
            if safe_write_wav(FAKE_DIR / f"asv2019_{fake_idx:05d}.wav", audio):
                fake_c += 1
                fake_idx += 1
                if fake_c % 100 == 0:
                    logger.info(f"  fake: {fake_c}/{samples_per_class}")

    logger.info(f"ASVspoof2019 LA: {real_c} real + {fake_c} fake (de {seen} vistas)")


# ---------------------------------------------------------------------------
# 7. WaveFake — 6 vocoders TTS (fake only, MIT License)
# ---------------------------------------------------------------------------

def download_wavefake(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """WaveFake (Zenodo 5642694 / HF espelho) — 6 vocoders, fake only.

    Vocoders: MelGAN, ParallelWaveGAN, HiFi-GAN, WaveGlow, MB-MelGAN, FB-MelGAN.
    Licença: MIT
    Zenodo: https://zenodo.org/records/5642694
    """
    logger.info("=" * 60)
    logger.info("WAVEFAKE (fake — 6 vocoders TTS)")
    logger.info("=" * 60)

    existing = count_wavs(FAKE_DIR, "wavefake")
    if existing >= max_samples:
        logger.info(f"Já existem {existing} amostras WaveFake. Pulando.")
        return

    logger.info(f"Iniciando (já tem {existing} amostras)...")

    # Tentar HuggingFace primeiro (vários espelhos conhecidos)
    _hf_repos = [
        ("speech-detection/WaveFake",      {"split": "train", "streaming": True}),
        ("halflingknight/wavefake",         {"split": "train", "streaming": True}),
        ("SantiagoCalderon/WaveFake",       {"split": "train", "streaming": True}),
    ]
    count = existing
    idx = next_index(FAKE_DIR, "wavefake")
    used_hf = False

    try:
        from datasets import load_dataset
        for repo, kwargs in _hf_repos:
            try:
                ds = load_dataset(repo, **kwargs)
                ds = cast_no_decode(ds)  # evita exigir torchcodec
                logger.info(f"  Usando HuggingFace: {repo}")
                for item in ds:
                    if count >= max_samples:
                        break
                    if "audio" not in item:
                        continue
                    _arr, _sr = extract_audio(item)
                    if _arr is None:
                        continue
                    audio, ok = process_audio(_arr, _sr)
                    if not ok:
                        continue
                    if safe_write_wav(FAKE_DIR / f"wavefake_{idx:05d}.wav", audio):
                        count += 1
                        idx += 1
                        if count % 100 == 0:
                            logger.info(f"  wavefake: {count}/{max_samples}")
                used_hf = True
                break
            except Exception as e:
                logger.info(f"  {repo} indisponível: {e}")
    except ImportError:
        logger.warning("datasets não instalado, tentando Zenodo...")

    if not used_hf and count < max_samples:
        count = _download_wavefake_zenodo(count, max_samples)

    logger.info(f"WaveFake: {count} amostras fake")


def _download_wavefake_zenodo(current_count: int, max_samples: int) -> int:
    """Fallback: download direto do Zenodo 5642694 via API."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("Instale: pip install requests tqdm")
        return current_count

    zenodo_api = "https://zenodo.org/api/records/5642694"
    logger.info(f"  Consultando Zenodo API: {zenodo_api}")

    try:
        resp = requests.get(zenodo_api, timeout=30)
        resp.raise_for_status()
        record = resp.json()
    except Exception as e:
        logger.error(f"  Zenodo API indisponível: {e}")
        return current_count

    files = record.get("files", [])
    # Filtrar apenas ZIPs de arquiteturas fake (excluir LJSpeech real)
    fake_zips = [
        f for f in files
        if f["key"].endswith(".zip") and "LJSpeech-1.1" not in f["key"]
    ]
    logger.info(f"  {len(fake_zips)} ZIPs de vocoders encontrados no Zenodo")

    count = current_count
    idx = next_index(FAKE_DIR, "wavefake")
    wavefake_raw = RAW_DIR / "wavefake_cache"
    wavefake_raw.mkdir(parents=True, exist_ok=True)

    for file_info in fake_zips:
        if count >= max_samples:
            break
        url = file_info["links"]["self"]
        zip_name = file_info["key"]
        local_zip = wavefake_raw / Path(zip_name).name

        if not local_zip.exists():
            logger.info(f"  Baixando {zip_name} ({file_info['size'] // 1_000_000} MB)...")
            try:
                r = requests.get(url, stream=True, timeout=120)
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(local_zip, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True,
                                                       desc=Path(zip_name).name) as bar:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                        bar.update(len(chunk))
            except Exception as e:
                logger.error(f"  Erro ao baixar {zip_name}: {e}")
                continue

        logger.info(f"  Extraindo {local_zip.name}...")
        try:
            with zipfile.ZipFile(str(local_zip), "r") as zf:
                wav_names = [n for n in zf.namelist() if n.lower().endswith(".wav")]
                for wav_name in wav_names:
                    if count >= max_samples:
                        break
                    try:
                        with zf.open(wav_name) as af:
                            raw_data, sr = sf.read(io.BytesIO(af.read()))
                        # process_audio trata stereo→mono internamente
                        audio, ok = process_audio(raw_data, sr)
                        if not ok:
                            continue
                        if safe_write_wav(FAKE_DIR / f"wavefake_{idx:05d}.wav", audio):
                            count += 1
                            idx += 1
                    except Exception as e:
                        logger.debug(f"  Erro em {wav_name}: {e}")
        except Exception as e:
            logger.error(f"  Erro ao extrair {local_zip.name}: {e}")

        logger.info(f"  Total acumulado: {count}/{max_samples}")

    return count


# ---------------------------------------------------------------------------
# 8. In-the-Wild — 58 celebridades (real + deepfake, CC 4.0)
# ---------------------------------------------------------------------------

def download_in_the_wild(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """In-the-Wild (mueller91/In-The-Wild) — celebridades reais + deepfakes.

    Licença: CC BY 4.0
    HuggingFace: https://huggingface.co/datasets/mueller91/In-The-Wild
    """
    logger.info("=" * 60)
    logger.info("IN-THE-WILD (celebridades reais + deepfakes)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    samples_per_class = max_samples // 2
    real_count = count_wavs(REAL_DIR, "itw")
    fake_count = count_wavs(FAKE_DIR, "itw")

    if real_count >= samples_per_class and fake_count >= samples_per_class:
        logger.info(f"Já existem {real_count} real + {fake_count} fake In-the-Wild. Pulando.")
        return

    logger.info(f"Iniciando (já tem {real_count} real + {fake_count} fake)...")

    ds = None
    last_exc, last_repo = None, ""
    for repo in ["mueller91/In-The-Wild", "muellersebastian/In-The-Wild"]:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            logger.info(f"  Usando {repo}")
            break
        except Exception as e:
            last_exc, last_repo = e, repo
            logger.info(f"  {repo} indisponível: {e}")

    if ds is None:
        logger.error(diagnose_hf_error(
            last_exc or Exception("indisponível"), "In-the-Wild", last_repo))
        return

    ds = cast_no_decode(ds)  # evita exigir torchcodec
    real_c, fake_c = real_count, fake_count
    real_idx = next_index(REAL_DIR, "itw")
    fake_idx = next_index(FAKE_DIR, "itw")
    seen = 0
    _MAX_SEEN = max(50_000, max_samples * 30)
    _REAL_VALUES = {0, "0", "real", "bonafide", "genuine"}
    _FAKE_VALUES = {1, "1", "fake", "spoof", "deepfake", "synthetic"}

    for item in ds:
        if real_c >= samples_per_class and fake_c >= samples_per_class:
            break
        seen += 1
        if seen > _MAX_SEEN:
            logger.warning(f"  In-the-Wild: cap de iterações ({_MAX_SEEN}) atingido.")
            break
        if "audio" not in item:
            continue

        raw_label = item.get("label", item.get("class", item.get("type", None)))
        if raw_label is None:
            continue

        label_str = str(raw_label).lower()
        is_real = raw_label in _REAL_VALUES or label_str in _REAL_VALUES
        is_fake = raw_label in _FAKE_VALUES or label_str in _FAKE_VALUES

        if not is_real and not is_fake:
            continue
        if is_real and real_c >= samples_per_class:
            continue
        if is_fake and fake_c >= samples_per_class:
            continue

        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue

        if is_real:
            if safe_write_wav(REAL_DIR / f"itw_{real_idx:05d}.wav", audio):
                real_c += 1
                real_idx += 1
                if real_c % 50 == 0:
                    logger.info(f"  real: {real_c}/{samples_per_class}")
        else:
            if safe_write_wav(FAKE_DIR / f"itw_{fake_idx:05d}.wav", audio):
                fake_c += 1
                fake_idx += 1
                if fake_c % 50 == 0:
                    logger.info(f"  fake: {fake_c}/{samples_per_class}")

    logger.info(f"In-the-Wild: {real_c} real + {fake_c} fake (de {seen} vistas)")


# ---------------------------------------------------------------------------
# 9. Mozilla Common Voice PT — fala real PT-BR (CC0, standalone)
# ---------------------------------------------------------------------------

def download_common_voice_pt(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """Mozilla Common Voice v17 PT (mozilla-foundation/common_voice_17_0) — real PT-BR.

    Nota: diferente do --cetuc que usa CV como fallback; este é standalone com
    prefixo 'cvpt' e não depende de outros datasets.

    Licença: CC0
    HuggingFace: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
    Requer autenticação HF: huggingface-cli login
    """
    logger.info("=" * 60)
    logger.info("MOZILLA COMMON VOICE v17 PT (real)")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    existing = count_wavs(REAL_DIR, "cvpt")
    if existing >= max_samples:
        logger.info(f"Já existem {existing} amostras Common Voice PT. Pulando.")
        return

    logger.info(f"Iniciando (já tem {existing} amostras)...")

    # Tentar versões em ordem decrescente
    ds = None
    last_exc, last_repo = None, ""
    for repo, config in [
        ("mozilla-foundation/common_voice_17_0", "pt"),
        ("mozilla-foundation/common_voice_16_1", "pt"),
        ("mozilla-foundation/common_voice_13_0", "pt"),
        ("google/fleurs", "pt_br"),  # fallback sem token
    ]:
        try:
            ds = load_dataset(repo, config, split="train", streaming=True, trust_remote_code=True)
            logger.info(f"  Usando {repo} ({config})")
            break
        except Exception as e:
            last_exc, last_repo = e, repo
            logger.info(f"  {repo} ({config}) indisponível: {e}")

    if ds is None:
        logger.error(diagnose_hf_error(
            last_exc or Exception("indisponível"),
            "Common Voice PT", last_repo or "mozilla-foundation/common_voice_17_0"))
        return

    ds = cast_no_decode(ds)  # evita exigir torchcodec
    count = existing
    idx = next_index(REAL_DIR, "cvpt")
    for item in ds:
        if count >= max_samples:
            break
        if "audio" not in item:
            continue
        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue
        if safe_write_wav(REAL_DIR / f"cvpt_{idx:05d}.wav", audio):
            count += 1
            idx += 1
            if count % 100 == 0:
                logger.info(f"  cvpt: {count}/{max_samples}")

    logger.info(f"Common Voice PT: {count} amostras reais")


# ---------------------------------------------------------------------------
# 10. ASVspoof 5 (2024) — 20+ tipos de ataque (real + fake)
# ---------------------------------------------------------------------------

def download_asvspoof5(max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
    """ASVspoof 5 (jungjee/asvspoof5) — challenge 2024, 20+ ataques TTS/VC.

    O dataset mais recente da série ASVspoof — inclui ataques com LLMs de voz.
    Licença: CC BY 4.0
    HuggingFace: https://huggingface.co/datasets/jungjee/asvspoof5
    """
    logger.info("=" * 60)
    logger.info("ASVSPOOF 5 (2024) — bonafide + 20+ ataques")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Instale: pip install datasets>=4.0")
        return

    samples_per_class = max_samples // 2
    real_count = count_wavs(REAL_DIR, "asv5")
    fake_count = count_wavs(FAKE_DIR, "asv5")

    if real_count >= samples_per_class and fake_count >= samples_per_class:
        logger.info(f"Já existem {real_count} real + {fake_count} fake ASVspoof5. Pulando.")
        return

    logger.info(f"Iniciando (já tem {real_count} real + {fake_count} fake)...")

    ds = None
    last_exc, last_repo = None, ""
    for repo, kwargs in [
        ("jungjee/asvspoof5",    {"split": "train", "streaming": True}),
        ("LanceaKing/asvspoof5", {"split": "train", "streaming": True}),
        ("ASVspoof/asvspoof5",   {"split": "train", "streaming": True}),
    ]:
        try:
            ds = load_dataset(repo, **kwargs)
            logger.info(f"  Usando {repo}")
            break
        except Exception as e:
            last_exc, last_repo = e, repo
            logger.info(f"  {repo} indisponível: {e}")

    if ds is None:
        logger.error(diagnose_hf_error(
            last_exc or Exception("indisponível"),
            "ASVspoof5", last_repo or "jungjee/asvspoof5"))
        return

    ds = cast_no_decode(ds)  # evita exigir torchcodec
    real_c, fake_c = real_count, fake_count
    real_idx = next_index(REAL_DIR, "asv5")
    fake_idx = next_index(FAKE_DIR, "asv5")
    seen = 0
    _MAX_SEEN = max(50_000, max_samples * 30)
    _BONAFIDE_VALUES = {0, "0", "bonafide", "genuine", "real"}
    _SPOOF_VALUES    = {1, "1", "spoof", "fake", "synthetic", "attack"}

    for item in ds:
        if real_c >= samples_per_class and fake_c >= samples_per_class:
            break
        seen += 1
        if seen > _MAX_SEEN:
            logger.warning(f"  ASVspoof5: cap de iterações ({_MAX_SEEN}) atingido.")
            break
        if "audio" not in item:
            continue

        raw_label = item.get("label", item.get("class", item.get("key", None)))
        if raw_label is None:
            continue

        label_str = str(raw_label).lower()
        is_real = raw_label in _BONAFIDE_VALUES or label_str in _BONAFIDE_VALUES
        is_fake = raw_label in _SPOOF_VALUES    or label_str in _SPOOF_VALUES

        if not is_real and not is_fake:
            continue
        if is_real and real_c >= samples_per_class:
            continue
        if is_fake and fake_c >= samples_per_class:
            continue

        _arr, _sr = extract_audio(item)
        if _arr is None:
            continue
        audio, ok = process_audio(_arr, _sr)
        if not ok:
            continue

        if is_real:
            if safe_write_wav(REAL_DIR / f"asv5_{real_idx:05d}.wav", audio):
                real_c += 1
                real_idx += 1
                if real_c % 100 == 0:
                    logger.info(f"  real: {real_c}/{samples_per_class}")
        elif safe_write_wav(FAKE_DIR / f"asv5_{fake_idx:05d}.wav", audio):
            fake_c += 1
            fake_idx += 1
            if fake_c % 100 == 0:
                logger.info(f"  fake: {fake_c}/{samples_per_class}")

    logger.info(f"ASVspoof5: {real_c} real + {fake_c} fake")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── PT-BR ──────────────────────────────────────────────────────────────
    parser.add_argument("--all", action="store_true",
                        help="brspeech + cetuc + fake-voices (recomendado PT-BR)")
    parser.add_argument("--brspeech", action="store_true",
                        help="BRSpeech-DF (real + fake PT-BR)")
    parser.add_argument("--cetuc", action="store_true",
                        help="CETUC/CommonVoice PT-BR (real)")
    parser.add_argument("--fake-voices", action="store_true",
                        help="Fake Voices XTTS via ZIPs (fake)")
    parser.add_argument("--fleurs", action="store_true",
                        help="FLEURS PT-BR do Google (real)")
    parser.add_argument("--mlaad-pt", action="store_true",
                        help="MLAAD v9 subconjunto PT (fake, CC-BY-NC 4.0)")
    parser.add_argument("--common-voice-pt", action="store_true",
                        help="Mozilla Common Voice v17 PT (real, CC0)")
    # ── Internacional ──────────────────────────────────────────────────────
    parser.add_argument("--all-intl", action="store_true",
                        help="asvspoof2019 + wavefake + in-the-wild (referência anti-spoofing)")
    parser.add_argument("--asvspoof2019", action="store_true",
                        help="ASVspoof 2019 LA — benchmark padrão (real + fake, ODC-BY)")
    parser.add_argument("--wavefake", action="store_true",
                        help="WaveFake — 6 vocoders TTS (fake only, MIT)")
    parser.add_argument("--in-the-wild", action="store_true",
                        help="In-the-Wild — 58 celebridades (real + deepfake, CC 4.0)")
    parser.add_argument("--asvspoof5", action="store_true",
                        help="ASVspoof 5 (2024) — 20+ tipos de ataque (real + fake)")
    # ── Parâmetros ─────────────────────────────────────────────────────────
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"Máximo de amostras por classe (padrão: {DEFAULT_MAX_SAMPLES})")
    parser.add_argument("--max-speakers", type=int, default=20,
                        help="Máximo de falantes do Fake Voices (padrão: 20)")

    args = parser.parse_args()

    _any = any([
        args.all, args.all_intl,
        args.brspeech, args.cetuc, args.fake_voices, args.fleurs,
        args.mlaad_pt, args.common_voice_pt,
        args.asvspoof2019, args.wavefake, args.in_the_wild, args.asvspoof5,
    ])
    if not _any:
        parser.print_help()
        print("\nExemplos recomendados:")
        print("  python scripts/download_datasets.py --all --max-samples 2000")
        print("  python scripts/download_datasets.py --all-intl --max-samples 2000")
        print("  python scripts/download_datasets.py --asvspoof2019 --wavefake --max-samples 1000")
        return

    # Pré-check de dependências: avisa sobre TUDO que falta de uma vez,
    # antes de iniciar qualquer download (BUG.Datasets.2).
    needs_hub = args.all or args.fake_voices  # Fake Voices usa huggingface_hub
    core_deps = ["datasets"]
    if needs_hub:
        core_deps += ["huggingface_hub", "pandas"]
    missing = check_dependencies(core_deps)
    if missing:
        logger.warning(
            "Algumas dependências de download podem estar faltando:\n"
            f"    pip install {' '.join(missing)}\n"
            "  Continuando — fontes individuais avisarão se não puderem rodar."
        )

    setup_dirs()
    logger.info(f"Max amostras por classe: {args.max_samples}")

    # PT-BR
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

    if args.common_voice_pt:
        download_common_voice_pt(args.max_samples)

    # Internacional
    if args.all_intl or args.asvspoof2019:
        download_asvspoof2019(args.max_samples)

    if args.all_intl or args.wavefake:
        download_wavefake(args.max_samples)

    if args.all_intl or args.in_the_wild:
        download_in_the_wild(args.max_samples)

    if args.asvspoof5:
        download_asvspoof5(args.max_samples)

    print_report()
    logger.info("Download concluído!")


if __name__ == "__main__":
    main()
