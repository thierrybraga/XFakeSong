#!/usr/bin/env python3
"""Constroi o corpus pareado PT-BR: CETUC (bonafide) x Fake Voices/XTTS (spoof).

Garantia anti-vazamento so vale quando e *verificavel*. Um corpus que nao publica
identidade de falante nem de texto obriga a inferir os grupos do split — e a
inferencia tem limite de deteccao: sempre resta uma fracao de pares mesmo-falante
que nenhum limiar separa sem colapsar o corpus.

Este builder usa um par de fontes em que as duas identidades sao **publicadas**:

- `falabrasil/cetuc` — 101 locutores, cada um lendo as MESMAS 1000 frases
  foneticamente balanceadas, com a transcricao ao lado de cada WAV (16 kHz,
  PCM_16). Verificado: o texto do indice `i` e identico entre locutores.
- `unfake/fake_voices` — 56 desses locutores clonados com XTTS-v2 a partir das
  gravacoes do CETUC, 1000 clones por locutor (24 kHz, FLOAT).

O alinhamento `N_fake.wav` -> CETUC `{N-1:04d}` foi verificado
estatisticamente: a correlacao entre o tamanho do texto e a duracao do clone e
0,675 nesse offset contra 0,001 e -0,032 nos offsets vizinhos.

Disso decorre o desenho 2x2 completo (locutor x frase x classe):

- o MESMO locutor aparece nas duas classes -> locutor nao prediz a classe;
- a MESMA frase aparece nas duas classes -> texto nao prediz a classe;
- as duas identidades sao publicadas -> a disjuncao e *verificavel*, nao inferida;
- o par so entra no corpus quando as DUAS amostras passam na validacao, o que
  torna o balanceamento de classe exato por construcao, sem cotas.

Uso:
    python scripts/dataset/build_paired_pt_corpus.py --plan
    python scripts/dataset/build_paired_pt_corpus.py --build
    python scripts/dataset/build_paired_pt_corpus.py --build --speakers 2 --per-speaker 25
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset.download_datasets import (  # noqa: E402
    TARGET_SR,
    download_workers,
    process_audio,
    safe_write_wav,
)

logger = logging.getLogger("paired_pt_corpus")

# Revisoes fixadas: a aquisicao tem de ser reproduzivel byte a byte. O builder
# avisa (sem falhar) se o repositorio avancou desde a fixacao.
REPO_REAL = "falabrasil/cetuc"
REV_REAL = "6dbc8081d35fff5bdd133d8a12b8a11f91959b01"
REPO_FAKE = "unfake/fake_voices"
REV_FAKE = "541bf396da524f92a6d6594a0e9952210e7d7e7e"

# Prefixo UNICO para as duas classes. O exportador de NPZ deriva `source_ids` do
# prefixo do nome quando o manifesto nao diz outra coisa, e um prefixo por classe
# (`cetuc_` real / `xtts_` fake) daria um oraculo de fonte de 100%: bastaria ler o
# nome do arquivo para acertar o rotulo. Aqui a fonte e a mesma nas duas classes
# porque o material de fala e o mesmo: mesmos locutores, mesmas frases.
PREFIX = "ptpair"
SOURCE_ID = "ptpair"

# Nivel canonico das duas classes (ver `_normalize_loudness`).
TARGET_RMS_DB = -26.0
PEAK_CEILING_DB = -1.0

CORPUS_DIR = ROOT / "data" / "datasets" / "corpus"
CACHE_DIR = ROOT / "data" / "datasets" / "raw" / "corpus_cache"
MANIFEST_PATH = CORPUS_DIR / "manifest.jsonl"
SENTENCES_PATH = CORPUS_DIR / "sentences.json"
STATE_PATH = CORPUS_DIR / "state.json"
ACQUISITION_PATH = CORPUS_DIR / "acquisition.json"

SPEAKER_CODE = re.compile(r"([FM]\d{3})")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Identidade de texto
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Forma canonica para identidade de conteudo textual.

    Mantem os acentos (sao fonemicamente relevantes em portugues) e descarta
    caixa, pontuacao e espacos redundantes.
    """
    norm = unicodedata.normalize("NFC", text).strip().lower()
    norm = _PUNCT.sub(" ", norm)
    return _SPACES.sub(" ", norm).strip()


def text_id_for(text: str) -> str:
    """`text_id` derivado do proprio texto, nao do indice do arquivo.

    Usar o hash do texto em vez de `s0000` faz o agrupamento por conteudo
    continuar correto mesmo se algum locutor divergir da frase canonica daquele
    indice — a divergencia recebe outro grupo em vez de contaminar o split.
    """
    # `usedforsecurity=False`: este digest é um IDENTIFICADOR de conteúdo, não
    # uma credencial nem prova de integridade — agrupa as gravações da MESMA
    # frase para que o split não parta um grupo ao meio. Não há adversário a
    # resistir: um SHA-1 forjado só produziria um agrupamento errado no nosso
    # próprio corpus. Sem a flag, o bandit marca B324 (CWE-327) com severidade
    # HIGH e bloqueia a CI, que trata HIGH como falha.
    digest = hashlib.sha1(
        normalize_text(text).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"t{digest[:12]}"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Descoberta e pareamento de locutores
# ---------------------------------------------------------------------------


def _list_paired_speakers() -> tuple[list[dict], dict]:
    """Locutores presentes NAS DUAS fontes, com a particao oficial do CETUC."""
    from huggingface_hub import HfApi

    api = HfApi()
    real_files = api.list_repo_files(REPO_REAL, repo_type="dataset", revision=REV_REAL)
    fake_files = api.list_repo_files(REPO_FAKE, repo_type="dataset", revision=REV_FAKE)

    real_by_code: dict[str, dict] = {}
    for path in sorted(real_files):
        if not path.endswith(".tar.gz"):
            continue
        parts = path.split("/")
        if len(parts) < 3:
            continue
        found = SPEAKER_CODE.search(parts[2])
        if found:
            real_by_code[found.group(1)] = {
                "real_file": path,
                "official_split": parts[1],
                "speaker_name": parts[2],
            }

    fake_by_code: dict[str, str] = {}
    for path in sorted(fake_files):
        if not path.endswith(".zip"):
            continue
        found = SPEAKER_CODE.search(Path(path).name)
        if found:
            fake_by_code[found.group(1)] = path

    paired = []
    for code in sorted(set(real_by_code) & set(fake_by_code)):
        item = dict(real_by_code[code])
        item["code"] = code
        item["fake_file"] = fake_by_code[code]
        paired.append(item)

    coverage = {
        "cetuc_speakers": len(real_by_code),
        "fake_voices_speakers": len(fake_by_code),
        "paired": len(paired),
        "real_only": sorted(set(real_by_code) - set(fake_by_code)),
        "fake_only": sorted(set(fake_by_code) - set(real_by_code)),
    }
    return paired, coverage


def _live_revision(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return str(HfApi().dataset_info(repo).sha)
    except Exception as exc:  # noqa: BLE001
        logger.debug("revisao viva indisponivel para %s: %s", repo, exc)
        return None


# ---------------------------------------------------------------------------
# Download com poda de cache
# ---------------------------------------------------------------------------


def _download(repo: str, filename: str, revision: str, attempts: int = 5) -> str:
    """Baixa com retry.

    `HF_HUB_DOWNLOAD_TIMEOUT` importa porque sem ele um stall de HTTP fica
    pendurado indefinidamente e o retry nunca dispara. Os pacotes aqui tem
    centenas de MB e a taxa observada varia de 2 a 30 MB/s, entao o timeout e por
    leitura sem progresso — nao um prazo para o download inteiro.
    """
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(CACHE_DIR),
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = min(60, 2**attempt)
            logger.warning(
                "download falhou (%d/%d) %s: %s — nova tentativa em %ds",
                attempt,
                attempts,
                filename,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"download esgotou as tentativas: {filename}") from last


def _prune_cache() -> None:
    """Remove pacotes ja extraidos.

    Sem symlink no Windows o cache do Hub guarda DUAS copias de cada pacote (o
    blob e o arquivo do snapshot). Um zip do Fake Voices tem ~400 MB, entao os
    56 locutores somariam ~40 GB de cache inutil depois da extracao.
    """
    if not CACHE_DIR.exists():
        return
    for item in CACHE_DIR.rglob("*"):
        if not item.is_file():
            continue
        if item.suffix in {".zip", ".gz"} or "blobs" in item.parts:
            try:
                item.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Extracao e canonicalizacao
# ---------------------------------------------------------------------------


def _normalize_loudness(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    """Iguala o nivel das duas classes; devolve o ganho aplicado.

    MEDIDO e por isso obrigatorio: sem esta etapa, o RMS separa as classes com
    **AUC 0,9926** — os clones saem do XTTS com nivel praticamente constante
    (rms -17,24 dB, desvio 0,78 dB) contra a variacao natural do CETUC
    (-27,44 dB, desvio 4,85 dB). Um detector acertaria quase tudo medindo
    volume, sem detectar sintese alguma. Depois da normalizacao o RMS cai para
    AUC 0,55.

    A politica anterior do projeto ("preserve loudness, so atenua clipping") era
    correta para fontes de um mesmo canal; entre uma gravacao de estudio e a
    saida de um TTS que normaliza o proprio nivel, ela vira atalho.

    O teto de pico existe apenas para nao estourar o PCM16 e raramente atua: nos
    dois lados a mediana do pico normalizado fica ~9 dB abaixo do teto.
    """
    # Delega para a FONTE ÚNICA da política de nível
    # (`benchmark_frontend.normalize_corpus_level`), para que a inferência em
    # produção possa aplicar exatamente a mesma coisa. Enquanto a lógica vivia
    # só aqui, o app normalizava a −23 LUFS e o vetor tabular do SVM/RF chegava
    # 1,41x fora do treino.
    from app.domain.features.benchmark_frontend import normalize_corpus_level

    return normalize_corpus_level(
        audio, target_rms_dbfs=TARGET_RMS_DB, peak_ceiling_dbfs=PEAK_CEILING_DB
    )


def extract_window(audio: np.ndarray, samples: int) -> tuple[np.ndarray, int, int]:
    """Recorte central da janela exportada, com o nivel normalizado NA JANELA.

    Normalizar o arquivo inteiro **nao** normaliza a janela. Medido: com todos os
    arquivos em exatamente -26 dBFS, o RMS dos 3 s centrais ainda separava as
    classes com **AUC 0,7097** (-25,38 dB no bonafide contra -26,10 no clone). A
    causa e que a gravacao do CETUC e mais longa e carrega mais silencio nas
    pontas, entao seu miolo e mais denso em fala do que a media do arquivo; o
    clone, mais curto e uniforme, tem miolo parecido com a media.

    O modelo recebe a janela, entao e a janela que precisa estar nivelada. Esta
    funcao e a unica fonte da politica: o exportador a usa para gerar o NPZ e a
    auditoria a usa para medir os descritores, de modo que a auditoria descreve
    exatamente o que o modelo recebe.

    CORRECAO DE FORMATO (2026-08-19), aplicada AQUI e nesta ordem:

      1. passa-baixas a 7,5 kHz
      2. remocao de DC
      3. normalizacao de nivel (ja existente)

    As duas primeiras sao novas e existem porque as classes vinham com
    assinaturas de PROCESSAMENTO, nao de sintese:

    - BANDA. A bonafide vem do CETUC em 16 kHz nativo; a spoof, do fake_voices
      em 24 kHz reamostrado para 16 kHz. A taxa de origem prediz a classe com
      100% de acuracia (49.264 arquivos de cada lado, sem excecao), e o filtro
      anti-aliasing deixa rastro: energia relativa em 7,9-8,0 kHz de 4,3e-06 na
      bonafide contra 4,3e-10 na spoof. Uma regressao logistica sobre 40
      energias de banda alcancava **AUC 1,0000**; com o corte, 0,84.

    - DC. Conversores AD introduzem offset na gravacao; vocoders neurais saem
      com media ~zero. Medido: 2,04e-03 contra 1,38e-04, razao de 14,7x, o que
      separava as classes com AUC 0,7387.

    A ORDEM E O PONTO. `_normalize_loudness` calcula o RMS INCLUINDO o DC. Com
    a bonafide carregando 14,7x mais offset, normalizar antes de remove-lo
    deixa o RMS desigual assim que o DC sai -- medido: o atalho de RMS sobe de
    0,5126 para 0,7503. Corrigir a banda e o DC ANTES faz a normalizacao operar
    sobre o sinal ja limpo, e o RMS volta ao acaso.

    O que sobrevive de proposito: fator de crista (AUC 0,697) e curtose
    (0,650), candidatos legitimos a artefato de sintese -- a compressao de
    faixa dinamica que os clones exibem e que o `applied_gain_db` confirma
    (bonafide +3,3 dB com desvio 6,6; spoof -8,7 dB com desvio 0,8).

    Devolve (janela, duracao original em amostras, inicio do recorte).
    """
    from app.domain.features.benchmark_frontend import apply_band_correction

    audio = np.asarray(audio, dtype="float32")
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    original = int(len(audio))
    if original == 0:
        raise ValueError("audio vazio")
    if original >= samples:
        start = (original - samples) // 2
        window = audio[start : start + samples]
    else:
        # A particao descarta os pares que nao alcancam a janela, entao isto so
        # ocorre se a janela pedida for maior que a garantida — o exportador
        # recusa esse caso antes de chegar aqui.
        start = 0
        repeats = int(np.ceil(samples / original))
        window = np.tile(audio, repeats)[:samples]
    # Banda e DC ANTES do nivel -- ver a justificativa da ordem no docstring.
    # `renormalize_rms_dbfs=None` porque quem normaliza e a linha seguinte,
    # que ja aplica a politica do corpus e registra o ganho no manifesto.
    window = apply_band_correction(
        window[None, :], renormalize_rms_dbfs=None
    )[0]
    normalized, _ = _normalize_loudness(window)
    return normalized, original, start


def _canonicalize(raw: bytes) -> tuple[np.ndarray | None, dict]:
    """bytes WAV -> mono float32 16 kHz normalizado, com metadado da origem."""
    try:
        with sf.SoundFile(io.BytesIO(raw)) as handle:
            info = {
                "source_sample_rate": int(handle.samplerate),
                "source_subtype": str(handle.subtype),
                "source_channels": int(handle.channels),
                "source_duration_sec": round(len(handle) / handle.samplerate, 4),
            }
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("decode falhou: %s", exc)
        return None, {}
    audio, ok = process_audio(data, sr)
    if not ok:
        return None, info
    audio, loudness = _normalize_loudness(audio)
    info.update(loudness)
    return audio, info


def _write_side(
    jobs: Iterable[tuple[int, bytes, Path]],
    workers: int,
    batch: int = 128,
) -> tuple[dict[int, dict], int]:
    """Canonicaliza e grava um lado do par; devolve ({indice: metadado}, vistos).

    Consome em lotes: `Executor.map` submeteria os 1000 enunciados de uma vez e
    manteria todos os bytes brutos em memoria ao mesmo tempo.
    """
    written: dict[int, dict] = {}
    seen = 0

    def work(job: tuple[int, bytes, Path]) -> tuple[int, Path, dict] | None:
        index, raw, target = job
        audio, info = _canonicalize(raw)
        if audio is None:
            return None
        if not safe_write_wav(target, audio, TARGET_SR):
            return None
        info["duration_sec"] = round(len(audio) / TARGET_SR, 4)
        return index, target, info

    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            chunk = []
            for job in iterator:
                chunk.append(job)
                if len(chunk) >= batch:
                    break
            if not chunk:
                break
            seen += len(chunk)
            for result in pool.map(work, chunk):
                if result is None:
                    continue
                index, target, info = result
                info["path"] = target
                written[index] = info
    return written, seen


def _real_jobs(
    tar_path: str, code: str, out_dir: Path, per_speaker: int
) -> tuple[Iterator[tuple[int, bytes, Path]], dict[int, str]]:
    """Le o tar.gz do CETUC: WAVs e as transcricoes irmas."""
    archive = tarfile.open(tar_path)
    transcripts: dict[int, str] = {}
    wav_members = []
    for member in archive.getmembers():
        name = Path(member.name)
        if name.name.startswith("._") or not member.isfile():
            continue  # `._*` sao resource forks do macOS, nao audio
        try:
            index = int(name.stem.split("-")[-1])
        except ValueError:
            continue
        if name.suffix == ".wav":
            wav_members.append((index, member))
        elif name.suffix == ".txt":
            handle = archive.extractfile(member)
            if handle is not None:
                transcripts[index] = (
                    handle.read().decode("utf-8", errors="replace").strip()
                )

    wav_members.sort(key=lambda item: item[0])
    if per_speaker:
        wav_members = wav_members[:per_speaker]

    def gen() -> Iterator[tuple[int, bytes, Path]]:
        try:
            for index, member in wav_members:
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                name = f"{PREFIX}_{code}_{index:04d}_bonafide.wav"
                yield index, handle.read(), out_dir / name
        finally:
            archive.close()

    return gen(), transcripts


def _fake_jobs(
    zip_path: str, code: str, out_dir: Path, keep: set[int]
) -> Iterator[tuple[int, bytes, Path]]:
    """Le o zip do Fake Voices, mapeando `N_fake.wav` -> indice CETUC `N-1`."""
    archive = zipfile.ZipFile(zip_path)
    entries = []
    for info in archive.infolist():
        name = Path(info.filename)
        if name.suffix.lower() != ".wav" or name.name.startswith("._"):
            continue
        try:
            index = int(name.stem.split("_")[0]) - 1
        except ValueError:
            continue
        if index in keep:
            entries.append((index, info.filename))
    entries.sort(key=lambda item: item[0])

    def gen() -> Iterator[tuple[int, bytes, Path]]:
        try:
            for index, member in entries:
                yield index, archive.read(
                    member
                ), out_dir / f"{PREFIX}_{code}_{index:04d}_xttsv2.wav"
        finally:
            archive.close()

    return gen()


# ---------------------------------------------------------------------------
# Estado (resume)
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("estado corrompido em %s — recomecando", STATE_PATH)
    return {"done": {}, "sentences": {}, "disagreements": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


# ---------------------------------------------------------------------------
# Construcao
# ---------------------------------------------------------------------------


def _build_speaker(item: dict, per_speaker: int, workers: int, state: dict) -> dict:
    """Baixa, canonicaliza e grava o par completo de um locutor."""
    code = item["code"]
    real_dir = CORPUS_DIR / "real" / code
    fake_dir = CORPUS_DIR / "fake" / code
    # Um locutor so entra no estado depois de concluido, entao chegar aqui com
    # arquivos em disco significa que a execucao anterior foi interrompida no meio
    # dele. Os restos nao aparecem nos dicionarios desta execucao e escapariam da
    # poda de pares, virando amostras sem par no corpus.
    for directory in (real_dir, fake_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    tar_path = _download(REPO_REAL, item["real_file"], REV_REAL)
    jobs, transcripts = _real_jobs(tar_path, code, real_dir, per_speaker)
    real_written, real_seen = _write_side(jobs, workers)
    logger.info(
        "    real: %d/%d validos | %d transcricoes",
        len(real_written),
        real_seen,
        len(transcripts),
    )

    # Sem transcricao nao existe `text_id` publicado: a amostra fica fora, em vez
    # de entrar com identidade de conteudo inventada.
    keep = {index for index in real_written if index in transcripts}

    zip_path = _download(REPO_FAKE, item["fake_file"], REV_FAKE)
    fake_written, fake_seen = _write_side(
        _fake_jobs(zip_path, code, fake_dir, keep), workers
    )
    logger.info("    fake: %d/%d validos", len(fake_written), fake_seen)

    # Pareamento estrito: so o indice presente e valido nas DUAS classes entra.
    common = sorted(set(real_written) & set(fake_written))
    for index, info in list(real_written.items()) + list(fake_written.items()):
        if index not in common:
            try:
                Path(info["path"]).unlink()
            except OSError:
                pass

    # Verificacao (nao suposicao) de que a frase do indice e a mesma entre
    # locutores. Divergencias nao sao silenciadas: o `text_id` vem do texto.
    canonical: dict[str, str] = state["sentences"]
    disagreements: list = state["disagreements"]
    records = []
    for index in common:
        text = transcripts[index]
        key = f"{index:04d}"
        norm = normalize_text(text)
        if key not in canonical:
            canonical[key] = text
        elif normalize_text(canonical[key]) != norm:
            disagreements.append({"index": key, "speaker": code, "text": text})

        tid = text_id_for(text)
        for side, written, generator, repo, revision in (
            ("real", real_written, "bonafide", REPO_REAL, REV_REAL),
            ("fake", fake_written, "xtts_v2", REPO_FAKE, REV_FAKE),
        ):
            info = written[index]
            path = Path(info["path"])
            records.append(
                {
                    "path": path.relative_to(CORPUS_DIR).as_posix(),
                    "class": side,
                    "label": 0 if side == "real" else 1,
                    "source": SOURCE_ID,
                    "speaker_id": code,
                    "speaker_name": item["speaker_name"],
                    "cetuc_official_split": item["official_split"],
                    "utterance_id": f"{code}-{index:04d}",
                    "sentence_index": key,
                    "text_id": tid,
                    "generator_id": generator,
                    "acquisition_repo": repo,
                    "acquisition_revision": revision,
                    "sample_rate": TARGET_SR,
                    "duration_sec": info["duration_sec"],
                    "source_sample_rate": info.get("source_sample_rate"),
                    "source_subtype": info.get("source_subtype"),
                    "source_duration_sec": info.get("source_duration_sec"),
                    "source_rms_db": info.get("source_rms_db"),
                    "source_peak_db": info.get("source_peak_db"),
                    "applied_gain_db": info.get("applied_gain_db"),
                    "peak_limited": info.get("peak_limited"),
                    "content_sha256": _sha256(path),
                }
            )

    with MANIFEST_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _prune_cache()
    return {
        "pairs": len(common),
        "real_valid": len(real_written),
        "fake_valid": len(fake_written),
        "transcripts": len(transcripts),
        "dropped_unpaired": len(set(real_written) ^ set(fake_written)),
    }


def _trim_manifest(done: set[str]) -> None:
    """Descarta linhas de locutores que nao chegaram ao estado de concluido.

    O manifesto e gravado antes de `_save_state`; uma interrupcao nessa janela
    deixaria linhas de um locutor que sera reconstruido, duplicando-as.
    """
    if not MANIFEST_PATH.exists():
        return
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    kept = [line for line in lines if json.loads(line)["speaker_id"] in done]
    if len(kept) == len(lines):
        return
    logger.info("manifesto: descartando %d linhas orfas", len(lines) - len(kept))
    MANIFEST_PATH.write_text("".join(kept), encoding="utf-8")


def build(per_speaker: int, max_speakers: int, workers: int) -> None:
    paired, coverage = _list_paired_speakers()
    if max_speakers:
        paired = paired[:max_speakers]

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    _trim_manifest(set(state["done"]))
    started = time.time()

    for position, item in enumerate(paired, start=1):
        code = item["code"]
        if code in state["done"]:
            logger.info(
                "[%d/%d] %s ja concluido — pulando", position, len(paired), code
            )
            continue
        logger.info(
            "[%d/%d] %s (%s) particao oficial=%s",
            position,
            len(paired),
            code,
            item["speaker_name"],
            item["official_split"],
        )
        try:
            result = _build_speaker(item, per_speaker, workers, state)
        except Exception as exc:  # noqa: BLE001
            logger.error("    %s falhou: %s", code, exc)
            _prune_cache()
            continue
        state["done"][code] = result
        _save_state(state)
        total = sum(entry["pairs"] for entry in state["done"].values())
        elapsed = time.time() - started
        logger.info(
            "    %d pares | corpus: %d pares (%d amostras) | %.1f min",
            result["pairs"],
            total,
            total * 2,
            elapsed / 60.0,
        )

    SENTENCES_PATH.write_text(
        json.dumps(
            {
                "count": len(state["sentences"]),
                "sentences": state["sentences"],
                "disagreements": state["disagreements"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pairs = sum(entry["pairs"] for entry in state["done"].values())
    ACQUISITION_PATH.write_text(
        json.dumps(
            {
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "real": {
                    "repo": REPO_REAL,
                    "pinned_revision": REV_REAL,
                    "live_revision": _live_revision(REPO_REAL),
                    "generator_id": "bonafide",
                },
                "fake": {
                    "repo": REPO_FAKE,
                    "pinned_revision": REV_FAKE,
                    "live_revision": _live_revision(REPO_FAKE),
                    "generator_id": "xtts_v2",
                },
                "alignment": "fake `N_fake.wav` -> cetuc index N-1 (verificado)",
                "canonical_audio": {
                    "sample_rate": TARGET_SR,
                    "channels": 1,
                    "subtype": "PCM_16",
                    "resampler": "soxr_hq",
                    "min_duration_sec": 1.0,
                    "max_duration_sec": 30.0,
                    "amplitude_policy": (
                        f"RMS normalizado para {TARGET_RMS_DB} dBFS nas duas "
                        f"classes, teto de pico {PEAK_CEILING_DB} dBFS; ganho "
                        f"registrado por amostra em applied_gain_db"
                    ),
                },
                "speaker_coverage": coverage,
                "speakers_built": state["done"],
                "sentences": len(state["sentences"]),
                "sentence_disagreements": len(state["disagreements"]),
                "pairs": pairs,
                "samples": pairs * 2,
                "per_speaker_cap": per_speaker or "sem limite",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "corpus pronto: %d pares = %d amostras (%d reais + %d falsas) em %s",
        pairs,
        pairs * 2,
        pairs,
        pairs,
        CORPUS_DIR,
    )


def plan() -> None:
    paired, coverage = _list_paired_speakers()
    from collections import Counter

    print(f"\n{REPO_REAL} @ {REV_REAL[:12]} -> {coverage['cetuc_speakers']} locutores")
    print(
        f"{REPO_FAKE} @ {REV_FAKE[:12]} -> {coverage['fake_voices_speakers']} locutores"
    )
    for repo, pinned in ((REPO_REAL, REV_REAL), (REPO_FAKE, REV_FAKE)):
        live = _live_revision(repo)
        if live and live != pinned:
            print(f"  AVISO: {repo} avancou para {live[:12]} desde a fixacao")
    print(f"\nlocutores PAREADOS (nas duas classes): {coverage['paired']}")
    print(
        f"so real (sem clone): {len(coverage['real_only'])} -> {coverage['real_only']}"
    )

    counts = Counter(item["official_split"] for item in paired)
    print("\nparticao oficial do CETUC entre os pareados:")
    for split in ("train", "dev", "test"):
        print(f"  {split:>5}: {counts.get(split, 0):>3} locutores")
    print(
        f"\nprojecao: {coverage['paired']} locutores x 1000 frases x 2 classes = "
        f"{coverage['paired'] * 2000} amostras"
    )
    print(
        f"download: ~{coverage['paired'] * 0.48:.1f} GB (poda o cache a cada locutor)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Relata a cobertura e sai.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--per-speaker",
        type=int,
        default=0,
        help="Limite de enunciados por locutor (0 = todos os 1000).",
    )
    parser.add_argument(
        "--speakers", type=int, default=0, help="Limite de locutores (0 = todos)."
    )
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if args.plan or not args.build:
        plan()
        return 0
    build(
        per_speaker=args.per_speaker,
        max_speakers=args.speakers,
        workers=args.workers or download_workers(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
