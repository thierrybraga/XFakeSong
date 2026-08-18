#!/usr/bin/env python3
"""Baixa o CETUC pareado por locutor com o Fake Voices (classe real do dataset).

Contexto (2026-07-25). O BRSpeech-DF nao publica identidade de locutor em
nenhuma das suas tres configs (`audio`, `label`, `model`), o que tornava
impossivel verificar disjuncao de falante: o pipeline caia no fallback
`sample:<filename>` e cada amostra virava um grupo singleton, satisfazendo
"zero falante compartilhado" de forma vacua.

A saida esta documentada no card do PortuFake: os deepfakes do
`unfake/fake_voices` foram gerados A PARTIR de gravacoes do **CETUC**. Verificado:
os 56 codigos de locutor do Fake Voices (`F049`, `M001`, ...) casam com os nomes
do `falabrasil/cetuc` (`CarolinaMagalhaes_F050` <-> `CarolinaMagalhaes_F050_Fake`).

Isso da o desenho correto:

- os MESMOS locutores aparecem nas duas classes, logo o locutor nao prediz a
  classe e nao existe atalho de falante;
- a identidade e PUBLICADA, entao a disjuncao e verificavel em vez de inferida
  (dispensa por completo a inferencia por embeddings, cujo risco residual media
  26,7% de pares mesmo-falante nao detectaveis);
- o CETUC ja vem particionado **speaker-disjoint pelos autores** (train 81 /
  dev 10 / test 10 locutores, um tar.gz por locutor em exatamente uma
  particao) — usar a particao oficial e mais defensavel que um split proprio;
- cada .wav vem com o .txt da transcricao, o que permite agrupamento por
  conteudo e torna conhecivel o pareamento de texto entre bonafide e spoof.

Uso:
    python scripts/dataset/download_cetuc_paired.py --plan
    python scripts/dataset/download_cetuc_paired.py --download --per-speaker 150
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("cetuc_paired")

REPO = "falabrasil/cetuc"
DATASETS_DIR = ROOT / "data" / "datasets"
CETUC_DIR = DATASETS_DIR / "cetuc"
CACHE_DIR = DATASETS_DIR / "raw" / "cetuc_cache"
MANIFEST_PATH = DATASETS_DIR / "metadata" / "speaker_manifest.json"

SPEAKER_CODE = re.compile(r"([FM]\d{3})")


def _fkvoice_codes() -> set[str]:
    """Codigos de locutor presentes na classe fake, lidos do manifesto."""
    if not MANIFEST_PATH.exists():
        return set()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    codes: set[str] = set()
    for entry in manifest.values():
        if entry.get("source") != "fkvoice":
            continue
        speaker = str(entry.get("speaker_id") or "")
        found = SPEAKER_CODE.search(speaker)
        if found:
            codes.add(found.group(1))
    return codes


def _list_speakers() -> list[dict]:
    """tar.gz do CETUC com particao oficial e codigo de locutor."""
    from huggingface_hub import HfApi

    files = [
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".tar.gz")
    ]
    speakers: list[dict] = []
    for path in sorted(files):
        parts = path.split("/")
        if len(parts) < 3:
            continue
        split = parts[1]
        name = parts[2]
        found = SPEAKER_CODE.search(name)
        if not found:
            continue
        speakers.append(
            {
                "file": path,
                "split": split,
                "speaker_name": name,
                "code": found.group(1),
            }
        )
    return speakers


def _plan(paired_only: bool) -> list[dict]:
    fake_codes = _fkvoice_codes()
    speakers = _list_speakers()
    for item in speakers:
        item["has_fake"] = item["code"] in fake_codes
    selected = [s for s in speakers if s["has_fake"]] if paired_only else speakers

    from collections import Counter

    print(f"\nCETUC: {len(speakers)} locutores | Fake Voices: {len(fake_codes)} codigos")
    print(f"pareados (presentes nas duas classes): {sum(s['has_fake'] for s in speakers)}")
    print("\n=== PARTICAO OFICIAL (speaker-disjoint pelos autores) ===")
    total = Counter(s["split"] for s in speakers)
    paired = Counter(s["split"] for s in speakers if s["has_fake"])
    print(f"{'split':>8}{'locutores':>12}{'pareados':>11}")
    for split in ("train", "dev", "test"):
        print(f"{split:>8}{total.get(split, 0):>12}{paired.get(split, 0):>11}")
    print(f"\nselecionados para download: {len(selected)}")
    sem_fake = [s["code"] for s in speakers if not s["has_fake"]]
    if sem_fake:
        print(f"sem contraparte fake ({len(sem_fake)}): {sorted(sem_fake)[:12]}...")
    return selected


def _download(selected: list[dict], per_speaker: int, prune: bool) -> None:
    from huggingface_hub import hf_hub_download

    from scripts.dataset.download_datasets import (
        process_audio,
        record_sample_metadata_safe,
        safe_write_wav,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    written_total = 0

    for pos, item in enumerate(selected, start=1):
        out_dir = CETUC_DIR / item["split"] / item["code"]
        existing = len(list(out_dir.glob("*.wav"))) if out_dir.exists() else 0
        if existing >= per_speaker:
            logger.info(
                "[%d/%d] %s ja tem %d wavs — pulando",
                pos,
                len(selected),
                item["code"],
                existing,
            )
            continue

        logger.info(
            "[%d/%d] %s (%s) split=%s",
            pos,
            len(selected),
            item["code"],
            item["speaker_name"],
            item["split"],
        )
        local = None
        try:
            local = hf_hub_download(
                REPO, item["file"], repo_type="dataset", cache_dir=str(CACHE_DIR)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            with tarfile.open(local) as archive:
                members = sorted(
                    (
                        m
                        for m in archive.getmembers()
                        # `._*` sao resource forks do macOS, nao audio.
                        if m.name.endswith(".wav")
                        and not Path(m.name).name.startswith("._")
                    ),
                    key=lambda m: m.name,
                )
                for member in members:
                    if written >= per_speaker:
                        break
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    stem = Path(member.name).stem

                    # Transcricao irma: vira `text_id` e permite agrupamento por
                    # conteudo, alem de tornar conhecivel o pareamento de texto
                    # entre a gravacao real e o clone sintetico.
                    transcript = None
                    try:
                        txt = archive.extractfile(
                            member.name[: -len(".wav")] + ".txt"
                        )
                        if txt is not None:
                            transcript = (
                                txt.read().decode("utf-8", errors="replace").strip()
                            )
                    except KeyError:
                        pass

                    audio, ok = _decode(handle.read())
                    if not ok:
                        continue
                    audio, ok = process_audio(audio, 16_000)
                    if not ok:
                        continue
                    target = out_dir / f"cetuc_{item['code']}_{stem}.wav"
                    if safe_write_wav(target, audio):
                        record_sample_metadata_safe(
                            target,
                            source="cetuc",
                            speaker_id=item["speaker_name"],
                            utterance_id=stem,
                            text_id=transcript,
                            generator_id="bonafide",
                            label=0,
                        )
                        written += 1
            written_total += written
            logger.info("    %d wavs escritos (total %d)", written, written_total)
        except Exception as exc:
            logger.warning("    falhou %s: %s", item["code"], exc)
        finally:
            # Cada tar.gz pesa ~138 MB e os 56 pareados somariam ~7,7 GB de
            # cache inutil depois da extracao.
            if local and prune:
                try:
                    os.remove(local)
                except OSError:
                    pass

    logger.info("CETUC: %d wavs no total", written_total)


def _decode(raw: bytes) -> tuple[object, bool]:
    """Decodifica WAV em memoria para float32 mono."""
    import io

    import numpy as np
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:
        return None, False
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        target = int(round(len(audio) * 16_000 / max(sr, 1)))
        if target <= 0:
            return None, False
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target),
            np.arange(len(audio)),
            audio.astype("float64"),
        ).astype("float32")
    return audio, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Relata e sai.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--per-speaker", type=int, default=150)
    parser.add_argument(
        "--all-speakers",
        action="store_true",
        help="Inclui locutores SEM contraparte fake (rompe o pareamento e "
        "reintroduz atalho de fonte; use apenas para analise).",
    )
    parser.add_argument("--no-prune", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    selected = _plan(paired_only=not args.all_speakers)
    if args.plan or not args.download:
        return 0
    _download(selected, args.per_speaker, prune=not args.no_prune)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
