"""Manifesto de falantes do dataset (identificacao de falante, aditiva).

Os nomes de arquivo do XFakeSong seguem `<fonte>_NNNNN.wav` e NAO embutem o
falante. Este modulo mantem um sidecar `data/datasets/speaker_manifest.json` que
mapeia o nome do arquivo -> falante, quando a fonte expoe esse identificador
(Fake Voices por ZIP, Common Voice por `client_id`, In-the-Wild por celebridade,
ASVspoof pelo `speaker` do protocolo). Para fontes que nao expoem falante, o
agrupamento cai para o nivel de FONTE (`<prefixo>`), o grupo mais fino disponivel.

Usado por todos os tiers para preencher `speaker_ids` no `.npz` quando houver ID
real de falante. O tier `large` tambem usa esse manifesto para split DISJUNTO
POR FALANTE (usuarios nao vistos) e pelo benchmark
(`--speaker-split`/`--unseen-speaker`).

Importante: este modulo NAO renomeia arquivos — e puramente aditivo, entao todos
os parsers de prefixo existentes (catalogo, auditoria, benchmark) permanecem
intactos.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("speaker_manifest")

# parents[3] = raiz do repositório (app/domain/dataset_metadata/ → 3 níveis).
# BUG pré-existente descoberto na consolidação de 2026-07-14: o antigo
# `parent.parent.parent / "app" / "datasets"` resolvia para app/app/datasets
# (caminho inexistente) — o sidecar era gravado num diretório fantasma.
BASE_DIR = Path(__file__).resolve().parents[3]

DATASETS_DIR = BASE_DIR / "data" / "datasets"
SPEAKER_MANIFEST_PATH = DATASETS_DIR / "speaker_manifest.json"

class MissingSampleMetadataError(ValueError):
    """Raised when a strict scientific protocol lacks required provenance."""


# Cache em memoria + lock; grava em disco de forma preguicosa (atexit + periodico)
# para nao sofrer I/O por arquivo durante downloads de dezenas de milhares de WAVs.
_LOCK = threading.RLock()
_CACHE: Optional[Dict[str, dict]] = None
_DIRTY = False
_FLUSH_EVERY = 1000
_PENDING = 0
_ATEXIT_REGISTERED = False


def _basename(path: str | Path) -> str:
    return Path(str(path).replace("\\", "/")).name


def _manifest_key(path: str | Path) -> str:
    """Keep real/fake homonyms separate while surviving split relocation."""
    normalized = str(path).replace("\\", "/")
    parts = [part.lower() for part in normalized.split("/") if part]
    basename = _basename(path)
    for label in ("real", "fake"):
        if label in parts:
            return f"{label}/{basename}"
    return basename


def _entry_for_path(path: str | Path) -> Dict[str, Any]:
    manifest = load_manifest()
    return manifest.get(_manifest_key(path)) or manifest.get(_basename(path)) or {}



def _infer_prefix(path: str | Path) -> str:
    """Prefixo de fonte (`<prefixo>` antes do primeiro `_`), em minusculas."""
    # import local para evitar ciclo de import com dataset_catalog
    try:
        from app.domain.dataset_metadata.dataset_catalog import infer_prefix_from_path

        return infer_prefix_from_path(path)
    except Exception:
        stem = Path(str(path).replace("\\", "/")).stem.lower()
        return stem.split("_", 1)[0] if stem else "unknown"


def load_manifest() -> Dict[str, dict]:
    """Carrega (e cacheia) o manifesto. Retorna {} se nao existir/corrompido."""
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        if SPEAKER_MANIFEST_PATH.exists():
            try:
                _CACHE = json.loads(SPEAKER_MANIFEST_PATH.read_text(encoding="utf-8"))
                if not isinstance(_CACHE, dict):
                    _CACHE = {}
            except Exception as exc:
                logger.warning("Falha ao ler %s: %s", SPEAKER_MANIFEST_PATH, exc)
                _CACHE = {}
        else:
            _CACHE = {}
        return _CACHE


def flush() -> None:
    """Persiste o manifesto em disco, se houver alteracoes pendentes."""
    global _DIRTY, _PENDING
    with _LOCK:
        if not _DIRTY or _CACHE is None:
            return
        try:
            SPEAKER_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            SPEAKER_MANIFEST_PATH.write_text(
                json.dumps(_CACHE, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _DIRTY = False
            _PENDING = 0
        except Exception as exc:
            logger.warning("Falha ao gravar %s: %s", SPEAKER_MANIFEST_PATH, exc)


def record_speaker(
    path: str | Path, speaker_id: Optional[str], source: Optional[str] = None
) -> None:
    """Registra o falante de um arquivo (best-effort, no-op se `speaker_id` vazio).

    Chave = nome do arquivo (robusto a mover entre `datasets/` e `splits/`).
    `source` default = prefixo inferido do nome.
    """
    global _DIRTY, _PENDING, _ATEXIT_REGISTERED
    sid = (str(speaker_id).strip() if speaker_id is not None else "")
    if not sid:
        return  # sem falante -> usa fallback por fonte na leitura
    key = _manifest_key(path)
    src = (source or _infer_prefix(path)).strip().lower()
    with _LOCK:
        manifest = load_manifest()
        manifest.setdefault(key, {}).update({"speaker_id": sid, "source": src})
        _DIRTY = True
        _PENDING += 1
        if not _ATEXIT_REGISTERED:
            atexit.register(flush)
            _ATEXIT_REGISTERED = True
        if _PENDING >= _FLUSH_EVERY:
            flush()


def record_sample_metadata(
    path: str | Path,
    *,
    speaker_id: Optional[str] = None,
    source: Optional[str] = None,
    utterance_id: Optional[str] = None,
    text_id: Optional[str] = None,
    generator_id: Optional[str] = None,
    vocoder_id: Optional[str] = None,
    codec: Optional[str] = None,
    channel: Optional[str] = None,
    source_revision: Optional[str] = None,
    label: Optional[int | str] = None,
) -> None:
    """Record hierarchical sample provenance without inventing missing IDs."""
    global _DIRTY, _PENDING, _ATEXIT_REGISTERED
    key = _manifest_key(path)
    src = (source or _infer_prefix(path)).strip().lower()
    values: Dict[str, Any] = {
        "source": src,
        "speaker_id": speaker_id,
        "utterance_id": utterance_id,
        "text_id": text_id,
        "generator_id": generator_id,
        "vocoder_id": vocoder_id,
        "codec": codec,
        "channel": channel,
        "source_revision": source_revision,
        "label": label,
    }
    clean = {
        key: (str(value).strip() if key != "label" else value)
        for key, value in values.items()
        if value is not None and str(value).strip()
    }
    with _LOCK:
        manifest = load_manifest()
        manifest.setdefault(key, {}).update(clean)
        _DIRTY = True
        _PENDING += 1
        if not _ATEXIT_REGISTERED:
            atexit.register(flush)
            _ATEXIT_REGISTERED = True
        if _PENDING >= _FLUSH_EVERY:
            flush()


def sample_metadata_for_path(path: str | Path) -> Dict[str, Any]:
    """Return explicit provenance plus source/status fields for one sample."""
    entry = dict(_entry_for_path(path))
    entry.setdefault("source", _infer_prefix(path))
    entry["speaker_known"] = bool(entry.get("speaker_id"))
    entry["utterance_known"] = bool(entry.get("utterance_id"))
    entry["text_known"] = bool(entry.get("text_id"))
    entry["generator_known"] = bool(entry.get("generator_id"))
    return entry


def metadata_id_for_path(
    path: str | Path,
    field: str,
    *,
    strict: bool = False,
) -> str:
    """Return a namespaced identity and fail when strict metadata is absent."""
    allowed = {
        "speaker_id", "utterance_id", "text_id", "generator_id",
        "vocoder_id", "source",
    }
    if field not in allowed:
        raise ValueError(f"Unsupported sample metadata field: {field}")
    meta = sample_metadata_for_path(path)
    value = meta.get(field)
    source = str(meta.get("source") or _infer_prefix(path)).lower()
    if value:
        return f"{source}:{value}" if field != "source" else str(value).lower()
    if strict:
        raise MissingSampleMetadataError(
            f"{_basename(path)} has no explicit {field}; "
            "the requested disjoint protocol cannot be guaranteed"
        )
    return f"unknown:{field}:{source}:{_basename(path)}"


def speaker_for_path(path: str | Path, *, strict: bool = False) -> str:
    """Chave de falante alinhavel a uma amostra.

    Retorna `<fonte>:<speaker_id>` quando conhecido; caso contrario, o nivel de
    FONTE (`<prefixo>`) — grupo mais fino disponivel para aquela amostra.
    """
    entry = _entry_for_path(path)
    if entry and entry.get("speaker_id"):
        src = entry.get("source") or _infer_prefix(path)
        return f"{src}:{entry['speaker_id']}"
    if strict:
        return metadata_id_for_path(path, "speaker_id", strict=True)
    # Compatibility only. Scientific protocols must call strict=True.
    return _infer_prefix(path)


def speaker_ids_for_paths(paths: Iterable[str | Path]) -> List[str]:
    """Lista de chaves de falante alinhada a `paths` (para o array `speaker_ids`)."""
    return [speaker_for_path(p) for p in paths]


def summarize_speakers(paths: Iterable[str | Path]) -> dict:
    """Resumo de cobertura de falantes para as amostras dadas."""
    paths = list(paths)
    speakers: set[str] = set()
    identified: set[str] = set()
    by_source: Dict[str, dict] = {}
    for p in paths:
        spk = speaker_for_path(p)
        src = _infer_prefix(p)
        speakers.add(spk)
        if ":" in spk:
            identified.add(spk)
        bucket = by_source.setdefault(src, {"speakers": set(), "files": 0})
        bucket["speakers"].add(spk)
        bucket["files"] += 1
    return {
        "total_files": len(paths),
        "total_speakers": len(speakers),
        "identified_speakers": len(identified),
        "manifest_entries": len(load_manifest()),
        "by_source": {
            src: {"speakers": len(b["speakers"]), "files": b["files"]}
            for src, b in sorted(by_source.items())
        },
    }
