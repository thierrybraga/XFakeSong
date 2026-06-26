"""Manifesto de falantes do dataset (identificacao de falante, aditiva).

Os nomes de arquivo do XFakeSong seguem `<fonte>_NNNNN.wav` e NAO embutem o
falante. Este modulo mantem um sidecar `app/datasets/speaker_manifest.json` que
mapeia o nome do arquivo -> falante, quando a fonte expoe esse identificador
(Fake Voices por ZIP, Common Voice por `client_id`, In-the-Wild por celebridade,
ASVspoof pelo `speaker` do protocolo). Para fontes que nao expoem falante, o
agrupamento cai para o nivel de FONTE (`<prefixo>`), o grupo mais fino disponivel.

Usado pelo tier `large` para split DISJUNTO POR FALANTE (usuarios nao vistos) e
pelo benchmark (`speaker_ids` no `.npz`, `--speaker-split`/`--unseen-speaker`).

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
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("speaker_manifest")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "app" / "datasets"
SPEAKER_MANIFEST_PATH = DATASETS_DIR / "speaker_manifest.json"

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


def _infer_prefix(path: str | Path) -> str:
    """Prefixo de fonte (`<prefixo>` antes do primeiro `_`), em minusculas."""
    # import local para evitar ciclo de import com dataset_catalog
    try:
        from app.core.dataset_catalog import infer_prefix_from_path

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
    name = _basename(path)
    src = (source or _infer_prefix(path)).strip().lower()
    with _LOCK:
        manifest = load_manifest()
        manifest[name] = {"speaker_id": sid, "source": src}
        _DIRTY = True
        _PENDING += 1
        if not _ATEXIT_REGISTERED:
            atexit.register(flush)
            _ATEXIT_REGISTERED = True
        if _PENDING >= _FLUSH_EVERY:
            flush()


def speaker_for_path(path: str | Path) -> str:
    """Chave de falante alinhavel a uma amostra.

    Retorna `<fonte>:<speaker_id>` quando conhecido; caso contrario, o nivel de
    FONTE (`<prefixo>`) — grupo mais fino disponivel para aquela amostra.
    """
    manifest = load_manifest()
    entry = manifest.get(_basename(path))
    if entry and entry.get("speaker_id"):
        src = entry.get("source") or _infer_prefix(path)
        return f"{src}:{entry['speaker_id']}"
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
