"""Base do benchmark: ambiente do processo, nomes, caminhos e artefato do run.

Saiu de ``runner.py`` (P2-15), como os demais módulos do motor (``protocol``,
``fit_neural``, ``fit_classical``, ``split_audit``, ``provenance``,
``seed_aggregation``). Os nomes com ``_`` são privados ao PACOTE ``benchmarks``.

Importar este módulo aplica os defaults de ambiente abaixo, o que os scripts do
benchmark — inclusive o runner SSL, que reusa a proveniência — herdavam ao
importar o ``runner``.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.core.config.paths import resolve_results_output
from benchmarks.config import BenchmarkConfig

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# O projeto é Keras 3-nativo. transformers.modeling_tf_utils seta
# TF_USE_LEGACY_KERAS=1 no processo que o importa; se herdado antes do
# import do TensorFlow, tf.keras vira Keras 2 (tf_keras) e as camadas
# custom quebram. setdefault protege processos iniciados de shell limpa.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")
# Estrito por padrão SÓ no benchmark (não em ssl_utils.strict_ssl_guard, cujo
# default permanece opt-in — testes unitários e o wizard do Gradio exercitam
# de propósito o fallback CNN-1D sem torch/checkpoint disponível). Um
# benchmark que carimba "WavLM"/"HuBERT" num modelo que na verdade é uma
# CNN-1D treinada do zero (torch ausente, checkpoint indisponível, etc.)
# produz um artefato academicamente inválido sem erro nenhum — silencioso
# demais para o default ser permissivo aqui. setdefault permite override
# explícito (XFAKE_STRICT_SSL=0) para quem sabe o que está abrindo mão.
os.environ.setdefault("XFAKE_STRICT_SSL", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.file_storage import copy_file_atomic  # noqa: E402

# Modelos clássicos (sklearn) — não passam pelo TrainingService (Keras).
CLASSICAL_ARCH_SLUGS = {"svm", "randomforest"}

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _compact_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_classical_arch(name: str) -> bool:
    return _compact_slug(name) in CLASSICAL_ARCH_SLUGS


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _normalize_project_paths(cfg: BenchmarkConfig) -> None:
    """Ancora caminhos relativos na raiz do projeto, não no cwd do processo."""
    cfg.output_dir = str(
        resolve_results_output(
            cfg.output_dir,
            default_subdir="benchmark",
            base_dir=PROJECT_ROOT,
        )
    )
    cfg.models_dir = str(_project_path(cfg.models_dir))
    if cfg.dataset_path:
        cfg.dataset_path = str(_project_path(cfg.dataset_path))


def _architecture_dir(
    cfg: BenchmarkConfig, arch: str, training_seed: int | None = None
) -> Path:
    """Diretorio de trabalho da arquitetura, isolado POR SEMENTE quando ha mais de uma.

    Com uma unica semente (o caso canonico) o layout NAO muda — importante para
    o `--resume` e para os scripts de consolidacao, que procuram
    `architectures/<slug>/`.

    Com `--n-seeds > 1` o isolamento e obrigatorio. O diretorio guarda o
    `training_backup/` do `BackupAndRestore` e o `best_checkpoint.weights.h5`
    da selecao: compartilhado entre repeticoes, a semente 2 retomava do backup
    da semente 1 (herdando pesos, estado do otimizador e contador de epocas) e
    terminava carregando um checkpoint que a semente 1 tinha escrito. As N
    execucoes deixavam de ser independentes, e o "media +- desvio sobre N
    sementes" que o relatorio imprime media outra coisa.
    """
    base = _project_path(cfg.output_dir) / "architectures" / _slug(arch)
    if training_seed is None:
        return base
    try:
        sementes = list(cfg.training_seeds)
    except Exception:  # noqa: BLE001 — config antiga sem a propriedade
        sementes = []
    if len(sementes) > 1:
        return base / f"seed_{int(training_seed)}"
    return base


def _file_fingerprint(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Identidade do arquivo em disco: sha256, tamanho e mtime UTC.

    ``efficiency.size_mb`` diz o TAMANHO do artefato, não QUAL artefato — dois
    arquivos diferentes com o mesmo peso passam por iguais. Sem isto, a
    promoção não tem como recusar um artefato que foi trocado depois do run.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = p.stat()
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "saved_at_utc": (
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"
            )
        ),
        # Distingue "hash tirado pelo próprio run" de "hash reconstruído depois"
        # (ver backfill_artifact_metadata.py, que só consegue verificar por
        # tamanho declarado e nem sempre consegue nem isso).
        "integrity": "recorded_at_run",
    }


def _preserve_run_artifact(
    artifact: Optional[Path], arch_dir: Path
) -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Vincula o artefato treinado (e o sidecar ``_config.json``) ao run.

    MOTIVAÇÃO 2026-08-09: ``_models_dir`` devolve ``data/models`` — um
    diretório GLOBAL chaveado só pela arquitetura. Todo run (benchmark, smoke,
    retreino) grava em ``data/models/bench_<arch>.*``, e nada amarra o arquivo
    ao run que o produziu. Foi assim que o ``bench_svm.pkl`` do
    ``clean_benchmark_15k`` (63 features, 3,6 MB) virou um artefato de smoke de
    47 KB e 8 amostras: as métricas do SVM sobreviveram, o modelo não.

    Os neurais escapavam por acidente — o ``best_checkpoint.weights.h5`` fica no
    run —, mas pesos sem grafo nem sidecar não são um artefato promovível. O
    runner SSL já fazia o certo (grava o ``.pt`` dentro do run); aqui os
    caminhos Keras e clássico passam a fazer o mesmo.

    ``data/models/bench_*`` continua como a versão CORRENTE que a inferência
    carrega. A cópia do run é REAL, nunca hardlink: com o inode compartilhado,
    qualquer gravação in-place no modelo corrente (um smoke, um script, um
    ``joblib.dump``) alteraria em silêncio o artefato que o run registrou, que é
    o incidente acima. Custa o espaço de um modelo por run.
    """
    if artifact is None:
        return None, None
    src = Path(artifact)
    if not src.is_file():
        return None, None
    dest_dir = arch_dir / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.resolve() != src.resolve():
        copy_file_atomic(src, dest)
        sidecar = src.with_name(f"{src.stem}_config.json")
        if sidecar.is_file():
            copy_file_atomic(sidecar, dest_dir / sidecar.name)
    return dest, _file_fingerprint(dest)


def _models_dir(cfg: BenchmarkConfig, arch: str) -> Path:
    explicit = (
        os.getenv("MODELS_DIR")
        or os.getenv("DEEPFAKE_MODELS_DIR")
        or os.getenv("XFAKE_MODELS_DIR")
    )
    if explicit:
        return _project_path(explicit)
    storage = os.getenv("XFAKE_STORAGE_DIR") or os.getenv("DEEPFAKE_STORAGE_DIR")
    if storage:
        return _project_path(storage) / "models"
    return _project_path(cfg.models_dir)


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64").ravel()
    if np.any(~np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)
