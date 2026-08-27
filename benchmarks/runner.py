"""Orquestrador do benchmark: treina, avalia, mede robustez e eficiência."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.core.config.paths import resolve_results_output
from benchmarks.config import BenchmarkConfig
from benchmarks.data import (
    BenchmarkData,
    looks_like_raw_audio,
    prepare_input_for_architecture,
)
from benchmarks.efficiency import (
    count_params,
    file_size_mb,
    measure_latency_profile,
)
from benchmarks.evaluate import evaluate_grouped_scores, evaluate_scores
from benchmarks.planning import (
    apply_plan_to_config,
    build_benchmark_plan,
    write_benchmark_plan,
)
from benchmarks.stability import analyze_training_stability

logger = logging.getLogger("benchmark")

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    """Copia o artefato treinado (e o sidecar ``_config.json``) para o run.

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

    ``data/models/bench_*`` continua existindo como a cópia CORRENTE que a
    inferência carrega — pode ser sobrescrita à vontade sem destruir o run.
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
        shutil.copy2(src, dest)
        sidecar = src.with_name(f"{src.stem}_config.json")
        if sidecar.is_file():
            shutil.copy2(sidecar, dest_dir / sidecar.name)
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


def _classical_search_space(arch: str, seed: int) -> tuple[dict[str, list], Any, str]:
    """Grid, estimador-base e nome do passo do pipeline para SVM/RF.

    O GRID VEM DA ARQUITETURA, não daqui (2026-08-09). Até então esta função
    carregava uma cópia própria — a 4ª fonte de hiperparâmetros do projeto,
    ausente das três que o CLAUDE.md lista — e era ela que rodava, deixando os
    grids regularizados de `svm.py`/`random_forest.py` como código morto sem
    NENHUM chamador em `app/`, `benchmarks/`, `scripts/` ou `tests/`.
    """
    compact = _compact_slug(arch)
    if compact == "svm":
        from sklearn.svm import SVC

        from app.domain.models.architectures.svm import SVM_PARAM_GRID

        return (
            [dict(bloco) for bloco in SVM_PARAM_GRID],
            # `probability=False` DENTRO da busca (2026-08-09). Com `True`, o
            # libsvm roda uma validação cruzada interna de 5 dobras a cada
            # ajuste para calibrar Platt — 6 ajustes de SVC onde a busca pede 1.
            # E não compra nada: o `scoring` é `roc_auc`, que é baseado em
            # ORDENAÇÃO, e a sigmoide de Platt é monotônica, então a AUC sobre
            # `predict_proba` é idêntica à sobre `decision_function`. O modelo
            # FINAL continua com probabilidade — quem a fornece lá é a
            # calibração isotônica, não o Platt interno.
            SVC(probability=False, random_state=seed),
            "svm",
        )

    from sklearn.ensemble import RandomForestClassifier

    from app.domain.models.architectures.random_forest import (
        RANDOM_FOREST_PARAM_GRID,
    )

    return (
        dict(RANDOM_FOREST_PARAM_GRID),
        # `n_jobs=1` DENTRO da busca (2026-08-09): o `GridSearchCV` já roda com
        # `n_jobs=-1`, e uma floresta que também pede todos os núcleos cria
        # sobre-inscrição de threads — os workers disputam os mesmos núcleos e o
        # grid fica mais lento que em série. Quem paraleliza aqui é a busca, que
        # tem 540 ajustes independentes para distribuir. O ajuste FINAL continua
        # com `n_jobs=-1` (vem do `create_random_forest_model`, sem laço externo).
        RandomForestClassifier(random_state=seed, n_jobs=1),
        "rf",
    )


#: Dobras da validação cruzada dos clássicos.
#:
#: Eram 3 (`min(3, min_class_count)`). Com 5 o desvio entre dobras cai e a
#: escolha do grid deixa de ser decidida por ruído de partição: no
#: `clean_benchmark_15k` o `std_test_score` do SVM era 0,0638 contra 0,0025 de
#: distância entre o 1º e o 3º colocado — o grid escolhia a dobra, não o
#: candidato.
_CLASSICAL_CV_FOLDS = 5


def _classical_cv_splitter(
    y: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
) -> tuple[Any, int, str, str]:
    """Splitter da CV dos clássicos: agrupado por cluster quando possível.

    VAZAMENTO CORRIGIDO EM 2026-08-09. A CV era um `StratifiedKFold` simples
    (o inteiro `cv=3` que o `GridSearchCV` interpreta assim), SEM `groups`. O
    Protocolo de Dataset é PAREADO — cada enunciado aparece como original CETUC
    e como clone XTTS-v2 do mesmo locutor e da mesma frase —, então uma
    partição aleatória põe metade do par no treino da dobra e a outra metade na
    validação dela. O modelo não precisa detectar síntese para acertar: basta
    reconhecer o enunciado que acabou de ver. `StratifiedGroupKFold` sobre
    `cluster_ids` mantém o par inteiro do mesmo lado.

    Sem `cluster_ids` degrada para o comportamento anterior, mas DECLARADO no
    artefato (`cv_kind`/`cv_grouping`) em vez de silenciosamente.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y_arr = np.asarray(y).ravel()
    values, counts = np.unique(y_arr, return_counts=True)
    min_class_count = int(counts.min()) if len(values) >= 2 else 0
    if min_class_count < 2:
        return None, min_class_count, "none", "amostras insuficientes por classe"

    if groups is None:
        folds = min(_CLASSICAL_CV_FOLDS, min_class_count)
        return (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
            folds,
            "StratifiedKFold",
            "SEM agrupamento: cluster_ids indisponíveis — pares locutor×frase "
            "podem se dividir entre treino e validação da dobra",
        )

    groups_arr = np.asarray(groups).ravel()
    if len(groups_arr) != len(y_arr):
        raise RuntimeError(
            f"cluster_ids desalinhados no tuning clássico: {len(groups_arr)} "
            f"grupos para {len(y_arr)} amostras"
        )
    n_groups = int(len(np.unique(groups_arr)))
    folds = min(_CLASSICAL_CV_FOLDS, min_class_count, n_groups)
    if folds < 2:
        return None, folds, "none", "grupos insuficientes para validação cruzada"
    return (
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed),
        folds,
        "StratifiedGroupKFold",
        f"agrupado por cluster_ids (locutor × frase): {n_groups} grupos",
    )


def _run_classical_tuning(
    arch: str,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Otimiza SVM/RF e grava o histórico do grid search.

    ``groups`` são os ``cluster_ids`` (locutor × frase) das linhas de ``X``.
    Com eles a validação cruzada é AGRUPADA — ver :func:`_classical_cv_splitter`.
    """
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    grid, estimator, step_name = _classical_search_space(arch, seed)
    splitter, cv, cv_kind, cv_note = _classical_cv_splitter(y, groups, seed)
    plan = {
        "enabled": True,
        "method": "GridSearchCV",
        "scoring": "roc_auc",
        "cv": cv,
        "cv_kind": cv_kind,
        "cv_grouping": cv_note,
        "param_grid": grid,
        # O grid pode ser um dicionário (produto cartesiano único) ou uma LISTA
        # de dicionários — a forma que o SVM usa para não cruzar `gamma` com o
        # kernel linear, que o ignora. `ParameterGrid` conta as duas certo; o
        # `np.prod` sobre `.values()` quebrava na lista.
        "n_candidates": len(ParameterGrid(grid)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if cv < 2:
        plan.update(
            {
                "status": "skipped",
                "reason": "amostras insuficientes por classe para validação cruzada",
            }
        )
        (output_dir / "hyperparameter_tuning.json").write_text(
            json.dumps(_json_safe(plan), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return plan

    pipeline = Pipeline([("scaler", StandardScaler()), (step_name, estimator)])
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=splitter,
        n_jobs=-1,
        refit=False,
        return_train_score=True,
        verbose=0,
    )
    started = time.time()
    search.fit(X, y, groups=groups if cv_kind == "StratifiedGroupKFold" else None)
    elapsed = round(time.time() - started, 3)

    rows = []
    results = search.cv_results_
    for idx, params in enumerate(results["params"]):
        rows.append(
            {
                "rank": int(results["rank_test_score"][idx]),
                "mean_test_score": float(results["mean_test_score"][idx]),
                "std_test_score": float(results["std_test_score"][idx]),
                "mean_train_score": float(
                    results.get("mean_train_score", [np.nan])[idx]
                ),
                "params_json": json.dumps(_json_safe(params), ensure_ascii=False),
            }
        )
    rows.sort(key=lambda row: row["rank"])
    with (output_dir / "hyperparameter_tuning.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
                "params_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_model_params = {
        k.replace(f"{step_name}__", ""): v
        for k, v in search.best_params_.items()
        if k.startswith(f"{step_name}__")
    }
    plan.update(
        {
            "status": "ok",
            "elapsed_s": elapsed,
            "best_score": float(search.best_score_),
            "best_params": _json_safe(search.best_params_),
            "best_model_params": _json_safe(best_model_params),
            "top_candidates": rows[:5],
        }
    )
    (output_dir / "hyperparameter_tuning.json").write_text(
        json.dumps(_json_safe(plan), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return plan


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64").ravel()
    if np.any(~np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)


def _git_provenance() -> Dict[str, Any]:
    """Commit que gerou os resultados — sem isto o run não é rastreável.

    Um artigo cujos números não apontam para uma revisão exata do código não é
    reproduzível: a mesma arquitetura pode ter mudado entre execuções (foi o
    caso em 2026-07-27, quando sete arquiteturas mudaram no mesmo dia).
    `dirty=True` sinaliza que havia alterações não commitadas — o run continua,
    mas o registro deixa a ressalva explícita.
    """
    import subprocess

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    status = _run(["git", "status", "--porcelain"])
    return {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def _library_versions() -> Dict[str, Any]:
    """Versões das bibliotecas que influenciam o resultado numérico."""
    versions: Dict[str, Any] = {}
    for module_name, attr in (
        ("tensorflow", "__version__"),
        ("keras", "__version__"),
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("sklearn", "__version__"),
        ("librosa", "__version__"),
        ("torch", "__version__"),
        ("transformers", "__version__"),
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, attr, None)
        except Exception:  # noqa: BLE001
            versions[module_name] = None
    return versions


def _pretrained_checkpoints() -> Dict[str, Any]:
    """Checkpoints pré-treinados que entram no grafo dos modelos.

    AST, WavLM e HuBERT partem de pesos externos: o identificador do checkpoint
    é parte da definição do experimento e precisa constar no artefato.
    """
    checkpoints: Dict[str, Any] = {}
    try:
        from app.domain.models.architectures.ast_pretrained import (
            DEFAULT_AST_CHECKPOINT,
        )

        checkpoints["SpectrogramTransformer"] = DEFAULT_AST_CHECKPOINT
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.domain.models.architectures.registry import architecture_registry

        wavlm = architecture_registry.get_architecture("WavLM").default_params
        hubert = architecture_registry.get_architecture("HuBERT").default_params
        checkpoints["WavLM"] = wavlm.get("wavlm_model")
        checkpoints["HuBERT"] = hubert.get("model_name")
    except Exception:  # noqa: BLE001
        pass
    return checkpoints


def _env_snapshot() -> Dict[str, Any]:
    snap = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "git": _git_provenance(),
        "libraries": _library_versions(),
        "pretrained_checkpoints": _pretrained_checkpoints(),
    }
    try:
        import tensorflow as tf

        snap["tensorflow"] = tf.__version__
        snap["gpu"] = bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        snap["tensorflow"] = None
        snap["gpu"] = False
    try:
        from app.core.gpu import describe_gpu_setup

        snap["device"] = describe_gpu_setup()
    except Exception:
        snap["device"] = "?"
    return snap


def _architecture_provenance(arch: str) -> Dict[str, Any]:
    """Entrada do manifesto oficial correspondente à arquitetura.

    Os rótulos `variant`/`family`/`runner` existiam em `benchmarks/config.py`
    mas NUNCA chegavam a nenhum artefato — proveniência que ninguém lia. Agora
    acompanham cada resultado.
    """
    try:
        from benchmarks.config import (
            EXTENDED_MODEL_MANIFEST,
            OFFICIAL_TCC_MODEL_MANIFEST,
        )
    except Exception:  # noqa: BLE001
        return {}
    for item in list(OFFICIAL_TCC_MODEL_MANIFEST) + list(EXTENDED_MODEL_MANIFEST):
        if item.get("benchmark_name") == arch:
            prov = {
                key: item.get(key)
                for key in ("variant", "family", "runner", "scope", "result_key")
                if item.get(key) is not None
            }
            _apply_ssl_backbone_status(arch, item, prov)
            return prov
    return {}


def _apply_ssl_backbone_status(
    arch: str, item: Dict[str, Any], prov: Dict[str, Any]
) -> None:
    """Substitui o rótulo declarado pelo backbone SSL REALMENTE construído.

    WavLM/HuBERT no caminho Keras degradam para um CNN-1D do zero quando o
    checkpoint não está acessível. Publicar o `variant` do manifesto nesse caso
    faria o artefato alegar backbone pré-treinado onde não houve — erro
    indetectável depois do run. `ssl_utils` registra o que foi montado; aqui
    isso vira proveniência.
    """
    try:
        from app.domain.models.architectures.ssl_utils import (
            get_ssl_backbone_status,
        )

        status = get_ssl_backbone_status(arch)
    except Exception:  # noqa: BLE001
        status = None
    if not status:
        return
    prov["ssl_backbone"] = status
    if not status.get("pretrained") and item.get("fallback_variant"):
        prov["declared_variant"] = prov.get("variant")
        prov["variant"] = item["fallback_variant"]


def _stratified_test_labels(
    y: np.ndarray,
    seed: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> np.ndarray:
    """Retorna apenas os labels do teste, sem copiar o tensor X completo."""
    y = np.asarray(y)
    try:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(y))
        _train_idx, temp_idx = train_test_split(
            idx,
            test_size=val_frac + test_frac,
            stratify=y,
            random_state=seed,
        )
        rel_test = test_frac / (val_frac + test_frac)
        _val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test,
            stratify=y[temp_idx],
            random_state=seed,
        )
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_test = max(1, int(len(idx) * test_frac))
        test_idx = idx[:n_test]
    return y[test_idx]


def _audit_split_overlap(raw_splits, fail_on_overlap: bool = True) -> Dict[str, Any]:
    """Detecta amostras binariamente idênticas entre treino, validação e teste."""

    names = ("train", "val", "test")
    arrays = (raw_splits[0], raw_splits[2], raw_splits[4])
    labels = (raw_splits[1], raw_splits[3], raw_splits[5])
    fingerprints: Dict[str, set[str]] = {}
    split_fingerprints: Dict[str, Any] = {
        "method": "sha256_ordered_blake2b_sample_hashes_and_labels"
    }
    for name, values, y in zip(names, arrays, labels):
        current: set[str] = set()
        ordered = hashlib.sha256()
        y_arr = np.ascontiguousarray(np.asarray(y, dtype="int64"))
        ordered.update(str(np.asarray(values).shape).encode("ascii"))
        ordered.update(y_arr.view(np.uint8))
        for sample in np.asarray(values):
            contiguous = np.ascontiguousarray(sample)
            digest_bytes = hashlib.blake2b(
                contiguous.view(np.uint8), digest_size=16
            ).digest()
            current.add(digest_bytes.hex())
            ordered.update(digest_bytes)
        fingerprints[name] = current
        split_fingerprints[name] = {
            "n": int(len(y_arr)),
            "sha256": ordered.hexdigest(),
        }

    pair_counts = {
        "train_val": len(fingerprints["train"] & fingerprints["val"]),
        "train_test": len(fingerprints["train"] & fingerprints["test"]),
        "val_test": len(fingerprints["val"] & fingerprints["test"]),
    }
    total = sum(pair_counts.values())
    audit = {
        "method": "blake2b-128_exact_array_bytes",
        "pairwise_overlap_counts": pair_counts,
        "split_fingerprints": split_fingerprints,
        "passed": total == 0,
    }
    if total and fail_on_overlap:
        raise ValueError(
            "Contaminação entre partições: amostras idênticas detectadas "
            f"{pair_counts}. Corrija o NPZ antes do benchmark."
        )
    return audit


def _split_fingerprint(raw_splits) -> Dict[str, Any]:
    """SHA-256 determinístico da identidade e ordem de cada partição."""

    result: Dict[str, Any] = {"method": "sha256_ordered_array_bytes_and_labels"}
    for name, X, y in (
        ("train", raw_splits[0], raw_splits[1]),
        ("val", raw_splits[2], raw_splits[3]),
        ("test", raw_splits[4], raw_splits[5]),
    ):
        digest = hashlib.sha256()
        y_arr = np.ascontiguousarray(np.asarray(y, dtype="int64"))
        digest.update(str(np.asarray(X).shape).encode("ascii"))
        digest.update(y_arr.view(np.uint8))
        for sample in np.asarray(X):
            contiguous = np.ascontiguousarray(sample)
            digest.update(contiguous.view(np.uint8))
        result[name] = {"n": int(len(y_arr)), "sha256": digest.hexdigest()}
    return result


def _audit_split_provenance(data: BenchmarkData) -> Dict[str, Any]:
    """Relata sobreposição de fonte/locutor nas partições efetivamente usadas."""

    indices = data.last_split_indices or {}
    required = {"train", "val", "test"}
    if not required.issubset(indices):
        return {"available": False, "reason": "no_effective_split_indices"}
    result: Dict[str, Any] = {"available": True}
    for field, values in (("groups", data.groups), ("speakers", data.speakers)):
        if values is None:
            result[field] = {"available": False}
            continue
        sets = {
            split: set(np.asarray(values)[indices[split]].astype(str).tolist())
            for split in ("train", "val", "test")
        }
        intersections = {
            "train_val": sorted(sets["train"] & sets["val"]),
            "train_test": sorted(sets["train"] & sets["test"]),
            "val_test": sorted(sets["val"] & sets["test"]),
        }
        result[field] = {
            "available": True,
            "unique_counts": {key: len(value) for key, value in sets.items()},
            "overlap_counts": {key: len(value) for key, value in intersections.items()},
            "train_test_examples": intersections["train_test"][:10],
            "disjoint": all(not value for value in intersections.values()),
        }
    return result


def _audit_source_label_shortcut(
    data: BenchmarkData,
    *,
    threshold: float = 0.55,
    fail: bool = False,
) -> Dict[str, Any]:
    """Measure how accurately a source-majority oracle predicts the label."""
    if data.groups is None:
        audit = {
            "available": False,
            "passed": False,
            "reason": "source_ids_missing",
        }
        if fail:
            raise ValueError("Auditoria fonte-rotulo exige source_ids explicitos")
        return audit

    groups = np.asarray(data.groups).astype(str)
    labels = np.asarray(data.y).astype(int)
    counts: Dict[str, Dict[str, int]] = {}
    correct = 0
    for source in sorted(set(groups.tolist())):
        source_labels = labels[groups == source]
        real_n = int(np.sum(source_labels == 0))
        fake_n = int(np.sum(source_labels == 1))
        counts[source] = {"real": real_n, "fake": fake_n}
        correct += max(real_n, fake_n)
    accuracy = correct / max(len(labels), 1)
    audit = {
        "available": True,
        "method": "source_majority_oracle",
        "accuracy": accuracy,
        "threshold": float(threshold),
        "counts": counts,
        "passed": bool(accuracy <= threshold),
    }
    if fail and not audit["passed"]:
        raise ValueError(
            f"Atalho fonte-rotulo: oraculo={accuracy:.4f} > limite={threshold:.4f}"
        )
    return audit


def _audit_predefined_provenance(data: BenchmarkData) -> Dict[str, Any]:
    """Alias legado; audita agora as partições efetivamente usadas."""

    return _audit_split_provenance(data)


def _stamp_eval_crop_strategy(
    config_path: Optional[Path],
    input_contract: Optional[Dict[str, Any]],
    protocol: Dict[str, Any],
) -> None:
    """Completa `crop_strategy` no contrato APÓS a avaliação resolvê-la.

    O contrato é carimbado antes do treino, quando `eval_crop_strategy` ainda é
    o placeholder ``"resolved_at_eval"``. A inferência ativa o multicrop por
    ``"multicrop" in crop_strategy``, então gravar o placeholder fazia produção
    rodar 1 crop enquanto o benchmark media com 3 e média dos scores — as
    métricas do artigo não valiam para a predição do app.
    """
    estrategia = protocol.get("eval_crop_strategy")
    if not config_path or not estrategia or estrategia == "resolved_at_eval":
        return
    valor = (
        f"train_{protocol.get('train_crop_strategy') or 'center'}"
        f"_eval_{estrategia}"
    )
    if isinstance(input_contract, dict):
        input_contract["crop_strategy"] = valor
    caminho = Path(config_path)
    try:
        if not caminho.exists():
            return
        payload = json.loads(caminho.read_text(encoding="utf-8"))
        contrato = dict(payload.get("input_contract") or {})
        contrato["crop_strategy"] = valor
        contrato["eval_num_crops"] = int(protocol.get("eval_num_crops") or 1)
        contrato["eval_score_aggregation"] = protocol.get("eval_score_aggregation")
        payload["input_contract"] = contrato
        caminho.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        logger.warning(
            "Não foi possível gravar crop_strategy resolvido em %s: %s",
            caminho,
            exc,
        )


def _band_correction_policy(
    cfg: BenchmarkConfig, data: Any = None
) -> Optional[Dict[str, Any]]:
    """Política de correção de formato do run, ou ``None`` se não houver.

    Vai para o protocolo e, por ele, para o `input_contract` do artefato — é o
    que permite a inferência aplicar a MESMA correção que o treino aplicou.

    DUAS ORIGENS, e as duas contam:

    1. ``cfg.band_correction_hz`` — o runner aplicou a correção às partições.
    2. ``data.metadata["band_correction"]`` — o dataset JÁ veio corrigido do
       build (`extract_window` aplica desde 2026-08-20) e o runner, corretamente,
       recusa a segunda aplicação.

    Considerar apenas (1) deixava um buraco: com um NPZ pré-corrigido a flag é
    recusada, o contrato saía sem o campo e a inferência rodava full-band contra
    um modelo treinado em banda limitada — exatamente a divergência que a
    política existe para fechar. O modelo viu áudio corrigido nos dois casos;
    o contrato tem de dizer isso nos dois casos.
    """
    ja_no_dataset = (getattr(data, "metadata", None) or {}).get("band_correction") or {}
    if ja_no_dataset.get("applied"):
        politica = {
            chave: ja_no_dataset.get(chave)
            for chave in ("cutoff_hz", "taps", "remove_dc", "renormalize_rms_dbfs")
            if ja_no_dataset.get(chave) is not None
        }
        politica["origem"] = "dataset_build"
        return politica

    corte = getattr(cfg, "band_correction_hz", None)
    if not corte:
        return None
    from app.domain.features.benchmark_frontend import (
        BAND_CORRECTION_RMS_DBFS,
        BAND_CORRECTION_TAPS,
    )

    return {
        "cutoff_hz": float(corte),
        "taps": int(BAND_CORRECTION_TAPS),
        "remove_dc": True,
        "renormalize_rms_dbfs": float(BAND_CORRECTION_RMS_DBFS),
        "origem": "runner",
    }


def _prepare_protocol_splits(
    arch: str,
    cfg: BenchmarkConfig,
    raw_splits,
    training_seed: int | None = None,
    data: Any = None,
):
    """Prepara treino/val/teste preservando AWGN no domínio da forma de onda."""
    Xtr_raw, ytr, Xv_raw, yv, Xte_raw, yte = raw_splits
    waveform_domain = (
        looks_like_raw_audio(Xtr_raw)
        and looks_like_raw_audio(Xv_raw)
        and looks_like_raw_audio(Xte_raw)
    )
    if (
        cfg.strict_waveform_awgn
        and cfg.dataset_path
        and cfg.snr_levels_db
        and not waveform_domain
    ):
        raise ValueError(
            "O protocolo exige formas de onda no NPZ para aplicar AWGN antes "
            "dos frontends. Use um dataset raw-audio ou desative "
            "strict_waveform_awgn explicitamente para testes legados."
        )

    # Semente da REPETIÇÃO: crop aleatório e ruído de treino fazem parte da
    # aleatoriedade de TREINO e variam entre seeds. O split (teste selado) e o
    # ruído de AVALIAÇÃO continuam presos a `cfg.seed`.
    train_seed = int(cfg.seed if training_seed is None else training_seed)
    Xtr_clean, input_type = prepare_input_for_architecture(
        Xtr_raw,
        arch,
        crop_strategy="random",
        seed=train_seed,
    )
    Xv, _ = prepare_input_for_architecture(Xv_raw, arch)
    Xte, _ = prepare_input_for_architecture(Xte_raw, arch)
    ytr_clean = np.asarray(ytr)
    prepared_shape = list(Xtr_clean.shape[1:])
    assigned_counts: dict[str, int] = {}

    # AJUSTE 2026-07-15 (diagnóstico do retreino 20260714): AASIST/RawGAT-ST
    # overfitavam a cópia AWGN ESTÁTICA (mesma realização toda época — AASIST:
    # val_acc pico 88,8% na época 11 caindo a 79,5% na 100). Para esses dois,
    # a cópia estática é substituída por augmentation DINÂMICO na forma de
    # onda (AudioAugmenter por época, domínio fisicamente válido — mesmo
    # regime do run de 2026-07-07 em que o AASIST fez 95,8%). Custo por época
    # idêntico (2× o treino limpo). Demais arquiteturas seguem o protocolo
    # padrão (1 cópia AWGN estática, augmenter interno desligado).
    dynamic_aug_archs = {"aasist", "rawgatst"}
    use_dynamic_augmenter = (
        waveform_domain
        and cfg.architecture_specific_augmentation
        and _compact_slug(arch) in dynamic_aug_archs
    )
    use_train_noise = (
        waveform_domain
        and not use_dynamic_augmenter
        and cfg.waveform_noise_augmentation
        and cfg.train_noise_copies > 0
        and bool(cfg.train_aug_snr_db)
    )
    train_noise_seeds: set[int] = set()

    # Aloca o tensor de treino UMA vez e escreve cada bloco no lugar.
    #
    # O caminho anterior (listas + `np.concatenate`) mantinha vivos ao mesmo
    # tempo o bloco limpo, o bloco ruidoso e o resultado concatenado — pico de
    # ~4x o tamanho de um bloco. Em raw-audio um bloco é
    # 33.226 x 48.000 x 4 B = 6,4 GB, ou seja ~25 GB de pico, acima do limite
    # de memória do container de treino (DOCKER_TRAIN_MEMORY_LIMIT) e da VM do
    # WSL2 numa estação com RTX 3060. Escrevendo em fatias o pico cai para o
    # tamanho do tensor final (~12,8 GB) mais o bloco corrente.
    #
    # Os valores e a ORDEM são idênticos aos do concatenate: bloco limpo em
    # [0, n), cópia k em [n*(k+1), n*(k+2)).
    n_clean = len(ytr_clean)
    n_copies = int(cfg.train_noise_copies) if use_train_noise else 0
    Xtr = np.empty(
        (n_clean * (1 + n_copies), *Xtr_clean.shape[1:]), dtype=Xtr_clean.dtype
    )
    Xtr[:n_clean] = Xtr_clean
    del Xtr_clean
    ytr_fit = np.concatenate([ytr_clean] * (1 + n_copies), axis=0)

    if use_train_noise:
        noise_batch = max(1, int(cfg.waveform_noise_batch_size))
        for copy_index in range(n_copies):
            offset = n_clean * (1 + copy_index)
            base_seed = train_seed + 10000 + copy_index
            assigned = BenchmarkData.balanced_snr_assignments(
                len(Xtr_raw), cfg.train_aug_snr_db, seed=base_seed
            )
            for start in range(0, len(Xtr_raw), noise_batch):
                stop = min(start + noise_batch, len(Xtr_raw))
                assigned_chunk = assigned[start:stop]
                noisy_chunk = BenchmarkData.add_awgn_assigned(
                    Xtr_raw[start:stop], assigned_chunk, seed=base_seed + start
                )
                prepared_chunk, _ = prepare_input_for_architecture(
                    noisy_chunk,
                    arch,
                    crop_strategy="random",
                    seed=base_seed + start,
                )
                Xtr[offset + start : offset + stop] = prepared_chunk
                # As sementes REAIS do RNG, não o `seed` de entrada:
                # `add_awgn_assigned` deriva uma por nível presente na chunk.
                train_noise_seeds.update(
                    BenchmarkData.assigned_awgn_seeds(
                        assigned_chunk, base_seed + start
                    ).values()
                )
            values, counts = np.unique(assigned, return_counts=True)
            for value, count in zip(values, counts):
                key = str(int(value))
                assigned_counts[key] = assigned_counts.get(key, 0) + int(count)

    # Disjunção treino↔avaliação das sementes de ruído, VERIFICADA.
    #
    # Uma colisão faria o modelo treinar sobre a mesma realização de AWGN usada
    # no teste, então a checagem precisa comparar as sementes que o gerador
    # REALMENTE recebe. Até 2026-07-29 ela comparava o `seed` de ENTRADA de
    # `add_awgn_assigned` e ignorava o termo `1009 * (offset + 1)` que aquela
    # função deriva por nível (benchmarks/data.py). Nos parâmetros canônicos
    # (lote 64) não havia colisão, mas o guard dava FALSO NEGATIVO: com
    # `waveform_noise_batch_size=8` e duas cópias de treino, as sementes reais
    # 20010 e 20020 colidem com a avaliação em 10 e 20 dB e o run era aprovado.
    # Agora as sementes vêm de `BenchmarkData.assigned_awgn_seeds`, a mesma
    # fonte que `add_awgn_assigned` consome.
    eval_noise_seeds = {
        int(cfg.seed) + 20000 + int(snr) for snr in (cfg.snr_levels_db or [])
    }
    seed_collisions = sorted(train_noise_seeds & eval_noise_seeds)
    if seed_collisions:
        raise ValueError(
            "colisão de sementes de ruído treino↔avaliação: "
            f"{seed_collisions}. O modelo treinaria sobre a mesma realização de "
            "AWGN usada no teste. Ajuste waveform_noise_batch_size ou os "
            "offsets 10000/20000 em _prepare_protocol_splits/_benchmark_one."
        )

    protocol = {
        "evaluation_domain": "waveform" if waveform_domain else "input_space_fallback",
        "frontend_after_noise": bool(waveform_domain),
        "training_augmentation_domain": (
            "waveform_dynamic_augmenter"
            if use_dynamic_augmenter
            else ("waveform" if use_train_noise else "disabled")
        ),
        "train_aug_snr_db": [int(v) for v in cfg.train_aug_snr_db],
        "train_noise_copies": int(cfg.train_noise_copies if use_train_noise else 0),
        "waveform_noise_batch_size": int(cfg.waveform_noise_batch_size),
        "assigned_snr_counts": assigned_counts,
        "train_noise_seed_count": len(train_noise_seeds),
        "train_eval_noise_seeds_disjoint": True,
        "clean_train_samples": int(len(ytr)),
        "fit_train_samples": int(len(ytr_fit)),
        "input_type": input_type,
        "original_shape": list(np.asarray(Xtr_raw).shape[1:]),
        "prepared_shape": prepared_shape,
        # CORREÇÃO DE FORMATO aplicada às três partições antes de chegar aqui
        # (ver `run_benchmark`). Vai para o protocolo — e daí para o
        # `input_contract` do artefato — porque a INFERÊNCIA precisa reproduzi-la:
        # um modelo treinado com passa-baixas de 7,5 kHz e sem DC recebia, em
        # produção, áudio full-band com offset. Sem o campo, o preparador não
        # tem como saber que a política existe.
        "band_correction": _band_correction_policy(cfg, data),
        "train_crop_strategy": "random" if input_type == "raw_audio" else None,
        # Resolvidos em `_benchmark_one`, quando se sabe se as formas de onda
        # cruas estão disponíveis para o multicrop. Declarar aqui produzia um
        # protocolo que contradizia a avaliação realmente executada.
        "eval_crop_strategy": "resolved_at_eval",
        "eval_num_crops": None,
    }
    return (
        Xtr,
        ytr_fit,
        Xv,
        np.asarray(yv),
        Xte,
        np.asarray(yte),
        int(len(ytr)),
        protocol,
    )


def _declared_n_fft(arch: str) -> int | None:
    """`n_fft` que a arquitetura declara no registry, se declarar."""
    try:
        from app.domain.models.architectures.registry import architecture_registry

        req = architecture_registry.get_architecture(arch).input_requirements or {}
        valor = req.get("n_fft")
        return int(valor) if valor else None
    except Exception:  # noqa: BLE001
        return None


def _stamp_benchmark_frontend(
    config_path: Path,
    input_contract: Dict[str, Any],
    protocol: Dict[str, Any],
    arch: str = "",
) -> Dict[str, Any]:
    """Grava no sidecar o front-end com que o modelo foi REALMENTE treinado.

    Os modelos do benchmark são os promovidos para produção, e a inferência só
    reproduz o preparo do treino quando o `input_contract` declara um
    `feature_frontend` conhecido — é o que faz o `FeaturePreparer` rotear para
    `app/domain/features/benchmark_frontend.py`. Sem isso, o app cai no
    front-end próprio (log-magnitude-mel, hop 128, sem z-score), que NÃO
    reproduz o do benchmark: as métricas do artigo deixam de valer para a
    predição em produção.

    Até 2026-07-28 apenas AASIST e RawGAT-ST declaravam o campo (via
    `registry.input_requirements`), e as outras dez dependiam de um passo
    pós-hoc (`scripts/reporting/rebuild_inference_contracts.py`) que, além de
    manual, cobria só nove arquiteturas.

    O carimbo é feito AQUI, e não no `TrainingService`, de propósito: só o
    benchmark sabe que preparou os dados com este front-end. Modelos treinados
    pelo assistente do Gradio usam outro preparo e não devem alegar paridade.
    """
    from app.domain.features.benchmark_frontend import (
        DEFAULT_SAMPLE_RATE,
        DEFAULT_SOURCE_SAMPLES,
        frontend_for_input_type,
    )

    frontend = frontend_for_input_type(protocol.get("input_type"))
    if not frontend:
        return input_contract

    contract = dict(input_contract or {})
    contract["feature_frontend"] = frontend
    contract.setdefault("input_type", protocol.get("input_type"))
    contract.setdefault("sample_rate", DEFAULT_SAMPLE_RATE)
    contract["source_samples"] = int(
        protocol.get("original_shape", [DEFAULT_SOURCE_SAMPLES])[0]
        if protocol.get("original_shape")
        else DEFAULT_SOURCE_SAMPLES
    )
    prepared = protocol.get("prepared_shape") or []
    if frontend == "benchmark_raw_v1" and prepared:
        contract["target_sequence_length"] = int(prepared[0])
    elif frontend == "benchmark_logmel_v1" and len(prepared) >= 2:
        contract["time_steps"] = int(prepared[0])
        contract["feature_dim"] = int(prepared[1])
        # A janela de análise faz parte da definição da feature: sem gravá-la,
        # a inferência não tem como reproduzir o espectrograma do treino.
        from app.domain.features.benchmark_frontend import resolve_n_fft

        origem = protocol.get("original_shape") or [48000]
        salto = max(64, int(-(-int(origem[0]) // max(int(prepared[0]), 1))))
        contract["n_fft"] = resolve_n_fft(salto, _declared_n_fft(arch))
        contract["hop_length"] = salto
    elif frontend == "benchmark_tabular_v1" and prepared:
        contract["feature_dim"] = int(prepared[0])
    # CROP DE AVALIAÇÃO — só carimba quando JÁ ESTÁ RESOLVIDO.
    #
    # `eval_crop_strategy` nasce como o placeholder "resolved_at_eval"
    # (`_prepare_protocol_splits`) e só vira "multicrop"/"center" no fim da
    # avaliação. Como o contrato era carimbado ANTES disso, os artefatos
    # promovidos guardavam literalmente `train_random_eval_resolved_at_eval`.
    # A inferência decide o multicrop por `"multicrop" in crop_strategy`, e essa
    # string não contém "multicrop": produção rodava 1 crop enquanto o benchmark
    # media com 3 e média dos scores. Gravar o placeholder é pior que não gravar
    # — `_stamp_eval_crop_strategy` completa o campo depois da avaliação.
    estrategia_eval = protocol.get("eval_crop_strategy")
    if estrategia_eval and estrategia_eval != "resolved_at_eval":
        contract["crop_strategy"] = (
            f"train_{protocol.get('train_crop_strategy') or 'center'}"
            f"_eval_{estrategia_eval}"
        )
    # Política de correção de formato, para a inferência reproduzi-la.
    if protocol.get("band_correction"):
        contract["band_correction"] = dict(protocol["band_correction"])
    contract["normalization"] = "per_sample_zscore"

    # Reescreve o sidecar para que o artefato promovido já nasça com o contrato
    # correto, sem depender de um passo de correção posterior.
    try:
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["input_contract"] = contract
            config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        logger.warning(
            "Não foi possível gravar feature_frontend em %s: %s", config_path, exc
        )
    return contract


def _run_neural(
    arch: str,
    cfg: BenchmarkConfig,
    splits,
    tmp: Path,
    models_dir: Path,
    training_seed: int | None = None,
):
    """Treina via TrainingService e devolve callables de inferência."""
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.training_service import TrainingService

    Xtr, ytr, Xv, yv, _Xte, _yte = splits[:6]
    protocol = splits[7] if len(splits) > 7 else {}
    name = f"bench_{_slug(arch)}"
    npz = tmp / "ds.npz"
    np.savez(npz, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv)

    models_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = _architecture_dir(cfg, arch, training_seed)
    svc = TrainingService(models_dir=str(models_dir))
    train_config = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "model_name": name,
        "verbose": 0,
        "progress_log_interval": 1,
        "progress_label": arch,
    }
    train_config.update(cfg.training_overrides.get(arch, {}))
    # O orçamento é um controle experimental, não um hiperparâmetro do modelo.
    train_config["epochs"] = int(cfg.epochs)
    train_config["model_name"] = name
    train_config["verbose"] = 0
    compact = _compact_slug(arch)
    if compact == "rawnet2":
        # Mantém os hiperparâmetros do retreino anterior. No protocolo comum,
        # o aumento interno será desligado após esta seleção porque a cópia
        # ruidosa já foi gerada no domínio da forma de onda.
        train_config.update(
            {
                "learning_rate": 1e-4,
                "use_augmentation": True,
                "use_mixed_precision": False,
            }
        )
        # TOPOLOGIA: baseline ANTI-SPOOFING (Tak et al., 2021), não o
        # "Improved RawNet" de VERIFICAÇÃO DE LOCUTOR (Jung et al., 2020).
        #
        # São arquiteturas diferentes com o mesmo nome. O escopo oficial roda
        # RawNet2 e AASIST como baselines da Track 1 do ASVspoof, e é o de Tak
        # que a literatura de anti-spoofing compara por EER — Sinc 20, blocos
        # [20,20,128,128,128,128], 3×GRU(1024). O `registry.default_params`
        # descreve a variante de locutor, então sem esta promoção o benchmark
        # construía ~7,0M parâmetros da arquitetura errada.
        #
        # Os valores vêm de `NEURAL_BENCHMARK_HPARAMS["rawnet2"]` (via
        # `train_config`), NÃO de um import da arquitetura: a interface Gradio
        # resolve seus defaults por `planning.effective_hyperparameters()`, que
        # lê o mesmo dicionário. Uma topologia que vivesse só aqui faria a
        # interface treinar uma arquitetura e o benchmark outra.
        #
        # ESTE BLOCO PRECISA VIVER NO PRIMEIRO RAMO. Ele nasceu (2026-08-20)
        # como um `elif compact == "rawnet2"` mais abaixo na mesma cadeia — e
        # portanto INALCANÇÁVEL, porque este `if` já casa. O efeito era mudo: o
        # benchmark seguia construindo a variante de locutor enquanto manifesto,
        # plano e tabela de custo declaravam a anti-spoofing.
        model_params = train_config.setdefault("parameters", {})
        for key in (
            "sinc_filters",
            "sinc_kernel_size",
            "res_filters",
            "gru_units",
            "gru_layers",
            "dense_units",
        ):
            if key in train_config:
                model_params[key] = train_config[key]
    elif compact in {"aasist", "rawgatst"}:
        # Compile-respect: LR e CosineDecay pertencem ao construtor.
        model_params = train_config.setdefault("parameters", {})
        for key in (
            "learning_rate",
            "min_learning_rate",
            "decay_steps",
            "dropout_rate",
            "l2_reg_strength",
            "classifier_head",
            # Só o RawGAT-ST declara `global_clipnorm` no plano; o AASIST
            # divide este ramo mas não o expõe no construtor, e a promoção é
            # condicionada a `if key in train_config`, então não o alcança.
            "global_clipnorm",
        ):
            if key in train_config:
                model_params.setdefault(key, train_config[key])
        train_config.pop("learning_rate", None)
        train_config.update(
            {
                "use_augmentation": True,
                "reduce_lr_on_plateau": False,
            }
        )
    elif compact == "efficientnetlstm":
        model_params = train_config.setdefault("parameters", {})
        if "dropout_rate" in train_config:
            model_params.setdefault("dropout_rate", float(train_config["dropout_rate"]))
        if "lstm_units" in train_config:
            model_params.setdefault("lstm_units", int(train_config["lstm_units"]))
        elif "hidden_units" in train_config:
            first_units = str(train_config["hidden_units"]).split("/", 1)[0]
            try:
                model_params.setdefault("lstm_units", int(first_units))
            except ValueError:
                pass
        if "pretrained" in train_config:
            model_params.setdefault("pretrained", bool(train_config["pretrained"]))
        train_config.update({"learning_rate": 1e-4})
    elif compact in {"hybridcnntransformer", "conformer", "spectrogramtransformer"}:
        train_config.update({"reduce_lr_on_plateau": False})
        # Compile-respect: LR/schedule vão ao CONSTRUTOR da arquitetura
        # (WarmupCosineDecay próprio), não ao TrainingConfig — inclusive o
        # CCT (2026-07-14; antes o compile do CCT era hardcoded e o
        # learning_rate/decay_steps do plano nunca chegavam ao modelo).
        model_params = train_config.setdefault("parameters", {})
        for key in (
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "decay_steps",
            "alpha",
            "dropout_rate",
        ):
            if key in train_config:
                model_params.setdefault(key, train_config[key])
        train_config.pop("learning_rate", None)
        if compact in {"conformer", "hybridcnntransformer"}:
            for key in ("clipnorm", "label_smoothing"):
                if key in train_config:
                    model_params.setdefault(key, train_config[key])
        elif compact == "spectrogramtransformer":
            # `pretrained` é parâmetro do CONSTRUTOR do AST (transferência dos
            # pesos AudioSet) — sem promovê-lo para `parameters` a flag do
            # plano não chegaria ao modelo.
            if "pretrained" in train_config:
                model_params.setdefault("pretrained", bool(train_config["pretrained"]))
            # P1 — retreino obrigatório. Liga augmentation (ruído SNR + SpecAug)
            # e força a restauração do MELHOR checkpoint (val_loss, agora
            # GUARDADA — validada no val antes de aceitar), atacando o colapso
            # val→teste por sobreajuste. Hiperparâmetros vêm do plano/defaults
            # (2026-07-14: pre-LN, lr de pico 1e-5, weight_decay 1e-5).
            train_config.update({"use_augmentation": True, "checkpoint_best": True})

    # Controles experimentais comuns; não alteram LR, dropout, regularização,
    # otimizador, scheduler ou batch customizados por arquitetura.
    train_config["calibrate_under_noise"] = False
    if cfg.fixed_epoch_budget:
        train_config["early_stopping"] = False
    # O AWGN científico já foi aplicado à forma de onda antes do frontend.
    # Desativa o AudioAugmenter legado para impedir novo ruído no log-Mel ou
    # em outra representação. SpecAugment/RawBoost são ablações separadas.
    # Exceção (2026-07-15): AASIST/RawGAT-ST usam o AudioAugmenter DINÂMICO
    # na forma de onda no lugar da cópia estática (ver
    # _prepare_protocol_splits) — para eles use_augmentation permanece True.
    if protocol.get("training_augmentation_domain") == "waveform":
        train_config["use_augmentation"] = False
        train_config["waveform_noise_protocol"] = protocol
    elif protocol.get("training_augmentation_domain") == "waveform_dynamic_augmenter":
        train_config["use_augmentation"] = True
        train_config["waveform_noise_protocol"] = protocol

    checkpoint_path = None
    checkpoint_requested = bool(train_config.pop("checkpoint_best", False))
    if cfg.select_best_checkpoint or checkpoint_requested:
        checkpoint_dir = arch_dir / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # `.weights.h5` → ModelCheckpoint salva SÓ os pesos (2026-07-14):
        # o artefato serve apenas à restauração guardada via load_weights;
        # o modelo completo de inferência é salvo separadamente pelo
        # TrainingService (save_inference_keras).
        checkpoint_path = checkpoint_dir / "best_checkpoint.weights.h5"
        train_config["checkpoint_path"] = str(checkpoint_path)
        # Sem esta linha o `checkpoint_monitor` do BenchmarkConfig não chega ao
        # ModelTrainer: o TrainingService descarta toda chave que não seja
        # campo do TrainingConfig, e o trainer lê `self.config.checkpoint_
        # monitor`. O smoke `_smoke_eer` (2026-08-17) gravou `val_loss` no
        # best.json pedindo `val_eer` justamente por causa disso.
        train_config["checkpoint_monitor"] = str(
            getattr(cfg, "checkpoint_monitor", "") or "val_loss"
        )

    # Backup operacional separado do checkpoint de selecao cientifica.
    # Preserva pesos, otimizador e contador de epocas para retomada apos
    # reinicio do host/Docker. E removido automaticamente ao concluir fit().
    backup_dir = arch_dir / "training_backup"
    train_config["backup_dir"] = str(backup_dir)

    res = svc.train_model(
        architecture=arch,
        dataset_path=str(npz),
        config=train_config,
    )
    if res.status != ProcessingStatus.SUCCESS:
        raise RuntimeError(f"train_model falhou: {res.errors}")
    train_data = (res.data.metrics or {}) if res.data else {}
    history = train_data.get("history")
    model = (res.metadata or {}).get("model")
    if model is None:
        raise RuntimeError("TrainingService não retornou o modelo treinado em memória")

    # AJUSTE 2026-07-14 (rigor da avaliação): modelos com saída LINEAR emitem
    # LOGITS crus (ex.: AASIST/AMSoftmax, faixa ≈[-15, 15]). Antes, pred[:, 1]
    # ia direto para _finite_scores, que CLIPA em [0, 1] — quantizando os
    # scores em ~{0, 1} (observado: 4 valores distintos em 2250 predições) e
    # invalidando EER/ROC/min-tDCF, que exigem scores contínuos. Além do clip,
    # o ranking de p_fake é monotônico em (z1 − z0), não em z1 isolado.
    # Normaliza (softmax/sigmoid) ANTES de extrair p_fake.
    # Mesma funcao que alimenta `output_is_logits` no contrato e que a producao
    # consulta — um criterio so, para os dois caminhos nao divergirem.
    from app.domain.services.detection.predictor import model_emits_logits

    _from_logits = bool(model_emits_logits(model))

    def _normalize_probs(pred: np.ndarray) -> np.ndarray:
        if not _from_logits:
            return pred
        if pred.ndim > 1 and pred.shape[-1] > 1:
            z = pred - pred.max(axis=-1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=-1, keepdims=True)
        return 1.0 / (1.0 + np.exp(-pred))

    config_path = models_dir / f"{name}_config.json"
    input_contract: dict[str, Any] = {}
    if config_path.exists():
        try:
            input_contract = json.loads(config_path.read_text(encoding="utf-8")).get(
                "input_contract", {}
            )
        except (OSError, json.JSONDecodeError):
            input_contract = {}

    # CORREÇÃO 2026-07-27 — escala de score coerente com o limiar calibrado.
    #
    # O `eer_threshold` do contrato é derivado no conjunto de VALIDAÇÃO sobre
    # probabilidades COM temperature scaling (trainer._compute_eer_threshold), e
    # `auto_calibrate_temperature` é True por padrão. O benchmark aplicava esse
    # limiar a scores SEM temperatura: como o scaling é monótono, o EER não muda,
    # mas o PONTO DE OPERAÇÃO se desloca — `accuracy_at_calibrated_threshold`,
    # justamente a métrica não-oráculo, ficava medida no limiar errado.
    #
    # Aplicar T aqui também alinha o benchmark ao detector implantado (o
    # Predictor aplica a mesma T) e faz o ECE medir a calibração do sistema
    # entregue, não a da saída crua. EER/AUC/min-tDCF não mudam (transformação
    # monótona); `accuracy` no limiar fixo 0,5 pode mudar, e deve — é a decisão
    # que o sistema realmente toma.
    _temperature = input_contract.get("temperature")
    try:
        _temperature = float(_temperature) if _temperature is not None else 1.0
    except (TypeError, ValueError):
        _temperature = 1.0
    if not np.isfinite(_temperature) or _temperature <= 0:
        _temperature = 1.0

    def _apply_temperature(probs: np.ndarray) -> np.ndarray:
        if _temperature == 1.0:
            return probs
        from app.domain.services.detection.predictor import (
            apply_temperature_scaling,
        )

        return np.asarray(
            apply_temperature_scaling(probs, _temperature), dtype="float64"
        )

    def predict_p_fake(X: np.ndarray) -> np.ndarray:
        pred = model.predict(np.asarray(X, dtype="float32"), verbose=0)
        pred = _normalize_probs(np.asarray(pred, dtype="float64"))
        pred = _apply_temperature(pred)
        if pred.ndim > 1 and pred.shape[-1] > 1:
            return pred[:, 1]
        return pred.reshape(-1)

    def predict_fn(xb):  # latência: forward puro do modelo
        return model.predict(np.asarray(xb, dtype="float32"), verbose=0)

    model_path = models_dir / f"{name}.keras"
    reported_training_config = dict(train_data.get("training_config") or train_config)
    model_parameters = dict(train_config.get("parameters") or {})
    for key in (
        "epochs",
        "batch_size",
        "use_augmentation",
        "use_mixed_precision",
        "reduce_lr_on_plateau",
        "checkpoint_path",
    ):
        if key in train_config and reported_training_config.get(key) is None:
            reported_training_config[key] = train_config[key]
    if checkpoint_path is not None:
        reported_training_config["checkpoint_best"] = True
        reported_training_config["best_checkpoint_path"] = str(checkpoint_path)
        # Qual métrica elegeu a época. O TrainingService filtra as chaves que
        # não são campos de TrainingConfig, então o valor precisa ser
        # reafirmado aqui para chegar ao metrics.json — é ele que o
        # diagnóstico de estabilidade e o rodapé do TCC consultam.
        reported_training_config["checkpoint_monitor"] = str(
            getattr(cfg, "checkpoint_monitor", "") or "val_loss"
        )
    if model_parameters:
        if "learning_rate" in model_parameters:
            reported_training_config["learning_rate"] = model_parameters[
                "learning_rate"
            ]
        reported_training_config["model_parameters"] = model_parameters
    # A temperatura faz parte da definição do score avaliado — sem registrá-la,
    # os números não são reproduzíveis a partir do modelo salvo.
    reported_training_config["calibrated_temperature"] = float(_temperature)
    reported_training_config["scores_temperature_scaled"] = bool(_temperature != 1.0)

    input_contract = _stamp_benchmark_frontend(
        config_path, input_contract, protocol, arch
    )

    checkpoint_restore = dict(train_data.get("checkpoint_restore") or {})
    if checkpoint_restore.get("attempted") and not checkpoint_restore.get("restored"):
        # Não aborta: os pesos da última época ainda são um resultado, e matar
        # um treino de horas na última linha seria pior. Mas o run.log precisa
        # dizer, na altura em que alguém lê, que este artefato NÃO é o
        # checkpoint selecionado — o aviso equivalente do guard fica a dezenas
        # de milhares de linhas de distância.
        logger.warning(
            "[%s] o checkpoint eleito por %s foi DESCARTADO na restauração: o "
            "artefato avaliado tem os pesos da ÚLTIMA ÉPOCA. O resultado não "
            "corresponde ao critério de seleção declarado — retreine antes de "
            "publicar.",
            arch,
            checkpoint_restore.get("monitor", "?"),
        )

    run_artifact, artifact_fingerprint = _preserve_run_artifact(model_path, arch_dir)

    return {
        "predict_p_fake": predict_p_fake,
        "predict_fn": predict_fn,
        # Necessário para RECARIMBAR o contrato depois da avaliação, quando a
        # estratégia de crop deixa de ser o placeholder "resolved_at_eval".
        #
        # SÃO DUAS CÓPIAS. `config_path` é o sidecar GLOBAL
        # (`data/models/bench_<arch>_config.json`), que a inferência carrega;
        # `run_config_path` é a cópia dentro do run, feita por
        # `_preserve_run_artifact` ANTES da avaliação — e é ela que
        # `sync_completed_benchmark_artifacts.py` promove para
        # `data/models/benchmark_final/`. Recarimbar só a global deixava o
        # artefato PROMOVIDO sem `crop_strategy`, e a inferência caía em 1 crop
        # enquanto o benchmark mediu com 3.
        "config_path": config_path,
        "run_config_path": (
            run_artifact.with_name(f"{run_artifact.stem}_config.json")
            if run_artifact is not None
            else None
        ),
        "params": count_params(model),
        "size_mb": file_size_mb(model_path),
        "history": history,
        "training_config": reported_training_config,
        "model_parameters": model_parameters,
        "final_metrics": train_data.get("final_metrics") or {},
        "input_contract": input_contract,
        "model_artifact": str(run_artifact or model_path),
        "model_artifact_shared_copy": str(model_path),
        "model_artifact_fingerprint": artifact_fingerprint,
        # Contraparte do bloco declarado no caminho clássico: o neural ajusta só
        # no treino e reserva a validação para escolher a época. Sem os dois
        # lados declarados, o `fit_samples` de 25.780 dos clássicos contra os
        # 24.324 dos neurais parecia divergência de dados, não de protocolo.
        "fit_strategy": {
            "kind": "fixed_epoch_budget_then_checkpoint_selection",
            "estimator": "keras",
            "fit_splits": ["train"],
            "fit_samples": int(len(ytr)),
            # DERIVADO do monitor real. O literal "menor val_loss limpa" que
            # ficava aqui contradizia os runs com `--checkpoint-monitor
            # val_eer`: o metrics.json declarava perda enquanto o
            # ModelCheckpoint selecionava por EER.
            "validation_role": (
                "seleção de checkpoint (menor "
                f"{str(getattr(cfg, 'checkpoint_monitor', '') or 'val_loss')}"
                " limpa)"
            ),
            "checkpoint_monitor": str(
                getattr(cfg, "checkpoint_monitor", "") or "val_loss"
            ),
            # QUAIS pesos foram avaliados. Um artefato que declara seleção por
            # val_eer mas guarda os pesos da última época não é um detalhe de
            # implementação: é o resultado sob outro protocolo.
            "checkpoint_restore": checkpoint_restore,
            "val_samples": int(len(yv)),
        },
    }


def _classical_input_contract(
    arch: str,
    models_dir: Path,
    name: str,
    n_features: int,
    protocol: Dict[str, Any],
    predict_p_fake: Callable,
    Xv: np.ndarray,
    yv: np.ndarray,
) -> Dict[str, Any]:
    """Monta e grava o sidecar de inferência de um modelo clássico.

    Espelha o que o `TrainingService` já fazia pelos neurais: front-end do
    benchmark, forma da entrada e limiar de EER derivado da VALIDAÇÃO. Sem
    isso, SVM/RandomForest chegavam à produção sem contrato algum — a
    inferência não reproduzia o vetor tabular e decidia sempre em 0,5.

    Desde 2026-08-09 a validação está FORA do ajuste (ver `_run_classical`),
    então o limiar aqui é genuinamente held-out; antes vinha de dados que o
    modelo já tinha visto.
    """
    from app.domain.features.benchmark_frontend import (
        DEFAULT_SAMPLE_RATE,
        DEFAULT_SOURCE_SAMPLES,
        FRONTEND_TABULAR,
        N_TABULAR_FEATURES,
        N_TABULAR_FEATURES_V2,
        frontend_for_input_type,
    )

    # `tabular_audio_features` = o vetor SAIU do front-end do benchmark (a
    # entrada era forma de onda). Só nesse caso o contrato pode alegar paridade
    # com o treino — e aí a largura tem de ser uma das duas conhecidas, senão o
    # erro precisa aparecer aqui e não em produção.
    #
    # `tabular_flattened` = o NPZ já continha features prontas e o front-end
    # não rodou. Declarar `benchmark_tabular_*` nesse caso é falso: a inferência
    # calcularia 183 descritores para um modelo ajustado noutro espaço. Fica sem
    # front-end declarado, com o motivo registrado.
    from_benchmark_frontend = protocol.get("input_type") == "tabular_audio_features"
    if from_benchmark_frontend and int(n_features) not in (
        N_TABULAR_FEATURES,
        N_TABULAR_FEATURES_V2,
    ):
        raise RuntimeError(
            f"[{arch}] vetor tabular com {n_features} colunas — esperado "
            f"{N_TABULAR_FEATURES} (v1) ou {N_TABULAR_FEATURES_V2} (v2)"
        )

    contract: Dict[str, Any] = {
        "architecture": arch,
        "type": "features",
        "format": "tabular",
        "input_type": "tabular",
        # Derivado da LARGURA, não do `input_type`: os dois front-ends tabulares
        # compartilham o mesmo `input_type`, e um contrato que declarasse v2
        # sobre um vetor de 63 colunas mandaria a inferência preparar 183.
        "feature_frontend": (
            (
                frontend_for_input_type("tabular")
                if int(n_features) == N_TABULAR_FEATURES_V2
                else FRONTEND_TABULAR  # o v1, para artefatos de 63 colunas
            )
            if from_benchmark_frontend
            else None
        ),
        "input_shape": [int(n_features)],
        "feature_dim": int(n_features),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "source_samples": int(
            (protocol.get("original_shape") or [DEFAULT_SOURCE_SAMPLES])[0]
        ),
        # O artefato é um Pipeline sklearn com o scaler DENTRO; a inferência não
        # deve aplicar normalização externa nenhuma.
        "normalization": "pipeline_interno",
        "scaler_applied": False,
        "feature_frontend_reason": (
            "vetor produzido pelo front-end tabular do benchmark"
            if from_benchmark_frontend
            else "NPZ ja continha features; sem paridade com o front-end do "
            "benchmark, entao nenhum e declarado"
        ),
        # `predict_proba` já devolve probabilidade: não há logit para escalar.
        "temperature": 1.0,
        "label_classes": [0, 1],
    }
    try:
        from app.domain.models.training.metrics import MetricsCalculator

        scores = _finite_scores(predict_p_fake(Xv))
        y_val = np.asarray(yv).ravel().astype(int)
        if len(np.unique(y_val)) > 1:
            eer, thr = MetricsCalculator().calculate_eer(y_val, scores)
            if np.isfinite(thr):
                contract["eer_threshold"] = float(thr)
                contract["eer_value"] = float(eer)
                contract["threshold_source"] = "validation_eer"
    except Exception as exc:  # noqa: BLE001 — contrato sem limiar ainda serve
        logger.warning("[%s] limiar de validação indisponível: %s", arch, exc)

    config_path = models_dir / f"{name}_config.json"
    try:
        config_path.write_text(
            json.dumps(
                {
                    "architecture": arch,
                    "model_type": "classical",
                    "input_shape": [int(n_features)],
                    "num_classes": 2,
                    "label_classes": [0, 1],
                    "input_contract": contract,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
    except OSError as exc:  # noqa: BLE001
        logger.warning("Não foi possível gravar %s: %s", config_path, exc)
    return contract


def _run_classical(
    arch: str,
    cfg: BenchmarkConfig,
    splits,
    tmp: Path,
    models_dir: Path,
    training_seed: int | None = None,
    fit_context: Dict[str, np.ndarray] | None = None,
):
    """Treina um modelo clássico (SVM/RF) diretamente (sklearn).

    ``fit_context`` traz ``train_cluster_ids`` — os grupos locutor × frase das
    amostras de treino — para que a busca de hiperparâmetros use CV agrupada.
    """
    train_seed = int(cfg.seed if training_seed is None else training_seed)
    fit_context = fit_context or {}
    Xtr, ytr, Xv, yv, _Xte, _yte = splits[:6]
    clean_train_count = int(splits[6]) if len(splits) > 6 else len(ytr)
    protocol = splits[7] if len(splits) > 7 else {}
    n_features = int(np.asarray(Xtr).reshape(len(Xtr), -1).shape[1])

    from app.domain.models.architectures.classical_ml_helpers import (
        unwrap_calibrated,
    )

    if "svm" in arch.lower():
        from app.domain.models.architectures.svm import create_svm_model as factory
    else:
        from app.domain.models.architectures.random_forest import (
            create_random_forest_model as factory,
        )
    if os.getenv("XFAKE_BENCHMARK_VERBOSE", "0") != "1":
        for logger_name in (
            "app.domain.models.architectures.svm",
            "app.domain.models.architectures.random_forest",
            "app.domain.models.architectures.classical_ml_helpers",
        ):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.disabled = True
    name = f"bench_{_slug(arch)}"
    models_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = _architecture_dir(cfg, arch, training_seed)

    X_train_2d = np.asarray(Xtr).reshape(len(Xtr), -1)
    y_train = np.asarray(ytr).ravel()

    # ASSIMETRIA REMOVIDA (2026-08-09): a validação SAIU do ajuste.
    #
    # Até aqui os clássicos ajustavam em treino+validação — declarado no
    # artefato, mas com duas consequências. (1) O n efetivo divergia das
    # neurais, que ajustam só no treino e reservam a validação para escolher a
    # época. (2) Pior: o `eer_threshold` gravado no contrato de inferência sai
    # justamente desse val (`_classical_input_contract`), ou seja, era um
    # limiar IN-SAMPLE — derivado de dados que o modelo já tinha visto. Agora o
    # ajuste usa só o treino (limpo + cópia AWGN) e a validação fica intacta
    # para calibrar a saída e fixar o ponto de operação.
    X_fit_2d = X_train_2d
    y_fit = y_train

    # Compatibilidade legada, desativada por padrão. A comparação científica
    # usa exclusivamente a cópia ruidosa produzida na forma de onda em
    # _prepare_protocol_splits; ruído direto no vetor tabular não é AWGN acústico.
    aug_snrs = []
    if (
        getattr(cfg, "classical_noise_augmentation", False)
        and protocol.get("training_augmentation_domain") != "waveform"
    ):
        aug_snrs = list(cfg.snr_levels_db)
    if aug_snrs:
        extra_X = [
            BenchmarkData.add_awgn(X_fit_2d, snr, seed=train_seed + i)
            for i, snr in enumerate(aug_snrs)
        ]
        X_fit_2d = np.concatenate([X_fit_2d, *extra_X], axis=0)
        y_fit = np.concatenate([y_fit] * (1 + len(aug_snrs)), axis=0)
        logging.getLogger("benchmark").info(
            "[%s] augmentation clássico: +%d cópias ruidosas (SNRs=%s) → %d amostras",
            arch,
            len(aug_snrs),
            aug_snrs,
            len(y_fit),
        )

    # Grupos da CV: os cluster_ids do treino, repetidos uma vez por bloco.
    # `X_fit_2d` é [limpo | cópia AWGN], e a cópia k da amostra i ocupa a
    # posição n_clean·(k+1)+i — a mesma frase do mesmo locutor. Sem repetir o
    # grupo junto, o par limpo/ruidoso da MESMA amostra cairia em dobras
    # diferentes: vazamento ainda mais direto que o do par real/clone.
    train_clusters = fit_context.get("train_cluster_ids")
    cv_groups = None
    if train_clusters is not None and clean_train_count:
        blocks, rest = divmod(len(y_fit), int(clean_train_count))
        if rest == 0 and len(train_clusters) == clean_train_count:
            cv_groups = np.tile(np.asarray(train_clusters).ravel(), blocks)
        else:
            logging.getLogger("benchmark").warning(
                "[%s] cluster_ids do treino não cobrem o conjunto de ajuste "
                "(%d grupos, %d amostras, %d limpas) — CV cai para não agrupada",
                arch,
                len(train_clusters),
                len(y_fit),
                clean_train_count,
            )

    tuning = {"enabled": False, "status": "disabled"}
    model_kwargs: dict[str, Any] = {}
    if cfg.optimize_hyperparameters:
        # REGIME DA CV = REGIME DO AJUSTE (2026-08-09). Antes a busca rodava só
        # sobre as amostras LIMPAS (`X_train_2d[:clean_train_count]`) e o
        # modelo final era ajustado sobre limpo+ruidoso: os hiperparâmetros
        # eram escolhidos num regime em que o modelo nunca opera, justamente o
        # que o protocolo mede a 10 e 5 dB.
        tuning = _run_classical_tuning(
            arch=arch,
            X=X_fit_2d,
            y=y_fit,
            output_dir=arch_dir,
            seed=train_seed,
            groups=cv_groups,
        )
        if tuning.get("status") == "ok":
            model_kwargs.update(tuning.get("best_model_params") or {})

    # A semente da REPETIÇÃO precisa alcançar o estimador: sem isto o
    # `random_state` fica preso ao default (42) e as N execuções de SVM/RF
    # produzem resultados IDÊNTICOS — desvio zero, repetição sem informação.
    model_kwargs.setdefault("random_state", train_seed)

    # CALIBRAÇÃO LIGADA (2026-08-09). `wrap_calibration` existia desde sempre
    # com `calibrate=False`, e o benchmark nunca a acionou. Os dois clássicos
    # fecharam o `clean_benchmark_15k` com os PIORES ECE do escopo oficial
    # (RandomForest 0,1212; SVM 0,0945 — o terceiro pior é 0,0416), e o
    # protocolo decide em limiar FIXO de 0,5: probabilidade mal calibrada vira
    # erro de classificação direto. É o que a 5 dB levava o SVM a recall
    # 0,0000 com AUC 0,849 — a ordenação sobrevivia, o ponto de operação não.
    #
    # Isotônica, e não Platt: as duas famílias têm distorção não monotônica em
    # forma de S (RF por média de votos de árvore, SVM por Platt interno sobre
    # margem), que a sigmoide de Platt não corrige. O custo é neutro ou menor —
    # com `calibrate=True` o `_create_pipeline` do SVM desliga o
    # `probability=True` do SVC (cujo Platt interno é 5-fold) e a calibração
    # externa passa a usar `decision_function` com cv=3.
    model_kwargs.setdefault("calibrate", True)

    # DOBRAS DA CALIBRAÇÃO = DOBRAS DA BUSCA.
    #
    # `wrap_calibration` usava o default `cv=3`, um inteiro — que o sklearn
    # interpreta como StratifiedKFold SIMPLES, sem grupos. Enquanto o grid
    # search já rodava agrupado (`StratifiedGroupKFold` por locutor×frase), o
    # `cross_val_predict` interno da calibração via, para cada amostra retida, a
    # cópia AWGN da MESMA gravação e o par real/clone do MESMO enunciado. A
    # isotônica é monótona, então AUC e EER não mudam; o que sai enviesado é o
    # ponto de operação — acurácia@0,5, F1 e ECE, exatamente as colunas da
    # tabela principal e a métrica que motivou ligar a calibração.
    if cv_groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            n_grupos = int(len(np.unique(cv_groups)))
            n_dobras = max(2, min(3, n_grupos))
            model_kwargs["calibration_cv"] = list(
                StratifiedGroupKFold(
                    n_splits=n_dobras, shuffle=True, random_state=train_seed
                ).split(X_fit_2d, y_fit, groups=cv_groups)
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("benchmark").warning(
                "[%s] não foi possível agrupar a CV da calibração (%s); "
                "cai para o cv=3 não agrupado",
                arch,
                exc,
            )

    model = factory(input_shape=(n_features,), num_classes=2, **model_kwargs)
    model.fit(X_fit_2d, y_fit)

    path = models_dir / f"{name}.pkl"
    try:
        model.save(str(path))
    except Exception:
        path = None

    def predict_p_fake(X: np.ndarray) -> np.ndarray:
        X2 = np.asarray(X).reshape(len(X), -1)
        proba = model.predict_proba(X2)
        return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()

    def predict_fn(xb):
        return model.predict_proba(np.asarray(xb).reshape(len(xb), -1))

    # Score de ORDENAÇÃO do detector, antes da calibração (2026-08-09).
    #
    # A isotônica é uma função escada: no SVM do `clean_benchmark_15k` ela
    # colapsou 1.382 margens distintas em 52 degraus, e os empates custaram
    # 0,75 pp de AUC (0,9731 no bruto contra 0,9656 no calibrado). AUC, EER e
    # min t-DCF medem ORDENAÇÃO — devem ver a margem, não o degrau. A decisão
    # em 0,5 e o ECE continuam sobre a probabilidade calibrada, que é o que a
    # calibração existe para consertar.
    #
    # Devolve `None` quando não há calibração: aí `p_fake` já É o score do
    # detector e passar os dois seria declarar uma separação que não existe.
    def _raw_ranking_score():
        if not model_kwargs.get("calibrate"):
            return None
        pipeline = getattr(model, "pipeline", None)
        if pipeline is None:
            return None
        final = pipeline.steps[-1][1]
        inner = unwrap_calibrated(final)
        if inner is final:  # não é CalibratedClassifierCV
            return None

        def _score(X: np.ndarray) -> np.ndarray:
            Xt = np.asarray(X).reshape(len(X), -1)
            for _nome, passo in pipeline.steps[:-1]:
                Xt = passo.transform(Xt)
            if hasattr(inner, "decision_function"):
                return np.asarray(inner.decision_function(Xt)).ravel()
            proba = inner.predict_proba(Xt)
            return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()

        return _score

    predict_ranking = _raw_ranking_score()

    # Contrato de inferência dos clássicos.
    #
    # Até 2026-07-28 o caminho clássico salvava só o `.pkl`: nenhum sidecar,
    # portanto NENHUM `input_contract` — sem `feature_frontend` (a inferência
    # não roteava para o front-end tabular do benchmark) e sem limiar
    # calibrado (o app decidia sempre em 0,5). Os neurais já saíam com os dois.
    # O limiar sai do conjunto de VALIDAÇÃO, nunca do teste.
    classical_contract = _classical_input_contract(
        arch, models_dir, name, n_features, protocol, predict_p_fake, Xv, yv
    )

    fit_strategy = {
        "kind": "single_fit",
        "estimator": "sklearn",
        "fit_samples": int(len(y_fit)),
        "n_features": n_features,
        # ASSIMETRIA REMOVIDA (2026-08-09): era ["train", "val"]. Os clássicos
        # ajustavam também na validação — n efetivo diferente do das neurais e,
        # pior, o limiar do contrato saía desse mesmo val, portanto in-sample.
        # Agora ajustam só no treino (limpo + cópia AWGN), como as neurais, e a
        # validação fica para calibrar e fixar o ponto de operação.
        "fit_splits": ["train"],
        "validation_role": (
            "held-out: calibração isotônica e limiar de EER do contrato "
            "(sem seleção de checkpoint); a busca de hiperparâmetros usa CV "
            "agrupada por cluster sobre o MESMO conjunto do ajuste"
            if len(yv)
            else "indisponível"
        ),
        "probability_calibration": {
            "applied": bool(model_kwargs.get("calibrate")),
            "method": "isotonic",
            "wrapper": "sklearn.calibration.CalibratedClassifierCV",
            "ensemble": False,
            "fitted_on": "predições out-of-fold do conjunto de ajuste",
        },
        "clean_train_samples": int(clean_train_count),
        "val_samples": int(len(yv)),
    }
    if tuning.get("enabled"):
        fit_strategy.update(
            {
                "kind": "grid_search_cv_then_refit",
                "cv": tuning.get("cv"),
                "n_candidates": tuning.get("n_candidates"),
                "n_fits": (
                    int(tuning.get("cv", 0)) * int(tuning.get("n_candidates", 0))
                    if tuning.get("cv") and tuning.get("n_candidates")
                    else None
                ),
                "final_refit": True,
                "scoring": tuning.get("scoring"),
            }
        )
        if fit_strategy.get("n_fits"):
            fit_strategy["total_fit_calls_estimate"] = int(fit_strategy["n_fits"]) + 1

    run_artifact, artifact_fingerprint = _preserve_run_artifact(path, arch_dir)

    return {
        "predict_p_fake": predict_p_fake,
        "predict_ranking": predict_ranking,
        "predict_fn": predict_fn,
        "params": None,
        "size_mb": file_size_mb(path) if path else None,
        "history": None,
        "epochs": None,
        "fit_strategy": fit_strategy,
        "training_config": {
            "model_family": "classical",
            "batch_size": None,
            "epochs": None,
            "fit_strategy": fit_strategy,
            "fit_samples": int(len(y_fit)),
            "n_features": n_features,
            "hyperparameter_tuning": tuning,
            "best_hyperparameters": tuning.get("best_model_params") or model_kwargs,
        },
        "final_metrics": {
            "fit_samples": int(len(y_fit)),
            "n_features": n_features,
            "classes": [int(v) for v in np.unique(y_fit).tolist()],
            "hyperparameter_tuning_status": tuning.get("status"),
            "hyperparameter_tuning_best_score": tuning.get("best_score"),
            "hyperparameter_tuning_best_params": tuning.get("best_model_params"),
        },
        "model_artifact": str(run_artifact or path) if (run_artifact or path) else None,
        "model_artifact_shared_copy": str(path) if path else None,
        "model_artifact_fingerprint": artifact_fingerprint,
        "input_contract": classical_contract,
        "hyperparameter_tuning": tuning,
    }


#: Métricas cujo agregado entre repetições é reportado como média ± desvio.
_SEED_AGGREGATED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "eer",
    "min_tdcf",
    # `accuracy_at_eer_oracle` é a coluna Acur.@EER das tabelas; sem estar aqui,
    # ela seria a da primeira semente enquanto as vizinhas mostram a média.
    "ece",
    "accuracy_at_eer",
    "accuracy_at_eer_oracle",
    "accuracy_at_calibrated_threshold",
)


def _aggregate_metric_block(blocks: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Média entre repetições + desvio-padrão amostral por métrica."""
    base = dict(blocks[0])
    if len(blocks) == 1:
        return base
    for metric in _SEED_AGGREGATED_METRICS:
        values = [
            float(b[metric])
            for b in blocks
            if isinstance(b.get(metric), (int, float)) and np.isfinite(b[metric])
        ]
        if not values:
            continue
        base[metric] = float(np.mean(values))
        # ddof=1: desvio AMOSTRAL — estamos estimando a variabilidade do
        # procedimento de treino a partir de N execuções, não descrevendo uma
        # população completa.
        base[f"{metric}_seed_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        base[f"{metric}_seed_values"] = values
    base["n_seeds"] = len(blocks)
    return base


def _aggregate_seed_runs(runs: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Consolida N repetições da MESMA arquitetura em um resultado único.

    Rigor acadêmico: com uma execução por modelo, o benchmark não distingue
    "arquitetura A é melhor que B" de "esta execução de A foi melhor". As
    métricas viram média ± desvio entre sementes de treino, e cada execução
    fica preservada em `seed_runs` para auditoria.

    O artefato PROMOVIDO é o da PRIMEIRA semente — nunca o da melhor. Escolher
    a melhor execução pelo teste seria seleção no conjunto de teste.
    """
    if len(runs) == 1:
        return runs[0]

    ok_runs = [r for r in runs if r.get("status") == "ok"]
    if not ok_runs:
        base = dict(runs[0])
        base["seed_runs"] = runs
        base["n_seeds"] = len(runs)
        return base

    # A primeira execução BEM-SUCEDIDA é a representativa (artefato promovido).
    base = dict(ok_runs[0])
    base["clean"] = _aggregate_metric_block([r["clean"] for r in ok_runs])

    robustness_keys = sorted({k for r in ok_runs for k in (r.get("robustness") or {})})
    if robustness_keys:
        aggregated_rob: Dict[str, Any] = {}
        for key in robustness_keys:
            blocks = [
                r["robustness"][key]
                for r in ok_runs
                if isinstance((r.get("robustness") or {}).get(key), dict)
            ]
            if blocks:
                aggregated_rob[key] = _aggregate_metric_block(blocks)
        base["robustness"] = aggregated_rob

    base["n_seeds"] = len(ok_runs)
    base["training_seeds"] = [r.get("training_seed") for r in ok_runs]
    base["promoted_artifact_seed"] = ok_runs[0].get("training_seed")
    # `clean`/`robustness` são a MÉDIA entre repetições, mas `scores_clean`,
    # `scores_robustness`, `history` e `efficiency` vêm de `ok_runs[0]`. Ou
    # seja: as tabelas mostram a média e as figuras (ROC, DET, matriz de
    # confusão) e os CSVs de predição mostram UMA execução. Sem este campo,
    # nada no artefato dizia qual — e as legendas não podiam declarar.
    base["scores_seed"] = ok_runs[0].get("training_seed")
    base["seed_runs"] = [
        {
            "training_seed": r.get("training_seed"),
            "status": r.get("status"),
            "clean": r.get("clean"),
            "robustness": r.get("robustness"),
            # O runner grava `wall_time_s` (não `duration_sec`): a chave errada
            # deixava o tempo de cada repetição sempre nulo em `seed_runs`.
            "wall_time_s": r.get("wall_time_s"),
            # MOTIVO DA FALHA (2026-08-18). `_benchmark_one` ja devolve
            # `error` quando a repeticao levanta, mas esta projecao copiava
            # cinco campos e deixava esse de fora -- entao o artefato
            # registrava `status: "error"` sem uma palavra sobre o porque.
            # Observado no estudo de sementes do Conformer: a semente 44
            # falhou em 50,8 s, `n_seeds` caiu de 3 para 2, e nao havia como
            # saber depois o que aconteceu. `None` nas repeticoes que deram
            # certo mantem o schema estavel.
            "error": r.get("error"),
        }
        for r in runs
    ]
    return base


def _benchmark_one(
    arch: str,
    cfg: BenchmarkConfig,
    raw_splits,
    eval_context: Dict[str, np.ndarray] | None = None,
    training_seed: int | None = None,
    fit_context: Dict[str, np.ndarray] | None = None,
    data: Any = None,
) -> Dict[str, Any]:
    """Treina e avalia uma arquitetura com perturbações no áudio canônico.

    `training_seed` controla a aleatoriedade de TREINO (inicialização, dropout,
    ordem de batch, crop e ruído de augmentation). O split e o ruído de
    AVALIAÇÃO permanecem presos a `cfg.seed`, para que todas as repetições
    sejam medidas no mesmo teste e nas mesmas condições.
    """
    train_seed = int(cfg.seed if training_seed is None else training_seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(train_seed)
    except Exception:  # noqa: BLE001 — ambiente clássico sem TF
        import random

        random.seed(train_seed)
        np.random.seed(train_seed)
    raw_Xte, raw_yte = raw_splits[4], raw_splits[5]
    t0 = time.time()
    eval_context = eval_context or {}
    cluster_ids = eval_context.get("cluster_ids")
    source_ids = eval_context.get("source_ids")
    generator_ids = eval_context.get("generator_ids")
    speaker_ids = eval_context.get("speaker_ids")
    models_dir = _models_dir(cfg, arch)
    try:
        splits = _prepare_protocol_splits(
            arch, cfg, raw_splits, training_seed=train_seed, data=data
        )
        _Xtr, _ytr, _Xv, _yv, Xte, yte = splits[:6]
        # Mesma instância (não cópia): `_run_neural` guarda este dict em
        # `training_config.waveform_noise_protocol`, e os campos de avaliação só
        # são resolvidos mais abaixo — com uma cópia, aquela via ficaria com os
        # valores provisórios.
        protocol = splits[7]
        with tempfile.TemporaryDirectory(prefix="bench_") as td:
            tmp = Path(td)
            is_classical = _is_classical_arch(arch)
            if is_classical:
                r = _run_classical(
                    arch,
                    cfg,
                    splits,
                    tmp,
                    models_dir,
                    training_seed=train_seed,
                    fit_context=fit_context,
                )
            else:
                r = _run_neural(
                    arch, cfg, splits, tmp, models_dir,
                    training_seed=train_seed,
                )
            predict_p_fake: Callable = r["predict_p_fake"]

            # Multicrop de avaliação para TODAS as arquiteturas de áudio bruto.
            #
            # Até 2026-07-27 valia só para AASIST/RawGAT-ST: média de 3 crops
            # nesses dois, crop central único em RawNet2/WavLM/HuBERT. Média
            # sobre crops reduz a variância do score e melhora EER/AUC — ou
            # seja, duas arquiteturas competiam na tabela principal com
            # test-time augmentation e as demais sem. Diferente do augmentation
            # dinâmico de treino, que ao menos está atrás de
            # `architecture_specific_augmentation` e é declarado como ablação,
            # essa assimetria não tinha flag nem registro.
            #
            # Quando a janela canônica cobre o sinal inteiro, os 3 crops
            # coincidem e a média devolve exatamente o score do crop único
            # (`raw_audio_multicrop_batch` repete o crop) — nenhuma arquitetura
            # é penalizada pela uniformização.
            use_multicrop = protocol.get("input_type") == "raw_audio"

            def predict_eval(
                prepared: np.ndarray,
                raw_waveforms: Optional[np.ndarray] = None,
            ) -> np.ndarray:
                if not use_multicrop or raw_waveforms is None:
                    return _finite_scores(predict_p_fake(prepared))
                from app.domain.features.benchmark_frontend import (
                    raw_audio_multicrop_batch,
                )

                crops = raw_audio_multicrop_batch(
                    raw_waveforms,
                    target_len=int(np.asarray(Xte).shape[1]),
                    num_crops=3,
                )
                n_samples, n_crops = crops.shape[:2]
                flat_crops = crops.reshape(n_samples * n_crops, *crops.shape[2:])
                crop_scores = _finite_scores(predict_p_fake(flat_crops))
                return crop_scores.reshape(n_samples, n_crops).mean(axis=1)

            # Score de ORDENAÇÃO, quando o modelo aplica calibração pós-hoc.
            # Só os clássicos calibram, e eles nunca fazem multicrop (a entrada
            # é tabular, não forma de onda), então não há caminho de crops aqui.
            # Sem `_finite_scores`: ele recorta em [0,1], o que destruiria a
            # ordenação de um `decision_function` centrado em zero.
            predict_ranking: Optional[Callable] = r.get("predict_ranking")

            def ranking_eval(
                prepared: np.ndarray,
                raw_waveforms: Optional[np.ndarray] = None,
            ) -> Optional[np.ndarray]:
                if predict_ranking is None:
                    return None
                return np.asarray(predict_ranking(prepared), dtype="float64").ravel()

            # O bloco de protocolo publicado precisa refletir a avaliação que
            # de fato ocorreu. Antes ele declarava `multicrop`/3 crops para toda
            # arquitetura raw-audio, inclusive as que rodavam com crop central.
            multicrop_effective = bool(use_multicrop and raw_Xte is not None)
            protocol["eval_crop_strategy"] = (
                "multicrop" if multicrop_effective else "center"
            )
            protocol["eval_num_crops"] = 3 if multicrop_effective else 1
            protocol["eval_score_aggregation"] = (
                "mean_over_crops" if multicrop_effective else "single_crop"
            )
            # Agora que a estratégia está RESOLVIDA, completa o contrato do
            # artefato. `_stamp_benchmark_frontend` roda antes da avaliação e
            # deixa `crop_strategy` de fora justamente para não gravar o
            # placeholder; sem este passo, a inferência nunca ativaria o
            # multicrop que o benchmark usou para medir.
            for _caminho in (r.get("config_path"), r.get("run_config_path")):
                _stamp_eval_crop_strategy(
                    _caminho, r.get("input_contract"), protocol
                )

            n_boot = int(getattr(cfg, "bootstrap_ci_samples", 0) or 0)
            pf_clean = predict_eval(Xte, raw_Xte)
            rank_clean = ranking_eval(Xte, raw_Xte)
            calibrated_threshold = (r.get("input_contract") or {}).get("eer_threshold")
            clean = evaluate_scores(
                yte,
                pf_clean,
                threshold=cfg.decision_threshold,
                n_bootstrap=n_boot,
                cluster_ids=cluster_ids,
                calibrated_threshold=calibrated_threshold,
                ranking_scores=rank_clean,
            )
            grouped_clean: Dict[str, Any] = {}
            if source_ids is not None:
                grouped_clean["source"] = evaluate_grouped_scores(
                    yte, pf_clean, source_ids, threshold=cfg.decision_threshold
                )
            # ACRÉSCIMO 2026-08-09: o agrupamento por LOCUTOR é o único que
            # informa neste dataset. `source` colapsa em 1 grupo (fonte única,
            # `ptpair`) e `generator` em 2, que são as próprias classes
            # (bonafide/xtts_v2) — nenhum dos dois mede dispersão. O protocolo é
            # speaker-disjoint (34 treino / 11 val / 11 teste, zero overlap),
            # então o pior locutor é a leitura honesta da generalização: no
            # `clean_benchmark_15k` o agregado de 95,88% do RawNet2 esconde
            # 74,2% em M026, e o de 93,92% do HuBERT esconde 71,0% em M028.
            if speaker_ids is not None:
                grouped_clean["speaker"] = evaluate_grouped_scores(
                    yte, pf_clean, speaker_ids, threshold=cfg.decision_threshold
                )
            generator_known = eval_context.get("generator_known")
            if generator_ids is not None and (
                generator_known is None or np.all(generator_known)
            ):
                grouped_clean["generator"] = evaluate_grouped_scores(
                    yte, pf_clean, generator_ids, threshold=cfg.decision_threshold
                )
            converged = (
                not np.isnan(clean.get("auc_roc", float("nan")))
                and clean["auc_roc"] >= cfg.converge_auc_threshold
                and clean.get("accuracy", 0.0) >= cfg.converge_accuracy_threshold
            )

            robustness: Dict[str, Any] = {}
            scores_robustness: Dict[str, Any] = {}
            # Níveis efetivamente usados no augmentation de treino DESTA
            # arquitetura: com augmentation desligado, nenhuma condição é
            # casada e todas as colunas medem generalização.
            train_snr_levels = (
                {int(v) for v in cfg.train_aug_snr_db}
                if protocol.get("training_augmentation_domain") != "disabled"
                else set()
            )
            protocol["matched_snr_levels_db"] = sorted(train_snr_levels)
            protocol["unseen_snr_levels_db"] = sorted(
                int(s) for s in cfg.snr_levels_db if int(s) not in train_snr_levels
            )
            for snr in cfg.snr_levels_db:
                if protocol["evaluation_domain"] == "waveform":
                    # Semente da AVALIAÇÃO: seed+20000+snr — mesma realização
                    # de ruído para todas as arquiteturas (comparabilidade).
                    # Aqui `add_awgn` é chamada direto, então esta É a semente
                    # do RNG. No treino ela passa por `add_awgn_assigned`, que
                    # deriva uma por nível; a disjunção entre os dois conjuntos
                    # não é presumida — `_prepare_protocol_splits` compara as
                    # sementes REAIS dos dois lados e aborta se houver
                    # interseção.
                    noisy_raw = BenchmarkData.add_awgn(
                        raw_Xte, snr, seed=cfg.seed + 20000 + int(snr)
                    )
                    Xn, _ = prepare_input_for_architecture(noisy_raw, arch)
                else:
                    Xn = BenchmarkData.add_awgn(
                        Xte, snr, seed=cfg.seed + 20000 + int(snr)
                    )
                    noisy_raw = None
                pf_noisy = predict_eval(Xn, noisy_raw)
                scores_robustness[str(snr)] = [round(float(v), 6) for v in pf_noisy]
                block = evaluate_scores(
                    raw_yte,
                    pf_noisy,
                    threshold=cfg.decision_threshold,
                    n_bootstrap=n_boot,
                    cluster_ids=cluster_ids,
                    calibrated_threshold=calibrated_threshold,
                    ranking_scores=ranking_eval(Xn, noisy_raw),
                )
                # Condição CASADA (o nível esteve no augmentation de treino) ou
                # NÃO VISTA. Sem esta marca, a tabela de robustez não distingue
                # "aprendeu a lidar com ruído" de "decorou os níveis do treino"
                # — e o leitor não tem como saber qual coluna é qual.
                block["noise_condition"] = (
                    "matched" if int(snr) in train_snr_levels else "unseen"
                )
                robustness[str(snr)] = block

            # Robustez a CODEC (opt-in): round-trip com perdas na FORMA DE
            # ONDA, antes dos frontends — mesmo ponto do protocolo do AWGN.
            # Determinístico (sem semente); mesma degradação p/ todas as
            # arquiteturas (comparabilidade pareada).
            codec_robustness: Dict[str, Any] = {}
            codecs = list(getattr(cfg, "codec_eval", []) or [])
            # AJUSTE 2026-08-09: um `{}` não dizia se a avaliação de codec foi
            # PEDIDA e não achou nada, se foi ignorada por domínio incompatível
            # ou se simplesmente não foi solicitada — as três leituras cabiam no
            # mesmo dict vazio, e no `clean_benchmark_15k` a terceira era a
            # verdadeira. O status fica num campo irmão para não colidir com os
            # nomes de codec, que são as chaves de `codec_robustness`.
            codec_eval_status: Dict[str, Any] = {
                "requested": [str(c) for c in codecs],
                "status": "not_requested" if not codecs else "ok",
            }
            if codecs and protocol["evaluation_domain"] != "waveform":
                codec_eval_status.update(
                    status="skipped",
                    reason=(
                        "avaliação fora do domínio da forma de onda "
                        f"({protocol['evaluation_domain']})"
                    ),
                )
            if codecs and protocol["evaluation_domain"] == "waveform":
                from benchmarks.perturbations import codec_roundtrip

                for codec in codecs:
                    try:
                        degraded = codec_roundtrip(raw_Xte, codec)
                        Xc, _ = prepare_input_for_architecture(degraded, arch)
                        codec_robustness[codec] = evaluate_scores(
                            raw_yte,
                            predict_eval(Xc, degraded),
                            threshold=cfg.decision_threshold,
                            n_bootstrap=n_boot,
                            cluster_ids=cluster_ids,
                            calibrated_threshold=calibrated_threshold,
                            ranking_scores=ranking_eval(Xc, degraded),
                        )
                    except Exception as exc:  # noqa: BLE001 — opt-in, não derruba o run
                        logger.warning(
                            "[%s] avaliação de codec '%s' falhou: %s",
                            arch,
                            codec,
                            exc,
                        )
                        codec_robustness[codec] = {"status": "error", "error": str(exc)}
                        codec_eval_status["status"] = "partial"
            elif codecs:
                logger.warning(
                    "[%s] codec_eval ignorado: avaliação não está no domínio "
                    "da forma de onda.",
                    arch,
                )

            latency_profile = measure_latency_profile(
                r["predict_fn"],
                Xte[0],
                runs=cfg.latency_runs,
                runtime="sklearn" if is_classical else "keras",
            )
            latency = latency_profile.get("median_ms")
            training_config = dict(r.get("training_config") or {})
            training_config.setdefault("waveform_noise_protocol", protocol)
            history = r.get("history")
            effective_epochs = r.get("epochs")
            if history:
                for values in history.values():
                    if isinstance(values, (list, tuple)):
                        effective_epochs = len(values)
                        training_config.setdefault(
                            "epochs_budget", training_config.get("epochs")
                        )
                        training_config["epochs_executed"] = effective_epochs
                        break
            if effective_epochs is None:
                effective_epochs = training_config.get("epochs")
            if effective_epochs is None and not is_classical:
                effective_epochs = cfg.epochs

            # `converged` fala do CHECKPOINT selecionado; `training_stability`
            # fala do TREINO que levou até ele. Um modelo que colapsou na época
            # 14 e teve o checkpoint da 10 promovido é `converged: True` e
            # `training_stability.status: "collapsed"` — foi exatamente o caso
            # do Conformer no `clean_benchmark_15k`, e nada no artefato dizia.
            training_stability = analyze_training_stability(
                history,
                epochs_budget=training_config.get(
                    "epochs_budget", training_config.get("epochs")
                ),
                # Sem isto o diagnóstico reportava como "melhor época" a de
                # menor val_loss mesmo nos runs selecionados por val_eer.
                checkpoint_monitor=str(
                    training_config.get("checkpoint_monitor")
                    or getattr(cfg, "checkpoint_monitor", "")
                    or "val_loss"
                ),
            )
            if training_stability.get("stable") is False:
                logger.warning(
                    "[%s] treino INSTÁVEL (%s): %s",
                    arch,
                    training_stability.get("status"),
                    training_stability.get("reason"),
                )

            return {
                "status": "ok",
                "type": "classical" if is_classical else "neural",
                "input_shape": list(np.asarray(Xte).shape[1:]),
                "input_preparation": protocol,
                "noise_protocol": protocol,
                "converged": bool(converged),
                "convergence_criteria": {
                    "auc_roc_min": cfg.converge_auc_threshold,
                    "accuracy_min": cfg.converge_accuracy_threshold,
                    "scope": "checkpoint_selecionado",
                },
                "training_stability": training_stability,
                "clean": clean,
                "grouped_clean": grouped_clean,
                "scores_clean": [round(float(v), 6) for v in pf_clean],
                # Score bruto do detector quando há calibração pós-hoc: é
                # sobre ELE que AUC/EER/min t-DCF são medidos, então sem
                # gravá-lo o artefato deixa de ser reverificável.
                "ranking_scores_clean": (
                    None
                    if rank_clean is None
                    else [round(float(v), 6) for v in rank_clean]
                ),
                "robustness": robustness,
                "scores_robustness": scores_robustness,
                "codec_robustness": codec_robustness,
                "codec_eval_status": codec_eval_status,
                "efficiency": {
                    "params": r["params"],
                    "size_mb": r["size_mb"],
                    "latency_ms": latency,
                    "latency_profile": latency_profile,
                },
                "history": history,
                "training_config": training_config,
                "model_parameters": r.get("model_parameters") or {},
                "final_training_metrics": r.get("final_metrics") or {},
                "fit_strategy": r.get("fit_strategy"),
                "model_artifact": r.get("model_artifact"),
                # A cópia em `data/models/` é compartilhada entre runs e pode
                # ser sobrescrita; a promoção compara o sha256 contra a cópia
                # preservada no run antes de publicar qualquer coisa.
                "model_artifact_shared_copy": r.get("model_artifact_shared_copy"),
                "model_artifact_fingerprint": r.get("model_artifact_fingerprint"),
                "training_artifacts_dir": str(_architecture_dir(cfg, arch)),
                "epochs": (
                    int(effective_epochs) if effective_epochs is not None else None
                ),
                "wall_time_s": round(time.time() - t0, 1),
            }
    except Exception as e:  # noqa: BLE001 — isola falha por arquitetura
        logger.warning("Arquitetura %s falhou: %s", arch, e)
        return {
            "status": "error",
            "error": str(e),
            "wall_time_s": round(time.time() - t0, 1),
        }


def _load_and_validate_data(cfg: BenchmarkConfig) -> BenchmarkData:
    if cfg.dataset_path:
        data = BenchmarkData.from_npz(cfg.dataset_path)
        # Guarda de sanidade (2026-07-14): um stub de smoke com 64 amostras
        # foi encontrado com o MESMO nome do dataset real de 15k — um run
        # completo sobre ele terminaria sem nenhum erro visível. Não falha
        # (NPZs de smoke pequenos são legítimos), mas avisa alto e deixa
        # rastro no log do run.
        if len(data.y) < 1000:
            logger.warning(
                "Dataset '%s' tem apenas %d amostras — se isto é um "
                "benchmark científico, confira se o caminho não aponta para "
                "um stub de smoke (dataset real: data/datasets/, ~15k).",
                data.name,
                len(data.y),
            )
    else:
        data = BenchmarkData.synthetic(cfg.synthetic_n, cfg.synthetic_shape, cfg.seed)
    data.validate()
    return data


def plan_benchmark(cfg: BenchmarkConfig, write: bool = True) -> Dict[str, Any]:
    """Valida dataset/configuração e grava o plano antes de treinar."""
    _normalize_project_paths(cfg)
    data = _load_and_validate_data(cfg)
    _audit_source_label_shortcut(
        data,
        threshold=cfg.source_oracle_threshold,
        fail=cfg.fail_on_source_shortcut,
    )
    plan = build_benchmark_plan(cfg, data)
    if write:
        write_benchmark_plan(plan, cfg.output_dir)
    return plan


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    """Executa o benchmark completo e grava os artefatos. Retorna o dict total."""
    import matplotlib

    matplotlib.use("Agg")
    _normalize_project_paths(cfg)

    # GPU/threads configurados como no app (idempotente)
    try:
        from app.core.gpu import setup_gpu

        setup_gpu(log_level=logging.WARNING)
    except Exception:
        pass

    data = _load_and_validate_data(cfg)
    source_shortcut_audit = _audit_source_label_shortcut(
        data,
        threshold=cfg.source_oracle_threshold,
        fail=cfg.fail_on_source_shortcut,
    )
    plan = build_benchmark_plan(cfg, data)
    write_benchmark_plan(plan, cfg.output_dir)
    apply_plan_to_config(cfg, plan)

    # Uma única partição no domínio bruto é reutilizada por todas as famílias.
    # Isso garante as mesmas amostras e as mesmas realizações de AWGN antes dos
    # frontends raw/log-Mel/tabular.
    raw_splits = data.stratified_split(
        cfg.seed,
        group_split=cfg.group_split,
        holdout_generator=cfg.holdout_generator,
        speaker_split=cfg.speaker_split,
        holdout_speaker=cfg.holdout_speaker,
        preserve_predefined=cfg.preserve_predefined_splits,
    )
    # CORREÇÃO DE BANDA (2026-08-19) -- aplicada UMA vez, sobre a partição
    # bruta, antes de qualquer frontend, do AWGN e da auditoria.
    #
    # Por que aqui e não no build do corpus: o efeito é idêntico e isto evita
    # reconstruir 15 GB, além de tornar a correção um PARÂMETRO DECLARADO do
    # protocolo (vai para o artefato) em vez de propriedade opaca dos arquivos.
    #
    # Por que antes do AWGN: o passa-baixas corrige o SINAL; o ruído é o canal.
    # Invertendo a ordem, o filtro removeria também a banda alta do ruído e
    # mudaria a SNR efetiva na faixa retida.
    #
    # Por que antes da auditoria: as impressões digitais de partição têm de
    # descrever o dado REALMENTE usado.
    if getattr(cfg, "band_correction_hz", None):
        from app.domain.features.benchmark_frontend import apply_band_correction

        # GUARDA DE DUPLA APLICAÇÃO. Desde 2026-08-19 `extract_window` também
        # aplica a correção no build do corpus, e o exportador grava a política
        # em `metadata_json.band_correction`. Um .npz reexportado com o código
        # atual já vem corrigido; aplicar de novo cascatearia dois FIR de 255
        # taps (a resposta em magnitude ao quadrado), produzindo uma banda que
        # não é a do protocolo declarado. Os .npz anteriores a essa data não
        # têm o bloco e seguem pelo caminho normal.
        _ja_corrigido = (getattr(data, "metadata", None) or {}).get(
            "band_correction"
        ) or {}
        if _ja_corrigido.get("applied"):
            raise RuntimeError(
                "O dataset já traz a correção de banda aplicada no build "
                f"(cutoff {_ja_corrigido.get('cutoff_hz')} Hz, gravada por "
                f"{_ja_corrigido.get('applied_by')}), e --band-correction-hz "
                f"pediria uma SEGUNDA aplicação em cascata. Rode sem a flag "
                "(o dataset já está no protocolo corrigido) ou reexporte o "
                "NPZ sem a correção no build."
            )

        corte = float(cfg.band_correction_hz)
        Xtr, ytr, Xv, yv, Xte, yte = raw_splits
        raw_splits = (
            apply_band_correction(Xtr, corte),
            ytr,
            apply_band_correction(Xv, corte),
            yv,
            apply_band_correction(Xte, corte),
            yte,
        )
        logger.info(
            "[protocolo] correção de banda ativa: passa-baixas %.0f Hz nas "
            "três partições, nas DUAS classes",
            corte,
        )

    split_overlap_audit = _audit_split_overlap(
        raw_splits, fail_on_overlap=cfg.fail_on_split_overlap
    )
    split_fingerprints = split_overlap_audit["split_fingerprints"]
    provenance_overlap_audit = _audit_split_provenance(data)
    if not data.last_split_indices or "test" not in data.last_split_indices:
        raise RuntimeError("Indices do teste indisponiveis para auditoria agrupada")
    test_idx = np.asarray(data.last_split_indices["test"], dtype="int64")
    eval_context: Dict[str, np.ndarray] = {}
    context_vectors = {
        "cluster_ids": data.cluster_ids,
        "source_ids": data.groups,
        "generator_ids": data.generators,
        "generator_known": data.generator_known,
        # `data.speakers` já era extraído por BenchmarkData (`speaker_ids` do
        # .npz) mas só alimentava --speaker-split/--holdout-speaker. Aqui ele
        # passa a alimentar também `grouped_clean["speaker"]` e o IC por
        # locutor da consolidação.
        "speaker_ids": data.speakers,
    }
    for name, values in context_vectors.items():
        if values is not None:
            aligned = np.asarray(values)
            if len(aligned) != len(data.y):
                raise RuntimeError(f"Vetor {name} desalinhado antes da avaliacao")
            eval_context[name] = aligned[test_idx]

    # Contexto do AJUSTE (não da avaliação): os cluster_ids do TREINO, que a
    # validação cruzada dos clássicos usa para não partir um par locutor×frase
    # entre a dobra de treino e a de validação. `eval_context` só carrega a
    # fatia de teste, então não serve para isto.
    fit_context: Dict[str, np.ndarray] = {}
    train_idx = data.last_split_indices.get("train")
    if train_idx is not None and data.cluster_ids is not None:
        fit_context["train_cluster_ids"] = np.asarray(data.cluster_ids)[
            np.asarray(train_idx, dtype="int64")
        ]
    elif any(_is_classical_arch(a) for a in cfg.architectures):
        logger.warning(
            "cluster_ids indisponiveis: a validacao cruzada de SVM/RandomForest "
            "sera NAO agrupada e o par real/clone da mesma frase pode se dividir "
            "entre as dobras (registrado em hyperparameter_tuning.cv_grouping)"
        )
    y_test_base = np.asarray(raw_splits[5])
    n_test = len(y_test_base)
    logger.info(
        "Dataset '%s': %d amostras | teste held-out: %d",
        data.name,
        len(data.y),
        n_test,
    )

    per_arch: Dict[str, Any] = {}
    training_seeds = cfg.training_seeds
    for arch in cfg.architectures:
        runs = []
        for repetition, train_seed in enumerate(training_seeds, start=1):
            if len(training_seeds) > 1:
                logger.info(
                    "=== Benchmark: %s (repetição %d/%d, seed de treino %d) ===",
                    arch,
                    repetition,
                    len(training_seeds),
                    train_seed,
                )
            else:
                logger.info("=== Benchmark: %s ===", arch)
            run = _benchmark_one(
                arch,
                cfg,
                raw_splits,
                eval_context,
                training_seed=train_seed,
                fit_context=fit_context,
                # Para `_band_correction_policy` saber se o DATASET já veio
                # corrigido do build — nesse caso o runner recusa a flag, mas o
                # contrato precisa declarar a política mesmo assim.
                data=data,
            )
            run["training_seed"] = int(train_seed)
            runs.append(run)
        aggregated = _aggregate_seed_runs(runs)
        provenance = _architecture_provenance(arch)
        if provenance:
            aggregated["provenance"] = provenance
        per_arch[arch] = aggregated

    results: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "preflight": plan,
        "environment": _env_snapshot(),
        "dataset": {
            "name": data.name,
            "n_total": int(len(data.y)),
            "n_test": int(n_test),
            "input_shape": list(np.asarray(data.X).shape[1:]),
            "metadata": data.metadata or {},
            "split_source": (
                "predefined_npz"
                if data.predefined_split_indices
                and cfg.preserve_predefined_splits
                and not any(
                    (
                        cfg.group_split,
                        cfg.holdout_generator,
                        cfg.speaker_split,
                        cfg.holdout_speaker,
                    )
                )
                else "generated_by_protocol"
            ),
            "split_overlap_audit": split_overlap_audit,
            "split_fingerprints": split_fingerprints,
            "test_split_sha256": split_fingerprints["test"]["sha256"],
            "test_lock": getattr(cfg, "test_lock", None),
            "provenance_overlap_audit": provenance_overlap_audit,
            "source_shortcut_audit": source_shortcut_audit,
            "source": (data.metadata or {}).get("source"),
            "balance_test": {
                "real": int((y_test_base == 0).sum()),
                "fake": int((y_test_base == 1).sum()),
            },
            "y_test": [int(v) for v in y_test_base],
            # Unidade de reamostragem do teste (frase/locutor). Já era usada nos
            # IC 95% por modelo, mas não era persistida — sem ela, a comparação
            # PAREADA entre dois modelos (benchmarks/significance.py) só pode
            # reamostrar por amostra e subestima a incerteza, porque amostras da
            # mesma frase não são independentes.
            "test_cluster_ids": (
                [str(v) for v in eval_context["cluster_ids"]]
                if eval_context.get("cluster_ids") is not None
                else None
            ),
            # A unidade acima é a FRASE (`cluster_ids == text_ids`: 183 no
            # teste). A alegação do protocolo, porém, é sobre LOCUTORES não
            # vistos, e são 11 — reamostrar frases trata frases do mesmo
            # locutor como independentes e estreita o IC. Persistir os dois
            # deixa a consolidação reportar as duas unidades em vez de escolher
            # por nós.
            "test_speaker_ids": (
                [str(v) for v in eval_context["speaker_ids"]]
                if eval_context.get("speaker_ids") is not None
                else None
            ),
        },
        "architectures": per_arch,
    }

    if cfg.run_api_probe:
        try:
            from benchmarks.api_probe import run_api_probe

            results["api"] = run_api_probe()
        except Exception as e:  # noqa: BLE001
            results["api"] = {"status": "error", "error": str(e)}

    # SQLite é a fonte canônica estruturada; JSON/CSV/figuras abaixo são
    # projeções compatíveis para análise, publicação e intercâmbio.
    try:
        from app.core.db.experiment_store import experiment_store

        experiment_store.ensure_schema()
        snapshot_uid = experiment_store.record_system_snapshot(purpose="benchmark")
        run_uid = experiment_store.persist_benchmark_results(
            results,
            output_dir=cfg.output_dir,
            source="benchmarks.runner",
        )
        results["persistence"] = {
            "backend": "sqlite",
            "database": "data/app.db",
            "run_uid": run_uid,
            "system_snapshot_uid": snapshot_uid,
            "json_csv_role": "export_projection",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha ao persistir benchmark no SQLite: %s", exc)
        results["persistence"] = {
            "backend": "sqlite",
            "status": "error",
            "error": str(exc),
        }
    from benchmarks.report import write_all

    write_all(results, cfg.output_dir)
    return results
