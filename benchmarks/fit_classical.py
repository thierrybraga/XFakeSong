"""Treino de SVM e RandomForest no benchmark, direto em sklearn.

Grid vindo da arquitetura, CV agrupada por locutor × frase, calibração isotônica
e o sidecar de inferência dos clássicos. Saiu de ``runner.py`` (P2-15).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from benchmarks.common import (
    _architecture_dir,
    _compact_slug,
    _finite_scores,
    _json_safe,
    _preserve_run_artifact,
    _slug,
)
from benchmarks.config import BenchmarkConfig
from benchmarks.data import BenchmarkData
from benchmarks.efficiency import file_size_mb

logger = logging.getLogger("benchmark")


def _classical_search_space(arch: str, seed: int) -> tuple[dict[str, list], Any, str]:
    """Grid, estimador-base e nome do passo do pipeline para SVM/RF.

    O GRID VEM DA ARQUITETURA, não daqui (2026-08-09). Até então esta função
    carregava uma cópia própria — a 4ª fonte de hiperparâmetros do projeto,
    ausente das três que o CLAUDE.md lista — e era ela que rodava, deixando os
    grids regularizados de `svm.py`/`random_forest.py` como código morto sem
    NENHUM chamador em `app/`, `benchmarks/`, `tools/` ou `tests/`.
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
    # Política de correção de formato, para a inferência reproduzi-la: o mesmo campo
    # que `_stamp_benchmark_frontend` grava nas neurais. Sem ele a inferência prepara
    # áudio full-band para um modelo treinado em banda limitada, e o SVM caía a 48%.
    # Só com front-end declarado: sem ele, a inferência não prepara o áudio.
    if from_benchmark_frontend and protocol.get("band_correction"):
        contract["band_correction"] = dict(protocol["band_correction"])
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
