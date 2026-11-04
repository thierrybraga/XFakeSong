"""Orquestrador do benchmark: treina, avalia, mede robustez e eficiência.

Conduz o run: dados → partição auditada → uma arquitetura por vez
(``_benchmark_one``) → artefatos. As etapas vivem ao lado: ``protocol`` (preparo e
contrato), ``fit_neural``/``fit_classical`` (treino), ``split_audit``,
``provenance``, ``seed_aggregation`` e ``common``.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.utils.file_storage import break_hardlink, break_hardlinks_in_tree
from benchmarks.common import (
    _architecture_dir,
    _finite_scores,
    _is_classical_arch,
    _models_dir,
    _normalize_project_paths,
    _slug,
)
from benchmarks.config import BenchmarkConfig
from benchmarks.data import BenchmarkData, prepare_input_for_architecture
from benchmarks.efficiency import measure_latency_profile
from benchmarks.evaluate import evaluate_grouped_scores, evaluate_scores
from benchmarks.fit_classical import _run_classical
from benchmarks.fit_neural import _run_neural
from benchmarks.planning import (
    apply_plan_to_config,
    build_benchmark_plan,
    write_benchmark_plan,
)
from benchmarks.protocol import _prepare_protocol_splits, _stamp_eval_crop_strategy
from benchmarks.provenance import _architecture_provenance, _env_snapshot
from benchmarks.seed_aggregation import _aggregate_seed_runs
from benchmarks.split_audit import (
    _audit_source_label_shortcut,
    _audit_split_overlap,
    _audit_split_provenance,
)
from benchmarks.stability import analyze_training_stability

logger = logging.getLogger("benchmark")


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
    arch_dir = _architecture_dir(cfg, arch, training_seed)
    break_hardlinks_in_tree(arch_dir)
    model_name = f"bench_{_slug(arch)}"
    for suffix in (".keras", ".h5", ".pkl", ".pt", "_config.json"):
        break_hardlink(models_dir / f"{model_name}{suffix}")
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
                    arch,
                    cfg,
                    splits,
                    tmp,
                    models_dir,
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
                _stamp_eval_crop_strategy(_caminho, r.get("input_contract"), protocol)

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
        break_hardlinks_in_tree(Path(cfg.output_dir))
        write_benchmark_plan(plan, cfg.output_dir)
    return plan


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    """Executa o benchmark completo e grava os artefatos. Retorna o dict total."""
    import matplotlib

    matplotlib.use("Agg")
    _normalize_project_paths(cfg)
    output_root = Path(cfg.output_dir)
    break_hardlinks_in_tree(
        output_root,
        exclude=lambda path: "_superseded" in path.relative_to(output_root).parts,
    )

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
