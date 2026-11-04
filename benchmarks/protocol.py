"""Protocolo de dados por arquitetura: partições preparadas (AWGN na forma de
onda, antes do front-end), política de banda e os carimbos do ``input_contract``
que dão à inferência o preparo do treino. Saiu de ``runner.py`` (P2-15).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from benchmarks.common import _compact_slug
from benchmarks.config import BenchmarkConfig
from benchmarks.data import (
    BenchmarkData,
    looks_like_raw_audio,
    prepare_input_for_architecture,
)

logger = logging.getLogger("benchmark")


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
        f"train_{protocol.get('train_crop_strategy') or 'center'}" f"_eval_{estrategia}"
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
    pós-hoc (`tools/reporting/rebuild_inference_contracts.py`) que, além de
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
