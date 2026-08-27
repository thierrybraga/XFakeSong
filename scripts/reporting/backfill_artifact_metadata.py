#!/usr/bin/env python3
"""Preenche, em runs antigos, os campos de metadata introduzidos em 2026-08-09.

MOTIVACAO. O run `clean_benchmark_15k` foi produzido antes de
`training_stability`, `test_cluster_ids`, `latency_profile.runtime`,
`fit_strategy.fit_splits` e `codec_eval_status` existirem. Reexecutar as 9
arquiteturas restantes so para ganhar metadata custaria ~60 h de GPU; todos
esses campos, porem, sao DERIVAVEIS do que ja esta gravado no proprio artefato
(mais o `.npz` do dataset, no caso dos cluster_ids).

O que mais importa aqui e a honestidade do resultado: nada e estimado,
arredondado ou "melhorado". Cada bloco escrito carrega um carimbo `backfill`
com a data, o script, os campos tocados e a base da derivacao, para que ninguem
confunda um artefato completado com um artefato produzido por uma execucao que
ja emitia esses campos.

NAO faz parte do escopo reconstruir historicos truncados por retomada (RawNet2
gravou 17 de 100 epocas, RawGAT-ST 91). A serie completa esta no `run.log`, mas
recupera-la e parsing de log, nao derivacao — o backfill apenas SINALIZA a
discrepancia via `training_stability.warnings`.

Uso:

    # inspeciona sem tocar em nada (padrao)
    python scripts/reporting/backfill_artifact_metadata.py \\
        --run data/results/clean_benchmark_15k \\
        --dataset data/datasets/benchmark_dataset_15k.npz

    # aplica, criando <arquivo>.pre-backfill.bak
    python scripts/reporting/backfill_artifact_metadata.py \\
        --run data/results/clean_benchmark_15k \\
        --dataset data/datasets/benchmark_dataset_15k.npz --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# numpy e o pacote `benchmarks` (que puxa TensorFlow pelo runner) entram SOB
# DEMANDA: so os campos que recalculam metrica dependem deles. Os demais reparos
# — declaracao de score de ordenacao, fingerprint, runtime de latencia — sao
# leitura e reescrita de JSON, e precisam rodar num checkout sem o ambiente de
# treino, que e onde artefatos costumam ser auditados.
def _numpy():
    import numpy as np

    return np


def _evaluate_grouped_scores(*args, **kwargs):
    from benchmarks.evaluate import evaluate_grouped_scores

    return evaluate_grouped_scores(*args, **kwargs)


def _analyze_training_stability(*args, **kwargs):
    from benchmarks.stability import analyze_training_stability

    return analyze_training_stability(*args, **kwargs)

SCRIPT_ID = "scripts/reporting/backfill_artifact_metadata.py"

#: Runtime por tipo de modelo. O escopo oficial roda os neurais Keras em
#: TensorFlow e os classicos em scikit-learn; WavLM/HuBERT Original usam um
#: runner PyTorch proprio que JA emite o campo, entao nao passam por aqui.
_RUNTIME_BY_TYPE = {"classical": "sklearn", "neural": "keras"}
_LIBRARY_BY_RUNTIME = {"keras": "tensorflow", "sklearn": "sklearn", "pytorch": "torch"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(fields: List[str], derived_from: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "applied_at_utc": _now(),
        "script": SCRIPT_ID,
        "reason": (
            "campos introduzidos em 2026-08-09; este run e anterior e nao foi "
            "reexecutado"
        ),
        "fields": sorted(fields),
        "derived_from": derived_from,
        "note": (
            "valores DERIVADOS do proprio artefato (e do .npz, para "
            "test_cluster_ids). Nenhuma metrica foi recalculada, estimada ou "
            "alterada."
        ),
    }


def provenance_ids_from_npz(
    dataset: Path, key: str, expected_y_test: List[int]
) -> Tuple[List[str], Dict[str, Any]]:
    """Extrai um vetor de proveniencia do TESTE sem materializar o audio.

    `BenchmarkData.from_npz` concatena X_train, X_val e X_test nessa ordem e
    numera os indices em sequencia, entao o teste e sempre a fatia final. Os
    vetores de proveniencia (`cluster_ids`, `speaker_ids`) ja vem alinhados a
    essa ordem. Ler so `y_*` e o vetor pedido evita carregar os ~2,9 GB de
    forma de onda.

    O alinhamento e VERIFICADO comparando o `y_test` derivado do .npz com o que
    o artefato gravou; divergencia aborta em vez de gravar ids desalinhados.
    """
    np = _numpy()
    with np.load(dataset, allow_pickle=False, mmap_mode="r") as data:
        for required in ("y_train", "y_val", "y_test", key):
            if required not in data:
                raise KeyError(f"{dataset.name} nao tem '{required}'")
        n_train = int(len(data["y_train"]))
        n_val = int(len(data["y_val"]))
        y_test = np.asarray(data["y_test"]).ravel().astype(int)
        values = np.asarray(data[key]).ravel()
        start = n_train + n_val
        stop = start + len(y_test)
        if stop > len(values):
            raise ValueError(
                f"{key} tem {len(values)} entradas, mas o teste termina em {stop}"
            )
        test_values = [str(v) for v in values[start:stop]]

    np = _numpy()
    stored = np.asarray(expected_y_test).ravel().astype(int)
    if len(stored) != len(y_test) or not np.array_equal(stored, y_test):
        raise ValueError(
            f"y_test do artefato nao bate com o do .npz — os {key} ficariam "
            "desalinhados. Backfill abortado."
        )
    return test_values, {
        "dataset": str(dataset),
        "key": key,
        "slice": [int(start), int(stop)],
        "n_groups": int(len(set(test_values))),
        "y_test_match": True,
    }


def test_cluster_ids_from_npz(
    dataset: Path, expected_y_test: List[int]
) -> Tuple[List[str], Dict[str, Any]]:
    """Compatibilidade: `provenance_ids_from_npz` com key='cluster_ids'."""
    values, basis = provenance_ids_from_npz(dataset, "cluster_ids", expected_y_test)
    basis["n_clusters"] = basis["n_groups"]
    return values, basis


def _artifact_fingerprint_block(arch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Identidade do artefato, com veredito honesto sobre o que da pra afirmar.

    O run nao gravava sha256 do modelo, entao nao ha como PROVAR que o arquivo
    em `model_artifact` e o que a execucao produziu. O que ha e uma checagem
    derivavel: nos caminhos Keras e classico o `efficiency.size_mb` e
    literalmente `file_size_mb(model_path)` — o tamanho DAQUELE arquivo. Se o
    tamanho em disco nao bate, o arquivo foi trocado depois do run.

    Foi assim que o `bench_svm.pkl` do clean_benchmark_15k se denunciou: o run
    registrou 3,61 MB e o arquivo tem 0,045 MB (um artefato de smoke de 8
    amostras gravado no mesmo caminho global `data/models/`).

    O runner SSL calcula `size_mb` como cabeca + backbone congelado, entao ali a
    divergencia e esperada e o veredito e `not_verifiable` — nao `size_mismatch`.
    """
    raw = arch.get("model_artifact")
    if not raw:
        return None
    text = str(raw)
    path = (
        Path(text.replace("/app/", "", 1)) if text.startswith("/app/") else Path(text)
    )
    if not path.is_absolute():
        path = _ROOT / path

    declared = (arch.get("efficiency") or {}).get("size_mb")
    runner = str((arch.get("provenance") or {}).get("runner") or "")
    ssl_runner = "run_wavlm_original_benchmark" in runner

    if not path.is_file():
        return {
            "path": text,
            "integrity": "missing_artifact",
            "declared_size_mb": declared,
            "reason": f"artefato declarado nao existe em disco: {path}",
            "inferred": True,
        }

    size_bytes = path.stat().st_size
    actual_mb = size_bytes / (1024 * 1024)
    block: Dict[str, Any] = {
        "path": text,
        "size_bytes": int(size_bytes),
        "size_mb_on_disk": round(actual_mb, 3),
        "declared_size_mb": declared,
        "inferred": True,
    }
    if ssl_runner:
        block["integrity"] = "not_verifiable"
        block["reason"] = (
            "runner SSL soma cabeca + backbone congelado em efficiency.size_mb, "
            "entao o tamanho declarado nao descreve o arquivo .pt sozinho"
        )
        return block
    if not isinstance(declared, (int, float)):
        block["integrity"] = "not_verifiable"
        block["reason"] = "run nao declarou efficiency.size_mb"
        return block
    # 0,02 MB absorve o arredondamento de 2 casas do `size_mb`.
    if abs(actual_mb - float(declared)) > 0.02:
        block["integrity"] = "size_mismatch"
        block["reason"] = (
            f"o run registrou {declared} MB para este artefato e o arquivo em "
            f"disco tem {actual_mb:.3f} MB — foi substituido depois da execucao"
        )
        return block
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    block["integrity"] = "verified_by_size"
    block["sha256"] = digest.hexdigest()
    block["reason"] = (
        "tamanho em disco bate com o declarado pelo run; sha256 tirado agora "
        "serve de referencia para deteccao de troca daqui em diante"
    )
    return block


def _runtime_block(arch: Dict[str, Any], environment: Dict[str, Any]) -> Optional[Dict]:
    runtime = _RUNTIME_BY_TYPE.get(str(arch.get("type")))
    if runtime is None:
        return None
    libraries = (environment or {}).get("libraries") or {}
    version = libraries.get(_LIBRARY_BY_RUNTIME[runtime])
    block: Dict[str, Any] = {
        "runtime": runtime,
        # A versao vem do bloco `environment` do PROPRIO run — e a que
        # executou a medicao, nao a instalada agora.
        "runtime_version": str(version) if version else None,
        "cross_runtime_comparable": False,
    }
    if runtime == "sklearn":
        block["device"] = "cpu"
    else:
        block["device"] = "gpu" if (environment or {}).get("gpu") else "cpu"
    return block


def _fit_strategy_block(
    arch: Dict[str, Any], dataset_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Declara quais splits alimentaram o ajuste, sem inventar contagem.

    Se a arquitetura JA tem `fit_strategy`, so as chaves de DECLARACAO sao
    acrescentadas — `kind`/`estimator`/`fit_samples` sao de quem executou o
    ajuste e nao podem ser deduzidos do tipo. WavLM/HuBERT Original sao
    `type: "neural"` mas rodam pelo runner PyTorch dedicado; carimba-los como
    `estimator: "keras"` seria fabricar.
    """
    splits = (dataset_meta or {}).get("splits") or {}
    n_train = (splits.get("train") or {}).get("samples")
    n_val = (splits.get("val") or {}).get("samples")
    if n_train is None or n_val is None:
        return None
    protocol = arch.get("noise_protocol") or {}
    copies = protocol.get("train_noise_copies")
    kind = str(arch.get("type"))
    ja_tem_estrategia = isinstance(arch.get("fit_strategy"), dict) and bool(
        arch.get("fit_strategy")
    )

    if kind == "classical":
        # O caminho classico ajusta em treino+validacao (nao ha checkpoint a
        # selecionar) e a busca de hiperparametros usa CV sobre o treino limpo.
        return {
            "fit_splits": ["train", "val"],
            "validation_role": (
                "incorporada ao ajuste (sem selecao de checkpoint); a busca de "
                "hiperparametros usa CV interna sobre o treino limpo"
            ),
            "clean_train_samples": int(n_train),
            "val_samples": int(n_val),
        }
    if kind != "neural":
        return None

    # Vale para todo neural do escopo: ajusta no treino e reserva a validacao
    # para escolher a epoca. E o que os dois runners fazem.
    block: Dict[str, Any] = {
        "fit_splits": ["train"],
        "validation_role": "selecao de checkpoint (menor val_loss limpa)",
        "val_samples": int(n_val),
    }
    if ja_tem_estrategia:
        # Runner proprio (SSL PyTorch) ja declarou como ajustou; nao sobrescreve.
        return block
    block["kind"] = "fixed_epoch_budget_then_checkpoint_selection"
    block["estimator"] = "keras"
    if copies is not None:
        block["fit_samples"] = int(n_train) * (1 + int(copies))
        block["fit_samples_derivation"] = (
            f"splits.train.samples ({n_train}) x (1 + train_noise_copies "
            f"({int(copies)}))"
        )
    return block


def _needs_stability_refresh(arch: Dict[str, Any]) -> bool:
    """True quando o bloco falta ou e de uma versao anterior do criterio.

    Os campos de oscilacao (`max_epoch_drop`, `monitor_std_tail`) e o
    `selection_gap` entraram depois deste run. Reanalisar e barato — a funcao
    so le o `history` que ja esta no artefato — e a ausencia deles no
    `criteria` e o marcador de versao. Blocos `unknown` (classicos, sem
    historico) nao tem `criteria` e ficam de fora, o que mantem a idempotencia.
    """
    ts = arch.get("training_stability")
    if not isinstance(ts, dict):
        return True
    criteria = ts.get("criteria")
    if not isinstance(criteria, dict):
        return False
    return "max_epoch_drop" not in criteria


#: Blocos de metrica que `evaluate_scores` produz e que passaram a declarar de
#: QUAL score saiu o AUC/EER (2026-08-09). Runs anteriores nao tem o campo.
_METRIC_BLOCKS = ("clean",)


def _ranking_disclosure(arch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Declara de qual score sairam AUC/EER/min-tDCF, para runs sem o campo.

    DERIVACAO, nao suposicao: `evaluate_scores` so separa ordenacao de
    probabilidade quando o chamador passa `ranking_scores`, e o runner passa
    exatamente nos casos em que grava `ranking_scores_clean`. Sem essa chave, a
    ordenacao FOI o proprio `p_fake` — por construcao, inclusive nos runs
    anteriores ao parametro existir, quando a funcao nem o aceitava.

    Sem isto, dois artefatos com o mesmo `auc_roc` podem ter medido coisas
    diferentes e nada no arquivo diz qual — que e a razao de o campo existir.
    """
    fonte = (
        "raw_detector_score"
        if arch.get("ranking_scores_clean") is not None
        else "p_fake"
    )
    alvos: Dict[str, Any] = {}
    for nome in _METRIC_BLOCKS:
        bloco = arch.get(nome)
        if isinstance(bloco, dict) and bloco and "ranking_score_source" not in bloco:
            alvos[nome] = fonte
    rob = arch.get("robustness")
    if isinstance(rob, dict):
        for snr, bloco in rob.items():
            if not isinstance(bloco, dict) or not bloco:
                continue
            if "ranking_score_source" not in bloco:
                alvos[f"robustness.{snr}"] = fonte
    return alvos or None


#: Assinatura do aviso que o guard de restauracao emite quando DESCARTA o
#: checkpoint eleito. Casada nas duas redacoes: a anterior a 2026-08-22 nomeava
#: val_loss porque o guard so sabia comparar perda.
_DESCARTE_RE = re.compile(
    r"Checkpoint\s+(?P<path>\S+)\s+descartado:\s+(?P<metrica>val_\w+)="
)


def _declared_checkpoint_monitor(arch: Dict[str, Any]) -> Optional[str]:
    """Monitor que REALMENTE selecionou a epoca, derivado do proprio artefato.

    Duas fontes, nesta ordem:

    1. ``training_config.checkpoint_monitor`` — gravado a partir de 2026-08-22.
    2. a presenca da serie ``val_eer`` no ``history``. Nao e palpite: o
       callback ``ValidationEER``, unico produtor dessa serie, so e registrado
       quando ``config.checkpoint_monitor == "val_eer"``
       (``ModelTrainer.train``). Serie presente <=> monitor era val_eer.

    Devolve None quando nenhuma das duas se aplica — a ausencia de evidencia
    NAO vira uma afirmacao de val_loss.
    """
    declarado = (arch.get("training_config") or {}).get("checkpoint_monitor")
    if declarado:
        return str(declarado)
    hist = arch.get("history")
    if isinstance(hist, dict) and hist.get("val_eer"):
        return "val_eer"
    return None


def _checkpoint_restore_from_log(
    run_log: Optional[Path], monitor: str
) -> Optional[Dict[str, Any]]:
    """Le no run.log se o checkpoint eleito foi mantido ou descartado.

    CORRECAO, nao complemento: um artefato cujo checkpoint foi descartado
    guarda os pesos da ULTIMA EPOCA, e nao ha nada nas metricas que denuncie
    isso — o unico registro do fato e este aviso. Foi o caso do Conformer na
    bateria corrigida.

    A base e declarada no bloco (``derivation``) justamente porque isto e
    leitura de log, e nao derivacao a partir de metrica gravada. Devolve None
    quando o log nao existe ou nao menciona restauracao nenhuma: sem evidencia,
    nada e afirmado.
    """
    if run_log is None or not run_log.is_file():
        return None
    try:
        texto = run_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    achado = _DESCARTE_RE.search(texto)
    if achado:
        return {
            "attempted": True,
            "restored": False,
            "monitor": monitor,
            "weights_evaluated": "last_epoch",
            "comparado_por": achado.group("metrica"),
            "derivation": (
                f"{run_log.name}: aviso de descarte do guard de restauracao "
                f"(comparacao por {achado.group('metrica')})"
            ),
        }
    mantido = (
        "Checkpoint validado no val set" in texto
        or "Melhor checkpoint restaurado" in texto
    )
    if mantido:
        return {
            "attempted": True,
            "restored": True,
            "monitor": monitor,
            "weights_evaluated": "best_checkpoint",
            "derivation": f"{run_log.name}: confirmacao de restauracao do guard",
        }
    return None


def _checkpoint_selection_correction(
    arch: Dict[str, Any], run_log: Optional[Path]
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Alinha o que o artefato DECLARA ao criterio que ele de fato usou.

    Ate 2026-08-22 o runner escrevia o literal "menor val_loss limpa" em
    ``fit_strategy.validation_role`` mesmo nos runs com ``--checkpoint-monitor
    val_eer``. Diferente dos demais reparos deste script, aqui ha um valor
    ERRADO a substituir, nao uma lacuna a preencher — entao o valor anterior
    fica gravado em ``validation_role_pre_backfill`` e o bloco carimba a base.
    Reescrever em silencio seria trocar uma declaracao falsa por outra
    indistinguivel de um artefato produzido corretamente.
    """
    monitor = _declared_checkpoint_monitor(arch)
    if monitor is None:
        return None, []

    fit = arch.get("fit_strategy")
    if not isinstance(fit, dict) or fit.get("estimator") != "keras":
        # Classicos nao selecionam epoca; o runner SSL declara o proprio bloco.
        return None, []

    correcao: Dict[str, Any] = {}
    campos: List[str] = []

    papel_atual = str(fit.get("validation_role") or "")
    papel_correto = f"seleção de checkpoint (menor {monitor} limpa)"
    if fit.get("checkpoint_monitor") != monitor:
        correcao["checkpoint_monitor"] = monitor
        campos.append("fit_strategy.checkpoint_monitor")
    if papel_atual and papel_atual != papel_correto:
        correcao["validation_role"] = papel_correto
        correcao["validation_role_pre_backfill"] = papel_atual
        campos.append("fit_strategy.validation_role")

    if "checkpoint_restore" not in fit:
        restore = _checkpoint_restore_from_log(run_log, monitor)
        if restore:
            correcao["checkpoint_restore"] = restore
            campos.append("fit_strategy.checkpoint_restore")

    return (correcao or None), campos


def plan_architecture(
    arch: Dict[str, Any],
    environment: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    speaker_ids: Optional[List[str]] = None,
    y_test: Optional[List[int]] = None,
    decision_threshold: float = 0.5,
    run_log: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Calcula as adicoes para UMA arquitetura. Nao muda nada existente."""
    additions: Dict[str, Any] = {}
    fields: List[str] = []

    monitor_real = _declared_checkpoint_monitor(arch) or "val_loss"
    ts_atual = arch.get("training_stability")
    historico = arch.get("history")
    # `tem_historico` e o que torna a reanalise IDEMPOTENTE e segura: sem serie
    # de epocas, `analyze_training_stability` devolve `unknown`, e reescrever um
    # bloco `stable` existente com `unknown` seria perder informacao para
    # ganhar um carimbo. Vale para artefatos cujo historico nao foi preservado.
    tem_historico = isinstance(historico, dict) and any(historico.values())
    monitor_defasado = (
        tem_historico
        and isinstance(ts_atual, dict)
        and ts_atual.get("checkpoint_monitor") != monitor_real
    )
    if _needs_stability_refresh(arch) or monitor_defasado:
        budget = (arch.get("training_config") or {}).get(
            "epochs_budget", (arch.get("training_config") or {}).get("epochs")
        )
        additions["training_stability"] = _analyze_training_stability(
            arch.get("history"),
            epochs_budget=budget,
            # Sem isto o diagnostico reporta como "melhor epoca" a de menor
            # val_loss mesmo nos runs selecionados por val_eer — foi o que os
            # artefatos da bateria corrigida gravaram.
            checkpoint_monitor=monitor_real,
        )
        fields.append("training_stability")

    if "model_artifact_fingerprint" not in arch:
        block = _artifact_fingerprint_block(arch)
        if block:
            additions["model_artifact_fingerprint"] = block
            fields.append("model_artifact_fingerprint")

    grouped = arch.get("grouped_clean")
    scores = arch.get("scores_clean")
    if (
        speaker_ids
        and y_test
        and scores
        and len(scores) == len(y_test) == len(speaker_ids)
        and not (isinstance(grouped, dict) and grouped.get("speaker"))
    ):
        # Derivavel do que ja esta gravado: os scores limpos por amostra, o
        # y_test e os speaker_ids do .npz (alinhamento ja verificado contra o
        # y_test antes de chegar aqui). Mesma funcao que o runner usa.
        np = _numpy()
        additions["_grouped_speaker"] = _evaluate_grouped_scores(
            np.asarray(y_test),
            np.asarray(scores, dtype="float64"),
            np.asarray(speaker_ids),
            threshold=decision_threshold,
        )
        fields.append("grouped_clean.speaker")

    if "codec_eval_status" not in arch:
        codecs = arch.get("codec_robustness")
        if not codecs:
            # Vazio/ausente com nenhuma chave de codec => nunca foi pedido. Se
            # tivesse sido pedido e falhado, haveria entradas com status error.
            additions["codec_eval_status"] = {
                "requested": [],
                "status": "not_requested",
                "inferred": True,
                "basis": "codec_robustness vazio e sem entradas de erro",
            }
            fields.append("codec_eval_status")

    profile = (arch.get("efficiency") or {}).get("latency_profile")
    if isinstance(profile, dict) and "runtime" not in profile:
        block = _runtime_block(arch, environment)
        if block:
            additions["_latency_profile_update"] = block
            fields.append("efficiency.latency_profile.runtime")

    existing_fit = arch.get("fit_strategy")
    if not isinstance(existing_fit, dict) or "fit_splits" not in existing_fit:
        block = _fit_strategy_block(arch, dataset_meta)
        if block:
            additions["_fit_strategy_update"] = block
            fields.append("fit_strategy.fit_splits")

    disclosure = _ranking_disclosure(arch)
    if disclosure:
        additions["_ranking_disclosure"] = disclosure
        fields.append("ranking_score_source")

    correcao, campos_correcao = _checkpoint_selection_correction(arch, run_log)
    if correcao:
        additions["_checkpoint_selection_correction"] = correcao
        fields.extend(campos_correcao)

    return additions, fields


def _sibling_metrics_files(results_path: Path, arch_name: str) -> List[Path]:
    """`metrics.json` da mesma arquitetura, ao lado do `results.json`.

    O runner grava a MESMA estrutura duas vezes: o bloco por arquitetura dentro
    de `results.json` (que a consolidacao le) e um `metrics.json` por
    arquitetura em `architectures/<slug>/`. Corrigir so o primeiro deixaria os
    dois arquivos do mesmo run declarando criterios de selecao diferentes — que
    e uma contradicao pior do que a declaracao errada original, porque nao ha
    como saber qual vale.

    O casamento e pelo campo `architecture` de dentro do arquivo, nao pelo nome
    do diretorio: e o proprio artefato dizendo de quem ele e.
    """
    base = results_path.parent / "architectures"
    if not base.is_dir():
        return []
    achados: List[Path] = []
    for candidato in sorted(base.glob("*/metrics.json")):
        try:
            payload = json.loads(candidato.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("architecture") or "") == arch_name:
            achados.append(candidato)
    return achados


def apply_architecture(arch: Dict[str, Any], additions: Dict[str, Any]) -> None:
    for key, value in additions.items():
        if key == "_latency_profile_update":
            arch["efficiency"]["latency_profile"].update(value)
        elif key == "_fit_strategy_update":
            current = arch.get("fit_strategy")
            arch["fit_strategy"] = {**(current or {}), **value}
        elif key == "_checkpoint_selection_correction":
            current = arch.get("fit_strategy")
            arch["fit_strategy"] = {**(current or {}), **value}
        elif key == "_grouped_speaker":
            current = arch.get("grouped_clean")
            arch["grouped_clean"] = {**(current or {}), "speaker": value}
        elif key == "_ranking_disclosure":
            for alvo, fonte in value.items():
                bloco = (
                    arch["robustness"][alvo.split(".", 1)[1]]
                    if alvo.startswith("robustness.")
                    else arch[alvo]
                )
                bloco["ranking_score_source"] = fonte
                bloco.setdefault(
                    "threshold_free_metrics", ["auc_roc", "eer", "min_tdcf"]
                )
        else:
            arch[key] = value


def _backup(path: Path) -> Optional[Path]:
    """Copia .pre-backfill.bak uma unica vez — nunca sobrescreve o original."""
    bak = path.with_suffix(path.suffix + ".pre-backfill.bak")
    if bak.exists():
        return None
    shutil.copy2(path, bak)
    return bak


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _protocol_guard_from_summary(
    payload: Dict[str, Any], run_dir: Path
) -> Optional[Dict[str, Any]]:
    """Reconstroi o `academic_protocol_guard` a partir do resumo do run.

    O guard e escrito pelo ORQUESTRADOR (`run_models_sequential.py`), nao pelo
    runner de cada modelo. Os dois runs SSL sairam sem ele, entao os artefatos
    que sustentam a alegacao de teste congelado eram os unicos do escopo que
    nao a carregavam por dentro.

    So reconstroi quando o `run_summary.json` traz o lock VALIDADO e o
    `test_split_sha256` gravado no proprio artefato bate com o que o resumo
    registrou para aquele run — a mesma igualdade que o orquestrador teria
    conferido. Divergiu, devolve None em vez de atestar o que nao pode.

    O bloco sai marcado como derivado: um leitor precisa poder distinguir isto
    de um guard escrito no momento da execucao.
    """
    if payload.get("academic_protocol_guard"):
        return None
    resumo_path = run_dir / "run_summary.json"
    if not resumo_path.is_file():
        return None
    try:
        resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    lock = resumo.get("test_lock") or {}
    if not (resumo.get("academic_protocol") and lock.get("validated")):
        return None

    sha_artefato = (payload.get("dataset") or {}).get("test_split_sha256")
    if not sha_artefato:
        return None
    nomes = set((payload.get("architectures") or {}).keys())
    for modelo in resumo.get("models") or []:
        if modelo.get("model") in nomes:
            if modelo.get("test_split_sha256") != sha_artefato:
                return None
            break
    else:
        return None

    return {
        "test_lock": lock,
        "test_split_sha256": sha_artefato,
        "validated_before_training": True,
        "derived": {
            "by": SCRIPT_ID,
            "from": "run_summary.json (test_lock validado do run)",
            "basis": (
                "o test_split_sha256 do artefato e identico ao que o resumo "
                "registrou para este modelo"
            ),
            "note": (
                "o orquestrador nao gravou o guard neste run; o bloco foi "
                "reconstruido, nao produzido pela execucao"
            ),
        },
    }


def process_run(run_dir: Path, dataset: Optional[Path], write: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {"run": str(run_dir), "architectures": [], "skipped": []}
    results_files = sorted(run_dir.glob("*/results.json"))
    if not results_files:
        results_files = sorted(run_dir.glob("results.json"))

    for results_path in results_files:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        archs = payload.get("architectures")
        if not isinstance(archs, dict):
            continue
        environment = payload.get("environment") or {}
        dataset_block = payload.get("dataset") or {}
        dataset_meta = dataset_block.get("metadata") or {}
        touched_fields: List[str] = []
        derived: Dict[str, Any] = {}
        y_test = dataset_block.get("y_test")
        threshold = float(
            (payload.get("config") or {}).get("decision_threshold", 0.5) or 0.5
        )

        # Os speaker_ids saem primeiro: alimentam tanto o campo do bloco
        # `dataset` quanto o `grouped_clean["speaker"]` de cada arquitetura.
        speaker_ids: Optional[List[str]] = dataset_block.get("test_speaker_ids")
        speaker_basis: Optional[Dict[str, Any]] = None
        if dataset and speaker_ids is None and y_test:
            speaker_ids, speaker_basis = provenance_ids_from_npz(
                dataset, "speaker_ids", y_test
            )

        for name, arch in archs.items():
            if arch.get("status") != "ok":
                continue
            # environment por arquitetura tem precedencia sobre o do run.
            env = arch.get("environment") or environment
            additions, fields = plan_architecture(
                arch,
                env,
                dataset_meta,
                speaker_ids=speaker_ids,
                y_test=y_test,
                decision_threshold=threshold,
                run_log=results_path.parent / "run.log",
            )
            if not fields:
                report["skipped"].append({"file": str(results_path), "arch": name})
                continue
            if write:
                apply_architecture(arch, additions)
                arch["backfill"] = _stamp(fields, {"environment": bool(env)})
                # Espelha no metrics.json da arquitetura: e o mesmo bloco,
                # gravado duas vezes pelo runner.
                for espelho in _sibling_metrics_files(results_path, name):
                    payload_espelho = json.loads(
                        espelho.read_text(encoding="utf-8")
                    )
                    apply_architecture(payload_espelho, additions)
                    payload_espelho["backfill"] = _stamp(
                        fields, {"espelho_de": str(results_path)}
                    )
                    _backup(espelho)
                    _write_json(espelho, payload_espelho)
                    report.setdefault("mirrored", []).append(str(espelho))
            touched_fields.extend(fields)
            report["architectures"].append(
                {
                    "file": str(results_path),
                    "arch": name,
                    "fields": fields,
                    "training_stability": (
                        additions.get("training_stability", {}) or {}
                    ).get("status"),
                }
            )

        dataset_fields: List[str] = []
        dataset_basis: Dict[str, Any] = {}
        if dataset and dataset_block.get("test_cluster_ids") is None and y_test:
            clusters, basis = test_cluster_ids_from_npz(dataset, y_test)
            if write:
                dataset_block["test_cluster_ids"] = clusters
            dataset_fields.append("test_cluster_ids")
            dataset_basis["test_cluster_ids"] = basis
            derived["test_cluster_ids"] = basis

        if speaker_basis is not None:
            if write:
                dataset_block["test_speaker_ids"] = speaker_ids
            dataset_fields.append("test_speaker_ids")
            dataset_basis["test_speaker_ids"] = speaker_basis
            derived["test_speaker_ids"] = speaker_basis

        if dataset_fields:
            if write:
                dataset_block["backfill"] = _stamp(dataset_fields, dataset_basis)
            touched_fields.extend(f"dataset.{f}" for f in dataset_fields)

        guard = _protocol_guard_from_summary(payload, run_dir)
        if guard:
            if write:
                payload["academic_protocol_guard"] = guard
            touched_fields.append("academic_protocol_guard")
            report["architectures"].append(
                {
                    "file": str(results_path),
                    "arch": "(run)",
                    "fields": ["academic_protocol_guard"],
                    "training_stability": None,
                }
            )

        if write and touched_fields:
            _backup(results_path)
            _write_json(results_path, payload)
            # metrics.json e a projecao por arquitetura do mesmo conteudo.
            for name, arch in archs.items():
                if arch.get("status") != "ok" or "backfill" not in arch:
                    continue
                for metrics_path in results_path.parent.glob(
                    "architectures/*/metrics.json"
                ):
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics.get("architecture") != name:
                        continue
                    _backup(metrics_path)
                    for key in (
                        "training_stability",
                        "codec_eval_status",
                        "fit_strategy",
                        "efficiency",
                        "grouped_clean",
                        "model_artifact_fingerprint",
                        "backfill",
                    ):
                        if key in arch:
                            metrics[key] = arch[key]
                    _write_json(metrics_path, metrics)

        if derived:
            report.setdefault("dataset", {})[str(results_path)] = derived

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="diretorio do run")
    parser.add_argument(
        "--dataset",
        default=None,
        help=".npz do run; necessario para preencher test_cluster_ids",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="aplica as mudancas (o padrao apenas inspeciona)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir
    if not run_dir.exists():
        print(f"ERRO: run inexistente: {run_dir}", file=sys.stderr)
        return 1
    dataset = None
    if args.dataset:
        dataset = Path(args.dataset)
        if not dataset.is_absolute():
            dataset = _ROOT / dataset
        if not dataset.exists():
            print(f"ERRO: dataset inexistente: {dataset}", file=sys.stderr)
            return 1

    report = process_run(run_dir, dataset, write=args.write)
    modo = "APLICADO" if args.write else "SIMULACAO (use --write para aplicar)"
    print(f"== Backfill de metadata — {modo}")
    print(f"   run: {report['run']}")
    for entry in report["architectures"]:
        estab = entry.get("training_stability") or "-"
        print(f"   + {entry['arch']:<24} {', '.join(entry['fields'])}")
        if "training_stability" in entry["fields"]:
            print(f"     {'':<24} training_stability = {estab}")
    for entry in report["skipped"]:
        print(f"   = {entry['arch']:<24} ja completo, nada a fazer")
    seen_basis: Dict[str, Dict[str, Any]] = {}
    for _path, derived in (report.get("dataset") or {}).items():
        seen_basis.update(derived)
    for field, basis in seen_basis.items():
        print(
            f"   + dataset.{field:<18} {basis['n_groups']} grupos "
            f"(fatia {basis['slice']}, y_test conferido)"
        )
    if not args.write:
        print("\n   Nada foi escrito. Repita com --write para aplicar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
