#!/usr/bin/env python3
"""Benchmark de SSL original (PyTorch/Hugging Face).

Este runner cobre WavLM e HuBERT reais via PyTorch. O backbone Hugging Face e
carregado congelado por padrao, e a cabeca de classificacao e treinada sobre
embeddings SSL reais. No fluxo WSL/Docker, HuBERT usa este caminho para evitar
o fallback Keras e preservar o backbone base congelado durante o treino.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from benchmarks.data import BenchmarkData

# Mesma coleta de versoes do runner Keras, reusada de proposito: duas listas
# paralelas divergiriam, e o campo so serve para comparar dois artefatos.
from benchmarks.runner import _library_versions

logger = logging.getLogger("ssl_original_benchmark")


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


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64").ravel()
    if np.any(~np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _fit_length(flat: np.ndarray, target_len: int) -> np.ndarray:
    if flat.shape[1] == target_len:
        return flat
    if flat.shape[1] > target_len:
        start = max(0, (flat.shape[1] - target_len) // 2)
        return flat[:, start : start + target_len]
    repeats = int(np.ceil(target_len / max(1, flat.shape[1])))
    return np.tile(flat, (1, repeats))[:, :target_len]


def _length_strategy(source_len: int, target_len: int) -> str:
    """Nome da operação que `_fit_length` realmente aplica.

    O artefato declarava `crop_strategy: "center"` de forma fixa, o que é falso
    quando o clipe é MENOR que a janela pedida — aí `_fit_length` REPETE o sinal
    (tiling), e não recorta. Com o dataset de 3 s (48.000) e o default legado de
    64.000 isso acontecia em todas as amostras, sem nada no artefato dizendo.
    """
    if source_len == target_len:
        return "identity"
    if source_len > target_len:
        return "center_crop"
    return "tile_repeat"


def _input_preparation_block(source_len: int, target_len: int) -> dict[str, Any]:
    """Descrição honesta do preparo de entrada, para gravar no artefato."""
    strategy = _length_strategy(source_len, target_len)
    block: dict[str, Any] = {
        "input_type": "raw_audio",
        "original_shape": [int(source_len), 1],
        "prepared_shape": [int(target_len), 1],
        "sample_rate": 16000,
        "crop_strategy": strategy,
    }
    if strategy == "tile_repeat":
        block["tiled_padding_samples"] = int(target_len - source_len)
        block["tiled_padding_ratio"] = round((target_len - source_len) / target_len, 4)
    return block


class _LoadedData:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str,
        metadata: dict[str, Any],
        original_shape: list[int],
        test_cluster_ids: np.ndarray | None = None,
        test_speaker_ids: np.ndarray | None = None,
    ):
        self.X = X
        self.y = y
        self.name = name
        self.metadata = metadata
        self.original_shape = original_shape
        # Necessário para o bootstrap por CLUSTER (ver _load_dataset).
        self.test_cluster_ids = test_cluster_ids
        # Unidade alternativa de reamostragem e base do grouped_clean["speaker"].
        self.test_speaker_ids = test_speaker_ids


def _band_correction_block(band_correction_hz: float | None) -> dict | None:
    """Politica de correcao de formato, no MESMO formato do runner Keras.

    Espelha `benchmarks/runner.py::_band_correction_policy` para que os
    artefatos das 11 entradas oficiais sejam comparaveis campo a campo.
    """
    if not band_correction_hz:
        return None
    from app.domain.features.benchmark_frontend import (
        BAND_CORRECTION_RMS_DBFS,
        BAND_CORRECTION_TAPS,
    )

    return {
        "cutoff_hz": float(band_correction_hz),
        "taps": int(BAND_CORRECTION_TAPS),
        "remove_dc": True,
        "renormalize_rms_dbfs": float(BAND_CORRECTION_RMS_DBFS),
        "origem": "runner_ssl",
    }


def _load_dataset(path: str, seed: int, band_correction_hz: float | None = None):
    """Carrega o NPZ preservando partições oficiais quando disponíveis.

    CORREÇÃO DE FORMATO (2026-08-19). Este runner tem caminho de dados PRÓPRIO
    -- não passa por `benchmarks.runner.run_benchmark` -- e por isso ficava de
    fora da correção que os outros nove modelos recebem. Treinar WavLM e HuBERT
    sobre o corpus não corrigido, enquanto os demais treinam sobre o corrigido,
    tornaria a tabela INCOMPARÁVEL: dois modelos com acesso ao atalho de
    reamostragem (AUC 0,98 sozinho) e nove sem.
    """

    p = Path(path)
    data = BenchmarkData.from_npz(str(p))
    splits = data.stratified_split(seed=seed, preserve_predefined=True)
    if band_correction_hz:
        from app.domain.features.benchmark_frontend import apply_band_correction

        Xtr, ytr, Xv, yv, Xte, yte = splits
        splits = (
            apply_band_correction(Xtr, float(band_correction_hz)),
            ytr,
            apply_band_correction(Xv, float(band_correction_hz)),
            yv,
            apply_band_correction(Xte, float(band_correction_hz)),
            yte,
        )
        logger.info(
            "[protocolo] correção de banda aplicada às três partições "
            "(passa-baixas %.0f Hz)", float(band_correction_hz)
        )
    metadata = dict(data.metadata or {"source": str(p), "npz_path": str(p)})
    metadata["split_source"] = (
        "predefined_npz" if data.predefined_split_indices else "stratified_seed"
    )
    from benchmarks.runner import _audit_split_provenance
    metadata["provenance_overlap_audit"] = _audit_split_provenance(data)
    original_shape = list(np.asarray(data.X).shape[1:])
    # AJUSTE 2026-08-06: sem os cluster_ids do teste, `evaluate_scores` cai no
    # bootstrap por AMOSTRA, enquanto os outros 9 modelos do escopo oficial
    # usam bootstrap por CLUSTER (183 clusters). ICs calculados com unidades
    # diferentes NÃO são comparáveis na mesma tabela — o por-amostra
    # subestima a largura porque trata amostras do mesmo locutor/frase como
    # independentes. Extraído aqui porque `data` é solto logo abaixo.
    test_cluster_ids = None
    test_speaker_ids = None
    test_idx = (getattr(data, "last_split_indices", None) or {}).get("test")
    if test_idx is not None and data.cluster_ids is not None:
        test_cluster_ids = np.asarray(data.cluster_ids)[np.asarray(test_idx, dtype=int)]
    # Mesma extração para os LOCUTORES: alimenta o grouped_clean["speaker"] e a
    # unidade de IC por locutor da consolidação. Como os cluster_ids, precisa
    # sair aqui — `data` é solto logo abaixo para liberar RAM.
    if test_idx is not None and getattr(data, "speakers", None) is not None:
        test_speaker_ids = np.asarray(data.speakers)[np.asarray(test_idx, dtype=int)]
    # `stratified_split` já copiou os splits (X_train/X_val/X_test) para
    # arrays independentes — o array cheio pre-split (data.X, todas as
    # amostras, maior que qualquer split individual) não é lido de novo
    # depois daqui. Sem soltar a referência, ele fica retido pelo objeto
    # `loaded` (usado até o fim do runner por causa de `loaded.y`) somando
    # vários GB de RAM ociosa ao pico já apertado do embed() SSL.
    loaded = _LoadedData(
        None,
        data.y,
        data.name,
        metadata,
        original_shape,
        test_cluster_ids=test_cluster_ids,
        test_speaker_ids=test_speaker_ids,
    )
    return loaded, splits


def _normalize_wave_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    flat = x.reshape(len(x), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    flat = (flat - mean) / np.maximum(std, 1e-6)
    return np.clip(flat, -5.0, 5.0).astype("float32")


def _add_awgn_raw(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """AWGN canônico no domínio da forma de onda."""
    return BenchmarkData.add_awgn(X, snr_db=snr_db, seed=seed)


def _count_torch_params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _path_size_mb(path: Path) -> float:
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / (1024 * 1024), 2)


# REMOVIDAS em 2026-08-06: `_write_predictions`, `_write_predictions_noisy` e
# `_write_robustness`. Escreviam `predictions_clean.csv`,
# `predictions_robustness.csv` e `robustness.csv` com um schema PRÓPRIO
# (`idx`/`snr_db`, sem `y_pred`/`correct`) e eram sobrescritas segundos depois
# por `benchmarks.report.write_all`, o writer canônico dos 11 modelos. A
# duplicação escondeu um bug real: como o dict de resultados não trazia
# `scores_robustness`, o writer canônico regravava o CSV de ruído só com o
# cabeçalho, e WavLM/HuBERT Original ficaram sem nenhuma predição por amostra
# sob ruído no clean_benchmark_15k. Este runner NÃO escreve esses três
# arquivos; ele alimenta o dict que o `write_all` consome.


def _write_model_card(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['display_name']}",
        "",
        f"Modelo treinado com backbone Hugging Face `{payload['model_class']}` real.",
        "",
        f"- Backbone: `{payload['model_name']}`",
        f"- Backbone congelado: `{payload['freeze_backbone']}`",
        f"- Epocas da cabeca: `{payload['epochs']}` "
        f"(treinadas: `{payload.get('epochs_trained', '?')}`, "
        f"melhor: `{payload.get('best_epoch', '?')}`)",
        f"- Janela de entrada: `{payload['input_shape'][0]}` amostras @16 kHz "
        f"(`{payload.get('input_preparation', {}).get('crop_strategy', '?')}` "
        f"sobre o clipe original)",
        f"- Batch embeddings: `{payload['feature_batch_size']}`",
        f"- Batch treino: `{payload['train_batch_size']}`",
        f"- Augmentation de ruido no treino: "
        f"`{payload.get('train_augmentation', False)}` "
        f"(SNRs `{payload.get('train_aug_snr_db', [])}` dB)",
        f"- Threshold de decisao calibrado: "
        f"`{payload.get('decision_threshold', 0.5)}`",
        f"- Artefato: `{payload['artifact']}`",
        f"- Backbone local: `{payload['backbone_artifact']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_scores(
    y_true: np.ndarray, p_fake: np.ndarray, threshold: float = 0.5,
    n_bootstrap: int = 1000, cluster_ids: np.ndarray | None = None
) -> dict[str, float]:
    """Delegado ao evaluate_scores canônico do benchmark.

    AJUSTE 2026-07-14: este runner tinha uma implementação própria de EER e
    principalmente de min t-DCF (fórmula simplificada com p_target=0.01),
    DIFERENTE do t-DCF ASVspoof2019 CM-only do MetricsCalculator usado para
    os outros 9 modelos — os números de WavLM/HuBERT Original não eram
    comparáveis na tabela consolidada. benchmarks.evaluate importa apenas
    numpy/sklearn (sem TensorFlow), então é seguro neste runner PyTorch.
    """
    from benchmarks.evaluate import evaluate_scores

    return evaluate_scores(
        y_true, p_fake, threshold=threshold, n_bootstrap=n_bootstrap,
        cluster_ids=cluster_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band-correction-hz", type=float, default=None,
        help="passa-baixas (Hz) aplicado as duas classes antes de qualquer "
             "frontend. Use 7500 para acompanhar os demais modelos "
             "do escopo; omitir deixa este runner com o atalho de "
             "reamostragem que os outros nove nao tem.")
    parser.add_argument(
        "--dataset",
        default="data/datasets/benchmark_audio_raw_balanced_15k_confirmatory_v2.npz",
        help="Dataset .npz balanceado com audio bruto.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Diretorio de saida.",
    )
    parser.add_argument(
        "--architecture",
        choices=["wavlm", "hubert"],
        default="hubert",
        help="Backbone SSL original a executar.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    # Avaliacao inclui o 5 dB NAO VISTO, em paridade com o caminho Keras.
    parser.add_argument("--snr", nargs="+", type=int, default=[30, 20, 10, 5])
    parser.add_argument(
        "--train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anexa copias do treino com AWGN (paridade com o caminho Keras; "
        "sem isso o recall colapsa sob ruido).",
    )
    parser.add_argument(
        "--train-aug-snr",
        nargs="+",
        type=int,
        default=[30, 20, 10],
        help="SNRs (dB) das copias de treino com ruido.",
    )
    parser.add_argument("--waveform-noise-batch-size", type=int, default=64)
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablação: interrompe antes das 100 épocas; o protocolo principal "
        "sempre restaura o melhor checkpoint em validação limpa.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--calibrate-under-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calibra opcionalmente o threshold em validação com ruído; "
        "desativado por padrão para manter o limiar comum de 0,5.",
    )
    parser.add_argument(
        "--calibration-snr",
        nargs="+",
        type=int,
        default=[20, 10],
        help="SNRs (dB) da validação ruidosa usada somente quando a calibração opcional está ativa.",
    )
    parser.add_argument("--latency-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Congela o backbone original e treina apenas a cabeca.",
    )
    # AJUSTE 2026-07-15 (acurácia): antes o SSL via só 1 s central (16000)
    # dos 5 s — os espectrais veem os 5 s inteiros. 4 s quadruplica a
    # evidência por clipe a custo pequeno (T≈199 frames no transformer).
    #
    # AJUSTE 2026-08-09: 64000 (4 s) virou dívida quando o dataset canônico
    # passou a ter clipes de 3 s (48.000 amostras). Como 48.000 < 64.000,
    # `_fit_length` deixava de recortar e passava a REPETIR o primeiro segundo
    # de cada clipe — 25% da janela era sinal duplicado, em treino e em teste,
    # sem nada no artefato registrando. 48000 = o clipe inteiro, sem recorte e
    # sem repetição, que é o que os espectrais também veem.
    parser.add_argument(
        "--target-samples",
        type=int,
        default=48000,
        help=(
            "Janela da forma de onda em amostras @16 kHz. O default casa com "
            "o clipe de 3 s do dataset canônico: nem recorte, nem repetição. "
            "Valores MAIORES que o clipe fazem tiling (histórico: 64000); "
            "menores recortam o centro (legado: 16000)."
        ),
    )
    parser.add_argument(
        "--layer-pooling",
        choices=["weighted", "last"],
        default="weighted",
        help="'weighted' = soma ponderada aprendida sobre TODAS as camadas "
        "(padrão SUPERB; camadas intermediárias carregam os artefatos); "
        "'last' = comportamento legado.",
    )
    parser.add_argument(
        "--time-pooling",
        choices=["meanstd", "mean"],
        default="meanstd",
        help="Pooling temporal por camada: média⊕desvio (default) ou média.",
    )
    # ── Back-end e fine-tuning (2026-08-09) ────────────────────────────────
    parser.add_argument(
        "--backend",
        choices=["mlp", "aasist"],
        default="mlp",
        help=(
            "'mlp' (default): receita de probing do SUPERB — pooling temporal "
            "global e MLP sobre embeddings em CACHE, backbone necessariamente "
            "congelado. 'aasist': a sequência inteira alimenta o grafo "
            "espectro-temporal (Jung et al., ICASSP 2022) e o treino é "
            "fim-a-fim, sem cache — é a receita das submissões de campeonato "
            "(Tak et al., Odyssey 2022) e permite destravar o backbone."
        ),
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help=(
            "LR do backbone no fine-tuning (discriminativo: duas ordens de "
            "grandeza abaixo do back-end, que usa --learning-rate). Ignorado "
            "com o backbone congelado."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Retoma de `training_state_<arch>.pt` se existir. O fim-a-fim leva "
            "~10 h por modelo e este runner não tinha checkpoint intermediário: "
            "uma queda perdia tudo."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Grava o estado de retomada a cada N épocas (0 desliga). O custo é "
            "uma escrita de ~380 MB contra ~6 min de época."
        ),
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help=(
            "Passos de acumulação de gradiente. O lote EFETIVO é "
            "--train-batch-size x --grad-accum; serve para manter o lote "
            "efetivo quando a VRAM força lotes pequenos no fim-a-fim."
        ),
    )
    args = parser.parse_args()

    # O caminho de cache de embeddings calcula o backbone UMA vez sob
    # `no_grad()`: destravá-lo ali não treinaria nada (era o bug de
    # `--no-freeze-backbone` até 2026-08-09, que produzia um run congelado
    # rotulado como treinável). Em vez de aceitar a combinação e mentir no
    # artefato, ela é recusada na entrada.
    if not args.freeze_backbone and args.backend == "mlp":
        parser.error(
            "--no-freeze-backbone exige --backend aasist: o caminho 'mlp' "
            "pré-calcula embeddings sob no_grad() e não propaga gradiente ao "
            "backbone. Use --backend aasist (treino fim-a-fim) ou mantenha o "
            "backbone congelado."
        )

    # A combinação SIMÉTRICA mente do outro lado. `--backend aasist` renomeia a
    # entrada para "WavLM AASIST"/"HuBERT AASIST" (mais abaixo), e o manifesto
    # oficial declara essas duas com o variante
    # `...:pytorch_finetuned_aasist_graph`. Com o backbone CONGELADO o artefato
    # sairia sob esse nome e esse variante tendo treinado só o back-end — a
    # ablação errada rotulada como a receita de Tak et al.
    #
    # `--freeze-backbone` é o DEFAULT do parser (serve às entradas `Original`),
    # então a combinação perigosa é a que se obtém esquecendo uma flag, não a
    # que se pede de propósito. Recusar na entrada é o mesmo tratamento dado à
    # combinação acima: não existe entrada de manifesto para um AASIST
    # congelado, então produzir um seria produzir um resultado sem lugar.
    if args.freeze_backbone and args.backend == "aasist":
        parser.error(
            "--backend aasist exige --no-freeze-backbone: as entradas 'WavLM "
            "AASIST'/'HuBERT AASIST' do manifesto oficial declaram backbone "
            "AJUSTADO (pytorch_finetuned_aasist_graph). Congelado, o artefato "
            "sairia rotulado como fine-tuning tendo treinado só o back-end. "
            "Para o contraste congelado, use as entradas 'Original' "
            "(--backend mlp)."
        )

    arch_meta = {
        "wavlm": {
            "display": "WavLM Original",
            "compact": "wavlm_original",
            "model_class": "WavLMModel",
            # AJUSTE 2026-07-15 (acurácia): base→base-plus. Mesmo tamanho/
            # arquitetura, pré-treinado em 94k h (vs 960 h) — ganho documentado
            # em robustez/anti-spoofing sem custo de inferência.
            "default_model": "microsoft/wavlm-base-plus",
            "default_out": "data/results/benchmark_wavlm_original_gpu_100e",
            "artifact": "bench_wavlm_original.pt",
        },
        "hubert": {
            "display": "HuBERT Original",
            "compact": "hubert_original",
            "model_class": "HubertModel",
            "default_model": "facebook/hubert-base-ls960",
            "default_out": "data/results/benchmark_hubert_original_gpu_100e",
            "artifact": "bench_hubert_original.pt",
        },
    }[args.architecture]
    # A variante com grafo é uma ENTRADA DISTINTA no manifesto oficial
    # (`WavLM AASIST`/`HuBERT AASIST`), não uma reconfiguração da congelada:
    # as duas convivem na mesma tabela como ablação. Sem renomear aqui, as
    # duas gravariam sob a mesma chave em `architectures` e a segunda
    # sobrescreveria a primeira na consolidação.
    if args.backend == "aasist":
        base = arch_meta["display"].split()[0]          # "WavLM" / "HuBERT"
        arch_meta = {
            **arch_meta,
            "display": f"{base} AASIST",
            "compact": f"{base.lower()}_aasist",
            "artifact": f"bench_{base.lower()}_aasist.pt",
            "default_out": (
                f"data/results/benchmark_{base.lower()}_aasist_gpu_100e"
            ),
        }
    if args.model_name is None:
        args.model_name = arch_meta["default_model"]
    if args.out is None:
        args.out = arch_meta["default_out"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _set_seed(args.seed)

    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import HubertModel, WavLMModel

    from benchmarks.report import write_all
    BackboneModel = WavLMModel if args.architecture == "wavlm" else HubertModel

    out = Path(args.out)
    arch_out = out / "architectures" / arch_meta["compact"]
    models_dir = arch_out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device PyTorch: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Carregando dataset: %s", args.dataset)
    data, splits = _load_dataset(
        args.dataset, args.seed,
        band_correction_hz=getattr(args, "band_correction_hz", None),
    )
    from benchmarks.runner import (
        _architecture_provenance,
        _audit_split_overlap,
        _file_fingerprint,
    )
    split_overlap_audit = _audit_split_overlap(splits, fail_on_overlap=True)
    split_fingerprints = split_overlap_audit["split_fingerprints"]
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    logger.info("Carregando %s: %s", arch_meta["display"], args.model_name)
    backbone = BackboneModel.from_pretrained(args.model_name).to(device)

    from app.domain.models.architectures.torch_ssl_aasist import (
        SSLAASISTModel,
        configure_finetuning,
    )

    # A declaração de treinabilidade sai da CONTAGEM de parâmetros com
    # `requires_grad`, não da flag: era a flag que mentia antes.
    finetune_info = configure_finetuning(backbone, args.freeze_backbone)
    logger.info("Treinabilidade do backbone: %s", finetune_info)

    from app.domain.models.inference.ssl_head import (
        build_ssl_classifier,
        pool_hidden_states,
    )

    hidden_size = int(backbone.config.hidden_size)
    # Contrato de embedding (2026-07-15): gravado no checkpoint e honrado
    # pelo wrapper de inferência (TorchSSLOriginalModel) — paridade garantida.
    num_hidden_layers = int(getattr(backbone.config, "num_hidden_layers", 12))
    feature_dim = hidden_size * (2 if args.time_pooling == "meanstd" else 1)
    embedding_config = {
        "target_samples": int(args.target_samples),
        "layer_pooling": args.layer_pooling,
        "time_pooling": args.time_pooling,
        "num_layers": (
            num_hidden_layers + 1 if args.layer_pooling == "weighted" else 1
        ),
        "feature_dim": int(feature_dim),
    }
    end_to_end = args.backend == "aasist"
    if end_to_end:
        # No fim-a-fim o pooling temporal NÃO acontece antes do classificador —
        # a sequência inteira vai ao grafo. O contrato registra isso para que a
        # inferência não tente reconstruir a receita de pooling do caminho MLP.
        embedding_config.update(
            {
                "backend": "aasist_graph",
                "time_pooling": "none_sequence_to_graph",
                "feature_dim": int(hidden_size),
            }
        )
    logger.info("Contrato de embedding: %s", embedding_config)

    if end_to_end:
        model = SSLAASISTModel(
            backbone,
            num_layers=num_hidden_layers + 1,
            hidden_size=hidden_size,
            dropout_rate=args.dropout,
        ).to(device)
        classifier = model.backend
        logger.info(
            "Back-end AASIST: %s params treináveis (backbone: %s)",
            f"{sum(p.numel() for p in model.backend.parameters()):,}",
            f"{finetune_info['backbone_trainable_params']:,}",
        )
    else:
        model = None
        classifier = build_ssl_classifier(embedding_config, args.dropout).to(device)

    need_hidden_states = args.layer_pooling == "weighted" or end_to_end

    def embed(X: np.ndarray, label: str) -> np.ndarray:
        # AJUSTE 2026-07-31: `_fit_length`/`_normalize_wave_batch` eram
        # aplicados ao split INTEIRO antes do DataLoader existir — para o
        # split de treino (maior) isso materializa uma copia extra do
        # tamanho de `X` (e.g. ~33k amostras x 64.000 x 4 B ~ 8,5 GB) ao
        # mesmo tempo em que `X_train` continua vivo no escopo externo (usado
        # depois pelo loop de augmentation AWGN), dobrando o pico logo no
        # 1o `embed(X_train, ...)` — mesma classe de OOM de host RAM ja
        # corrigida em benchmarks/runner.py (bloco cru + bloco preparado
        # vivos ao mesmo tempo). Recorte/normalizacao agora rodam por
        # micro-lote (`feature_batch_size`), entao o pico extra por chamada
        # cai do tamanho do split inteiro para o de um lote.
        backbone.eval()
        n = len(X)
        batch = max(1, int(args.feature_batch_size))
        n_batches = (n + batch - 1) // batch
        chunks = []
        started = time.time()
        with torch.no_grad():
            for step, start in enumerate(range(0, n, batch), start=1):
                stop = min(start + batch, n)
                raw = np.asarray(X[start:stop], dtype="float32").reshape(
                    stop - start, -1
                )
                Xn = _normalize_wave_batch(_fit_length(raw, int(args.target_samples)))
                xb = torch.from_numpy(Xn).to(device, non_blocking=True)
                with torch.amp.autocast(
                    "cuda", enabled=(device.type == "cuda"), dtype=torch.float16
                ):
                    outp = backbone(
                        xb, output_hidden_states=need_hidden_states
                    )
                    pooled = pool_hidden_states(outp, embedding_config)
                chunks.append(pooled.detach().float().cpu().numpy())
                if step == 1 or step % 50 == 0 or step == n_batches:
                    logger.info(
                        "Embeddings %s: %d/%d batches", label, step, n_batches
                    )
        logger.info(
            "Embeddings %s concluidos em %.1fs", label, time.time() - started
        )
        return np.concatenate(chunks, axis=0).astype("float32")

    def _prepare_waves(X: np.ndarray) -> np.ndarray:
        """Recorte/repetição + normalização, sem passar pelo backbone.

        No fim-a-fim o backbone roda DENTRO do laço de treino (é o que permite
        o gradiente chegar nele), então o pré-processamento tem de acontecer
        antes, uma vez só.
        """
        flat = np.asarray(X, dtype="float32").reshape(len(X), -1)
        return _normalize_wave_batch(_fit_length(flat, int(args.target_samples)))

    if end_to_end:
        Z_train = _prepare_waves(X_train)
        Z_val = _prepare_waves(X_val)
        Z_test = _prepare_waves(X_test)
    else:
        Z_train = embed(X_train, "train")
        Z_val = embed(X_val, "val")
        Z_test = embed(X_test, "test")

    # Uma cópia AWGN por amostra é criada na forma de onda canônica antes
    # do recorte e da extração de embeddings, como nas demais arquiteturas.
    Z_train_fit = Z_train
    y_train_fit = y_train
    assigned_snr_counts: dict[str, int] = {}
    if args.train_augmentation and args.train_aug_snr:
        if args.waveform_noise_batch_size <= 0:
            raise ValueError("--waveform-noise-batch-size deve ser > 0")
        base_seed = args.seed + 10000
        assigned = BenchmarkData.balanced_snr_assignments(
            len(X_train), args.train_aug_snr, seed=base_seed
        )
        noisy_embedding_chunks = []
        for start in range(0, len(X_train), args.waveform_noise_batch_size):
            stop = min(start + args.waveform_noise_batch_size, len(X_train))
            noisy_chunk = BenchmarkData.add_awgn_assigned(
                X_train[start:stop],
                assigned[start:stop],
                seed=base_seed + start,
            )
            # MESMA realização de ruído nos dois caminhos (mesma semente, mesmo
            # troceamento): o que muda é só o que se guarda — embedding no
            # caminho MLP, forma de onda preparada no fim-a-fim.
            noisy_embedding_chunks.append(
                _prepare_waves(noisy_chunk) if end_to_end
                else embed(noisy_chunk, f"train_awgn_{start}_{stop}")
            )
        Z_train_noisy = np.concatenate(noisy_embedding_chunks, axis=0)
        Z_train_fit = np.concatenate([Z_train, Z_train_noisy], axis=0)
        y_train_fit = np.tile(y_train, 2)
        values, counts = np.unique(assigned, return_counts=True)
        assigned_snr_counts = {
            str(int(value)): int(count)
            for value, count in zip(values, counts)
        }
        logger.info(
            "Treino aumentado no waveform: %d -> %d amostras (SNRs %s; %s)",
            len(y_train), len(y_train_fit), args.train_aug_snr,
            assigned_snr_counts,
        )

    # A seleção do checkpoint usa validação limpa. Validação ruidosa só é
    # construída se a ablação de calibração for solicitada explicitamente.
    Z_val_monitor = Z_val
    y_val_monitor = y_val
    if args.calibrate_under_noise and args.calibration_snr:
        val_chunks = [Z_val]
        for snr in args.calibration_snr:
            noisy_val = _add_awgn_raw(X_val, snr, seed=args.seed + 2000 + int(snr))
            val_chunks.append(embed(noisy_val, f"val_snr_{snr}"))
        Z_val_monitor = np.concatenate(val_chunks, axis=0)
        y_val_monitor = np.tile(y_val, len(val_chunks))

    train_ds = TensorDataset(
        torch.from_numpy(Z_train_fit), torch.from_numpy(y_train_fit.astype("int64"))
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=False
    )
    criterion = nn.CrossEntropyLoss()

    # `net` é o que treina: no fim-a-fim inclui o backbone; no caminho MLP é
    # só a cabeça sobre embeddings em cache.
    net = model if end_to_end else classifier

    if end_to_end and not args.freeze_backbone:
        # LR discriminativo: o backbone já está pré-treinado e um passo grande
        # apaga a representação (catastrophic forgetting); o back-end começa do
        # zero e precisa de passo maior. Duas ordens de grandeza de diferença é
        # a prática dos trabalhos que destravam o front-end SSL.
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                {
                    "params": list(model.backend.parameters())
                    + list(model.layer_sum.parameters()),
                    "lr": args.learning_rate,
                },
            ],
            weight_decay=args.weight_decay,
        )
        logger.info(
            "Otimizador discriminativo: backbone lr=%.2e | back-end lr=%.2e",
            args.backbone_lr, args.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and end_to_end)

    def _forward(xb: torch.Tensor) -> torch.Tensor:
        return net(xb)

    @torch.no_grad()
    def _evaluate(Z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Avaliação em LOTES.

        O caminho MLP cabia num tensor só na GPU (embeddings são pequenos); com
        o backbone no grafo, um lote de 1.456 formas de onda de 3 s estoura a
        VRAM. Lotes também aqui.
        """
        net.eval()
        total_loss = 0.0
        correct = 0
        n = len(Z)
        bs = max(1, int(args.feature_batch_size) if end_to_end
                 else int(args.train_batch_size))
        yt_all = torch.from_numpy(y.astype("int64"))
        for start in range(0, n, bs):
            stop = min(start + bs, n)
            xb = torch.from_numpy(Z[start:stop]).to(device, non_blocking=True)
            yb = yt_all[start:stop].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = _forward(xb)
            logits = logits.float()
            total_loss += float(criterion(logits, yb)) * (stop - start)
            correct += int((logits.argmax(dim=1) == yb).sum())
        return total_loss / max(n, 1), correct / max(n, 1)

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None

    # ── Retomada (2026-08-09) ─────────────────────────────────────────────
    #
    # O caminho congelado treina a cabeça sobre embeddings em cache e leva ~5
    # min: perder o run e refazer não custava nada. O fim-a-fim leva ~10 h por
    # modelo, e este runner não tinha checkpoint intermediário NENHUM — uma
    # queda de energia na hora 9 jogava fora as 9 horas. O caminho Keras já se
    # protege com `BackupAndRestore`; aqui não havia equivalente.
    #
    # Grava estado a cada época (pesos do melhor checkpoint + histórico +
    # contador), com escrita ATÔMICA: uma queda no meio da gravação deixa o
    # arquivo anterior intacto em vez de um `.pt` truncado.
    state_path = models_dir / f"training_state_{arch_meta['compact']}.pt"
    start_epoch = 1
    if args.resume and state_path.is_file():
        try:
            saved = torch.load(state_path, map_location=device, weights_only=False)
            net.load_state_dict(saved["net_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            history = saved["history"]
            best_val_loss = saved["best_val_loss"]
            best_epoch = saved["best_epoch"]
            best_state = saved["best_state"]
            start_epoch = int(saved["epoch"]) + 1
            logger.info(
                "Retomando da epoca %d (melhor val_loss=%.4f na epoca %d)",
                start_epoch, best_val_loss, best_epoch,
            )
        except Exception as exc:  # noqa: BLE001
            # Estado corrompido não pode virar treino silenciosamente errado.
            logger.warning(
                "Estado de retomada ilegivel (%s) — recomecando do zero", exc
            )
            start_epoch = 1

    def _save_training_state(epoch: int) -> None:
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": int(epoch),
                "net_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "args": vars(args),
            },
            tmp,
        )
        tmp.replace(state_path)

    logger.info(
        "Treinando %s (%s) por %d epocas",
        arch_meta["display"],
        (
            "fim-a-fim: backbone + grafo"
            if end_to_end and not args.freeze_backbone
            else "grafo sobre backbone congelado"
            if end_to_end
            else "cabeca sobre embeddings em cache"
        ),
        args.epochs,
    )
    started_train = time.time()
    epochs_without_improvement = 0
    accum = max(1, int(args.grad_accum))
    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        if end_to_end and args.freeze_backbone:
            # BatchNorm/dropout do backbone precisam ficar em modo eval quando
            # ele está congelado, senão as estatísticas correntes continuam
            # sendo atualizadas por um módulo que não treina.
            model.backbone.eval()
        losses = []
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits = _forward(xb)
                loss = criterion(logits.float(), yb)
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(dim=1) == yb).sum().detach())
            total += int(yb.numel())
            if end_to_end and (step == 1 or step % 200 == 0):
                logger.info(
                    "  epoca %d: lote %d/%d loss=%.4f",
                    epoch, step, len(train_loader), losses[-1],
                )

        val_loss, val_acc = _evaluate(Z_val_monitor, y_val_monitor)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_acc = float(correct / max(total, 1))
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoca %03d/%03d - loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        # Early stopping em val_loss (val com ruido) + restore_best_weights,
        # como nos callbacks do ModelTrainer. Sem isto a cabeca roda as 100
        # epocas superajustando (val_loss minima ~ep13 nos runs anteriores).
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            # `net`, não `classifier`: no fim-a-fim os pesos do backbone também
            # mudam a cada época e precisam entrar no checkpoint, senão a
            # restauração devolveria um backbone de outra época.
            best_state = {
                k: v.detach().clone().cpu() for k, v in net.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                logger.info(
                    "Early stopping na epoca %d (melhor val_loss=%.4f na epoca %d)",
                    epoch,
                    best_val_loss,
                    best_epoch,
                )
                _save_training_state(epoch)
                break

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            _save_training_state(epoch)

    epochs_trained = len(history["loss"])
    if best_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info("Pesos restaurados da melhor epoca (%d)", best_epoch)

    # Treino concluído: o estado de retomada deixa de fazer sentido e, se
    # ficasse, um `--resume` posterior retomaria de uma execução encerrada
    # (foi exatamente esse o defeito que invalidou o Conformer em 2026-08-06).
    if state_path.is_file():
        state_path.unlink()

    train_time_s = round(time.time() - started_train, 3)

    def predict_scores_from_embeddings(Z: np.ndarray) -> np.ndarray:
        net.eval()
        bs = max(1, int(args.feature_batch_size) if end_to_end
                 else int(args.train_batch_size))
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Z.astype("float32"))),
            batch_size=bs,
            shuffle=False,
        )
        scores = []
        with torch.no_grad():
            for (xb,) in loader:
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=torch.float16
                ):
                    logits = _forward(xb.to(device, non_blocking=True))
                scores.append(
                    torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                )
        return _finite_scores(np.concatenate(scores))

    # A comparação principal usa o limiar comum de 0,5. A calibração em
    # validação ruidosa permanece disponível apenas como ablação explícita.
    decision_threshold = 0.5
    calibration_info: dict[str, Any] = {
        "calibrated_under_noise": bool(args.calibrate_under_noise),
        "calibration_snr_db": [int(s) for s in args.calibration_snr],
        "threshold_source": "default_0.5",
    }
    if args.calibrate_under_noise:
        val_scores = predict_scores_from_embeddings(Z_val_monitor)
        val_eval = _evaluate_scores(y_val_monitor, val_scores)
        thr = val_eval.get("eer_threshold")
        if thr is not None and np.isfinite(thr):
            decision_threshold = float(thr)
            calibration_info["threshold_source"] = "val_noisy_eer"
            calibration_info["val_eer"] = val_eval.get("eer")
    calibration_info["decision_threshold"] = decision_threshold
    logger.info(
        "Threshold de decisao: %.4f (%s)",
        decision_threshold,
        calibration_info["threshold_source"],
    )

    test_cluster_ids = getattr(data, "test_cluster_ids", None)
    if test_cluster_ids is None:
        logger.warning(
            "cluster_ids do teste indisponíveis — os IC 95%% sairão por AMOSTRA "
            "e NÃO serão comparáveis com os dos modelos Keras (por cluster)."
        )

    test_speaker_ids = getattr(data, "test_speaker_ids", None)

    scores_clean = predict_scores_from_embeddings(Z_test)
    clean = _evaluate_scores(
        y_test, scores_clean, threshold=decision_threshold,
        cluster_ids=test_cluster_ids,
    )

    # PARIDADE 2026-08-09: até aqui o runner SSL não emitia `grouped_clean`
    # nenhum, enquanto os 9 modelos Keras traziam o bloco — WavLM/HuBERT
    # simplesmente não apareciam nas comparações por grupo. Usa o mesmo
    # `evaluate_grouped_scores` do runner Keras, para que a coluna de pior
    # locutor exista nos 11.
    grouped_clean: dict[str, Any] = {}
    if test_speaker_ids is not None:
        from benchmarks.evaluate import evaluate_grouped_scores

        grouped_clean["speaker"] = evaluate_grouped_scores(
            y_test, scores_clean, test_speaker_ids, threshold=decision_threshold
        )
    else:
        logger.warning(
            "speaker_ids do teste indisponíveis — sem grouped_clean por locutor; "
            "a coluna de pior locutor ficará vazia para esta arquitetura."
        )

    robustness: dict[str, dict[str, float]] = {}
    scores_robustness: dict[str, list[float]] = {}
    for snr in args.snr:
        noisy = _add_awgn_raw(X_test, snr, seed=args.seed + 20000 + int(snr))
        Z_noisy = (
            _prepare_waves(noisy) if end_to_end else embed(noisy, f"snr_{snr}")
        )
        scores_noisy = predict_scores_from_embeddings(Z_noisy)
        scores_robustness[str(snr)] = scores_noisy.tolist()
        robustness[str(snr)] = _evaluate_scores(
            y_test,
            scores_noisy,
            threshold=decision_threshold,
            cluster_ids=test_cluster_ids,
        )

    # Em LOTES, pela mesma razão da validação: com o backbone no caminho, o
    # conjunto de teste inteiro num tensor só estoura a VRAM (e no caminho MLP
    # o resultado é idêntico, só que dividido).
    test_loss, _ = _evaluate(Z_test, y_test)
    # Predicao final com o threshold calibrado (consistente com clean/robustez)
    y_pred = (scores_clean >= decision_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [float(v) for v in cm.ravel()]
    final_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores_clean)),
        "pr_auc": float(average_precision_score(y_test, scores_clean)),
        "eer": clean.get("eer"),
        "eer_threshold": clean.get("eer_threshold"),
        "min_tdcf": clean.get("min_tdcf"),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "specificity": float(tn / max(tn + fp, 1.0)),
        "sensitivity": float(tp / max(tp + fn, 1.0)),
        "false_positive_rate": float(fp / max(fp + tn, 1.0)),
        "false_negative_rate": float(fn / max(fn + tp, 1.0)),
        "total_samples": float(len(y_test)),
        "test_loss": test_loss,
        "decision_threshold": decision_threshold,
    }

    artifact = models_dir / arch_meta["artifact"]
    backbone_dir = models_dir / f"{args.architecture}_backbone"
    backbone.save_pretrained(backbone_dir, safe_serialization=False)
    torch.save(
        {
            "model_name": args.model_name,
            "architecture": arch_meta["display"],
            "model_class": arch_meta["model_class"],
            "backbone_config": backbone.config.to_dict(),
            "classifier_state_dict": classifier.state_dict(),
            # No fim-a-fim os pesos do BACKBONE também mudaram: sem eles, o
            # `.pt` não reconstrói o modelo avaliado. `classifier_state_dict`
            # continua existindo para o wrapper de inferência legado.
            "model_state_dict": (net.state_dict() if end_to_end else None),
            "backend": args.backend,
            "freeze_backbone": args.freeze_backbone,
            "finetune_info": finetune_info,
            "hidden_size": hidden_size,
            # Contrato de embedding (2026-07-15): o wrapper de inferência
            # (TorchSSLOriginalModel) reconstrói janela/pooling/cabeça a
            # partir daqui. Checkpoints sem esta chave caem no legado
            # (16000 / last / mean).
            "embedding_config": embedding_config,
            "labels": {"real": 0, "fake": 1},
            "input_shape": [int(args.target_samples), 1],
            "history": history,
            "clean_metrics": clean,
            "training_config": vars(args),
            "decision_threshold": decision_threshold,
            "calibration": calibration_info,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
        },
        artifact,
    )
    # Janela REAL usada no treino/avaliação. Até 2026-08-09 os quatro pontos
    # abaixo gravavam o literal [16000, 1] enquanto o run usava
    # `--target-samples` (64.000 por default): o sidecar, o model card e o
    # `metrics.json` declaravam 1 s para modelos treinados com 4 s. A inferência
    # nunca foi afetada — o wrapper lê `embedding_config` do `.pt` —, mas era
    # esta metadata que alimentava a seção de métodos.
    source_samples = (
        int(np.prod(data.original_shape))
        if data.original_shape
        else int(args.target_samples)
    )
    prepared_shape = [int(args.target_samples), 1]
    input_preparation = _input_preparation_block(
        source_samples, int(args.target_samples)
    )
    if input_preparation["crop_strategy"] == "tile_repeat":
        logger.warning(
            "Janela pedida (%d) MAIOR que o clipe (%d): %.0f%% de cada entrada "
            "é repetição do próprio sinal. Use --target-samples %d para casar "
            "com o clipe.",
            int(args.target_samples),
            source_samples,
            100.0 * input_preparation["tiled_padding_ratio"],
            source_samples,
        )

    config_payload = {
        "architecture": arch_meta["display"].replace(" ", ""),
        "display_name": arch_meta["display"],
        "model_class": arch_meta["model_class"],
        "model_name": args.model_name,
        "artifact": str(artifact),
        "backbone_artifact": str(backbone_dir),
        "input_shape": prepared_shape,
        "input_preparation": input_preparation,
        "freeze_backbone": args.freeze_backbone,
        "epochs": args.epochs,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "feature_batch_size": args.feature_batch_size,
        "train_batch_size": args.train_batch_size,
        "train_augmentation": bool(args.train_augmentation),
        "train_aug_snr_db": [int(s) for s in args.train_aug_snr],
        "noise_protocol": {
            "evaluation_domain": "waveform",
            "frontend_after_noise": True,
            "training_augmentation_domain": "waveform" if args.train_augmentation else "disabled",
            "train_noise_copies": 1 if args.train_augmentation else 0,
            "waveform_noise_batch_size": args.waveform_noise_batch_size,
            "assigned_snr_counts": assigned_snr_counts,
        },
        "decision_threshold": decision_threshold,
        "calibration": calibration_info,
    }
    (models_dir / f"bench_{arch_meta['compact']}_config.json").write_text(
        json.dumps(_json_safe(config_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_model_card(models_dir / "README.md", config_payload)

    latency_ms = None
    latency_profile: dict[str, Any] = {"status": "skipped", "runtime": "pytorch"}
    if args.latency_runs > 0:
        latency_raw = np.asarray(X_test[:1], dtype="float32").reshape(1, -1)
        latency_x = _normalize_wave_batch(
            _fit_length(latency_raw, int(args.target_samples))
        )
        x_latency = torch.from_numpy(latency_x).to(device)
        backbone.eval()
        net.eval()

        def _latency_forward() -> None:
            with torch.no_grad():
                if end_to_end:
                    # Mede o caminho REAL de inferência desta variante:
                    # backbone + soma de camadas + grafo, num só forward.
                    _ = net(x_latency)
                    return
                outp = backbone(
                    x_latency, output_hidden_states=need_hidden_states
                )
                _ = classifier(pool_hidden_states(outp, embedding_config))

        for _ in range(2):
            _latency_forward()
        times = []
        for _ in range(args.latency_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _latency_forward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        latency_ms = round(float(np.median(times)), 2)
        # Mesmo schema do runner Keras (benchmarks/efficiency.py), incluindo o
        # `runtime`: esta medição é PyTorch e não é comparável com a dos 7
        # modelos Keras nem com a dos clássicos em sklearn.
        from benchmarks.efficiency import describe_runtime

        values = np.asarray(times, dtype="float64")
        latency_profile = {
            "status": "ok",
            "component": "model_forward_only",
            "batch_size": 1,
            "warmup_runs": 2,
            "measured_runs": int(args.latency_runs),
            "median_ms": latency_ms,
            "p95_ms": round(float(np.percentile(values, 95)), 2),
            "mean_ms": round(float(np.mean(values)), 2),
            "std_ms": round(float(np.std(values)), 2),
            "includes_frontend": False,
            "includes_postprocessing": False,
            **describe_runtime("pytorch"),
        }

    size_mb = round(_path_size_mb(artifact) + _path_size_mb(backbone_dir), 2)
    params = _count_torch_params(backbone) + _count_torch_params(classifier)
    converged = bool(
        clean.get("auc_roc", 0.0) >= 0.70 and clean.get("accuracy", 0.0) >= 0.55
    )

    from benchmarks.stability import analyze_training_stability

    training_stability = analyze_training_stability(
        history, epochs_budget=int(args.epochs)
    )
    if training_stability.get("stable") is False:
        logger.warning(
            "Treino INSTÁVEL (%s): %s",
            training_stability.get("status"),
            training_stability.get("reason"),
        )

    results = {
        "config": {
            "preset_name": f"single:{arch_meta['display'].replace(' ', '')}",
            "architectures": [arch_meta["display"]],
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "snr_levels_db": args.snr,
            "seed": args.seed,
            "device_profile": "gpu" if device.type == "cuda" else "cpu",
            "model_name": args.model_name,
            "runner": "scripts/benchmark/run_wavlm_original_benchmark.py",
            "decision_threshold": decision_threshold,
            "fixed_epoch_budget": not bool(args.early_stopping),
            "checkpoint_selection": "minimum_clean_validation_loss",
            "validation_condition": "clean",
            # PROTOCOLO DE FORMATO — precisa aparecer aqui.
            #
            # `_load_dataset` aplica a correcao de banda as tres particoes
            # quando `--band-correction-hz` e passado (e o compose passa), mas
            # ate 2026-08-21 o bloco `config` nao a registrava: o results.json
            # das duas entradas SSL saia com `band_correction_hz` ausente
            # enquanto o caminho Keras gravava 7500.0. Quem comparasse os
            # artefatos concluiria que WavLM e HuBERT rodaram SEM a correcao —
            # uma assimetria de protocolo que nao existe, mas que o artefato
            # nao permitia descartar sem ir ao log. Num TCC onde proveniencia e
            # argumento, o artefato tem de dizer sob qual protocolo rodou.
            "band_correction_hz": (
                float(args.band_correction_hz)
                if getattr(args, "band_correction_hz", None)
                else None
            ),
            "band_correction": _band_correction_block(
                getattr(args, "band_correction_hz", None)
            ),
            # O runner SSL tem laco de treino proprio em PyTorch, sem o
            # ModelCheckpoint do Keras: `--checkpoint-monitor` nao se aplica e
            # nao e repassado (ver run_models_sequential.py). Declarado para o
            # artefato nao sugerir que o monitor foi simplesmente esquecido.
            "checkpoint_monitor": None,
            "checkpoint_monitor_nota": (
                "nao aplicavel: laco de treino proprio em PyTorch, sem "
                "ModelCheckpoint do Keras"
            ),
        },
        "preflight": {
            "status": "ok",
            "requires": ["torch", "transformers", arch_meta["model_class"]],
            "fallback_used": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            # Mesma chave que o runner Keras emite. Sem ela o artefato SSL nao
            # dizia em que versao de `transformers` o backbone foi carregado —
            # e e ela que decide como os pesos do checkpoint sao mapeados, entao
            # sem esse numero o resultado nao e reexecutavel.
            "libraries": _library_versions(),
            "pretrained_checkpoints": {arch_meta["display"]: args.model_name},
        },
        "dataset": {
            "name": data.name,
            "source": args.dataset,
            "n_total": int(len(data.y)),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "input_shape": data.original_shape,
            "prepared_shape": prepared_shape,
            "balance_test": {
                "real": int((y_test == 0).sum()),
                "fake": int((y_test == 1).sum()),
            },
            "y_test": y_test.astype(int).tolist(),
            "metadata": _json_safe(data.metadata or {}),
            "split_source": (data.metadata or {}).get("split_source"),
            "split_overlap_audit": split_overlap_audit,
            "split_fingerprints": split_fingerprints,
            "test_split_sha256": split_fingerprints["test"]["sha256"],
            # Mesma chave do runner Keras: habilita a comparação PAREADA por
            # cluster em benchmarks/significance.py.
            "test_cluster_ids": (
                [str(v) for v in test_cluster_ids]
                if test_cluster_ids is not None
                else None
            ),
            # Idem para a unidade LOCUTOR (11 no teste, contra 183 frases): é a
            # que casa com a alegação speaker-disjoint do protocolo.
            "test_speaker_ids": (
                [str(v) for v in test_speaker_ids]
                if test_speaker_ids is not None
                else None
            ),
            "provenance_overlap_audit": (
                (data.metadata or {}).get("provenance_overlap_audit")
            ),
        },
        "architectures": {
            arch_meta["display"]: {
                "status": "ok",
                "type": "neural",
                "input_shape": prepared_shape,
                "converged": converged,
                "convergence_criteria": {
                    "accuracy_min": 0.55,
                    "auc_roc_min": 0.70,
                    "scope": "checkpoint_selecionado",
                },
                # Mesmo bloco que o runner Keras grava: `converged` fala do
                # checkpoint promovido, `training_stability` fala do treino que
                # levou até ele. Ver benchmarks/stability.py.
                "training_stability": training_stability,
                "clean": clean,
                "grouped_clean": grouped_clean,
                "scores_clean": scores_clean.tolist(),
                "robustness": robustness,
                # AJUSTE 2026-08-06: SEM esta chave, `benchmarks.report.
                # write_all` (chamado no fim deste runner) regravava
                # `predictions_robustness.csv` com APENAS O CABEÇALHO, por cima
                # do arquivo correto que `_write_predictions_noisy` acabara de
                # escrever — o writer canônico lê `scores_robustness` do dict de
                # resultados, não do disco. Foi o que aconteceu no
                # clean_benchmark_15k: WavLM/HuBERT ficaram sem NENHUMA predição
                # por amostra sob ruído, então os agregados por SNR não eram
                # reverificáveis. Também alinha o schema do metrics.json com o
                # dos 9 modelos Keras, que já carregavam a chave.
                "scores_robustness": scores_robustness,
                # Paridade de schema com o runner Keras: este runner não
                # implementa `--codec-eval`, e a ausência da chave era
                # indistinguível de "rodou e não achou nada".
                "codec_robustness": {},
                "codec_eval_status": {
                    "requested": [],
                    "status": "not_supported",
                    "reason": "runner SSL dedicado não implementa --codec-eval",
                },
                "noise_protocol": {
                    "evaluation_domain": "waveform",
                    "frontend_after_noise": True,
                    "training_augmentation_domain": (
                        "waveform" if args.train_augmentation else "disabled"
                    ),
                    "train_aug_snr_db": [int(s) for s in args.train_aug_snr],
                    "train_noise_copies": 1 if args.train_augmentation else 0,
                    "waveform_noise_batch_size": args.waveform_noise_batch_size,
                    "assigned_snr_counts": assigned_snr_counts,
                },
                "efficiency": {
                    "params": params,
                    "size_mb": size_mb,
                    "latency_ms": latency_ms,
                    "latency_profile": latency_profile,
                },
                "history": history,
                "training_config": {
                    "model_family": "pytorch_transformers",
                    "model_name": args.model_name,
                    "architecture": arch_meta["display"],
                    "model_class": arch_meta["model_class"],
                    "freeze_backbone": args.freeze_backbone,
                    "epochs": args.epochs,
                    "feature_batch_size": args.feature_batch_size,
                    "train_batch_size": args.train_batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "train_augmentation": bool(args.train_augmentation),
                    "train_aug_snr_db": [int(s) for s in args.train_aug_snr],
                    "noise_protocol": {
                        "evaluation_domain": "waveform",
                        "frontend_after_noise": True,
                        "training_augmentation_domain": (
                            "waveform" if args.train_augmentation else "disabled"
                        ),
                        "train_noise_copies": 1 if args.train_augmentation else 0,
                        "waveform_noise_batch_size": args.waveform_noise_batch_size,
                        "assigned_snr_counts": assigned_snr_counts,
                    },
                    "early_stopping": bool(args.early_stopping),
                    "early_stopping_patience": args.early_stopping_patience,
                    "calibrate_under_noise": bool(args.calibrate_under_noise),
                    "calibration_snr_db": [int(s) for s in args.calibration_snr],
                    "fallback_used": False,
                },
                "calibration": calibration_info,
                "model_parameters": {
                    "backbone_params": _count_torch_params(backbone),
                    "classifier_params": _count_torch_params(classifier),
                    "hidden_size": hidden_size,
                    "backend": args.backend,
                    "backbone_lr": (
                        float(args.backbone_lr) if not args.freeze_backbone else None
                    ),
                    "grad_accum": int(args.grad_accum),
                    **finetune_info,
                },
                "final_training_metrics": final_metrics,
                "fit_strategy": {
                    "kind": (
                        "end_to_end_finetune_ssl_then_graph_backend"
                        if end_to_end and not args.freeze_backbone
                        else "frozen_backbone_sequence_then_graph_backend"
                        if end_to_end
                        else "frozen_backbone_embedding_then_classifier_fit"
                    ),
                    "backbone": args.model_name,
                    # Derivado da contagem de parâmetros com requires_grad, não
                    # da flag: `--no-freeze-backbone` chegou a declarar `true`
                    # num run integralmente congelado (corrigido em 2026-08-09).
                    "backbone_trainable": finetune_info["backbone_trainable"],
                    "backbone_trainable_params": finetune_info[
                        "backbone_trainable_params"
                    ],
                    "feature_encoder_frozen": finetune_info["feature_encoder_frozen"],
                    "train_time_s": train_time_s,
                },
                # AJUSTE 2026-08-06: era a única chave de proveniência que
                # faltava nestes dois (o `input_preparation` e o
                # `noise_protocol` já vinham). Sem ela, `metrics.json` de
                # WavLM/HuBERT Original não declarava variant/family/runner/
                # scope, e o `variant` importa aqui em particular: o rótulo
                # registra `wavlm-base-plus`, não `base`. Reaproveita o builder
                # canônico do runner Keras em vez de repetir os literais — foi
                # justamente a duplicação que já deixou esse rótulo defasado
                # uma vez.
                "provenance": _architecture_provenance(arch_meta["display"]),
                "model_artifact": str(artifact),
                # PARIDADE 2026-08-09 com o runner Keras. Sem o sha256 gravado
                # na hora, a promoção não tem como distinguir o `.pt` do run de
                # um arquivo trocado depois — e o `efficiency.size_mb` daqui NÃO
                # serve de proxy, porque soma cabeça + backbone congelado
                # (361 MB declarados contra ~1,5 MB de `.pt`).
                "model_artifact_fingerprint": _file_fingerprint(artifact),
                "backbone_artifact": str(backbone_dir),
                "training_artifacts_dir": str(arch_out),
                "epochs": epochs_trained,
                "best_epoch": best_epoch,
                "wall_time_s": train_time_s,
                "input_preparation": input_preparation,
            }
        },
    }

    # NÃO reescrever predictions_clean/predictions_robustness/robustness aqui:
    # `write_all` (abaixo) é o writer CANÔNICO desses três e regrava todos com
    # o schema comum aos 11 modelos (`sample_index`/`y_pred`/`correct`;
    # `condition` em vez de `snr_db`). As funções locais `_write_predictions*`
    # /`_write_robustness` usavam um schema próprio e eram sobrescritas de
    # imediato — mantê-las aqui só criava a ilusão de que este runner controla
    # esses arquivos, e foi assim que o `predictions_robustness.csv` vazio
    # passou despercebido. Ver o comentário em `scores_robustness` acima.
    arch_result = results["architectures"][arch_meta["display"]]
    (arch_out / "metrics.json").write_text(
        json.dumps(_json_safe(arch_result), indent=2),
        encoding="utf-8",
    )

    write_all(_json_safe(results), str(out))
    logger.info("Benchmark %s finalizado em: %s", arch_meta["display"], out)
    print(json.dumps(_json_safe({
        "output_dir": str(out),
        "model_artifact": str(artifact),
        "accuracy": clean.get("accuracy"),
        "auc_roc": clean.get("auc_roc"),
        "eer": clean.get("eer"),
        "latency_ms": latency_ms,
        "fallback_used": False,
    }), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
