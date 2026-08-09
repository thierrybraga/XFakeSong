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
    ):
        self.X = X
        self.y = y
        self.name = name
        self.metadata = metadata
        self.original_shape = original_shape
        # Necessário para o bootstrap por CLUSTER (ver _load_dataset).
        self.test_cluster_ids = test_cluster_ids


def _load_dataset(path: str, seed: int):
    """Carrega o NPZ preservando partições oficiais quando disponíveis."""

    p = Path(path)
    data = BenchmarkData.from_npz(str(p))
    splits = data.stratified_split(seed=seed, preserve_predefined=True)
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
    test_idx = (getattr(data, "last_split_indices", None) or {}).get("test")
    if test_idx is not None and data.cluster_ids is not None:
        test_cluster_ids = np.asarray(data.cluster_ids)[np.asarray(test_idx, dtype=int)]
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
    args = parser.parse_args()

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
    data, splits = _load_dataset(args.dataset, args.seed)
    from benchmarks.runner import _architecture_provenance, _audit_split_overlap
    split_overlap_audit = _audit_split_overlap(splits, fail_on_overlap=True)
    split_fingerprints = split_overlap_audit["split_fingerprints"]
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    logger.info("Carregando %s: %s", arch_meta["display"], args.model_name)
    backbone = BackboneModel.from_pretrained(args.model_name).to(device)
    backbone.train(not args.freeze_backbone)
    for param in backbone.parameters():
        param.requires_grad = not args.freeze_backbone

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
    logger.info("Contrato de embedding: %s", embedding_config)
    classifier = build_ssl_classifier(embedding_config, args.dropout).to(device)

    need_hidden_states = args.layer_pooling == "weighted"

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
            noisy_embedding_chunks.append(
                embed(noisy_chunk, f"train_awgn_{start}_{stop}")
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
    val_x = torch.from_numpy(Z_val_monitor).to(device)
    val_y = torch.from_numpy(y_val_monitor.astype("int64")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    logger.info("Treinando cabeca %s por %d epocas", arch_meta["display"], args.epochs)
    started_train = time.time()
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        losses = []
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().detach().cpu())
            total += int(yb.numel())

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_x)
            val_loss = float(criterion(val_logits, val_y).detach().cpu())
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == val_y).float().mean().detach().cpu())

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
            best_state = {
                k: v.detach().clone() for k, v in classifier.state_dict().items()
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
                break

    epochs_trained = len(history["loss"])
    if best_state is not None:
        classifier.load_state_dict(best_state)
        logger.info("Pesos restaurados da melhor epoca (%d)", best_epoch)

    train_time_s = round(time.time() - started_train, 3)

    def predict_scores_from_embeddings(Z: np.ndarray) -> np.ndarray:
        classifier.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Z.astype("float32"))),
            batch_size=args.train_batch_size,
            shuffle=False,
        )
        scores = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = classifier(xb.to(device, non_blocking=True))
                scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
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

    scores_clean = predict_scores_from_embeddings(Z_test)
    clean = _evaluate_scores(
        y_test, scores_clean, threshold=decision_threshold,
        cluster_ids=test_cluster_ids,
    )

    robustness: dict[str, dict[str, float]] = {}
    scores_robustness: dict[str, list[float]] = {}
    for snr in args.snr:
        noisy = _add_awgn_raw(X_test, snr, seed=args.seed + 20000 + int(snr))
        Z_noisy = embed(noisy, f"snr_{snr}")
        scores_noisy = predict_scores_from_embeddings(Z_noisy)
        scores_robustness[str(snr)] = scores_noisy.tolist()
        robustness[str(snr)] = _evaluate_scores(
            y_test,
            scores_noisy,
            threshold=decision_threshold,
            cluster_ids=test_cluster_ids,
        )

    test_tensor = torch.from_numpy(Z_test).to(device)
    y_test_tensor = torch.from_numpy(y_test.astype("int64")).to(device)
    with torch.no_grad():
        logits_test = classifier(test_tensor)
        test_loss = float(criterion(logits_test, y_test_tensor).detach().cpu())
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
            "freeze_backbone": args.freeze_backbone,
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
        classifier.eval()

        def _latency_forward() -> None:
            with torch.no_grad():
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
                },
                "final_training_metrics": final_metrics,
                "fit_strategy": {
                    "kind": "frozen_backbone_embedding_then_classifier_fit",
                    "backbone": args.model_name,
                    "backbone_trainable": not args.freeze_backbone,
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
