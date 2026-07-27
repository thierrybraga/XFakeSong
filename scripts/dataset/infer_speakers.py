#!/usr/bin/env python3
"""Infere identidade de falante por embeddings, validada contra verdade-base.

Motivacao (2026-07-25). O BRSpeech-DF nao publica falante: as tres configs do
repositorio (`bonafide`, `default`, `spoof`) expoem apenas `audio`, `label` e,
no spoof, `model`. Nenhum download recupera o campo. Sem identidade de falante,
um split agrupado por locutor e impossivel de verificar, e o pipeline antigo
caia no fallback `sample:<filename>` — cada amostra virava um grupo singleton, o
que satisfaz "zero falante compartilhado" de forma vacua: o mesmo locutor
podia estar em treino e teste livremente.

A unica saida tecnica e inferir o falante do proprio audio. Para nao trocar um
numero inventado por outro, este script **calibra contra verdade-base antes de
aplicar**: `mlspt` (22 falantes), `ttsport` (1) e `fkvoice` (56 vozes
sinteticas) tem `speaker_id` real no manifesto, com ~90-107 locucoes cada.
Varremos camada do WavLM x limiar de aglomeracao, medimos NMI/homogeneidade
contra os rotulos verdadeiros, e so entao aplicamos a melhor configuracao ao
brspeech.

Escolha de camada importa: em tarefas de falante (SUPERB SID/SV) as camadas
intermediarias-baixas do WavLM sao mais discriminativas de locutor que a
ultima, que carrega mais conteudo fonetico.

Uso:
    python scripts/dataset/infer_speakers.py --validate
    python scripts/dataset/infer_speakers.py --infer --layer 6 --threshold 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
logger = logging.getLogger("infer_speakers")

DATASETS_DIR = ROOT / "data" / "datasets"
MANIFEST_PATH = DATASETS_DIR / "metadata" / "speaker_manifest.json"
VALIDATION_CACHE = DATASETS_DIR / "metadata" / "speaker_val_embeddings.npz"
BRSPEECH_CACHE = DATASETS_DIR / "metadata" / "speaker_brspeech_embeddings.npz"

MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SAMPLES = 80_000  # 5 s @ 16 kHz, a mesma janela do NPZ
BATCH = 8


def _load_window(path: Path) -> np.ndarray | None:
    """Le o audio na MESMA janela de 5 s centrais usada no NPZ."""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        logger.debug("falha em %s: %s", path, exc)
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        target = int(round(len(audio) * 16_000 / max(sr, 1)))
        if target <= 0:
            return None
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target),
            np.arange(len(audio)),
            audio.astype("float64"),
        ).astype("float32")
    if len(audio) == 0:
        return None
    if len(audio) >= TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) // 2
        return audio[start : start + TARGET_SAMPLES]
    repeats = int(np.ceil(TARGET_SAMPLES / len(audio)))
    return np.tile(audio, repeats)[:TARGET_SAMPLES]


def _extract(paths: list[Path], all_layers: bool, layer: int | None) -> np.ndarray:
    """Embeddings mean-pooled do WavLM. (n, 13, 768) ou (n, 768)."""
    import torch
    from transformers import WavLMModel

    torch.set_num_threads(8)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.eval()

    chunks: list[np.ndarray] = []
    started = time.time()
    for start in range(0, len(paths), BATCH):
        batch_paths = paths[start : start + BATCH]
        waves = [_load_window(p) for p in batch_paths]
        keep = [w for w in waves if w is not None]
        if not keep:
            continue
        x = torch.from_numpy(np.stack(keep))
        with torch.no_grad():
            out = model(x, output_hidden_states=True)
        if all_layers:
            # (camadas, batch, tempo, dim) -> mean sobre tempo
            pooled = np.stack(
                [h.mean(dim=1).numpy() for h in out.hidden_states], axis=1
            )
        else:
            pooled = out.hidden_states[layer].mean(dim=1).numpy()[:, None, :]
        chunks.append(pooled.astype("float32"))
        if start % (BATCH * 25) == 0:
            done = start + len(batch_paths)
            rate = done / max(time.time() - started, 1e-6)
            logger.info(
                "  %d/%d embeddings (%.1f/s, resta ~%.1f min)",
                done,
                len(paths),
                rate,
                (len(paths) - done) / max(rate, 1e-6) / 60,
            )
    result = np.concatenate(chunks, axis=0)
    return result if all_layers else result[:, 0, :]


def _l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def _cluster(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Aglomeracao com distancia de cosseno e ligacao media."""
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(_l2(vectors))


def _known_speaker_files(per_speaker: int, seed: int) -> tuple[list[Path], list[str]]:
    """Arquivos com `speaker_id` real no manifesto, subamostrados por falante."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_speaker: dict[str, list[Path]] = defaultdict(list)

    # mlspt/ttsport vivem na reserva `real/`; fkvoice, nos splits (`fake/`).
    candidates: list[tuple[str, Path]] = []
    for path in DATASETS_DIR.glob("real/*.wav"):
        candidates.append((f"real/{path.name}", path))
    for path in DATASETS_DIR.glob("splits/*/fake/fkvoice_*.wav"):
        candidates.append((f"fake/{path.name}", path))

    for key, path in candidates:
        entry = manifest.get(key)
        if not entry:
            continue
        speaker = entry.get("speaker_id")
        if speaker:
            by_speaker[str(speaker)].append(path)

    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    labels: list[str] = []
    for speaker, files in sorted(by_speaker.items()):
        files = sorted(files)
        if len(files) > per_speaker:
            idx = rng.choice(len(files), per_speaker, replace=False)
            files = [files[i] for i in sorted(idx)]
        paths.extend(files)
        labels.extend([speaker] * len(files))
    return paths, labels


def _validate(per_speaker: int, seed: int, use_cache: bool) -> None:
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
    )

    if use_cache and VALIDATION_CACHE.exists():
        cached = np.load(VALIDATION_CACHE, allow_pickle=False)
        embeddings, truth = cached["embeddings"], cached["truth"].astype(str)
        logger.info("Cache de validacao: %s", embeddings.shape)
    else:
        paths, truth_list = _known_speaker_files(per_speaker, seed)
        logger.info(
            "Verdade-base: %d amostras, %d falantes",
            len(paths),
            len(set(truth_list)),
        )
        embeddings = _extract(paths, all_layers=True, layer=None)
        truth = np.asarray(truth_list[: len(embeddings)], dtype="U64")
        VALIDATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            VALIDATION_CACHE, embeddings=embeddings, truth=truth
        )
        logger.info("Cache gravado: %s", VALIDATION_CACHE)

    n_true = len(set(truth.tolist()))
    print(f"\nverdade-base: {len(truth)} amostras, {n_true} falantes reais")
    print("\n=== VARREDURA camada x limiar (NMI contra verdade-base) ===")
    header = f"{'camada':>7}" + "".join(f"{t:>9.2f}" for t in _THRESHOLDS)
    print(header)

    best = (-1.0, None, None)
    for layer in _LAYERS:
        row = f"{layer:>7}"
        for threshold in _THRESHOLDS:
            assignment = _cluster(embeddings[:, layer, :], threshold)
            nmi = normalized_mutual_info_score(truth, assignment)
            row += f"{nmi:>9.3f}"
            if nmi > best[0]:
                best = (nmi, layer, threshold)
        print(row)

    nmi, layer, threshold = best
    assignment = _cluster(embeddings[:, layer, :], threshold)
    print(f"\n=== MELHOR: camada {layer}, limiar {threshold:.2f} ===")
    print(f"  NMI              {nmi:.4f}")
    print(f"  homogeneidade    {homogeneity_score(truth, assignment):.4f}")
    print(f"  completude       {completeness_score(truth, assignment):.4f}")
    print(f"  ARI              {adjusted_rand_score(truth, assignment):.4f}")
    print(f"  clusters         {len(set(assignment.tolist()))} (real: {n_true})")
    print(
        "\nhomogeneidade alta = clusters nao misturam falantes (o que importa\n"
        "para um split agrupado: evita o mesmo locutor em treino e teste)."
    )


_LAYERS = (0, 2, 4, 6, 8, 10, 12)
_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def _infer(layer: int, threshold: float, use_cache: bool, limit: int | None) -> None:
    """Aplica a clusterizacao ao brspeech e grava pseudo-falantes no manifesto."""
    # Roda sobre os REPRESENTANTES do pool (um por cluster de conteudo), que sao
    # exatamente os candidatos do dataset final. Processar as gemeas seria
    # desperdicio: elas compartilham conteudo e falante por construcao.
    cache = DATASETS_DIR / "metadata" / "brspeech_fingerprints.npz"
    if cache.exists():
        from scripts.dataset.build_clean_dataset import _cluster as _content_cluster
        from scripts.dataset.build_clean_dataset import _representatives

        indexed = np.load(cache, allow_pickle=False)
        pool_paths = [str(p) for p in indexed["paths"]]
        roots = _content_cluster(indexed["fps"], 0.99)
        keep = _representatives(roots, [str(h) for h in indexed["hashes"]])
        paths = [ROOT / pool_paths[i] for i in keep.tolist()]
        logger.info(
            "representantes do pool: %d de %d arquivos", len(paths), len(pool_paths)
        )
    else:
        paths = sorted(DATASETS_DIR.glob("splits_v3/*/*/brspeech_*.wav"))
        if not paths:
            paths = sorted(DATASETS_DIR.glob("splits/*/*/brspeech_*.wav"))
    if limit:
        paths = paths[:limit]
    logger.info("brspeech a processar: %d", len(paths))

    if use_cache and BRSPEECH_CACHE.exists():
        cached = np.load(BRSPEECH_CACHE, allow_pickle=False)
        embeddings = cached["embeddings"]
        names = [str(x) for x in cached["names"]]
        logger.info("Cache brspeech: %s", embeddings.shape)
    else:
        embeddings = _extract(paths, all_layers=False, layer=layer)
        names = [p.name for p in paths[: len(embeddings)]]
        BRSPEECH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            BRSPEECH_CACHE,
            embeddings=embeddings,
            names=np.asarray(names, dtype="U128"),
            layer=np.asarray(layer),
        )
        logger.info("Cache gravado: %s", BRSPEECH_CACHE)

    assignment = _cluster(embeddings, threshold)
    n_clusters = len(set(assignment.tolist()))
    sizes = np.bincount(assignment - assignment.min())
    logger.info(
        "pseudo-falantes: %d (mediana %d utt, maior %d)",
        n_clusters,
        int(np.median(sizes)),
        int(sizes.max()),
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    updated = 0
    for name, cluster in zip(names, assignment.tolist()):
        pseudo = f"inferred:wavlm{layer}:{int(cluster):05d}"
        for label in ("real", "fake"):
            key = f"{label}/{name}"
            if key in manifest:
                # Prefixo `inferred:` deixa explicito que NAO e identidade
                # publicada pelo corpus — nao confundir com speaker_id real.
                manifest[key]["inferred_speaker_id"] = pseudo
                manifest[key]["inferred_speaker_method"] = (
                    f"wavlm-base-plus L{layer} agglomerative cosine "
                    f"threshold={threshold}"
                )
                updated += 1
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Manifesto: %d entradas com pseudo-falante", updated)

    out = DATASETS_DIR / "metadata" / "inferred_speakers.json"
    out.write_text(
        json.dumps(
            {
                "method": "wavlm-base-plus mean-pool + agglomerative cosine",
                "layer": layer,
                "threshold": threshold,
                "n_samples": len(names),
                "n_pseudo_speakers": n_clusters,
                "median_utterances": int(np.median(sizes)),
                "largest_cluster": int(sizes.max()),
                "assignments": dict(zip(names, [int(a) for a in assignment])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Atribuicoes gravadas: %s", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--per-speaker", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.validate:
        _validate(args.per_speaker, args.seed, use_cache=not args.no_cache)
        return 0
    if args.infer:
        _infer(args.layer, args.threshold, not args.no_cache, args.limit)
        return 0
    parser.error("escolha --validate ou --infer")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
