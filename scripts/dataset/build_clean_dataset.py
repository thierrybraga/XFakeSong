#!/usr/bin/env python3
"""Constroi um dataset limpo, sem reaproveitamento de dados entre particoes.

Contexto (2026-07-24). O dataset anterior
(`benchmark_audio_raw_balanced_15k_academic_v2.npz`) tinha tres defeitos
compostos, todos herdados da composicao do corpus e nao do codigo de treino:

1. ATALHO DE FONTE: tres das quatro fontes eram puras de classe (mlspt e
   ttsport apenas reais, fkvoice apenas fake). Um detector alcancava 87,6%
   apenas identificando o corpus de origem, sem detectar sintese
   (`scripts/dataset/audit_source_shortcut.py`).
2. SPEAKER-DISJOINTNESS VACUOSA: o BRSpeech-DF nao publica identidade de
   falante (schema oficial: `audio`, `label`, `model`), logo 76% das amostras
   caiam no fallback `sample:<filename>` de `cluster_ids` e cada uma virava um
   cluster singleton. A garantia "zero falante compartilhado" cobria apenas os
   24% restantes.
3. IDENTIDADE POR INDICE: nomes como `brspeech_00000.wav` eram reatribuidos a
   audios diferentes a cada re-download, e a chave do manifesto
   (`real/<basename>`) colidia entre copias distintas — quebrando
   reprodutibilidade e corrompendo o metadado de uma das copias.

Este script ataca os tres:

- Usa SOMENTE `brspeech`, a unica fonte pareada (bonafide + spoof). Com uma
  fonte unica cobrindo as duas classes, `P(classe|fonte) = 0,5`: o atalho de
  fonte deixa de existir por construcao, e nao por re-split.
- Na ausencia de identidade de falante/enunciado, agrupa as amostras por
  CONTEUDO ESPECTRAL (impressao digital log-Mel) e trata cada grupo como
  unidade indivisivel. Isso captura tanto re-codificacoes do mesmo audio
  quanto o par bonafide/spoof de um mesmo enunciado — cujo espalhamento entre
  particoes seria vazamento de conteudo. O grupo e a unidade honesta de
  agrupamento disponivel neste corpus, e fica registrado como
  `utterance_id = ndup:<hash>` para deixar explicito que e derivado, nao um ID
  autoritativo do corpus.
- Nomeia cada arquivo pelo hash do proprio conteudo
  (`brspeech_<sha256[:16]>.wav`), tornando a identidade estavel e imune a
  reatribuicao de indice.

As fontes puras de classe NAO entram no dataset principal: ficam reservadas
como conjunto externo (cross-generator/cross-corpus), onde tem valor de
validade externa em vez de virarem atalho.

Uso:
    python scripts/dataset/build_clean_dataset.py --analyze
    python scripts/dataset/build_clean_dataset.py --build --per-class 7500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_clean_dataset")

DATASETS_DIR = ROOT / "data" / "datasets"
FINGERPRINT_CACHE = DATASETS_DIR / "metadata" / "brspeech_fingerprints.npz"

# Fontes puras de classe: excluidas do dataset principal (viram atalho) e
# reservadas como conjunto externo de validade.
CLASS_PURE_SOURCES = ("mlspt", "ttsport", "fkvoice")

SAMPLE_RATE = 16000
# Janela exportada para o NPZ (5 s @ 16 kHz). A deduplicacao precisa comparar
# exatamente esta janela, nao o arquivo inteiro.
WINDOW_SAMPLES = 80_000
N_FFT = 512
HOP = 256
N_BANDS = 32
N_SEGMENTS = 8


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _collect_brspeech() -> dict[str, list[Path]]:
    """Reune o brspeech disponivel nas duas camadas que contem audio.

    `splits/` e as reservas `real/`+`fake/` guardam audios DIFERENTES sob os
    mesmos nomes (ver defeito 3), portanto ambas sao varridas e a deduplicacao
    por conteudo decide o que sobrevive.
    """
    pools: dict[str, list[Path]] = {}
    for label in ("real", "fake"):
        paths: list[Path] = []
        paths.extend(sorted(DATASETS_DIR.glob(f"splits/*/{label}/brspeech_*.wav")))
        paths.extend(sorted(DATASETS_DIR.glob(f"{label}/brspeech_*.wav")))
        pools[label] = paths
    return pools


def _load_wave(path: Path) -> np.ndarray | None:
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - depende de IO
        logger.warning("Falha ao ler %s: %s", path, exc)
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Reamostragem linear e suficiente: a impressao digital e grosseira.
        target = int(round(len(audio) * SAMPLE_RATE / max(sr, 1)))
        if target <= 0:
            return None
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target, dtype="float64"),
            np.arange(len(audio), dtype="float64"),
            audio.astype("float64"),
        ).astype("float32")
    if len(audio) == 0:
        return None
    # CORRECAO (2026-07-25): a impressao digital tem de ser calculada na MESMA
    # janela que vai para o NPZ (recorte central de 5 s, ou tile se menor) —
    # ver `_load_wav` em scripts/benchmark/run_tcc_pipeline.py. Medir o arquivo
    # inteiro comparava audio que o modelo nunca ve: duas copias da mesma
    # locucao com duracoes diferentes (ex. 10,2 s e 5,2 s) ficavam com
    # impressoes distintas e escapavam da deduplicacao, embora seus recortes
    # centrais de 5 s fossem quase identicos.
    if len(audio) >= WINDOW_SAMPLES:
        start = (len(audio) - WINDOW_SAMPLES) // 2
        return audio[start : start + WINDOW_SAMPLES]
    repeats = int(np.ceil(WINDOW_SAMPLES / len(audio)))
    return np.tile(audio, repeats)[:WINDOW_SAMPLES]


def _fingerprint(audio: np.ndarray) -> np.ndarray:
    """Impressao digital espectral compacta e L2-normalizada.

    log-Mel-ish: STFT -> bandas log-espacadas -> log1p -> media por segmento
    temporal. Invariante a ganho (normalizacao) e robusta a re-codificacao,
    mas sensivel a conteudo falado.
    """
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    peak = float(np.max(np.abs(audio)) or 1.0)
    audio = audio / peak

    n_frames = 1 + (len(audio) - N_FFT) // HOP
    if n_frames < 1:
        return np.zeros(N_BANDS * N_SEGMENTS, dtype="float32")
    window = np.hanning(N_FFT).astype("float32")
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP, audio.strides[0]),
        writeable=False,
    )
    spec = np.abs(np.fft.rfft(frames * window, axis=1))

    # Bandas log-espacadas sobre os bins de frequencia.
    n_bins = spec.shape[1]
    edges = np.unique(
        np.geomspace(1, n_bins, N_BANDS + 1).astype(int).clip(1, n_bins)
    )
    bands = []
    for start, stop in zip(edges[:-1], edges[1:]):
        bands.append(spec[:, start:stop].mean(axis=1))
    band_spec = np.log1p(np.stack(bands, axis=1))  # (frames, bandas)

    # Media por segmento temporal.
    seg_bounds = np.linspace(0, band_spec.shape[0], N_SEGMENTS + 1).astype(int)
    segments = [
        band_spec[a:b].mean(axis=0)
        if b > a
        else np.zeros(band_spec.shape[1], dtype="float32")
        for a, b in zip(seg_bounds[:-1], seg_bounds[1:])
    ]
    vec = np.concatenate(segments).astype("float32")
    if vec.shape[0] < N_BANDS * N_SEGMENTS:
        vec = np.pad(vec, (0, N_BANDS * N_SEGMENTS - vec.shape[0]))
    vec = vec[: N_BANDS * N_SEGMENTS]
    norm = float(np.linalg.norm(vec) or 1.0)
    return (vec / norm).astype("float32")


def _build_index(use_cache: bool = True) -> dict:
    """Indexa o pool: hash, rotulo e impressao digital por audio unico."""
    if use_cache and FINGERPRINT_CACHE.exists():
        cached = np.load(FINGERPRINT_CACHE, allow_pickle=False)
        logger.info(
            "Cache de impressoes digitais: %d audios unicos", len(cached["hashes"])
        )
        return {
            "hashes": [str(h) for h in cached["hashes"]],
            "labels": [str(x) for x in cached["labels"]],
            "paths": [str(p) for p in cached["paths"]],
            "fps": cached["fps"],
        }

    pools = _collect_brspeech()
    seen: dict[str, int] = {}
    hashes: list[str] = []
    labels: list[str] = []
    paths: list[str] = []
    fps: list[np.ndarray] = []
    duplicates = 0

    for label, files in pools.items():
        logger.info("Indexando %s: %d arquivos", label, len(files))
        for pos, path in enumerate(files, start=1):
            digest = _sha256(path)
            if digest in seen:
                duplicates += 1
                continue
            audio = _load_wave(path)
            if audio is None:
                continue
            seen[digest] = len(hashes)
            hashes.append(digest)
            labels.append(label)
            paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            fps.append(_fingerprint(audio))
            if pos % 1000 == 0:
                logger.info("  %s: %d/%d", label, pos, len(files))

    logger.info(
        "Indexados %d audios unicos (%d duplicatas exatas descartadas)",
        len(hashes),
        duplicates,
    )
    index = {
        "hashes": hashes,
        "labels": labels,
        "paths": paths,
        "fps": np.stack(fps).astype("float32"),
    }
    FINGERPRINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FINGERPRINT_CACHE,
        hashes=np.asarray(hashes, dtype="U64"),
        labels=np.asarray(labels, dtype="U8"),
        paths=np.asarray(paths, dtype="U512"),
        fps=index["fps"],
    )
    logger.info("Cache gravado: %s", FINGERPRINT_CACHE)
    return index


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[b] = a


def _cluster(fps: np.ndarray, threshold: float, block: int = 2048) -> np.ndarray:
    """Componentes conexas por similaridade de cosseno >= threshold.

    Agrupa entre classes de proposito: o par bonafide/spoof do mesmo enunciado
    precisa cair na MESMA particao, senao o modelo ve o mesmo conteudo como
    real no treino e como fake no teste.
    """
    n = fps.shape[0]
    union = _UnionFind(n)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = fps[start:stop] @ fps.T  # (bloco, n)
        rows, cols = np.nonzero(sims >= threshold)
        for row, col in zip(rows, cols):
            idx = start + int(row)
            other = int(col)
            if other > idx:
                union.union(idx, other)
    return np.asarray([union.find(i) for i in range(n)], dtype=np.int64)


def _cluster_report(fps: np.ndarray, labels: Sequence[str], thresholds: Iterable[float]) -> None:
    n = fps.shape[0]
    logger.info("Analise de clusterizacao sobre %d audios", n)
    print(f"\n{'limiar':>8}{'clusters':>11}{'maior':>8}{'>1 amostra':>12}{'mistos':>9}")
    for threshold in thresholds:
        roots = _cluster(fps, threshold)
        sizes = Counter(roots.tolist())
        multi = sum(1 for v in sizes.values() if v > 1)
        biggest = max(sizes.values())
        mixed = 0
        by_root: dict[int, set[str]] = {}
        for root, label in zip(roots.tolist(), labels):
            by_root.setdefault(root, set()).add(label)
        mixed = sum(1 for v in by_root.values() if len(v) > 1)
        print(
            f"{threshold:>8.3f}{len(sizes):>11}{biggest:>8}{multi:>12}{mixed:>9}"
        )
    print(
        "\nmistos = clusters com real E fake juntos (par bonafide/spoof do mesmo\n"
        "enunciado detectado). 'maior' muito alto indica limiar baixo demais\n"
        "colapsando o corpus."
    )


def _representatives(
    roots: np.ndarray, hashes: Sequence[str]
) -> np.ndarray:
    """Um indice por cluster de conteudo, escolhido deterministicamente.

    Sem isto o dataset conta as duas copias de uma locucao re-codificada como
    duas amostras: no acervo antigo, 69% das amostras tinham uma gemea quase
    identica e os 15.003 do v3 correspondiam a apenas ~9.845 conteudos
    distintos. Manter as gemeas juntas evita vazamento entre particoes, mas nao
    torna o dataset mais diverso.
    """
    best: dict[int, tuple[str, int]] = {}
    for idx, root in enumerate(roots.tolist()):
        digest = hashes[idx]
        current = best.get(root)
        if current is None or digest < current[0]:
            best[root] = (digest, idx)
    return np.asarray(sorted(idx for _, idx in best.values()), dtype=np.int64)


def _speaker_groups(
    paths: Sequence[str],
    roots: np.ndarray,
    speaker_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Funde clusteres de conteudo com componentes conexas de falante.

    Por que componentes conexas e nao clusterizacao aglomerativa: nenhuma
    clusterizacao atinge completude 1,0 sem colapsar o corpus (medido em
    `mlspt`, 21 falantes reais: average-linkage plateia em 0,954; single-linkage
    so chega a 1,0 com um unico componente). Logo a garantia nao pode depender
    da qualidade da clusterizacao.

    Componentes conexas dao uma garantia POR CONSTRUCAO: se toda aresta com
    similaridade >= `speaker_threshold` esta dentro de um componente, e o
    componente e indivisivel no split, entao NENHUM par com similaridade
    >= `speaker_threshold` atravessa particoes. Calibrado par-a-par contra 78
    falantes reais: em 0,95 a precisao de "mesmo falante" e 1,000; em 0,85 o
    recall chega a 0,733. Agrupar em 0,85 confina, portanto, ~73% de todos os
    pares mesmo-falante e 100% dos pares certos. Fundir falantes distintos por
    engano (precisao 0,56 nesse limiar) e inofensivo para a validade: apenas
    torna os grupos mais grossos.
    """
    cache = DATASETS_DIR / "metadata" / "speaker_brspeech_embeddings.npz"
    stats: dict[str, Any] = {"available": False, "threshold": speaker_threshold}
    if not cache.exists():
        logger.warning(
            "sem %s — agrupando apenas por conteudo (rode infer_speakers.py "
            "--infer para a garantia de falante)",
            cache.name,
        )
        return roots, stats

    payload = np.load(cache, allow_pickle=False)
    emb_by_name = {
        str(name): idx for idx, name in enumerate(payload["names"])
    }
    embeddings = payload["embeddings"]
    embeddings = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
    )

    rows: list[int] = []
    vectors: list[np.ndarray] = []
    for idx, rel in enumerate(paths):
        pos = emb_by_name.get(Path(rel).name)
        if pos is not None:
            rows.append(idx)
            vectors.append(embeddings[pos])
    if not rows:
        logger.warning("nenhum embedding casou com o pool; so conteudo")
        return roots, stats

    logger.info(
        "embeddings de falante casados: %d/%d amostras", len(rows), len(paths)
    )
    matrix = np.stack(vectors).astype("float32")
    union = _UnionFind(len(paths))

    # Componentes conexas sobre as arestas de falante.
    edges = 0
    block = 2048
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        sims = matrix[start:stop] @ matrix.T
        src, dst = np.nonzero(sims >= speaker_threshold)
        for left, right in zip(src, dst):
            i = start + int(left)
            j = int(right)
            if j > i:
                union.union(rows[i], rows[j])
                edges += 1

    # O cluster de conteudo continua sendo unidade minima indivisivel.
    first_of: dict[int, int] = {}
    for idx, root in enumerate(roots.tolist()):
        if root in first_of:
            union.union(first_of[root], idx)
        else:
            first_of[root] = idx

    merged = np.asarray([union.find(i) for i in range(len(paths))], dtype=np.int64)
    sizes = np.bincount(np.unique(merged, return_inverse=True)[1])
    stats.update(
        {
            "available": True,
            "method": "wavlm-base-plus L0 mean-pool, connected components",
            "matched_samples": len(rows),
            "speaker_edges": edges,
            "groups": int(len(sizes)),
            "largest_group": int(sizes.max()),
            "median_group": int(np.median(sizes)),
            "calibration": {
                "ground_truth_speakers": 78,
                "precision_at_0.95": 1.0,
                "recall_at_threshold": 0.733,
            },
        }
    )
    logger.info(
        "grupos apos fundir falante+conteudo: %d (maior=%d, mediana=%d)",
        len(sizes),
        int(sizes.max()),
        int(np.median(sizes)),
    )
    return merged, stats


def _split_clusters(
    roots: np.ndarray,
    labels: Sequence[str],
    per_class: int,
    ratios: tuple[float, float, float],
    seed: int,
    allowed: set[int] | None = None,
) -> dict[str, list[int]]:
    """Distribui GRUPOS (nunca amostras) entre treino/val/teste.

    Cada grupo e indivisivel, o que garante zero conteudo E zero falante
    compartilhado entre particoes. `allowed` restringe as amostras elegiveis
    (um representante por cluster de conteudo) sem quebrar o grupo: o grupo
    segue sendo a unidade de decisao, apenas contribui menos amostras.
    """
    rng = np.random.default_rng(seed)
    members: dict[int, list[int]] = {}
    for idx, root in enumerate(roots.tolist()):
        if allowed is not None and idx not in allowed:
            continue
        members.setdefault(root, []).append(idx)

    order = list(members.keys())
    rng.shuffle(order)

    targets = {
        "train": {"real": int(per_class * ratios[0]), "fake": int(per_class * ratios[0])},
        "val": {"real": int(per_class * ratios[1]), "fake": int(per_class * ratios[1])},
        "test": {"real": int(per_class * ratios[2]), "fake": int(per_class * ratios[2])},
    }
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    counts = {s: {"real": 0, "fake": 0} for s in assigned}

    def deficit(split: str, cluster: list[int]) -> float:
        """Quanto o cluster ajuda a fechar a cota faltante do split."""
        need = 0.0
        for idx in cluster:
            label = labels[idx]
            remaining = targets[split][label] - counts[split][label]
            if remaining <= 0:
                return -1.0
            need += remaining / max(targets[split][label], 1)
        return need / max(len(cluster), 1)

    for root in order:
        cluster = members[root]
        scored = [(deficit(s, cluster), s) for s in ("train", "val", "test")]
        scored.sort(reverse=True)
        best_score, best_split = scored[0]
        if best_score < 0:
            continue  # todas as cotas cheias para as classes deste cluster
        assigned[best_split].extend(cluster)
        for idx in cluster:
            counts[best_split][labels[idx]] += 1

    logger.info("Distribuicao por particao: %s", counts)
    return assigned


def _write_dataset(
    index: dict,
    assigned: dict[str, list[int]],
    roots: np.ndarray,
    out_dir: Path,
    groups: np.ndarray,
    speaker_stats: dict[str, Any],
) -> dict:
    """Materializa os splits com nomes enderecados por conteudo."""
    if out_dir.exists():
        raise SystemExit(
            f"{out_dir} ja existe — remova ou escolha outro --out-dir "
            "(este script nunca sobrescreve audio)"
        )

    from app.domain.dataset_metadata import speaker_manifest

    manifest_entries: dict[str, dict] = {}
    written = {"train": Counter(), "val": Counter(), "test": Counter()}
    rows: list[dict] = []

    for split, indices in assigned.items():
        for label in ("real", "fake"):
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)
        for idx in indices:
            digest = index["hashes"][idx]
            label = index["labels"][idx]
            src = ROOT / index["paths"][idx]
            name = f"brspeech_{digest[:16]}.wav"
            dest = out_dir / split / label / name
            shutil.copy2(src, dest)
            written[split][label] += 1

            cluster_key = f"ndup:{int(roots[idx]):08d}"
            manifest_entries[f"{label}/{name}"] = {
                "source": "brspeech",
                "generator_id": "bonafide" if label == "real" else "spoof",
                "label": 0 if label == "real" else 1,
                "content_sha256": digest,
                # ID derivado de agrupamento espectral, NAO um ID autoritativo
                # do corpus. Explicito no prefixo para nao induzir a erro.
                "utterance_id": cluster_key,
                "speaker_id": None,
                "speaker_identity_available": False,
            }
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "name": name,
                    "sha256": digest,
                    "cluster": cluster_key,
                    "group": int(groups[idx]),
                    "origin": index["paths"][idx],
                }
            )

    # Funde no manifesto existente (chaves novas nao colidem: sao por conteudo).
    existing = dict(speaker_manifest.load_manifest())
    existing.update(manifest_entries)
    manifest_path = DATASETS_DIR / "metadata" / "speaker_manifest.json"
    manifest_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Manifesto atualizado: +%d entradas", len(manifest_entries))

    (out_dir / "build_manifest.json").write_text(
        json.dumps(
            {
                "schema": "xfakesong-clean-dataset-v4",
                "source": "brspeech",
                "rationale": (
                    "fonte unica pareada: P(classe|fonte)=0,5, atalho de fonte "
                    "eliminado por construcao"
                ),
                "grouping_unit": (
                    "speaker_connected_component + ndup_spectral_cluster"
                    if speaker_stats.get("available")
                    else "ndup_spectral_cluster"
                ),
                "one_sample_per_content_cluster": True,
                "speaker_identity_available": False,
                "speaker_identity_note": (
                    "BRSpeech-DF nao publica falante (schema: audio/label/model, "
                    "identico nas 3 configs); a identidade e INFERIDA de "
                    "embeddings e usada apenas como unidade de agrupamento"
                ),
                "speaker_grouping": speaker_stats,
                "class_pure_sources_excluded": list(CLASS_PURE_SOURCES),
                "counts": {s: dict(c) for s, c in written.items()},
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {s: dict(c) for s, c in written.items()}


def _audit(out_dir: Path) -> None:
    """Prova que nao ha reaproveitamento entre particoes."""
    payload = json.loads((out_dir / "build_manifest.json").read_text(encoding="utf-8"))
    rows = payload["samples"]
    by_split: dict[str, set[str]] = {}
    clusters: dict[str, set[str]] = {}
    groups: dict[str, set[int]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["sha256"])
        clusters.setdefault(row["split"], set()).add(row["cluster"])
        if "group" in row:
            groups.setdefault(row["split"], set()).add(int(row["group"]))

    print("\n=== AUDITORIA DE CONTAMINACAO ===")
    ok = True
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_audio = by_split.get(left, set()) & by_split.get(right, set())
        shared_cluster = clusters.get(left, set()) & clusters.get(right, set())
        shared_group = groups.get(left, set()) & groups.get(right, set())
        ok &= not shared_audio and not shared_cluster and not shared_group
        print(
            f"{left:>6} x {right:<6} audio={len(shared_audio):>4}  "
            f"cluster={len(shared_cluster):>4}  grupo(falante)={len(shared_group):>4}"
        )

    # Duplicata dentro de qualquer particao tambem e defeito: o dataset deve ter
    # um unico representante por conteudo.
    all_hashes = [row["sha256"] for row in rows]
    all_clusters = [row["cluster"] for row in rows]
    dup_audio = len(all_hashes) - len(set(all_hashes))
    dup_cluster = len(all_clusters) - len(set(all_clusters))
    ok &= dup_audio == 0 and dup_cluster == 0
    print(
        f"\nrepeticao interna: audio duplicado={dup_audio}  "
        f"conteudo duplicado={dup_cluster}  (ambos devem ser 0)"
    )

    spk = payload.get("speaker_grouping") or {}
    print("\n=== GARANTIA DE FALANTE ===")
    if spk.get("available"):
        print(f"metodo         {spk.get('method')}")
        print(f"limiar         {spk.get('threshold')}")
        print(
            f"grupos         {spk.get('groups')} "
            f"(maior={spk.get('largest_group')}, mediana={spk.get('median_group')})"
        )
        cal = spk.get("calibration", {})
        print(
            f"calibracao     precisao 1,000 em 0,95; recall "
            f"{cal.get('recall_at_threshold')} no limiar de agrupamento "
            f"({cal.get('ground_truth_speakers')} falantes reais)"
        )
        print(
            "garantia       nenhum par com similaridade de falante >= limiar\n"
            "               atravessa particoes (por construcao: componentes\n"
            "               conexas sao indivisiveis no split)"
        )
    else:
        print("INDISPONIVEL — agrupado apenas por conteudo")

    counts = payload["counts"]
    print("\n=== COMPOSICAO ===")
    total = 0
    for split in ("train", "val", "test"):
        real = counts.get(split, {}).get("real", 0)
        fake = counts.get(split, {}).get("fake", 0)
        total += real + fake
        print(f"{split:>6}: real={real:>5} fake={fake:>5} total={real + fake:>5}")
    print(f"{'TOTAL':>6}: {total}")

    print("\n=== ATALHO DE FONTE ===")
    print("fonte unica (brspeech) cobrindo as duas classes -> P(classe|fonte)=0,50")
    print("teto de acuracia por regra de fonte = 50,0% (acaso)")
    print("anterior (4 fontes, 3 puras de classe) = 87,6%")
    print(f"\nRESULTADO: {'SEM CONTAMINACAO' if ok else 'FALHOU'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true", help="Relata clusterizacao e sai.")
    parser.add_argument("--build", action="store_true", help="Constroi os splits.")
    parser.add_argument("--audit", action="store_true", help="Audita um build existente.")
    parser.add_argument("--per-class", type=int, default=7500)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=0.85,
        help="Similaridade de falante para componentes conexas. Calibrado "
        "par-a-par contra 78 falantes reais: precisao 1,000 em 0,95, recall "
        "0,733 em 0,85. Fundir demais e seguro; de menos vaza.",
    )
    parser.add_argument(
        "--out-dir", default="data/datasets/splits_v4", help="Destino dos splits."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if args.audit:
        _audit(out_dir)
        return 0

    index = _build_index(use_cache=not args.no_cache)
    labels = index["labels"]
    print(f"\npool unico: {Counter(labels)}")

    if args.analyze:
        _cluster_report(
            index["fps"], labels, (0.9999, 0.999, 0.995, 0.99, 0.98, 0.95)
        )
        return 0

    if not args.build:
        parser.error("escolha --analyze, --build ou --audit")

    roots = _cluster(index["fps"], args.threshold)
    sizes = Counter(roots.tolist())
    logger.info(
        "Clusters de conteudo: %d (maior=%d, com >1 amostra=%d)",
        len(sizes),
        max(sizes.values()),
        sum(1 for v in sizes.values() if v > 1),
    )

    groups, speaker_stats = _speaker_groups(
        index["paths"], roots, args.speaker_threshold
    )

    # Um representante por cluster de conteudo: elimina a repeticao em vez de
    # apenas mante-la dentro da mesma particao.
    keep = _representatives(roots, index["hashes"])
    logger.info(
        "representantes: %d de %d arquivos (%.0f%% eram redundantes)",
        len(keep),
        len(index["hashes"]),
        100.0 * (1 - len(keep) / max(len(index["hashes"]), 1)),
    )
    keep_set = set(keep.tolist())

    ratios = (0.70, 0.15, 0.15)
    assigned = _split_clusters(
        groups, labels, args.per_class, ratios, args.seed, allowed=keep_set
    )
    counts = _write_dataset(index, assigned, roots, out_dir, groups, speaker_stats)
    logger.info("Splits gravados em %s: %s", out_dir, counts)
    _audit(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
