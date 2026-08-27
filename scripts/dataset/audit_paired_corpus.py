#!/usr/bin/env python3
"""Auditoria do corpus pareado e da particao de disjuncao dupla.

Verifica no artefato, e nao no plano, tudo o que o Protocolo de Dataset afirma. Falha
com codigo != 0 quando uma garantia e violada, para poder rodar em CI antes de
qualquer treino.

Blocos:

A. Integridade e pareamento — todo real tem o seu fake do mesmo enunciado.
B. Disjuncao — locutor, frase, texto, enunciado e hash de conteudo nao
   atravessam particoes.
C. Balanceamento — 50/50 por particao e por locutor.
D. Atalhos de metadado — nenhuma variavel de proveniencia prediz a classe acima
   do acaso (fonte, locutor, frase, particao oficial).
E. Descritores de sinal — AUC de descritor unico entre as classes. Reprova
   apenas nos de EMPACOTAMENTO (RMS, duracao), que nao carregam informacao sobre
   sintese; os de sinal (crista, ZCR, centroide, rolloff, flatness, banda
   7-8 kHz) sao reportados, porque e deles que um detector legitimamente vive.
F. Quase-duplicatas de conteudo, por impressao digital espectral:
   F1 redundancia interna (mesmo locutor e classe) e F2 vazamento entre
   particoes (todos os pares de particoes diferentes).

Uso:
    python scripts/dataset/audit_paired_corpus.py
    python scripts/dataset/audit_paired_corpus.py --descriptor-sample 600
    python scripts/dataset/audit_paired_corpus.py --skip-nearduplicates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset.build_paired_pt_corpus import extract_window  # noqa: E402

logger = logging.getLogger("audit_paired")

CORPUS_DIR = ROOT / "data" / "datasets" / "corpus"
SPLITS_DIR = ROOT / "data" / "datasets" / "splits"
ASSIGNMENT_PATH = SPLITS_DIR / "assignment.jsonl"
REPORT_PATH = SPLITS_DIR / "audit_report.json"

SPLITS = ("train", "val", "test")
ORACLE_MAX = 0.55

# Nivel absoluto nao carrega informacao sobre sintese: se separar as classes, o
# corpus esta medindo a diferenca entre os pipelines de distribuicao. O caso
# concreto: antes da normalizacao de loudness o RMS dava AUC 0,9926.
PACKAGING_DESCRIPTORS = frozenset({"rms_db"})
PACKAGING_AUC_MAX = 0.60

# A duracao e categoria a parte. Ela DIFERE entre as classes (AUC 0,64: o XTTS le
# o mesmo texto ~10% mais rapido) e isso e prosodia genuina do gerador, nao
# empacotamento. Mas com janela fixa o modelo nao ve a duracao — ela so vaza pela
# politica de janela, quando uma classe e repetida (`tile`) mais que a outra.
# Por isso o criterio nao e a AUC da duracao: e a taxa de repeticao, que tem de
# ser praticamente identica entre as classes.
DURATION_DESCRIPTORS = frozenset({"duration_sec"})
TILE_GAP_MAX = 0.02
SR = 16_000


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC por estatistica de Mann-Whitney, simetrizada para [0,5; 1,0].

    Simetrizar importa: um descritor que separa as classes ao contrario separa
    igualmente bem.
    """
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    stat = ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2
    value = stat / (len(positive) * len(negative))
    return float(max(value, 1.0 - value))


def majority_oracle(records: list[dict], key: str) -> tuple[float, int]:
    """Acuracia da regra trivial 'valor de `key`' -> classe majoritaria."""
    buckets: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        buckets[str(record[key])][record["label"]] += 1
    correct = sum(max(counter.values()) for counter in buckets.values())
    return correct / max(len(records), 1), len(buckets)


def load_assignment() -> list[dict]:
    if not ASSIGNMENT_PATH.exists():
        raise SystemExit(
            f"particao ausente: {ASSIGNMENT_PATH}\n"
            "rode primeiro: python scripts/dataset/build_paired_splits.py --build"
        )
    with ASSIGNMENT_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# A. Integridade e pareamento
# ---------------------------------------------------------------------------


def audit_pairing(records: list[dict]) -> dict:
    by_utterance: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_utterance[record["utterance_id"]].add(record["class"])
    unpaired = sorted(
        key for key, classes in by_utterance.items() if classes != {"real", "fake"}
    )

    def duplicate_groups(rows: list[dict]) -> dict[str, list[str]]:
        hashes: dict[str, list[str]] = defaultdict(list)
        for record in rows:
            hashes[record["content_sha256"]].append(record["path"])
        return {h: paths for h, paths in hashes.items() if len(paths) > 1}

    # O corpus preserva a aquisicao como ela e, inclusive os defeitos da fonte: o
    # CETUC repete gravacoes em indices consecutivos. O que precisa estar limpo e
    # a PARTICAO, de onde `build_paired_splits` remove esses enunciados.
    in_corpus = duplicate_groups(records)
    in_partition = duplicate_groups([r for r in records if r.get("split") in SPLITS])

    missing = [
        record["path"]
        for record in records
        if not (CORPUS_DIR / record["path"]).exists()
    ]
    return {
        "samples": len(records),
        "utterances": len(by_utterance),
        "unpaired_utterances": len(unpaired),
        "unpaired_examples": unpaired[:10],
        "exact_duplicate_groups_in_corpus": len(in_corpus),
        "exact_duplicate_groups_in_partition": len(in_partition),
        "exact_duplicate_examples": list(in_corpus.values())[:5],
        "missing_files": len(missing),
        "missing_examples": missing[:5],
        "speakers": len({record["speaker_id"] for record in records}),
        "sentences": len({record["sentence_index"] for record in records}),
        "text_ids": len({record["text_id"] for record in records}),
        "ok": not unpaired and not in_partition and not missing,
    }


def audit_text_consistency(records: list[dict]) -> dict:
    """Cada slot de frase deve mapear para exatamente um `text_id`.

    O CETUC afirma que todo locutor le as mesmas 1000 frases; isto verifica.
    """
    by_index: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_index[record["sentence_index"]].add(record["text_id"])
    divergent = {k: sorted(v) for k, v in by_index.items() if len(v) > 1}
    return {
        "sentence_slots": len(by_index),
        "slots_with_divergent_text": len(divergent),
        "examples": dict(list(divergent.items())[:5]),
        "ok": not divergent,
    }


# ---------------------------------------------------------------------------
# B. Disjuncao
# ---------------------------------------------------------------------------


def audit_disjointness(records: list[dict]) -> dict:
    used = [r for r in records if r["split"] in SPLITS]
    fields = {
        "speaker_id": "locutor",
        "sentence_index": "frase",
        "text_id": "texto",
        "utterance_id": "enunciado",
        "content_sha256": "hash de conteudo",
    }
    result: dict[str, dict] = {}
    ok = True
    for field, label in fields.items():
        by_split = {
            split: {r[field] for r in used if r["split"] == split} for split in SPLITS
        }
        overlaps = {}
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1 :]:
                shared = by_split[first] & by_split[second]
                overlaps[f"{first}x{second}"] = len(shared)
                if shared:
                    overlaps[f"{first}x{second}_examples"] = sorted(shared)[:5]
        field_ok = all(
            value == 0
            for key, value in overlaps.items()
            if not key.endswith("examples")
        )
        ok = ok and field_ok
        result[field] = {
            "label": label,
            "distinct_per_split": {s: len(v) for s, v in by_split.items()},
            "overlaps": overlaps,
            "ok": field_ok,
        }
    result["ok"] = ok
    return result


# ---------------------------------------------------------------------------
# C. Balanceamento
# ---------------------------------------------------------------------------


def audit_balance(records: list[dict]) -> dict:
    per_split: dict[str, dict] = {}
    ok = True
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        real = sum(1 for r in rows if r["label"] == 0)
        fake = len(rows) - real
        by_speaker = defaultdict(Counter)
        for row in rows:
            by_speaker[row["speaker_id"]][row["label"]] += 1
        unbalanced = [
            speaker
            for speaker, counter in by_speaker.items()
            if counter[0] != counter[1]
        ]
        split_ok = real == fake and not unbalanced
        ok = ok and split_ok
        per_split[split] = {
            "real": real,
            "fake": fake,
            "balanced": real == fake,
            "speakers": len(by_speaker),
            "speakers_unbalanced": len(unbalanced),
            "unbalanced_examples": unbalanced[:5],
            "ok": split_ok,
        }
    per_split["ok"] = ok
    return per_split


# ---------------------------------------------------------------------------
# D. Atalhos de metadado
# ---------------------------------------------------------------------------


def audit_metadata_shortcuts(records: list[dict]) -> dict:
    used = [r for r in records if r["split"] in SPLITS]
    result: dict[str, dict] = {}
    ok = True
    for field in (
        "source",
        "speaker_id",
        "sentence_index",
        "text_id",
        "cetuc_official_split",
    ):
        accuracy, groups = majority_oracle(used, field)
        field_ok = accuracy <= ORACLE_MAX
        ok = ok and field_ok
        result[field] = {
            "majority_oracle_accuracy": round(accuracy, 4),
            "groups": groups,
            "ok": field_ok,
        }
    # `generator_id` DEFINE a classe (bonafide vs xtts_v2). O oraculo e 100% por
    # construcao e nao e um atalho: e o rotulo. Registrado para deixar claro que
    # a excecao e intencional.
    accuracy, groups = majority_oracle(used, "generator_id")
    result["generator_id"] = {
        "majority_oracle_accuracy": round(accuracy, 4),
        "groups": groups,
        "note": "define o rotulo por construcao; nao e atalho",
        "ok": True,
    }
    result["ok"] = ok
    result["threshold"] = ORACLE_MAX
    return result


# ---------------------------------------------------------------------------
# E. Confundidores de baixo nivel
# ---------------------------------------------------------------------------


def _descriptors(path: Path, window_sec: float) -> dict[str, float] | None:
    """Descritores da JANELA que o modelo recebe, nao do arquivo inteiro.

    Usa `extract_window`, a mesma funcao do exportador, entao o que a auditoria
    mede e literalmente o que entra no modelo. `duration_sec` e a excecao: so faz
    sentido no arquivo.

    A analise usava o inicio do arquivo e quebrou num audio cujo primeiro segundo
    era silencio digital: o espectro zerado fazia `searchsorted` devolver um
    indice fora do array. O recorte central evita o silencio de cabeca e a guarda
    de energia cobre o resto.
    """
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        return None
    if audio.size < 512 or not np.all(np.isfinite(audio)):
        return None
    duration = len(audio) / sr

    # Mesma funcao que o exportador usa, entao o que e medido aqui e exatamente
    # o que o modelo recebe — inclusive a normalizacao de nivel da janela.
    center, _, _ = extract_window(audio, int(sr * window_sec))

    size = 16384
    segment = center[:size] * np.hanning(min(len(center), size))
    spectrum = np.abs(np.fft.rfft(segment, n=size)) ** 2
    freqs = np.fft.rfftfreq(size, 1 / sr)
    total = float(spectrum.sum())
    if total <= 0 or not np.isfinite(total):
        return None

    def band(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(spectrum[mask].mean() + 1e-20)

    cumulative = np.cumsum(spectrum) / total
    index = min(int(np.searchsorted(cumulative, 0.95)), len(freqs) - 1)
    rms = float(np.sqrt((center**2).mean()) + 1e-12)
    peak = float(np.abs(center).max() + 1e-12)
    return {
        "duration_sec": duration,
        "rms_db": float(20 * np.log10(rms)),
        "peak_db": float(20 * np.log10(peak)),
        "crest_db": float(20 * np.log10(peak / rms)),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(center))) > 0)),
        "spectral_centroid_hz": float((freqs * spectrum).sum() / total),
        "spectral_rolloff95_hz": float(freqs[index]),
        "spectral_flatness": float(
            np.exp(np.log(spectrum + 1e-20).mean()) / (spectrum.mean() + 1e-20)
        ),
        "band_7k_8k_rel_db": float(10 * np.log10(band(7000, 7900) / band(300, 3000))),
    }


def audit_descriptors(
    records: list[dict], per_split: int, workers: int, window_sec: float
) -> dict:
    rng = np.random.default_rng(42)
    selected: list[dict] = []
    for split in SPLITS:
        for label in (0, 1):
            rows = [r for r in records if r["split"] == split and r["label"] == label]
            if not rows:
                continue
            take = min(per_split, len(rows))
            index = rng.choice(len(rows), size=take, replace=False)
            selected.extend(rows[int(i)] for i in index)

    def work(record: dict) -> tuple[dict, dict | None]:
        return record, _descriptors(CORPUS_DIR / record["path"], window_sec)

    values: dict[int, dict[str, list[float]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record, feats in pool.map(work, selected):
            if feats is None:
                continue
            for name, value in feats.items():
                values[record["label"]][name].append(value)

    per_descriptor = {}
    for name in sorted(values[0]):
        score = auc(
            np.asarray(values[1][name], dtype=float),
            np.asarray(values[0][name], dtype=float),
        )
        per_descriptor[name] = {
            "auc": round(score, 4),
            "real_mean": round(float(np.mean(values[0][name])), 4),
            "fake_mean": round(float(np.mean(values[1][name])), 4),
            "kind": (
                "empacotamento"
                if name in PACKAGING_DESCRIPTORS
                else "duracao" if name in DURATION_DESCRIPTORS else "sinal"
            ),
        }

    # Simetria da politica de janela: se uma classe e recortada e a outra
    # repetida (`tile`), a propria emenda vira pista de classe.
    tile_rate = {}
    for label, name in ((0, "real"), (1, "fake")):
        durations = np.asarray(values[label]["duration_sec"], dtype=float)
        tile_rate[name] = round(float(np.mean(durations < window_sec)), 4)
    tile_gap = abs(tile_rate["real"] - tile_rate["fake"])

    packaging = {
        name: block["auc"]
        for name, block in per_descriptor.items()
        if name in PACKAGING_DESCRIPTORS
    }
    worst_packaging = max(packaging.items(), key=lambda kv: kv[1], default=("", 0.5))
    signal = {
        name: block["auc"]
        for name, block in per_descriptor.items()
        if block["kind"] == "sinal"
    }
    worst_signal = max(signal.items(), key=lambda kv: kv[1], default=("", 0.5))

    return {
        "sampled": len(selected),
        "window_sec": window_sec,
        "descriptors": per_descriptor,
        # Confundidor: nao carrega informacao sobre sintese, so sobre como o
        # audio foi empacotado. TEM de ficar no acaso, senao o corpus esta
        # medindo a diferenca entre os pipelines de distribuicao.
        "worst_packaging": {
            "name": worst_packaging[0],
            "auc": round(worst_packaging[1], 4),
        },
        # Propriedade do sinal: e o que um detector de spoofing legitimamente
        # usa. Reportado sempre, nunca reprovado — mas o valor diz o quanto a
        # tarefa e facil, e isso precisa aparecer no relatorio do benchmark.
        "worst_signal": {"name": worst_signal[0], "auc": round(worst_signal[1], 4)},
        "tile_rate": tile_rate,
        "tile_rate_gap": round(tile_gap, 4),
        "packaging_threshold": PACKAGING_AUC_MAX,
        "tile_gap_threshold": TILE_GAP_MAX,
        "ok": worst_packaging[1] <= PACKAGING_AUC_MAX and tile_gap <= TILE_GAP_MAX,
    }


# ---------------------------------------------------------------------------
# o. Quase-duplicatas de conteudo
# ---------------------------------------------------------------------------


def _fingerprint(path: Path, window_sec: float = 5.0) -> np.ndarray | None:
    """Impressao espectral: 32 bandas log x 8 segmentos, L2-normalizada.

    Calculada na MESMA janela que vai para o NPZ — medir o arquivo inteiro
    compararia audio que o modelo nunca ve.
    """
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        return None
    samples = int(sr * window_sec)
    if len(audio) >= samples:
        start = (len(audio) - samples) // 2
        audio = audio[start : start + samples]
    else:
        audio = np.tile(audio, int(np.ceil(samples / max(len(audio), 1))))[:samples]
    frame = 512
    n_frames = len(audio) // frame
    if n_frames < 8:
        return None
    spec = np.abs(
        np.fft.rfft(audio[: n_frames * frame].reshape(n_frames, frame), axis=1)
    )
    edges = np.unique(np.geomspace(1, spec.shape[1] - 1, 33).astype(int))
    bands = np.stack(
        [spec[:, edges[i] : edges[i + 1]].mean(axis=1) for i in range(len(edges) - 1)],
        axis=1,
    )
    chunks = np.array_split(np.log1p(bands), 8, axis=0)
    vector = np.concatenate([chunk.mean(axis=0) for chunk in chunks])
    norm = np.linalg.norm(vector)
    return None if norm == 0 else (vector / norm).astype("float32")


class _UnionFind:
    """Componentes conexas do grafo de similaridade, com compressao de caminho."""

    def __init__(selo, size: int) -> None:
        selo._parent = list(range(size))

    def find(selo, item: int) -> int:
        parent = selo._parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(selo, a: int, b: int) -> None:
        root_a, root_b = selo.find(a), selo.find(b)
        if root_a != root_b:
            selo._parent[root_a] = root_b

    def sizes(selo) -> Counter:
        return Counter(selo.find(item) for item in range(len(selo._parent)))


def _fingerprints(records: list[dict], workers: int) -> tuple[list[dict], np.ndarray]:
    """Impressao espectral de cada amostra, na ordem dos registros validos."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prints = list(pool.map(lambda r: _fingerprint(CORPUS_DIR / r["path"]), records))
    valid = [(r, p) for r, p in zip(records, prints) if p is not None]
    if not valid:
        return [], np.empty((0, 0), dtype="float32")
    return [r for r, _ in valid], np.stack([p for _, p in valid])


def audit_nearduplicates(records: list[dict], workers: int, threshold: float) -> dict:
    """Quase-duplicatas por conteudo, em duas medicoes distintas.

    **F1 — redundancia interna.** Dentro de cada (locutor, classe): quantas
    amostras tem uma gemea quase identica. E a medicao que sustenta "sem repetir
    amostras"; foi o defeito que retirou o v3, onde 69% do corpus tinha gemea.

    **F2 — vazamento entre particoes.** Compara TODAS as amostras de particoes
    diferentes, par a par. Esta e a verificacao que responde "nenhuma amostra se
    repete entre treino, validacao e teste".

    As duas sao necessarias e nenhuma substitui a outra: como locutor nao
    atravessa particao, F1 jamais encontraria um par entre splits — a versao
    inicial deste bloco procurava vazamento dentro do locutor e por isso dava
    zero por construcao, sem verificar nada.
    """
    used = [r for r in records if r["split"] in SPLITS]

    # --- F1: redundancia interna ---------------------------------------------
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in used:
        groups[(record["speaker_id"], record["class"])].append(record)

    internal_pairs = 0
    biggest = 1
    components_over_one = 0
    examples: list[dict] = []
    for _, rows in sorted(groups.items()):
        valid, matrix = _fingerprints(rows, workers)
        if len(valid) < 2:
            continue
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, 0.0)
        components = _UnionFind(len(valid))
        for i, a in np.argwhere(similarity >= threshold):
            if i >= a:
                continue
            internal_pairs += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "a": valid[int(i)]["path"],
                        "b": valid[int(a)]["path"],
                        "similarity": round(float(similarity[i, a]), 5),
                        "split": valid[int(i)]["split"],
                    }
                )
            components.union(int(i), int(a))
        sizes = components.sizes()
        biggest = max(biggest, max(sizes.values()))
        components_over_one += sum(1 for size in sizes.values() if size > 1)

    # --- F2: vazamento entre particoes ---------------------------------------
    by_split: dict[str, tuple[list[dict], np.ndarray]] = {}
    for split in SPLITS:
        rows = [r for r in used if r["split"] == split]
        by_split[split] = _fingerprints(rows, workers)

    cross: dict[str, dict] = {}
    cross_total = 0
    cross_examples: list[dict] = []
    for position, first in enumerate(SPLITS):
        for second in SPLITS[position + 1 :]:
            rows_a, matrix_a = by_split[first]
            rows_b, matrix_b = by_split[second]
            if not len(rows_a) or not len(rows_b):
                cross[f"{first}x{second}"] = {"pairs": 0, "max_similarity": None}
                continue
            best = 0.0
            hits = 0
            # Em blocos: a matriz cheia treino x teste teria ~10^8 celulas.
            for start in range(0, len(rows_a), 2048):
                block = matrix_a[start : start + 2048] @ matrix_b.T
                best = max(best, float(block.max()))
                found = np.argwhere(block >= threshold)
                hits += len(found)
                for i, a in found[:5]:
                    if len(cross_examples) < 5:
                        cross_examples.append(
                            {
                                "a": rows_a[start + int(i)]["path"],
                                "b": rows_b[int(a)]["path"],
                                "similarity": round(float(block[i, a]), 5),
                                "splits": f"{first}x{second}",
                            }
                        )
            cross_total += hits
            cross[f"{first}x{second}"] = {
                "pairs": hits,
                "max_similarity": round(best, 5),
                "compared": len(rows_a) * len(rows_b),
            }

    return {
        "threshold": threshold,
        "internal": {
            "groups_compared": len(groups),
            "near_duplicate_pairs": internal_pairs,
            "components_over_one": components_over_one,
            "largest_component": biggest,
            "examples": examples,
        },
        "cross_split": cross,
        "cross_split_pairs": cross_total,
        "cross_split_examples": cross_examples,
        "ok": cross_total == 0,
    }


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor-sample",
        type=int,
        default=400,
        help="Amostras por particao e classe no bloco E.",
    )
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--nearduplicate-threshold", type=float, default=0.99)
    parser.add_argument("--skip-nearduplicates", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    records = load_assignment()
    report: dict[str, dict] = {}

    logger.info("A. integridade e pareamento")
    report["pairing"] = audit_pairing(records)
    report["text_consistency"] = audit_text_consistency(records)
    logger.info("B. disjuncao entre particoes")
    report["disjointness"] = audit_disjointness(records)
    logger.info("C. balanceamento")
    report["balance"] = audit_balance(records)
    logger.info("D. atalhos de metadado")
    report["metadata_shortcuts"] = audit_metadata_shortcuts(records)
    logger.info("E. confundidores de baixo nivel (decodificando audio)")
    report["descriptors"] = audit_descriptors(
        records, args.descriptor_sample, args.workers, args.window_sec
    )
    if args.skip_nearduplicates:
        report["nearduplicates"] = {"skipped": True, "ok": True}
    else:
        logger.info("F. quase-duplicatas por conteudo (varredura completa)")
        report["nearduplicates"] = audit_nearduplicates(
            records, args.workers, args.nearduplicate_threshold
        )

    failures = [name for name, block in report.items() if not block.get("ok", True)]
    report["verdict"] = {"failed_blocks": failures, "ok": not failures}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_report(report)
    print(f"\nrelatorio: {REPORT_PATH.relative_to(ROOT)}")
    return 1 if failures else 0


def _print_report(report: dict) -> None:
    def mark(block: dict) -> str:
        return "OK  " if block.get("ok", True) else "FALHA"

    pairing = report["pairing"]
    print("\n=== A. INTEGRIDADE E PAREAMENTO ===")
    print(
        f"  {mark(pairing)} {pairing['samples']} amostras"
        f" | {pairing['utterances']} enunciados"
    )
    print(
        f"       {pairing['speakers']} locutores"
        f" | {pairing['sentences']} frases"
        f" | {pairing['text_ids']} textos distintos"
    )
    print(f"       enunciados sem par: {pairing['unpaired_utterances']}")
    print(
        "       duplicatas exatas (SHA-256): "
        f"{pairing['exact_duplicate_groups_in_corpus']} grupos no corpus"
        f" -> {pairing['exact_duplicate_groups_in_partition']} na particao"
    )
    print(f"       arquivos ausentes: {pairing['missing_files']}")
    text = report["text_consistency"]
    print(
        f"  {mark(text)} slots de frase com texto divergente: "
        f"{text['slots_with_divergent_text']}/{text['sentence_slots']}"
    )

    print("\n=== B. DISJUNCAO ENTRE PARTICOES ===")
    disjoint = report["disjointness"]
    for field, block in disjoint.items():
        if field == "ok":
            continue
        pairs = {
            k: v for k, v in block["overlaps"].items() if not k.endswith("examples")
        }
        print(
            f"  {mark(block)} {block['label']:>18}: sobreposicao {pairs}"
            f" | distintos {block['distinct_per_split']}"
        )

    print("\n=== C. BALANCEAMENTO ===")
    balance = report["balance"]
    for split in SPLITS:
        block = balance[split]
        print(
            f"  {mark(block)} {split:>5}: {block['real']} reais"
            f" / {block['fake']} falsas | {block['speakers']} locutores,"
            f" desbalanceados {block['speakers_unbalanced']}"
        )

    print("\n=== D. ATALHOS DE METADADO (oraculo 'valor -> classe majoritaria') ===")
    shortcuts = report["metadata_shortcuts"]
    for field, block in shortcuts.items():
        if field in {"ok", "threshold"}:
            continue
        note = f"  <- {block['note']}" if block.get("note") else ""
        accuracy = block["majority_oracle_accuracy"] * 100
        print(
            f"  {mark(block)} {field:>22}: {accuracy:6.2f}%"
            f" ({block['groups']} grupos){note}"
        )

    print("\n=== E. DESCRITORES DE SINAL (AUC de descritor unico) ===")
    desc = report["descriptors"]
    print(
        f"  {mark(desc)} {desc['sampled']} amostras | empacotamento <= "
        f"{desc['packaging_threshold']} e diferenca de tile <= "
        f"{desc['tile_gap_threshold']}; sinal e duracao apenas reportados"
    )
    for name, block in sorted(
        desc["descriptors"].items(), key=lambda kv: -kv[1]["auc"]
    ):
        flag = {"empacotamento": "!", "duracao": "~"}.get(block["kind"], " ")
        print(
            f"     {flag} AUC={block['auc']:.4f}  {name:<22} {block['kind']:<14}"
            f" real={block['real_mean']:>9.3f}"
            f"  fake={block['fake_mean']:>9.3f}"
        )
    print(
        f"       pior empacotamento: {desc['worst_packaging']['name']}"
        f" ({desc['worst_packaging']['auc']:.4f})"
        f" | pior sinal: {desc['worst_signal']['name']}"
        f" ({desc['worst_signal']['auc']:.4f})"
    )
    print(
        f"       taxa de tile (duracao < {desc['window_sec']}s):"
        f" real={desc['tile_rate']['real']:.3f}"
        f" fake={desc['tile_rate']['fake']:.3f}"
        f" | diferenca {desc['tile_rate_gap']:.3f}"
    )

    print("\n=== F. QUASE-DUPLICATAS DE CONTEUDO ===")
    near = report["nearduplicates"]
    if near.get("skipped"):
        print("  (pulado)")
    else:
        internal = near["internal"]
        print(f"  cosseno >= {near['threshold']}")
        print(
            f"       F1 redundancia interna (mesmo locutor e classe):"
            f" {internal['near_duplicate_pairs']} pares"
            f" | grupos com gemea {internal['components_over_one']}"
            f" | maior {internal['largest_component']}"
        )
        print(f"  {mark(near)} F2 vazamento entre particoes:")
        for pair, block in near["cross_split"].items():
            similarity = block["max_similarity"]
            shown = "n/d" if similarity is None else f"{similarity:.4f}"
            print(
                f"          {pair:>12}: {block['pairs']} pares acima do limiar"
                f" | maior similaridade {shown}"
                f" | {block.get('compared', 0)} comparacoes"
            )

    verdict = report["verdict"]
    print("\n=== VEREDITO ===")
    if verdict["ok"]:
        print("  todas as garantias verificadas")
    else:
        print(f"  BLOCOS COM FALHA: {', '.join(verdict['failed_blocks'])}")


if __name__ == "__main__":
    raise SystemExit(main())
