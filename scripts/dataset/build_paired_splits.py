#!/usr/bin/env python3
"""Particiona o corpus pareado com disjuncao DUPLA: locutor x frase.

O corpus produzido por `build_paired_pt_corpus.py` e um grid completo
locutor x frase x classe. Isso permite uma garantia que nenhum protocolo
anterior do projeto conseguiu dar:

    treino, validacao e teste nao compartilham NEM locutor NEM frase.

A particao e o bloco diagonal do grid:

    split(amostra) = S   <=>   locutor(amostra) in S_loc  E  frase(amostra) in S_frase

Amostras fora da diagonal (locutor de treino lendo frase de teste, etc.) sao
EXCLUIDAS. Nao ha como aproveita-las sem quebrar uma das duas disjuncoes; o
custo esta declarado no manifesto (`coverage`).

Duas estrategias para a particao de LOCUTORES:

- `stratified` (padrao): 60/20/20 dos 56 locutores pareados, estratificado por
  sexo. Da 33/11/12 locutores, contra 44/5/7 da oficial. Onze a doze locutores em
  cada conjunto de avaliacao sustentam um intervalo de confianca honesto; cinco
  nao sustentam — com cinco, um unico locutor atipico move a metrica inteira.
- `official`: a particao publicada pelo CETUC (`data/<split>/`). Mais defensavel
  por vir dos autores, mas ela foi desenhada para ASR sobre os 101 locutores, e
  o recorte de 56 pareados ja distorce as proporcoes originais (81/10/10 vira
  44/5/7).

As duas sao disjuntas por locutor; a escolha e entre autoridade da fonte e poder
estatistico na avaliacao. A particao de FRASES e sempre nossa, deterministica
pela semente, porque o CETUC nao define uma.

Como o corpus e exatamente pareado (o par so existe quando as duas amostras
passam na validacao), cada particao sai 50/50 por construcao, sem cotas.

Uso:
    python scripts/dataset/build_paired_splits.py --plan
    python scripts/dataset/build_paired_splits.py --build
    python scripts/dataset/build_paired_splits.py --build --speaker-strategy official
    python scripts/dataset/build_paired_splits.py --build --sentences 600 200 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("paired_splits")

CORPUS_DIR = ROOT / "data" / "datasets" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.jsonl"
SPLITS_DIR = ROOT / "data" / "datasets" / "splits"

# A particao oficial do CETUC chama a validacao de `dev`; o resto do projeto usa
# `val` (nomes dos diretorios, chaves do NPZ, `X_val`).
OFFICIAL_TO_SPLIT = {"train": "train", "dev": "val", "test": "test"}
SPLITS = ("train", "val", "test")


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"manifesto ausente: {MANIFEST_PATH}\n"
            "rode primeiro: python scripts/dataset/build_paired_pt_corpus.py --build"
        )
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def speaker_partition(
    records: list[dict],
    strategy: str,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Distribui os locutores entre as particoes, sem repetir nenhum.

    `stratified` embaralha dentro de cada sexo antes de repartir, entao as tres
    particoes ficam com proporcao semelhante de vozes femininas e masculinas. Sem
    isso, um sorteio simples pode entregar um conjunto de teste quase todo de um
    sexo so — com 11 locutores isso e perfeitamente possivel, e a metrica passaria
    a medir tambem a diferenca entre vozes graves e agudas.

    O codigo do CETUC ja carrega o sexo na primeira letra (`F049`, `M001`).
    """
    speakers = sorted({record["speaker_id"] for record in records})
    if strategy == "official":
        official = {
            record["speaker_id"]: OFFICIAL_TO_SPLIT[record["cetuc_official_split"]]
            for record in records
        }
        return {speaker: official[speaker] for speaker in speakers}
    if strategy != "stratified":
        raise SystemExit(f"estrategia de locutor desconhecida: {strategy}")

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    # O resto da divisao e distribuido a partir do treino, entao os conjuntos de
    # avaliacao nunca ficam menores do que a fracao pedida por arredondamento.
    for sex in sorted({speaker[0] for speaker in speakers}):
        group = [speaker for speaker in speakers if speaker[0] == sex]
        order = rng.permutation(len(group))
        shuffled = [group[int(i)] for i in order]
        n_val = int(round(len(group) * fractions[1]))
        n_test = int(round(len(group) * fractions[2]))
        n_val = min(n_val, max(len(group) - 2, 0))
        n_test = min(n_test, max(len(group) - n_val - 1, 0))
        for position, speaker in enumerate(shuffled):
            if position < n_test:
                assignment[speaker] = "test"
            elif position < n_test + n_val:
                assignment[speaker] = "val"
            else:
                assignment[speaker] = "train"
    return assignment


def content_groups(records: list[dict]) -> dict[str, str]:
    """Agrupa os slots de frase por componente conexa de `indice <-> texto`.

    Nao basta agrupar pelo `sentence_index`: **medido no corpus**, 3 textos
    aparecem em DOIS slots diferentes (o CETUC repete algumas frases na lista de
    1000). Particionar por indice deixaria o mesmo texto em treino e teste — foi
    exatamente o que a auditoria acusou (`texto: trainxtest = 1`).

    Tambem nao basta agrupar pelo `text_id`: se um locutor divergir do texto
    canonico de um slot, o slot ganharia dois textos e se dividiria.

    Unir as duas relacoes cobre os dois casos: dois slots com o mesmo texto ficam
    no mesmo grupo, e dois textos no mesmo slot tambem.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for record in records:
        union(f"idx:{record['sentence_index']}", f"txt:{record['text_id']}")

    return {
        record["sentence_index"]: find(f"idx:{record['sentence_index']}")
        for record in records
    }


def sentence_partition(
    records: list[dict], sizes: tuple[int, int, int], seed: int
) -> dict[str, str]:
    """Distribui os grupos de conteudo entre as particoes.

    `sizes` conta grupos de conteudo (quase sempre = frases). O ultimo split
    absorve o resto, para que nenhum grupo fique sem particao quando o numero de
    grupos nao for exatamente 1000.
    """
    group_of = content_groups(records)
    groups = sorted(set(group_of.values()))
    if sizes[0] + sizes[1] >= len(groups):
        raise SystemExit(
            f"pedidos {sizes[0]} + {sizes[1]} grupos para treino e validacao, "
            f"o corpus tem {len(groups)}. Ajuste --sentences."
        )
    order = np.random.default_rng(seed).permutation(len(groups))
    shuffled = [groups[int(i)] for i in order]
    split_of_group: dict[str, str] = {}
    for index, group in enumerate(shuffled):
        if index < sizes[0]:
            split_of_group[group] = "train"
        elif index < sizes[0] + sizes[1]:
            split_of_group[group] = "val"
        else:
            split_of_group[group] = "test"
    logger.info(
        "grupos de conteudo: %d (de %d slots de frase) -> train %d / val %d / test %d",
        len(groups),
        len(group_of),
        sizes[0],
        sizes[1],
        len(groups) - sizes[0] - sizes[1],
    )
    return {index: split_of_group[group] for index, group in group_of.items()}


def duplicate_audio_utterances(records: list[dict]) -> set[str]:
    """Enunciados cujo audio e byte-identico ao de outro enunciado.

    **Medido no corpus**: o CETUC contem gravacoes duplicadas — o WAV do indice N
    reaparece como indice N+1, mesmo locutor, mesma classe, bytes identicos, mas
    com transcricao diferente. Ou seja, uma das duas tem o texto errado, e nao ha
    como saber qual sem reconhecimento de fala.

    Os dois lados sao descartados: manter um seria escolher arbitrariamente entre
    duas transcricoes das quais uma esta errada. O custo e minimo (15 grupos em
    12.556 enunciados na medicao parcial) e resolve de uma vez a repeticao de
    amostra e o rotulo de conteudo incorreto.
    """
    by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_hash[record["content_sha256"]].add(record["utterance_id"])
    affected: set[str] = set()
    for utterances in by_hash.values():
        if len(utterances) > 1:
            affected |= utterances
    return affected


def short_utterances(records: list[dict], min_duration: float) -> set[str]:
    """Enunciados em que ALGUM dos dois lados e mais curto que a janela.

    **Medido**: os clones falam mais rapido que o original no mesmo texto
    (mediana 4,41 s contra 4,90 s). Com janela fixa, quem nao alcanca a janela e
    repetido (`tile`) e quem passa e recortado — e a emenda da repeticao vira
    pista de classe. Com janela de 5 s a diferenca de taxa de repeticao era de
    **0,19** entre as classes (0,510 no real contra 0,700 no clone).

    Descartar o par inteiro quando qualquer lado nao alcanca a janela zera a
    repeticao: toda amostra vira recorte central puro, do mesmo tamanho, e a
    duracao deixa de ser visivel para o modelo. A 3 s isso custa 4,8% dos pares.
    """
    by_utterance: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_utterance[record["utterance_id"]].append(record["duration_sec"])
    return {
        utterance
        for utterance, durations in by_utterance.items()
        if min(durations) < min_duration
    }


def assign(
    records: list[dict],
    sentence_split: dict[str, str],
    speaker_split: dict[str, str] | None = None,
    min_duration: float = 0.0,
) -> list[dict]:
    """Rotula cada amostra com sua particao (ou o motivo da exclusao)."""
    duplicated = duplicate_audio_utterances(records)
    too_short = short_utterances(records, min_duration) if min_duration else set()
    if speaker_split is None:
        speaker_split = {
            record["speaker_id"]: OFFICIAL_TO_SPLIT[record["cetuc_official_split"]]
            for record in records
        }
    for record in records:
        speaker_split_of = speaker_split.get(record["speaker_id"])
        text_split = sentence_split.get(record["sentence_index"])
        if record["utterance_id"] in duplicated:
            record["split"] = "excluded_duplicate_audio"
        elif record["utterance_id"] in too_short:
            record["split"] = "excluded_shorter_than_window"
        elif speaker_split_of is None or text_split is None:
            record["split"] = "excluded_unassigned"
        elif speaker_split_of == text_split:
            record["split"] = speaker_split_of
        else:
            record["split"] = f"excluded_offdiagonal:{speaker_split_of}x{text_split}"
    return records


def _link(src: Path, dst: Path) -> str:
    """Hardlink quando possivel; copia como fallback.

    57 mil copias de WAV seriam ~7 GB duplicados sem necessidade — o conteudo e
    identico ao do corpus, que e a fonte de verdade.
    """
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize(records: list[dict]) -> dict:
    """Cria splits/<split>/<classe>/ com hardlinks para o corpus."""
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)
    for split in SPLITS:
        for cls in ("real", "fake"):
            (SPLITS_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    modes: Counter = Counter()
    for record in records:
        if record["split"] not in SPLITS:
            continue
        src = CORPUS_DIR / record["path"]
        dst = SPLITS_DIR / record["split"] / record["class"] / src.name
        modes[_link(src, dst)] += 1
        record["split_path"] = dst.relative_to(ROOT).as_posix()
    return dict(modes)


def sync_speaker_manifest(records: list[dict]) -> int:
    """Publica a proveniencia no manifesto do projeto, em lote.

    O exportador de NPZ e as auditorias existentes leem `speaker_manifest.json`
    por `<classe>/<basename>`. Gravar em lote (e nao por amostra) evita reescrever
    um JSON de dezenas de MB 112 mil vezes.
    """
    from app.domain.dataset_metadata import speaker_manifest as sm

    manifest = sm.load_manifest()
    for record in records:
        key = f"{record['class']}/{Path(record['path']).name}"
        manifest[key] = {
            "source": record["source"],
            "speaker_id": record["speaker_id"],
            "utterance_id": record["utterance_id"],
            "text_id": record["text_id"],
            "generator_id": record["generator_id"],
            "source_revision": record["acquisition_revision"],
            "label": record["label"],
            "content_sha256": record["content_sha256"],
        }
    sm._CACHE = manifest
    sm._DIRTY = True
    sm.flush()
    return len(manifest)


def summarize(
    records: list[dict],
    sentence_split: dict[str, str],
    speaker_split: dict[str, str],
    strategy: str,
    seed: int,
    min_duration: float,
) -> dict:
    by_split: dict[str, dict] = {}
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        speakers = sorted({r["speaker_id"] for r in rows})
        sentences = sorted({r["sentence_index"] for r in rows})
        by_split[split] = {
            "samples": len(rows),
            "real": sum(1 for r in rows if r["label"] == 0),
            "fake": sum(1 for r in rows if r["label"] == 1),
            "pairs": len(rows) // 2,
            "speakers": len(speakers),
            "speaker_ids": speakers,
            "speakers_female": sum(1 for s in speakers if s.startswith("F")),
            "speakers_male": sum(1 for s in speakers if s.startswith("M")),
            "sentences": len(sentences),
            "hours": round(sum(r["duration_sec"] for r in rows) / 3600.0, 2),
        }
    excluded = Counter(r["split"] for r in records if r["split"] not in SPLITS)
    used = sum(by_split[s]["samples"] for s in SPLITS)
    return {
        "seed": seed,
        "strategy": "speaker_x_sentence_double_disjoint_block_diagonal",
        # A janela de exportacao NAO pode ser maior que isto: acima disso volta a
        # existir repeticao (`tile`) e ela e assimetrica entre as classes.
        "min_duration_sec": min_duration,
        "speaker_partition_strategy": strategy,
        "speaker_partition": speaker_split,
        "sentence_partition_sizes": {
            split: sum(1 for v in sentence_split.values() if v == split)
            for split in SPLITS
        },
        "splits": by_split,
        "excluded": dict(excluded),
        "coverage": {
            "corpus_samples": len(records),
            "used_samples": used,
            "used_ratio": round(used / max(len(records), 1), 4),
            "excluded_samples": len(records) - used,
        },
    }


def build(
    sizes: tuple[int, int, int],
    seed: int,
    do_materialize: bool,
    strategy: str,
    fractions: tuple[float, float, float],
    min_duration: float,
) -> None:
    records = load_manifest()
    logger.info("corpus: %d amostras", len(records))
    speaker_split = speaker_partition(records, strategy, fractions, seed)
    counts = Counter(speaker_split.values())
    logger.info(
        "locutores (%s): train %d / val %d / test %d",
        strategy,
        counts["train"],
        counts["val"],
        counts["test"],
    )
    sentence_split = sentence_partition(records, sizes, seed)
    records = assign(records, sentence_split, speaker_split, min_duration)

    summary = summarize(
        records, sentence_split, speaker_split, strategy, seed, min_duration
    )
    SPLITS_DIR.parent.mkdir(parents=True, exist_ok=True)

    if do_materialize:
        summary["link_modes"] = materialize(records)
        logger.info("splits materializados: %s", summary["link_modes"])
        summary["speaker_manifest_entries"] = sync_speaker_manifest(records)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with (SPLITS_DIR / "assignment.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (SPLITS_DIR / "split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (SPLITS_DIR / "sentence_partition.json").write_text(
        json.dumps(sentence_split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(summary)


def _report(summary: dict) -> None:
    print("\n=== PARTICAO (disjuncao dupla locutor x frase) ===")
    print(
        f"{'split':>7}{'amostras':>10}{'reais':>8}{'falsas':>8}"
        f"{'locutores':>11}{'frases':>8}{'horas':>8}"
    )
    for split in SPLITS:
        info = summary["splits"][split]
        print(
            f"{split:>7}{info['samples']:>10}{info['real']:>8}{info['fake']:>8}"
            f"{info['speakers']:>11}{info['sentences']:>8}{info['hours']:>8.2f}"
            f"   ({info['speakers_female']}F/{info['speakers_male']}M)"
        )
    cov = summary["coverage"]
    print(
        f"\naproveitamento: {cov['used_samples']}/{cov['corpus_samples']} "
        f"({cov['used_ratio'] * 100:.1f}%) — {cov['excluded_samples']} fora da diagonal"
    )
    for reason, count in sorted(summary["excluded"].items()):
        print(f"   {reason}: {count}")


def plan(
    sizes: tuple[int, int, int],
    seed: int,
    strategy: str,
    fractions: tuple[float, float, float],
) -> None:
    records = load_manifest()
    sentences = len({r["sentence_index"] for r in records})
    print(f"\ncorpus: {len(records)} amostras | {sentences} frases distintas")

    for candidate in ("stratified", "official"):
        speaker_split = speaker_partition(records, candidate, fractions, seed)
        speakers = defaultdict(set)
        for speaker, split in speaker_split.items():
            speakers[split].add(speaker)
        marker = " (padrao)" if candidate == strategy else ""
        print(f"\nestrategia de locutor: {candidate}{marker}")
        total = 0
        for split, size in zip(SPLITS, sizes):
            group = sorted(speakers[split])
            female = sum(1 for s in group if s.startswith("F"))
            pairs = len(group) * size
            total += pairs * 2
            print(
                f"  {split:>5}: {len(group):>3} locutores"
                f" ({female}F/{len(group) - female}M)"
                f" x {size:>4} frases = {pairs:>6} pares = {pairs * 2:>6} amostras"
            )
        print(
            f"  total: {total} amostras de {len(records)} "
            f"({100 * total / max(len(records), 1):.1f}%) — teto sem contar"
            f" a cobertura desigual do upstream"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--sentences",
        type=int,
        nargs=3,
        default=(600, 200, 200),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Quantas frases (de 1000) em cada particao.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speaker-strategy",
        choices=("stratified", "official"),
        default="stratified",
        help=(
            "stratified: 60/20/20 dos pareados, estratificado por sexo (padrao). "
            "official: a particao publicada pelo CETUC."
        ),
    )
    parser.add_argument(
        "--speaker-fractions",
        type=float,
        nargs=3,
        default=(0.6, 0.2, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Fracoes de locutores (apenas para --speaker-strategy stratified).",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=3.0,
        help=(
            "Descarta o par quando algum lado e mais curto que isto. Tem de ser "
            ">= a janela de exportacao, senao volta a haver repeticao (`tile`) "
            "assimetrica entre as classes."
        ),
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Nao cria splits/<split>/ nem sincroniza o speaker_manifest.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    sizes = tuple(args.sentences)
    fractions = tuple(args.speaker_fractions)
    if args.plan or not args.build:
        plan(sizes, args.seed, args.speaker_strategy, fractions)
        return 0
    build(
        sizes,
        args.seed,
        do_materialize=not args.no_materialize,
        strategy=args.speaker_strategy,
        fractions=fractions,
        min_duration=args.min_duration_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
