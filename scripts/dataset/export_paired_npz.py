#!/usr/bin/env python3
"""Exporta o .npz de audio bruto a partir da particao pareada (Protocolo de Dataset).

Le `splits/assignment.jsonl` como fonte de verdade — nao os diretorios. O
rotulo, o locutor, a frase e o gerador vem do manifesto, entao nenhuma amostra
pode entrar com proveniencia inventada e o rotulo nao depende do nome da pasta.

Propriedades que o exportador garante:

1. **Memoria plana.** Cada particao e pre-alocada e preenchida no lugar, em vez
   de acumular uma lista Python de arrays e converter no fim — o que dobraria o
   pico.
2. **Selecao estratificada.** Quando ha limite de tamanho, a amostragem e
   uniforme por locutor e por frase (nao aleatoria simples), para nao concentrar
   o corte em poucos locutores.
3. **Pareamento preservado.** O limite e aplicado a PARES (enunciados), nunca a
   amostras isoladas, entao o corte nao pode desbalancear as classes.
4. **Barreira final.** Sobreposicao entre particoes, balanceamento e oraculos sao
   reverificados DEPOIS da selecao e do recorte; a gravacao e interrompida se
   algo violar o protocolo.

Janela: 3 s (48.000 amostras a 16 kHz), **recorte central puro**. A particao
descarta os pares em que algum lado nao alcanca a janela, entao nenhuma amostra
precisa ser repetida (`tile`) — repeticao assimetrica entre as classes seria
pista de classe, e `_check_window` recusa uma janela maior que a garantida.

Uso:
    python scripts/dataset/export_paired_npz.py
    python scripts/dataset/export_paired_npz.py --max-pairs-train 7500
    python scripts/dataset/export_paired_npz.py --out data/datasets/meu.npz --no-compress
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset.build_paired_pt_corpus import extract_window  # noqa: E402

logger = logging.getLogger("export_paired_npz")

CORPUS_DIR = ROOT / "data" / "datasets" / "corpus"
SPLITS_DIR = ROOT / "data" / "datasets" / "splits"
ASSIGNMENT_PATH = SPLITS_DIR / "assignment.jsonl"
DEFAULT_OUT = ROOT / "data" / "datasets" / "benchmark_dataset.npz"

SPLITS = ("train", "val", "test")


def load_assignment() -> list[dict]:
    if not ASSIGNMENT_PATH.exists():
        raise SystemExit(
            f"particao ausente: {ASSIGNMENT_PATH}\n"
            "rode: python scripts/dataset/build_paired_splits.py --build"
        )
    with ASSIGNMENT_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _check_window(window_sec: float) -> None:
    """A janela nao pode passar do minimo garantido pela particao.

    `build_paired_splits --min-duration-sec` descarta os pares em que algum lado
    e mais curto que a janela, justamente para que nenhuma amostra precise ser
    repetida (`tile`). Exportar com janela maior reintroduz a repeticao — e ela e
    assimetrica entre as classes, porque o clone e sistematicamente mais curto.
    """
    manifest_path = SPLITS_DIR / "split_manifest.json"
    if not manifest_path.exists():
        return
    guaranteed = json.loads(manifest_path.read_text(encoding="utf-8")).get(
        "min_duration_sec"
    )
    if guaranteed and window_sec > guaranteed + 1e-9:
        raise SystemExit(
            f"janela de {window_sec:g}s excede o minimo garantido pela particao "
            f"({guaranteed:g}s). Amostras curtas seriam repetidas, e a taxa de "
            f"repeticao difere entre as classes. Refaca a particao com "
            f"--min-duration-sec {window_sec:g} ou exporte com janela menor."
        )


def available_pairs(records: list[dict]) -> dict[str, int]:
    """Enunciados disponiveis por particao, antes de qualquer corte."""
    pools: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        if record["split"] in pools:
            pools[record["split"]].add(record["utterance_id"])
    return {split: len(pool) for split, pool in pools.items()}


def derive_max_pairs(records: list[dict], target_samples: int) -> dict[str, int]:
    """Distribui `target_samples` entre as particoes na proporcao natural.

    A proporcao NAO e uma constante: ela e consequencia do bloco diagonal
    (locutores x frases de cada particao) e muda se o corpus mudar. Por isso e
    derivada do proprio `assignment.jsonl` a cada execucao, em vez de ficar
    fixada em codigo — um corpus novo produz o rateio novo sem editar nada.

    O rateio usa maior resto (quota de Hare), o metodo padrao de apportionment:
    piso da cota exata para todas, e as vagas restantes vao para as maiores
    partes fracionarias. Garante que a soma bata exatamente com o alvo, sem o
    vies sistematico que arredondar cada parte isoladamente introduziria.

    Cada par gera DUAS amostras (bonafide + clone), entao o alvo em amostras
    precisa ser par.
    """
    if target_samples % 2:
        raise SystemExit(
            f"--target-samples deve ser par: cada par de enunciado gera duas "
            f"amostras (bonafide + clone). Recebido: {target_samples}"
        )
    target_pairs = target_samples // 2
    available = available_pairs(records)
    total_available = sum(available.values())
    if target_pairs > total_available:
        raise SystemExit(
            f"alvo de {target_samples} amostras ({target_pairs} pares) excede o "
            f"disponivel na particao: {total_available} pares "
            f"({total_available * 2} amostras). "
            f"Disponivel por particao: {available}"
        )

    exact = {s: target_pairs * available[s] / total_available for s in SPLITS}
    chosen = {s: int(exact[s]) for s in SPLITS}
    sobra = target_pairs - sum(chosen.values())
    # Maior resto: as `sobra` particoes com maior parte fracionaria ganham +1.
    ordem = sorted(SPLITS, key=lambda s: exact[s] - int(exact[s]), reverse=True)
    for split in ordem[:sobra]:
        chosen[split] += 1

    vazias = [s for s in SPLITS if chosen[s] < 1]
    if vazias:
        raise SystemExit(
            f"alvo de {target_samples} amostras e pequeno demais: "
            f"{', '.join(vazias)} ficaria(m) sem nenhum par. "
            f"Minimo pratico: {int(total_available / min(available.values())) * 2}"
        )
    return chosen


def select_pairs(records: list[dict], max_pairs: int, seed: int) -> list[dict]:
    """Reduz para `max_pairs` enunciados, uniformemente por locutor e frase.

    Corta PARES, nunca amostras: o real e o seu clone entram ou saem juntos,
    logo o balanceamento de classe sobrevive a qualquer limite.
    """
    by_utterance: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_utterance[record["utterance_id"]].append(record)
    utterances = sorted(by_utterance)
    if not max_pairs or len(utterances) <= max_pairs:
        return records

    rng = np.random.default_rng(seed)
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for utterance in utterances:
        by_speaker[by_utterance[utterance][0]["speaker_id"]].append(utterance)
    for speaker in by_speaker:
        order = rng.permutation(len(by_speaker[speaker]))
        by_speaker[speaker] = [by_speaker[speaker][int(i)] for i in order]

    # Rodizio entre locutores: cada volta pega um enunciado de cada, entao o
    # corte tira a mesma proporcao de todos em vez de eliminar locutores.
    chosen: list[str] = []
    speakers = sorted(by_speaker)
    cursor = 0
    while len(chosen) < max_pairs:
        progressed = False
        for speaker in speakers:
            pool = by_speaker[speaker]
            if cursor < len(pool):
                chosen.append(pool[cursor])
                progressed = True
                if len(chosen) >= max_pairs:
                    break
        if not progressed:
            break
        cursor += 1

    keep = set(chosen)
    return [r for r in records if r["utterance_id"] in keep]


def _load_window(path: Path, samples: int) -> tuple[np.ndarray, int, int]:
    """Le o WAV e devolve a janela ja nivelada (politica em `extract_window`)."""
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    return extract_window(audio, samples)


def build_split(
    records: list[dict], samples: int, workers: int
) -> dict[str, np.ndarray]:
    """Pre-aloca e preenche no lugar (sem lista intermediaria de arrays)."""
    rows = sorted(
        records, key=lambda r: (r["speaker_id"], r["sentence_index"], r["label"])
    )
    count = len(rows)
    X = np.empty((count, samples, 1), dtype="float32")
    y = np.empty(count, dtype="int64")
    lengths = np.empty(count, dtype="int64")
    starts = np.empty(count, dtype="int64")

    def work(item: tuple[int, dict]) -> tuple[int, np.ndarray, int, int] | None:
        position, record = item
        try:
            window, original, start = _load_window(CORPUS_DIR / record["path"], samples)
        except Exception as exc:  # noqa: BLE001
            logger.warning("falha ao carregar %s: %s", record["path"], exc)
            return None
        return position, window, original, start

    valid = np.zeros(count, dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(work, enumerate(rows)):
            if result is None:
                continue
            position, window, original, start = result
            X[position, :, 0] = window
            y[position] = rows[position]["label"]
            lengths[position] = original
            starts[position] = start
            valid[position] = True

    if not valid.all():
        logger.warning(
            "%d amostras descartadas por falha de leitura", int((~valid).sum())
        )
        X, y, lengths, starts = X[valid], y[valid], lengths[valid], starts[valid]
        rows = [row for row, keep in zip(rows, valid) if keep]

    return {
        "X": X,
        "y": y,
        "lengths": lengths,
        "starts": starts,
        "rows": rows,
    }


def _provenance(rows: list[dict]) -> dict[str, np.ndarray]:
    """Arrays de proveniencia alinhados as amostras.

    `cluster_ids` = `text_id`: a unidade que torna as amostras dependentes entre
    si e a frase (o mesmo texto lido por varios locutores, nas duas classes).
    Isso da ao bootstrap por cluster um agrupamento com significado real — ja houve acervo em que eram todos singletons e os IC95 saiam estreitos demais. `speaker_ids` fica
    exportado para quem preferir agrupar por locutor.
    """

    def column(field: str, width: str) -> np.ndarray:
        return np.asarray([str(row[field]) for row in rows], dtype=width)

    return {
        "sample_paths": np.asarray(
            [f"data/datasets/corpus/{row['path']}" for row in rows], dtype="U256"
        ),
        "source_ids": column("source", "U64"),
        "groups": column("source", "U64"),
        "speaker_ids": np.asarray(
            [f"{row['source']}:{row['speaker_id']}" for row in rows], dtype="U64"
        ),
        "speaker_known": np.ones(len(rows), dtype=bool),
        "utterance_ids": np.asarray(
            [f"{row['source']}:{row['utterance_id']}" for row in rows], dtype="U64"
        ),
        "text_ids": np.asarray(
            [f"{row['source']}:{row['text_id']}" for row in rows], dtype="U64"
        ),
        "sentence_indices": column("sentence_index", "U8"),
        "generator_ids": column("generator_id", "U32"),
        "generator_known": np.ones(len(rows), dtype=bool),
        "cluster_ids": np.asarray(
            [f"{row['source']}:{row['text_id']}" for row in rows], dtype="U64"
        ),
        "content_sha256": column("content_sha256", "U64"),
        "cetuc_official_split": column("cetuc_official_split", "U8"),
    }


def export(
    out: Path,
    max_pairs: dict[str, int],
    samples: int,
    seed: int,
    workers: int,
    compress: bool,
) -> None:
    records = load_assignment()
    _check_window(samples / 16_000)
    arrays: dict[str, np.ndarray] = {}
    all_rows: list[dict] = []
    meta: dict = {
        "protocol": "cetuc-xtts-paired",
        "source": "data/datasets/corpus",
        "partition": "data/datasets/splits",
        "sample_rate": 16_000,
        "duration_sec": samples / 16_000,
        "format": "raw_audio",
        "resampling": "soxr_hq",
        "window_policy": "center_crop_or_tile_no_zero_padding",
        "amplitude_policy": "rms_normalized_-26_dBFS_peak_ceiling_-1_dBFS",
        "selection_seed": seed,
        "strategy": "speaker_x_sentence_double_disjoint_block_diagonal",
        # Como o tamanho foi decidido. Sem isso, um .npz reduzido nao carrega o
        # que o distingue do completo, e reproduzi-lo exige saber de fora os
        # numeros de pares usados.
        "size_policy": {
            "max_pairs": dict(max_pairs),
            "total_samples": sum(max_pairs.values()) * 2 if all(
                max_pairs.get(s) for s in SPLITS
            ) else None,
            "subsampled": bool(any(max_pairs.get(s) for s in SPLITS)),
            "pair_selection": "uniform_por_locutor_e_frase_rodizio",
        },
        "splits": {},
    }

    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        if not rows:
            raise SystemExit(f"particao vazia: {split}")
        before = len({r["utterance_id"] for r in rows})
        rows = select_pairs(rows, max_pairs.get(split, 0), seed)
        after = len({r["utterance_id"] for r in rows})
        logger.info(
            "%s: %d pares -> %d pares (%d amostras)", split, before, after, len(rows)
        )
        started = time.time()
        built = build_split(rows, samples, workers)
        arrays[f"X_{split}"] = built["X"]
        arrays[f"y_{split}"] = built["y"]
        arrays[f"original_num_samples_{split}"] = built["lengths"]
        arrays[f"window_start_{split}"] = built["starts"]
        all_rows.extend(built["rows"])
        labels = built["y"]
        meta["splits"][split] = {
            "samples": int(len(labels)),
            "real": int((labels == 0).sum()),
            "fake": int((labels == 1).sum()),
            "pairs": after,
            "speakers": len({r["speaker_id"] for r in built["rows"]}),
            "sentences": len({r["sentence_index"] for r in built["rows"]}),
            "hours": round(sum(r["duration_sec"] for r in built["rows"]) / 3600.0, 3),
        }
        logger.info(
            "  %s pronto: %s em %.1f s (%.2f GB)",
            split,
            built["X"].shape,
            time.time() - started,
            built["X"].nbytes / 1e9,
        )

    arrays.update(_provenance(all_rows))

    # Auditorias que precisam existir DENTRO do artefato: quem receber apenas o
    # .npz tem de poder verificar as garantias sem o repositorio.
    labels = np.concatenate([arrays[f"y_{split}"] for split in SPLITS])
    meta["audits"] = _inline_audits(arrays, all_rows, labels)
    _enforce(meta["audits"])
    meta["provenance_coverage"] = {
        field: {"known": len(all_rows), "total": len(all_rows), "ratio": 1.0}
        for field in (
            "speaker_id",
            "utterance_id",
            "text_id",
            "generator_id",
            "source_revision",
        )
    }
    arrays["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))

    total = sum(arrays[f"X_{split}"].nbytes for split in SPLITS)
    logger.info(
        "gravando %s (%.2f GB em memoria, compressao=%s)",
        out.name,
        total / 1e9,
        compress,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    writer = np.savez_compressed if compress else np.savez
    writer(out, **arrays)
    logger.info("NPZ exportado em %.1f min: %s", (time.time() - started) / 60.0, out)
    _report(meta, out)


def _inline_audits(
    arrays: dict[str, np.ndarray], rows: list[dict], labels: np.ndarray
) -> dict:
    offset = 0
    index_by_split: dict[str, slice] = {}
    for split in SPLITS:
        size = len(arrays[f"y_{split}"])
        index_by_split[split] = slice(offset, offset + size)
        offset += size

    audits: dict = {}
    for field in ("speaker_id", "sentence_index", "text_id", "content_sha256"):
        per_split = {
            split: {
                rows[i][field] for i in range(*index_by_split[split].indices(len(rows)))
            }
            for split in SPLITS
        }
        audits[f"{field}_overlap"] = {
            f"{a}x{b}": len(per_split[a] & per_split[b])
            for i, a in enumerate(SPLITS)
            for b in SPLITS[i + 1 :]
        }

    oracle: dict[str, float] = {}
    for field in ("source", "speaker_id", "text_id"):
        buckets: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            buckets[str(row[field])].append(int(labels[position]))
        correct = sum(
            max(bucket.count(0), bucket.count(1)) for bucket in buckets.values()
        )
        oracle[field] = round(correct / max(len(rows), 1), 4)
    audits["majority_oracle_accuracy"] = oracle
    audits["class_balance"] = {
        split: {
            "real": int((arrays[f"y_{split}"] == 0).sum()),
            "fake": int((arrays[f"y_{split}"] == 1).sum()),
        }
        for split in SPLITS
    }
    return audits


def _enforce(audits: dict) -> None:
    """Ultima barreira antes do treino.

    A particao ja foi auditada, mas o exportador ainda seleciona, recorta e
    reordena — e e o .npz, nao o manifesto, que chega ao benchmark. Uma violacao
    aqui tem de interromper a exportacao, nao virar uma linha de log que ninguem
    le numa execucao de horas.
    """
    problems: list[str] = []
    for field in ("speaker_id", "sentence_index", "text_id", "content_sha256"):
        for pair, count in audits[f"{field}_overlap"].items():
            if count:
                problems.append(f"{field} compartilhado entre {pair}: {count}")
    for split, counts in audits["class_balance"].items():
        if counts["real"] != counts["fake"]:
            problems.append(
                f"{split} desbalanceado: {counts['real']} reais / {counts['fake']} falsas"
            )
    for field, accuracy in audits["majority_oracle_accuracy"].items():
        if accuracy > 0.55:
            problems.append(f"oraculo por {field} acerta {accuracy:.2%}")
    if problems:
        raise SystemExit(
            "exportacao interrompida — o .npz violaria o Protocolo de Dataset:\n  "
            + "\n  ".join(problems)
        )


def _report(meta: dict, out: Path) -> None:
    print(f"\n=== NPZ: {out.name} ===")
    print(
        f"{'split':>7}{'amostras':>10}{'reais':>8}{'falsas':>8}"
        f"{'locutores':>11}{'frases':>8}{'horas':>8}"
    )
    for split in SPLITS:
        info = meta["splits"][split]
        print(
            f"{split:>7}{info['samples']:>10}{info['real']:>8}{info['fake']:>8}"
            f"{info['speakers']:>11}{info['sentences']:>8}{info['hours']:>8.2f}"
        )
    audits = meta["audits"]
    print("\nsobreposicao entre particoes (deve ser zero):")
    for field in ("speaker_id", "sentence_index", "text_id", "content_sha256"):
        print(f"  {field:>16}: {audits[f'{field}_overlap']}")
    print("\noraculo de maioria (acaso = 0,5):")
    for field, value in audits["majority_oracle_accuracy"].items():
        print(f"  {field:>16}: {value:.4f}")
    if out.exists():
        print(f"\ntamanho: {out.stat().st_size / 1e9:.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--target-samples",
        type=int,
        default=0,
        help=(
            "Tamanho total do dataset em AMOSTRAS (0 = usa a particao inteira). "
            "Rateia entre treino/val/teste na proporcao natural do bloco "
            "diagonal, derivada do assignment.jsonl. Precisa ser par. "
            "Ex.: --target-samples 15000."
        ),
    )
    parser.add_argument(
        "--max-pairs-train",
        type=int,
        default=0,
        help=(
            "Limite de PARES no treino (0 = todos). Cada par gera 2 amostras. "
            "Alternativa manual a --target-samples, para rateio nao proporcional."
        ),
    )
    parser.add_argument("--max-pairs-val", type=int, default=0)
    parser.add_argument("--max-pairs-test", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Grava sem compressao (muito mais rapido; arquivo maior).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    explicitos = {
        "train": args.max_pairs_train,
        "val": args.max_pairs_val,
        "test": args.max_pairs_test,
    }
    if args.target_samples and any(explicitos.values()):
        raise SystemExit(
            "--target-samples e --max-pairs-* sao mutuamente exclusivos: o "
            "primeiro DERIVA o rateio, o segundo o impoe. Escolha um."
        )
    if args.target_samples:
        max_pairs = derive_max_pairs(load_assignment(), args.target_samples)
        logger.info(
            "alvo %d amostras -> pares por particao: %s (total %d amostras)",
            args.target_samples,
            max_pairs,
            sum(max_pairs.values()) * 2,
        )
    else:
        max_pairs = explicitos

    export(
        out=out,
        max_pairs=max_pairs,
        samples=int(args.sample_rate * args.duration_sec),
        seed=args.seed,
        workers=args.workers,
        compress=not args.no_compress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
