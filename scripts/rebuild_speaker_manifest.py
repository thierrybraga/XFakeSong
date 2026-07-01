#!/usr/bin/env python3
"""Reconstrói speaker_manifest.json a partir dos metadados locais disponíveis.

Este script não inventa falantes. Ele registra IDs reais apenas quando há uma
fonte local rastreável. No dataset atual, Fake Voices preserva o falante pelo
nome do ZIP e os WAVs foram gravados em blocos sequenciais por falante.
BRSpeech-DF e Common Voice PT ficam sem ID real quando o cache local não traz
colunas de falante.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "app" / "datasets"
FKVOICE_RE = re.compile(r"^fkvoice_(\d+)\.wav$", re.IGNORECASE)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _speaker_order(dataset_dir: Path) -> list[str]:
    done_path = dataset_dir / "raw" / "fkvoice_done.json"
    done = set(_load_json(done_path, []))
    zips = sorted(
        dataset_dir.glob(
            "raw/fake_voices_cache/datasets--unfake--fake_voices/"
            "snapshots/*/falabrasil-fake-voices/*.zip"
        )
    )
    order = [z.stem for z in zips if not done or z.stem in done]
    if not order and done:
        order = sorted(done)
    return order


def _infer_fkvoice_block_size(
    dataset_dir: Path, speakers: list[str], explicit: int | None
) -> int:
    if explicit:
        return explicit
    files = sorted((dataset_dir / "fake").glob("fkvoice_*.wav"))
    if speakers and files and len(files) % len(speakers) == 0:
        return len(files) // len(speakers)
    # O downloader usa limit=100 com janela paralela; no dataset atual o
    # overshoot por falante ficou em 107. Mantemos 100 como fallback conservador.
    return 100


def _add_fkvoice_manifest(
    dataset_dir: Path,
    manifest: dict[str, dict[str, Any]],
    block_size: int | None,
) -> dict[str, Any]:
    speakers = _speaker_order(dataset_dir)
    inferred_block = _infer_fkvoice_block_size(dataset_dir, speakers, block_size)
    candidates = {
        path.name: path
        for base in ("fake", "splits")
        for path in (dataset_dir / base).rglob("fkvoice_*.wav")
    }
    added = 0
    out_of_range = 0
    by_speaker: Counter[str] = Counter()
    for name in sorted(candidates):
        match = FKVOICE_RE.match(name)
        if not match:
            continue
        idx = int(match.group(1))
        speaker_idx = idx // inferred_block
        if speaker_idx >= len(speakers):
            out_of_range += 1
            continue
        speaker_id = speakers[speaker_idx]
        manifest[name] = {
            "speaker_id": speaker_id,
            "source": "fkvoice",
            "derivation": "fake_voices_zip_order",
            "block_size": inferred_block,
        }
        added += 1
        by_speaker[speaker_id] += 1
    return {
        "speaker_order_count": len(speakers),
        "block_size": inferred_block,
        "entries_added": added,
        "out_of_range": out_of_range,
        "by_speaker": dict(sorted(by_speaker.items())),
    }


def _inspect_brspeech_columns(dataset_dir: Path) -> dict[str, Any]:
    parquet_files = sorted((dataset_dir / "raw" / "brspeech_cache").rglob("*.parquet"))
    report: dict[str, Any] = {
        "parquet_files": len(parquet_files),
        "speaker_columns_found": [],
        "available_columns": [],
        "note": "pyarrow ausente ou nenhum parquet encontrado",
    }
    if not parquet_files:
        return report
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # noqa: BLE001
        report["note"] = f"pyarrow indisponível: {exc}"
        return report
    schema = pq.read_schema(parquet_files[0])
    columns = list(schema.names)
    speaker_cols = [
        col
        for col in columns
        if any(token in col.lower() for token in ("speaker", "client", "reader"))
    ]
    report.update(
        {
            "available_columns": columns,
            "speaker_columns_found": speaker_cols,
            "note": "sem coluna explícita de falante" if not speaker_cols else "ok",
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera app/datasets/speaker_manifest.json com IDs reais disponíveis."
    )
    parser.add_argument("--dataset-dir", default="app/datasets")
    parser.add_argument("--out", default="app/datasets/speaker_manifest.json")
    parser.add_argument("--report", default="app/datasets/speaker_manifest_report.json")
    parser.add_argument(
        "--fkvoice-block-size",
        type=int,
        default=None,
        help="Tamanho do bloco sequencial por falante. Se omitido, infere pelos WAVs.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    manifest = _load_json(out_path, {})
    if not isinstance(manifest, dict):
        manifest = {}

    fkvoice_report = _add_fkvoice_manifest(dataset_dir, manifest, args.fkvoice_block_size)
    brspeech_report = _inspect_brspeech_columns(dataset_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    total_wavs = list((dataset_dir / "splits").rglob("*.wav"))
    identified = sum(1 for p in total_wavs if p.name in manifest)
    report = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(out_path),
        "manifest_entries": len(manifest),
        "split_wavs": len(total_wavs),
        "split_identified": identified,
        "split_identified_ratio": round(identified / len(total_wavs), 6)
        if total_wavs
        else 0.0,
        "fkvoice": fkvoice_report,
        "brspeech": brspeech_report,
        "common_voice": {
            "note": "cache local do Common Voice não preserva client_id alinhável aos WAVs atuais"
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
