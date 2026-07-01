#!/usr/bin/env python3
"""Exporta tabela consolidada de falantes do dataset.

Gera CSV/JSONL com uma linha por WAV encontrado em `app/datasets`, incluindo
classe, split, fonte, speaker_id, speaker_key, status do ID, tamanho e duração.
IDs reais vêm de `speaker_manifest.json`; quando não há entrada no manifesto, o
status fica `fallback_source` e o speaker_key cai para o prefixo da fonte.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "app" / "datasets"


def _load_manifest(dataset_dir: Path) -> dict[str, dict]:
    path = dataset_dir / "speaker_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _prefix(path: Path) -> str:
    return path.stem.split("_", 1)[0].lower()


def _class_and_split(path: Path, dataset_dir: Path) -> tuple[str, str]:
    rel = path.relative_to(dataset_dir).parts
    if rel[0] == "real":
        return "real", "active"
    if rel[0] == "fake":
        return "fake", "active"
    if rel[0] == "splits" and len(rel) >= 3:
        return rel[2], rel[1]
    if rel[0] == "overflow" and len(rel) >= 2:
        return rel[1], "overflow"
    return "unknown", "unknown"


def _duration(path: Path) -> tuple[float | None, int | None, int | None]:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sr = wav.getframerate()
            channels = wav.getnchannels()
            return (round(frames / sr, 6) if sr else None, sr, channels)
    except Exception:
        return None, None, None


def _iter_wavs(dataset_dir: Path, scope: str) -> Iterable[Path]:
    roots = {
        "active": [dataset_dir / "real", dataset_dir / "fake"],
        "splits": [dataset_dir / "splits"],
        "all": [
            dataset_dir / "real",
            dataset_dir / "fake",
            dataset_dir / "splits",
            dataset_dir / "overflow",
        ],
    }[scope]
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.wav")):
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera tabela CSV/JSONL de IDs de falante por arquivo."
    )
    parser.add_argument("--dataset-dir", default="app/datasets")
    parser.add_argument("--scope", choices=["active", "splits", "all"], default="all")
    parser.add_argument(
        "--csv-out",
        default="app/datasets/speaker_table.csv",
        help="CSV de saída.",
    )
    parser.add_argument(
        "--jsonl-out",
        default="app/datasets/speaker_table.jsonl",
        help="JSONL de saída.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    csv_out = Path(args.csv_out)
    if not csv_out.is_absolute():
        csv_out = ROOT / csv_out
    jsonl_out = Path(args.jsonl_out)
    if not jsonl_out.is_absolute():
        jsonl_out = ROOT / jsonl_out

    manifest = _load_manifest(dataset_dir)
    rows = []
    for path in _iter_wavs(dataset_dir, args.scope):
        source = _prefix(path)
        cls, split = _class_and_split(path, dataset_dir)
        entry = manifest.get(path.name) or {}
        speaker_id = str(entry.get("speaker_id", "")).strip()
        if speaker_id:
            manifest_source = str(entry.get("source") or source).strip().lower()
            speaker_key = f"{manifest_source}:{speaker_id}"
            id_status = "real_id"
        else:
            manifest_source = source
            speaker_key = source
            id_status = "fallback_source"
        duration_sec, sample_rate, channels = _duration(path)
        rows.append(
            {
                "file": path.name,
                "relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "split": split,
                "class": cls,
                "source": source,
                "speaker_id": speaker_id,
                "speaker_key": speaker_key,
                "id_status": id_status,
                "manifest_source": manifest_source,
                "duration_sec": duration_sec,
                "sample_rate": sample_rate,
                "channels": channels,
                "size_bytes": path.stat().st_size,
            }
        )

    fieldnames = [
        "file",
        "relative_path",
        "split",
        "class",
        "source",
        "speaker_id",
        "speaker_key",
        "id_status",
        "manifest_source",
        "duration_sec",
        "sample_rate",
        "channels",
        "size_bytes",
    ]
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "csv_out": str(csv_out),
        "jsonl_out": str(jsonl_out),
        "real_id": sum(1 for row in rows if row["id_status"] == "real_id"),
        "fallback_source": sum(
            1 for row in rows if row["id_status"] == "fallback_source"
        ),
        "distinct_speaker_keys": len({row["speaker_key"] for row in rows}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
