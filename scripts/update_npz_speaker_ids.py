#!/usr/bin/env python3
"""Atualiza o array speaker_ids de um NPZ usando speaker_manifest.json."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.dataset_catalog import infer_prefix_from_path  # noqa: E402
from app.core.speaker_manifest import speaker_for_path  # noqa: E402


def _read_metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in data.files:
        return {}
    raw = data["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _paths_from_metadata(meta: dict) -> list[str]:
    paths: list[str] = []
    for split in ("train", "val", "test"):
        paths.extend((meta.get("splits", {}).get(split, {}) or {}).get("paths", []))
    return paths


def _speaker_summary(paths: list[str], speaker_ids: list[str]) -> dict:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for path, speaker_id in zip(paths, speaker_ids):
        source = infer_prefix_from_path(path)
        by_source[source]["files"] += 1
        if ":" in speaker_id:
            by_source[source]["identified"] += 1
        else:
            by_source[source]["fallback"] += 1
    identified = sum(1 for sid in speaker_ids if ":" in sid)
    return {
        "total_files": len(speaker_ids),
        "identified_files": identified,
        "fallback_files": len(speaker_ids) - identified,
        "identified_ratio": round(identified / len(speaker_ids), 6)
        if speaker_ids
        else 0.0,
        "distinct_speaker_keys": len(set(speaker_ids)),
        "by_source": {
            source: dict(counts) for source, counts in sorted(by_source.items())
        },
        "note": (
            "IDs com prefixo fonte:falante vêm de speaker_manifest.json; valores "
            "sem ':' são fallback por fonte por falta de ID real local."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regrava speaker_ids em um .npz sem alterar X/y/splits."
    )
    parser.add_argument(
        "--npz",
        default="app/datasets/benchmark_audio_raw_balanced_15k.npz",
        help="NPZ de entrada e saída.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Arquivo de saída. Se omitido, atualiza o NPZ in-place via arquivo temporário.",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    out_path = Path(args.out) if args.out else npz_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    tmp_path = out_path.with_name(out_path.name + ".tmp.npz")

    with np.load(npz_path, allow_pickle=False) as data:
        meta = _read_metadata(data)
        paths = _paths_from_metadata(meta)
        if not paths:
            raise SystemExit("metadata_json não contém paths por split; não é seguro alinhar speaker_ids")
        speaker_ids = [speaker_for_path(path) for path in paths]
        meta["speaker_manifest"] = _speaker_summary(paths, speaker_ids)
        arrays = {
            name: data[name]
            for name in data.files
            if name not in {"speaker_ids", "metadata_json"}
        }
        arrays["speaker_ids"] = np.asarray(speaker_ids, dtype="U256")
        arrays["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(tmp_path, **arrays)

    if out_path == npz_path:
        os.replace(tmp_path, out_path)
    else:
        if out_path.exists():
            out_path.unlink()
        os.replace(tmp_path, out_path)
    print(json.dumps(meta["speaker_manifest"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
