#!/usr/bin/env python3
"""Audita cobertura do speaker_manifest.json no dataset ativo.

O objetivo é separar IDs reais de falante de fallback por fonte (`brspeech`,
`cvpt`, `fkvoice`). O script não altera os dados; apenas resume e falha quando
a cobertura mínima configurada não é atingida.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "app" / "datasets"


def _prefix(path: Path) -> str:
    return path.stem.split("_", 1)[0].lower()


def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _collect_wavs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(base.rglob("*.wav"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita cobertura de IDs reais de falantes por arquivo."
    )
    parser.add_argument("--dataset-dir", default="app/datasets")
    parser.add_argument(
        "--scope",
        choices=["active", "splits"],
        default="active",
        help="active usa real/ + fake/; splits usa app/datasets/splits/.",
    )
    parser.add_argument(
        "--min-identified-ratio",
        type=float,
        default=0.50,
        help="Razão mínima de arquivos com speaker_id real no manifesto.",
    )
    parser.add_argument(
        "--json-out",
        default="app/datasets/speaker_audit.json",
        help="Relatório JSON gerado.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    manifest_path = dataset_dir / "speaker_manifest.json"
    manifest = _load_manifest(manifest_path)

    if args.scope == "splits":
        wavs = _collect_wavs(dataset_dir / "splits")
    else:
        wavs = _collect_wavs(dataset_dir / "real") + _collect_wavs(dataset_dir / "fake")

    by_prefix: dict[str, Counter[str]] = defaultdict(Counter)
    identified = 0
    missing = 0
    speakers_by_prefix: dict[str, set[str]] = defaultdict(set)

    for wav in wavs:
        name = wav.name
        prefix = _prefix(wav)
        entry = manifest.get(name)
        if entry and entry.get("speaker_id"):
            identified += 1
            sid = str(entry["speaker_id"])
            by_prefix[prefix]["identified"] += 1
            speakers_by_prefix[prefix].add(sid)
        else:
            missing += 1
            by_prefix[prefix]["missing"] += 1
        by_prefix[prefix]["files"] += 1

    total = len(wavs)
    identified_ratio = identified / total if total else 0.0
    report = {
        "scope": args.scope,
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "manifest_entries": len(manifest),
        "total_wavs": total,
        "identified": identified,
        "missing": missing,
        "identified_ratio": round(identified_ratio, 6),
        "min_identified_ratio": args.min_identified_ratio,
        "note": (
            "identified conta apenas arquivos presentes em speaker_manifest.json; "
            "arquivos ausentes continuam usando fallback por fonte no NPZ."
        ),
        "by_prefix": {
            prefix: {
                "files": counts["files"],
                "identified": counts["identified"],
                "missing": counts["missing"],
                "identified_ratio": round(
                    counts["identified"] / counts["files"], 6
                )
                if counts["files"]
                else 0.0,
                "distinct_speakers": len(speakers_by_prefix[prefix]),
            }
            for prefix, counts in sorted(by_prefix.items())
        },
    }

    out_path = Path(args.json_out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if identified_ratio < args.min_identified_ratio:
        raise SystemExit(
            f"speaker_manifest insuficiente: {identified_ratio:.1%} "
            f"< {args.min_identified_ratio:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
