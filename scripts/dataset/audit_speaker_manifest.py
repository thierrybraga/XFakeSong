#!/usr/bin/env python3
"""Audita cobertura do speaker_manifest.json no dataset ativo.

O objetivo é separar IDs reais de falante de fallback por fonte (`brspeech`,
`cvpt`, `fkvoice`). O script não altera os dados; apenas resume e falha quando
a cobertura mínima configurada não é atingida.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASETS_DIR = ROOT / "data" / "datasets"


def _prefix(path: Path) -> str:
    return path.stem.split("_", 1)[0].lower()


def _load_manifest(path: Path | None = None) -> dict[str, dict]:
    """Carrega o manifesto CANONICO, via o modulo que o projeto usa.

    Este script lia `data/datasets/speaker_manifest.json` e indexava por
    `wav.name`. Nenhuma das duas coisas confere: o manifesto canonico e
    `app.domain.dataset_metadata.speaker_manifest.SPEAKER_MANIFEST_PATH`
    (`data/datasets/metadata/speaker_manifest.json`), e a chave e a de
    `_manifest_key` (`<classe>/<basename>`, com queda para o basename). O
    resultado era cobertura 0% SEMPRE — a auditoria reprovava qualquer que
    fosse o estado real do manifesto, entao ninguem a rodava.

    `path` fica so para teste; em producao resolve pelo modulo.
    """
    if path is not None:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    from app.domain.dataset_metadata import speaker_manifest

    return speaker_manifest.load_manifest()


def _collect_wavs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(base.rglob("*.wav"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita cobertura de IDs reais de falantes por arquivo."
    )
    parser.add_argument("--dataset-dir", default="data/datasets")
    parser.add_argument(
        "--scope",
        choices=["active", "splits"],
        default="active",
        help="active usa real/ + fake/; splits usa data/datasets/splits/.",
    )
    parser.add_argument(
        "--min-identified-ratio",
        type=float,
        default=0.50,
        help="Razão mínima de arquivos com speaker_id real no manifesto.",
    )
    parser.add_argument(
        "--json-out",
        default="data/datasets/speaker_audit.json",
        help="Relatório JSON gerado.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    from app.domain.dataset_metadata import speaker_manifest

    manifest_path = speaker_manifest.SPEAKER_MANIFEST_PATH
    manifest = _load_manifest()

    if args.scope == "splits":
        wavs = _collect_wavs(dataset_dir / "splits")
    else:
        wavs = _collect_wavs(dataset_dir / "real") + _collect_wavs(dataset_dir / "fake")

    by_prefix: dict[str, Counter[str]] = defaultdict(Counter)
    identified = 0
    missing = 0
    speakers_by_prefix: dict[str, set[str]] = defaultdict(set)

    for wav in wavs:
        prefix = _prefix(wav)
        # Busca pela MESMA chave que o dominio usa (`<classe>/<basename>`, com
        # queda para o basename), nao por `wav.name` cru.
        entry = speaker_manifest.sample_metadata_for_path(wav)
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
