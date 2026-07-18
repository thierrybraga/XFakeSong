#!/usr/bin/env python3
"""Executa o benchmark canônico em múltiplas sementes, sem misturar artefatos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "benchmark" / "run_models_sequential.py"


def _test_fingerprints(seed_dir: Path) -> set[str]:
    values: set[str] = set()
    for result_path in seed_dir.glob("*/results.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        fingerprint = (payload.get("dataset") or {}).get("test_split_sha256")
        if not fingerprint:
            raise RuntimeError(f"resultado sem test_split_sha256: {result_path}")
        values.add(str(fingerprint))
    return values

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="results/retrain_waveform_awgn_multiseed")
    parser.add_argument("--test-lock", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device-profile", default="gpu")
    parser.add_argument("--timeout-min", type=int, default=120)
    args, extra = parser.parse_known_args()
    forbidden = {
        "--seed", "--out", "--epochs", "--dataset", "--test-lock",
        "--no-academic-protocol", "--academic-protocol",
    }
    if forbidden.intersection(extra):
        raise SystemExit(f"não repita argumentos controlados: {sorted(forbidden)}")

    if args.epochs != 100:
        parser.error("benchmark acadêmico multissemente exige exatamente 100 épocas")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contém valores duplicados")
    out_root = (ROOT / args.out).resolve()
    reference_test_sha256: str | None = None
    seed_fingerprints: dict[str, str] = {}
    for seed in args.seeds:
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            str((ROOT / args.dataset).resolve()),
            "--test-lock",
            str(Path(args.test_lock).resolve()),
            "--epochs",
            str(args.epochs),
            "--device-profile",
            args.device_profile,
            "--seed",
            str(seed),
            "--snr",
            "30",
            "20",
            "10",
            "--train-aug-snr",
            "30",
            "20",
            "10",
            "--train-noise-copies",
            "1",
            "--waveform-noise-batch-size",
            "64",
            "--waveform-train-augmentation",
            "--ssl-train-batch-size",
            "128",
            "--timeout-min",
            str(args.timeout_min),
            "--out",
            str(out_root / f"seed_{seed}"),
            *extra,
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if "--plan-only" not in extra:
            fingerprints = _test_fingerprints(out_root / f"seed_{seed}")
            if len(fingerprints) != 1:
                raise RuntimeError(
                    f"modelos da seed {seed} não compartilham um único teste: {fingerprints}"
                )
            current = next(iter(fingerprints))
            if reference_test_sha256 is None:
                reference_test_sha256 = current
            elif current != reference_test_sha256:
                raise RuntimeError(
                    "teste mudou entre sementes; resultados multissemente inválidos"
                )
            seed_fingerprints[str(seed)] = current
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "multiseed_protocol.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "epochs": args.epochs,
                "test_split_sha256": reference_test_sha256,
                "seed_test_fingerprints": seed_fingerprints,
                "test_frozen_across_seeds": bool(reference_test_sha256),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())