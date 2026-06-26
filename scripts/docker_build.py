#!/usr/bin/env python3
"""Build/run helper for XFakeSong Docker profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROFILES = {
    "inference-cpu": ("docker/compose/inference.cpu.yml", "inference-api"),
    "inference-nvidia": ("docker/compose/inference.nvidia.yml", "inference-api"),
    "train-cpu": ("docker/compose/train.cpu.yml", None),
    "train-nvidia": ("docker/compose/train.nvidia.yml", None),
    "benchmark-nvidia": ("docker/compose/benchmark.nvidia.yml", "benchmark"),
}


def _compose_cmd(compose_file: str, action: str, service: str | None,
                 extra: list[str]) -> list[str]:
    cmd = ["docker", "compose", "-f", compose_file]
    if action == "config":
        return [*cmd, "config", "--quiet"]
    if action == "build":
        return [*cmd, "build", *( [service] if service else [] ), *extra]
    if action == "up":
        return [*cmd, "up", "--build", *( [service] if service else [] ), *extra]
    if action == "run":
        if not service:
            raise SystemExit("Use --service with action=run for multi-service profiles.")
        return [*cmd, "run", "--rm", service, *extra]
    raise SystemExit(f"Unsupported action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Docker build/run helper for segmented XFakeSong profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("action", choices=["config", "build", "up", "run"])
    parser.add_argument("--service", help="Override compose service.")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Arguments appended after the compose action.")
    args = parser.parse_args()

    compose_file, default_service = PROFILES[args.profile]
    service = args.service or default_service
    cmd = _compose_cmd(compose_file, args.action, service, args.extra)
    print("Executing:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    raise SystemExit(main())

