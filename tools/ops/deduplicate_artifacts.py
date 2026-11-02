#!/usr/bin/env python3
"""Deduplica artefatos idênticos de ``data/models`` e ``data/results``.

Os caminhos são preservados porque manifests, notebooks e relatórios os usam.
Em NTFS, arquivos com o mesmo conteúdo passam a compartilhar um único arquivo
físico por hardlinks. A operação padrão é uma prévia; use ``--apply`` para
aplicar.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.file_storage import link_or_copy_file  # noqa: E402

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_groups() -> list[tuple[int, list[Path]]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for relative in (Path("data/models"), Path("data/results")):
        base = ROOT / relative
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not path.is_symlink():
                try:
                    by_size[path.stat().st_size].append(path)
                except OSError:
                    continue

    groups: list[tuple[int, list[Path]]] = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            try:
                by_hash[_sha256(path)].append(path)
            except OSError:
                continue
        groups.extend(
            (size, sorted(matches, key=lambda item: str(item).casefold()))
            for matches in by_hash.values()
            if len(matches) > 1
        )
    return sorted(
        groups,
        key=lambda item: (-item[0] * (len(item[1]) - 1), str(item[1][0])),
    )


def deduplicate(*, apply: bool) -> dict[str, int]:
    groups = _duplicate_groups()
    linked = 0
    copied = 0
    already_linked = 0
    bytes_reclaimed = 0
    bytes_potential = 0

    for size, paths in groups:
        canonical = paths[0]
        for duplicate in paths[1:]:
            try:
                if canonical.samefile(duplicate):
                    already_linked += 1
                    continue
            except OSError:
                pass
            if not apply:
                bytes_potential += size
                continue
            method = link_or_copy_file(canonical, duplicate)
            if method == "hardlink":
                linked += 1
                bytes_reclaimed += size
            else:
                copied += 1

    return {
        "groups": len(groups),
        "paths": sum(len(paths) for _, paths in groups),
        "hardlinks_created": linked,
        "copies_fallback": copied,
        "already_linked": already_linked,
        "bytes_reclaimed": bytes_reclaimed,
        "bytes_potential": bytes_potential,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="substitui cópias idênticas por hardlinks; padrão é somente auditoria",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = deduplicate(apply=args.apply)
    action = "Aplicada" if args.apply else "Prévia"
    space = result["bytes_reclaimed"] if args.apply else result["bytes_potential"]
    label = "recuperado" if args.apply else "recuperável"
    logger.info(
        f"{action}: {result['groups']} grupos, {result['paths']} caminhos; "
        f"{result['hardlinks_created']} hardlinks criados, "
        f"{result['copies_fallback']} cópias de fallback, "
        f"{result['already_linked']} já compartilhados; "
        f"espaço {label} {space / (1024 ** 3):.2f} GiB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
