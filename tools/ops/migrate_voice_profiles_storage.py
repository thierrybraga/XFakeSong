"""Move voice-profile files and persisted paths into the canonical data folder.

Dry-run is the default. Pass ``--apply`` to move the directory and update the
SQLite ``voice_profiles`` table. Path values are never printed.

Usage:
    python tools/ops/migrate_voice_profiles_storage.py
    python tools/ops/migrate_voice_profiles_storage.py --apply
    python tools/ops/migrate_voice_profiles_storage.py --database data/app.db --apply
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _replace_path_prefix(value: str, old_prefix: str, new_prefix: str) -> str:
    """Replace a whole path prefix while accepting either slash convention."""
    normalized = value.replace("\\", "/")
    old_normalized = old_prefix.replace("\\", "/").rstrip("/")
    new_normalized = new_prefix.replace("\\", "/").rstrip("/")
    folded = normalized.casefold()
    old_folded = old_normalized.casefold()
    if folded != old_folded and not folded.startswith(f"{old_folded}/"):
        return value
    return new_normalized + normalized[len(old_normalized) :]


def _rewrite_stored_path(value: str, root: Path) -> str:
    """Translate known legacy relative, repository-absolute, and container paths."""
    root = root.resolve()
    prefixes = (
        (
            str(root / "app" / "voice_profiles"),
            str(root / "data" / "voice_profiles"),
        ),
        ("app/voice_profiles", "data/voice_profiles"),
        ("/app/voice_profiles", "/app/data/voice_profiles"),
    )
    for old, new in prefixes:
        rewritten = _replace_path_prefix(value, old, new)
        if rewritten != value:
            return rewritten
    return value


def _pending_updates(
    connection: sqlite3.Connection, root: Path
) -> list[tuple[int, dict[str, str]]]:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'voice_profiles'"
    ).fetchone()
    if not table_exists:
        return []

    columns = [
        row[1] for row in connection.execute("PRAGMA table_info(voice_profiles)")
    ]
    path_columns = [
        name for name in columns if "path" in name.lower() or "dir" in name.lower()
    ]
    if not path_columns:
        return []

    selected = ", ".join(_quote_identifier(name) for name in path_columns)
    updates: list[tuple[int, dict[str, str]]] = []
    for row in connection.execute(f"SELECT rowid, {selected} FROM voice_profiles"):
        rowid = row[0]
        changes = {
            column: rewritten
            for column, value in zip(path_columns, row[1:])
            if isinstance(value, str)
            and (rewritten := _rewrite_stored_path(value, root)) != value
        }
        if changes:
            updates.append((rowid, changes))
    return updates


def migrate(database: Path, apply: bool, root: Path = PROJECT_ROOT) -> tuple[int, int]:
    """Report or apply the storage migration; return moved-directory and row counts."""
    source = root / "app" / "voice_profiles"
    target = root / "data" / "voice_profiles"
    if source.exists() and target.exists():
        raise FileExistsError(
            "Both legacy and canonical voice-profile directories exist; "
            "merge them manually."
        )

    updates: list[tuple[int, dict[str, str]]] = []
    if database.exists():
        if apply:
            connection = sqlite3.connect(database, timeout=10)
        else:
            connection = sqlite3.connect(
                f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=10
            )
        try:
            updates = _pending_updates(connection, root)
        finally:
            connection.close()

    should_move = source.exists()
    print(f"Legacy directory present: {should_move}")
    print(f"Database records with legacy paths: {len(updates)}")
    if not apply:
        print("Dry-run only. Re-run with --apply to perform the migration.")
        return int(should_move), len(updates)

    moved = False
    if should_move:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        moved = True

    try:
        if database.exists():
            connection = sqlite3.connect(database, timeout=10)
            try:
                connection.execute("BEGIN IMMEDIATE")
                updates = _pending_updates(connection, root)
                for rowid, changes in updates:
                    assignments = ", ".join(
                        f"{_quote_identifier(name)} = ?" for name in changes
                    )
                    connection.execute(
                        f"UPDATE voice_profiles SET {assignments} WHERE rowid = ?",
                        [*changes.values(), rowid],
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
    except Exception:
        if moved and target.exists() and not source.exists():
            os.replace(target, source)
        raise

    print(f"Moved legacy directory: {moved}")
    print(f"Updated database records: {len(updates)}")
    return int(moved), len(updates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "app.db",
        help="SQLite database to update (default: data/app.db)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="move files and update SQLite paths"
    )
    args = parser.parse_args()
    database = (
        args.database
        if args.database.is_absolute()
        else PROJECT_ROOT / args.database
    )
    migrate(database, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
