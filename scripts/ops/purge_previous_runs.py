#!/usr/bin/env python3
"""Apaga o historico de treino/benchmark anterior persistido no SQLite.

`experiment_store.persist_benchmark_results` grava cada execucao em
`data/app.db`. Quando o dataset canonico muda, esse historico continua la e
vira a unica copia sobrevivente de numeros obsoletos: as tabelas do artigo e as
figuras podem ser regeneradas, mas as linhas do banco nao caducam sozinhas e
alimentam a API de historico, a aba Gradio e qualquer consolidacao que leia o
store em vez de `data/results/`.

Por padrao apaga TUDO (`--all`, o caso de virada de protocolo). Com
`--keep-dataset` preserva as execucoes de um `.npz` especifico.

Nao toca em `users`, `voice_profiles`, `configuration_entries` nem
`architecture_configs`: sao configuracao/identidade, nao resultado de treino.

Exemplos:
  python scripts/ops/purge_previous_runs.py --dry-run
  python scripts/ops/purge_previous_runs.py --all
  python scripts/ops/purge_previous_runs.py \
      --keep-dataset data/datasets/benchmark_dataset.npz
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "app.db"

#: Ordem importa: filhos antes dos pais (o schema declara ON DELETE CASCADE,
#: mas `PRAGMA foreign_keys` e OFF por padrao no sqlite3 do Python).
RUN_TABLES = (
    "metric_records",
    "artifact_records",
    "model_runs",
    "experiment_runs",
)
#: Tabelas de historico sem vinculo com experiment_runs.
STANDALONE_TABLES = ("training_jobs", "analysis_results", "system_snapshots")


def _counts(conn: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in tables:
        try:
            result[table] = conn.execute(f"select count(*) from {table}").fetchone()[0]
        except sqlite3.OperationalError:
            continue
    return result


def _purge(conn: sqlite3.Connection, keep_dataset: str | None) -> None:
    if keep_dataset is None:
        for table in RUN_TABLES + STANDALONE_TABLES:
            try:
                conn.execute(f"delete from {table}")
            except sqlite3.OperationalError:
                continue
        return

    doomed = [
        row[0]
        for row in conn.execute(
            "select id from experiment_runs "
            "where dataset_path is null or dataset_path not like ?",
            (f"%{Path(keep_dataset).name}",),
        )
    ]
    if doomed:
        marks = ",".join("?" * len(doomed))
        for table in RUN_TABLES[:-1]:
            conn.execute(f"delete from {table} where experiment_id in ({marks})", doomed)
        conn.execute(f"delete from experiment_runs where id in ({marks})", doomed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="apaga todo o historico de execucoes (virada de protocolo)",
    )
    group.add_argument(
        "--keep-dataset",
        metavar="NPZ",
        help="preserva apenas as execucoes deste dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="nao copia o banco antes (por padrao grava <db>.bak-purge)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        parser.error(f"banco nao encontrado: {db_path}")

    tables = RUN_TABLES + STANDALONE_TABLES
    conn = sqlite3.connect(db_path)
    before = _counts(conn, tables)

    if args.dry_run:
        conn.execute("begin")
        _purge(conn, args.keep_dataset)
        after = _counts(conn, tables)
        conn.rollback()
        conn.close()
        print(f"[dry-run] nada foi gravado em {db_path}")
    else:
        conn.close()
        if not args.no_backup:
            backup = db_path.with_suffix(db_path.suffix + ".bak-purge")
            shutil.copy2(db_path, backup)
            print(f"[backup] {backup}")
        conn = sqlite3.connect(db_path)
        _purge(conn, args.keep_dataset)
        conn.commit()
        after = _counts(conn, tables)
        conn.execute("vacuum")
        conn.close()

    for table in tables:
        if table in before:
            print(f"  {table:20s} {before[table]:7d} -> {after.get(table, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
