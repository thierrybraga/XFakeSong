#!/usr/bin/env python3
"""Audita o ESQUEMA do SQLite do projeto (`data/app.db`) e remove o esquema morto.

Nao confundir com `consolidate_sqlite.py`, que faz o oposto: IMPORTA
configuracao e resultados legados PARA dentro do banco. Este aqui so olha a
forma do banco (quais tabelas existem) e derruba as que nao pertencem a ele.


O arquivo carregava 18 tabelas de OUTRO dominio (`workouts`, `exercises`,
`mentors`, `store_items`, `wallet_transactions`, `events`, `runs`,
`achievements`, ...) — nenhuma delas referenciada em lugar algum do
XFakeSong. Elas nao vieram do `create_all`: o `Base.metadata` do projeto so
conhece 11 tabelas. Vieram no proprio arquivo, reaproveitado de outro projeto.

Consequencias praticas de deixa-las: `PRAGMA integrity_check` e backups
carregam esquema morto; `follows`/`wallet_transactions`/`workout_sessions`
declaram FK para `users`, entao qualquer limpeza futura de usuarios esbarra em
dependencias inexistentes no codigo; e qualquer auditoria do banco precisa
explicar por que um detector de deepfake tem carteira e loja.

A lista de tabelas LEGITIMAS e derivada do `Base.metadata` do ORM, nunca
escrita a mao: uma lista fixa envelhece e passaria a derrubar tabela nova.

Uma tabela orfa com LINHAS nunca e derrubada — o script para e pede decisao
explicita, porque dado inesperado nao e lixo por definicao.

Exemplos:
  python scripts/ops/audit_database_schema.py                # auditoria
  python scripts/ops/audit_database_schema.py --apply        # remove as orfas
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "data" / "app.db"


def _orm_tables() -> tuple[set[str], str]:
    """Tabelas declaradas pelo ORM — a definicao de 'legitima'.

    Preferencia pelo `Base.metadata` real. Como importar o dominio arrasta
    numpy/TF, ha um fallback que le os `__tablename__` direto do fonte: mais
    fraco (nao resolve nome montado em tempo de execucao), mas ainda DERIVADO
    do codigo, e permite auditar num ambiente enxuto. O fallback nunca decide
    sozinho um `--apply`.
    """
    try:
        from app.core.db.session import Base
        from app.core.db.setup import _load_domain_models

        _load_domain_models()
        return set(Base.metadata.tables), "orm"
    except ImportError:
        pass

    names: set[str] = set()
    for path in (ROOT / "app" / "domain" / "models").rglob("*.py"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("__tablename__"):
                _, _, value = stripped.partition("=")
                names.add(value.strip().strip("\"'"))
    if not names:
        raise ImportError("nenhum __tablename__ encontrado em app/domain/models")
    return names, "fonte"


def _table_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'select count(*) from "{table}"').fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="derruba as tabelas orfas VAZIAS e compacta (default: so audita)",
    )
    parser.add_argument(
        "--force-drop-non-empty",
        action="store_true",
        help="derruba tambem tabelas orfas COM linhas (exige decisao consciente)",
    )
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        parser.error(f"banco nao encontrado: {db_path}")

    try:
        legit, origem = _orm_tables()
    except ImportError as exc:
        parser.error(
            f"nao foi possivel determinar o esquema legitimo ({exc}). "
            "Sem ele nao ha como distinguir tabela orfa de tabela nova — "
            "rode no ambiente do projeto."
        )
    if origem == "fonte" and args.apply:
        parser.error(
            "o metadata do ORM nao carregou (dependencias ausentes) e o "
            "fallback por leitura de fonte nao autoriza --apply: derrubar "
            "tabela exige a lista autoritativa. Rode no ambiente do projeto."
        )

    conn = sqlite3.connect(db_path)
    present = {
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type='table' "
            "and name not like 'sqlite_%'"
        )
    }
    orphans = sorted(present - legit)
    missing = sorted(legit - present)

    size_before = db_path.stat().st_size
    print(f"Banco: {db_path}  ({size_before / 1024 / 1024:.1f} MB)")
    print(
        f"Tabelas legitimas: {len(legit)} (origem: {origem}) | "
        f"no arquivo: {len(present)}"
    )

    integrity = conn.execute("pragma integrity_check").fetchone()[0]
    print(f"integrity_check: {integrity}")
    fk_broken = list(conn.execute("pragma foreign_key_check"))
    print(f"foreign_key_check: {len(fk_broken)} violacao(oes)")

    if missing:
        print(f"\nAUSENTES (o ORM espera, o arquivo nao tem): {', '.join(missing)}")
        print("  -> rode o bootstrap do banco antes de consolidar")

    if not orphans:
        print("\nNenhuma tabela orfa. Nada a consolidar.")
        conn.close()
        return 0

    populated = {t: _table_rows(conn, t) for t in orphans}
    print(f"\nORFAS ({len(orphans)}) — sem modelo no ORM e sem referencia no codigo:")
    for table in orphans:
        mark = "  " if populated[table] == 0 else " !"
        print(f" {mark} {table:26s} {populated[table]:6d} linha(s)")

    with_rows = [t for t, n in populated.items() if n]
    if with_rows and not args.force_drop_non_empty:
        print(
            f"\n{len(with_rows)} tabela(s) orfa(s) tem linhas: "
            f"{', '.join(with_rows)}.\n"
            "Dado inesperado nao e lixo por definicao — inspecione e, se for "
            "descartavel, repita com --force-drop-non-empty."
        )
        if args.apply:
            conn.close()
            return 1

    if not args.apply:
        print("\n[auditoria] nada foi alterado. Repita com --apply para consolidar.")
        conn.close()
        return 0

    conn.close()
    if not args.no_backup:
        backup = db_path.with_suffix(db_path.suffix + ".bak-consolidate")
        shutil.copy2(db_path, backup)
        print(f"\n[backup] {backup}")

    conn = sqlite3.connect(db_path)
    doomed = orphans if args.force_drop_non_empty else [t for t in orphans if not populated[t]]
    for table in doomed:
        conn.execute(f'drop table if exists "{table}"')
        print(f"  [drop] {table}")
    conn.commit()
    conn.execute("vacuum")
    conn.close()

    size_after = db_path.stat().st_size
    print(
        f"\n{len(doomed)} tabela(s) removida(s). "
        f"{size_before / 1024 / 1024:.1f} MB -> {size_after / 1024 / 1024:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
