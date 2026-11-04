"""Gate do pip-audit: bloqueia o que a lista de exceções registrada não cobre.

O ``pip-audit`` puro não separa um aviso novo de um já avaliado: com pinos
deliberados (``torch``, ``transformers``) ele falharia em toda release, e um gate
que sempre falha deixa de ser lido. Aqui cada aviso tolerado está em
``configs/security/pip-audit-exceptions.json``, com a versão exata, os IDs, a
justificativa e a data de revisão. O gate falha quando:

- surge um aviso que a lista não cobre, em qualquer pacote;
- a data de revisão de uma exceção passou;
- a versão instalada difere da registrada (a exceção avaliou outra versão);
- uma exceção aponta para um pacote ausente do ambiente auditado;
- um pacote não pôde ser auditado, ou o próprio pip-audit não terminou.

Um ID registrado que deixou de ser reportado só gera nota: a base de avisos pode
retirá-lo sem que nada tenha mudado no projeto.

Uso:
    python tools/ops/check_pip_audit.py                   # audita o ambiente atual
    python tools/ops/check_pip_audit.py --report a.json   # usa um relatório pronto

Saída: 0 aprovado, 1 reprovado, 2 não foi possível avaliar.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_EXCEPTIONS = Path("configs/security/pip-audit-exceptions.json")
_REQUIRED_TEXT_FIELDS = ("package", "version", "reason", "blocked_by")
# Códigos do pip-audit: 0 = sem avisos, 1 = com avisos. Outro código é erro; um
# erro com código 1 sai sem JSON na saída e é recusado na leitura do relatório.
_AUDIT_EXIT_CODES = {0, 1}


@dataclass(frozen=True)
class AuditException:
    """Uma exceção válida: pacote e versão exatos, IDs avaliados e prazo."""

    package: str
    version: str
    review_by: date
    ids: frozenset[str]


def normalize_name(name: str) -> str:
    """Nome canônico (PEP 503): ``Pillow`` e ``pillow`` são o mesmo pacote."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_exception(entry: Any) -> AuditException:
    if not isinstance(entry, dict):
        raise ValueError(f"exceção não é um objeto: {entry!r}")
    label = entry.get("package", "?")
    for field in _REQUIRED_TEXT_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: campo '{field}' ausente ou vazio")
    ids = entry.get("ids")
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(item, str) and item.strip() for item in ids)
    ):
        raise ValueError(f"{label}: 'ids' precisa listar ao menos um ID")
    try:
        review_by = date.fromisoformat(entry.get("review_by", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: 'review_by' não é uma data ISO") from exc
    return AuditException(
        package=normalize_name(entry["package"]),
        version=entry["version"],
        review_by=review_by,
        ids=frozenset(ids),
    )


def load_exceptions(path: Path) -> dict[str, AuditException]:
    """Lê e valida a lista; uma entrada incompleta invalida o arquivo inteiro."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("exceptions"), list):
        raise ValueError(f"{path}: esperado um objeto com a lista 'exceptions'")
    exceptions: dict[str, AuditException] = {}
    for entry in data["exceptions"]:
        parsed = _parse_exception(entry)
        if parsed.package in exceptions:
            raise ValueError(f"{parsed.package}: exceção duplicada")
        exceptions[parsed.package] = parsed
    return exceptions


def report_dependencies(report: Any) -> list[dict[str, Any]]:
    """Devolve os pacotes do relatório; um relatório sem pacotes não prova nada."""
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("relatório do pip-audit sem pacotes auditados")
    return dependencies


def evaluate(
    report: Any,
    exceptions: dict[str, AuditException],
    today: date,
) -> tuple[list[str], list[str]]:
    """Compara o relatório com as exceções; devolve ``(falhas, notas)``."""
    failures: set[str] = set()
    notes: set[str] = set()
    for exception in exceptions.values():
        if exception.review_by < today:
            failures.add(
                f"{exception.package}: a exceção venceu em {exception.review_by}; "
                "reavalie os avisos e registre uma nova data"
            )

    audited: set[str] = set()
    for dependency in report_dependencies(report):
        name = normalize_name(dependency["name"])
        audited.add(name)
        if dependency.get("skip_reason"):
            failures.add(f"{name}: não auditado ({dependency['skip_reason']})")
            continue

        version = dependency.get("version")
        exception = exceptions.get(name)
        if exception is not None and version != exception.version:
            failures.add(
                f"{name}: instalado {version}, mas a exceção avaliou "
                f"{exception.version}"
            )
            exception = None

        covered: set[str] = set()
        for vuln in dependency.get("vulns") or []:
            ids = {vuln["id"], *(vuln.get("aliases") or [])}
            matched = ids & exception.ids if exception is not None else set()
            if matched:
                covered |= matched
                continue
            fixes = ", ".join(vuln.get("fix_versions") or []) or "sem correção"
            failures.add(f"{name} {version}: {vuln['id']} fora da lista ({fixes})")

        if exception is not None and exception.ids - covered:
            notes.add(
                f"{name}: não reportados mais: "
                f"{', '.join(sorted(exception.ids - covered))}"
            )

    for name in sorted(set(exceptions) - audited):
        failures.add(f"{name}: há exceção, mas o pacote não está no ambiente auditado")
    return sorted(failures), sorted(notes)


def run_pip_audit() -> Any:
    """Audita o ambiente deste interpretador e devolve o relatório em JSON."""
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--format",
            "json",
            "--progress-spinner",
            "off",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode not in _AUDIT_EXIT_CODES:
        raise ValueError(
            f"pip-audit terminou com código {process.returncode}: "
            f"{process.stderr.strip()[-500:]}"
        )
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"pip-audit não devolveu JSON: {process.stderr.strip()[-500:]}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bloqueia avisos do pip-audit fora da lista de exceções."
    )
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    parser.add_argument(
        "--report",
        type=Path,
        help="relatório JSON do pip-audit; sem isto, audita o ambiente atual",
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=None,
        help="data de referência (AAAA-MM-DD) para simular o vencimento",
    )
    args = parser.parse_args(argv)

    try:
        exceptions = load_exceptions(args.exceptions)
        if args.report:
            report = json.loads(args.report.read_text(encoding="utf-8"))
        else:
            report = run_pip_audit()
        failures, notes = evaluate(report, exceptions, args.today or date.today())
    except (OSError, ValueError, KeyError) as exc:
        print(f"Não foi possível avaliar o pip-audit: {exc}", file=sys.stderr)
        return 2

    for note in notes:
        print(f"nota: {note}")
    if failures:
        print(f"pip-audit reprovado: {len(failures)} problema(s)")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    tolerated = sum(len(item.ids) for item in exceptions.values())
    print(
        f"pip-audit aprovado: {len(report_dependencies(report))} pacotes; "
        f"{tolerated} IDs tolerados em {len(exceptions)} exceções."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
