#!/usr/bin/env python3
"""Consolida runs, valida artefatos, gera tabelas/figuras e compila o artigo."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "data" / "results" / "paper"
CONSOLIDATED_DIR = ROOT / "data" / "results" / "paper" / "consolidated"
FINAL_DIR = PAPER_DIR / "final"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _discover_results(inputs: list[str]) -> list[Path]:
    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file() and path.name == "results.json":
            found.add(path.resolve())
        elif path.is_dir():
            direct = path / "results.json"
            if direct.exists():
                found.add(direct.resolve())
            else:
                found.update(item.resolve() for item in path.glob("*/results.json"))
    if not found:
        raise SystemExit("ERRO: nenhum results.json encontrado nos inputs.")
    return sorted(found)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_result(path: Path) -> list[str]:
    errors: list[str] = []
    data = _load_json(path)
    dataset = data.get("dataset") or {}
    if dataset.get("n_total") != 15000:
        errors.append(f"{path}: n_total={dataset.get('n_total')}, esperado 15000")
    overlap = dataset.get("split_overlap_audit") or {}
    if overlap.get("passed") is not True:
        errors.append(f"{path}: auditoria de duplicatas ausente ou reprovada")
    test_sha256 = dataset.get("test_split_sha256")
    if not isinstance(test_sha256, str) or len(test_sha256) != 64:
        errors.append(f"{path}: test_split_sha256 ausente ou inválido")
    if dataset.get("split_source") != "predefined_npz":
        errors.append(f"{path}: split principal não veio do NPZ predefinido")
    guard = data.get("academic_protocol_guard") or {}
    lock = guard.get("test_lock") or {}
    if guard.get("validated_before_training") is not True:
        errors.append(f"{path}: selo acadêmico não foi validado antes do treino")
    if lock.get("validated") is not True or lock.get("declared_untouched") is not True:
        errors.append(f"{path}: declaração de teste intocado ausente")
    if guard.get("test_split_sha256") != test_sha256:
        errors.append(f"{path}: fingerprint do guard diverge do resultado")
    config = data.get("config") or {}
    try:
        threshold = float(config.get("decision_threshold", -1.0))
    except (TypeError, ValueError):
        threshold = -1.0
    if threshold != 0.5:
        errors.append(f"{path}: limiar oficial diferente de 0,5")
    y_test = dataset.get("y_test") or []
    for name, result in (data.get("architectures") or {}).items():
        if result.get("status") != "ok":
            errors.append(f"{path}: {name} status={result.get('status')}")
            continue
        scores = result.get("scores_clean") or []
        if len(scores) != len(y_test):
            errors.append(
                f"{path}: {name} scores={len(scores)} e y_test={len(y_test)}"
            )
        robustness = result.get("robustness") or {}
        if set(robustness) != {"30", "20", "10"}:
            errors.append(f"{path}: {name} SNRs={sorted(robustness)}")
        protocol = result.get("noise_protocol") or result.get("input_preparation") or {}
        if protocol.get("evaluation_domain") != "waveform":
            errors.append(f"{path}: {name} AWGN não está em waveform")
        if protocol.get("frontend_after_noise") is not True:
            errors.append(f"{path}: {name} frontend_after_noise ausente")
        if result.get("type") != "classical":
            if result.get("epochs") != 100:
                errors.append(f"{path}: {name} épocas={result.get('epochs')}")
            history = result.get("history") or {}
            losses = history.get("loss") or []
            if len(losses) != 100:
                errors.append(f"{path}: {name} histórico possui {len(losses)} épocas")
        arch_dir = path.parent / "architectures" / _slug(name)
        for filename in ("metrics.json", "predictions_clean.csv", "robustness.csv"):
            if not (arch_dir / filename).exists():
                errors.append(f"{path}: ausente {arch_dir / filename}")
    return errors


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida runs e reconstrói tabelas, figuras e PDF do artigo."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="diretório sequencial, diretórios de modelo ou results.json",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compila main.tex com latexmk",
    )
    args = parser.parse_args()

    result_files = _discover_results(args.inputs)
    validation_errors: list[str] = []
    for result_file in result_files:
        validation_errors.extend(_validate_result(result_file))
    if validation_errors:
        separator = chr(10) + "- "
        raise SystemExit(
            "ERRO: artefatos de benchmark inválidos:" + separator
            + separator.join(validation_errors)
        )

    consolidate = ROOT / "scripts" / "reporting" / "consolidate_results.py"
    update_latex = ROOT / "scripts" / "reporting" / "update_tcc_latex.py"
    summary = CONSOLIDATED_DIR / "benchmark_summary.json"
    table = PAPER_DIR / "tabelas_benchmark.tex"

    _run(
        [
            sys.executable,
            str(consolidate),
            *[str(path) for path in result_files],
            "--prefer-last",
            "--out",
            str(CONSOLIDATED_DIR),
            "--copy-to",
            str(PAPER_DIR / "figures"),
        ]
    )
    _run(
        [
            sys.executable,
            str(update_latex),
            "--summary",
            str(summary),
            "--output",
            str(table),
            "--figures-dir",
            "figures",
        ]
    )

    main_tex = PAPER_DIR / "main.tex"
    main_text = main_tex.read_text(encoding="utf-8")
    table_text = table.read_text(encoding="utf-8")
    if r"input{tabelas_benchmark.tex}" not in main_text:
        raise SystemExit("ERRO: main.tex não inclui tabelas_benchmark.tex")
    for label in (
        "tab:resultados_consolidados",
        "tab:eficiencia_modelos",
        "tab:robustez_awgn",
        "tab:estabilidade_treinamento",
    ):
        if label not in table_text:
            raise SystemExit(f"ERRO: label ausente no fragmento: {label}")

    if args.compile:
        _run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=PAPER_DIR,
        )

    # Fonte canônica única: data/results/paper. A pasta final é apenas um espelho
    # de distribuição regenerado, nunca uma segunda fonte editável.
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_tex, FINAL_DIR / "main.tex")
    shutil.copy2(table, FINAL_DIR / "tabelas_benchmark.tex")
    shutil.copy2(
        PAPER_DIR / "MELHORIAS_POS_RETREINO.md",
        FINAL_DIR / "MELHORIAS_POS_RETREINO.md",
    )
    if (PAPER_DIR / "main.pdf").exists():
        shutil.copy2(PAPER_DIR / "main.pdf", FINAL_DIR / "main.pdf")
    shutil.copytree(PAPER_DIR / "figures", FINAL_DIR / "figures", dirs_exist_ok=True)
    (FINAL_DIR / "README.md").write_text(
        "# Espelho gerado\n\n"
        "Não edite esta pasta. A fonte canônica é `data/results/paper/main.tex`; "
        "regenere este espelho com `build_paper_from_benchmark.py`.\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in result_files],
        "summary": str(summary),
        "table": str(table),
        "pdf": str(PAPER_DIR / "main.pdf"),
        "sha256": {
            "benchmark_summary.json": _sha256(summary),
            "tabelas_benchmark.tex": _sha256(table),
            "main.pdf": _sha256(PAPER_DIR / "main.pdf"),
        },
    }
    (PAPER_DIR / "paper_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Artigo atualizado: {PAPER_DIR / 'main.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())