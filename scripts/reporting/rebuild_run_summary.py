#!/usr/bin/env python3
"""Reconstrói ``run_summary.json``/``.md`` a partir dos ``results.json`` do run.

MOTIVAÇÃO 2026-08-09. O ``run_summary`` é escrito UMA vez, quando
``run_models_sequential.py`` termina a bateria. Reavaliar um subconjunto depois
disso — sem repetir as outras arquiteturas — não tinha caminho: os
``results.json`` dos modelos reavaliados mudavam e o resumo continuava
anunciando os números antigos.

Foi exatamente o que aconteceu no ``clean_benchmark_15k``. WavLM e HuBERT
Original foram reavaliados em 2026-08-09 com a janela corrigida
(``--target-samples 48000``); os ``.pt`` e os ``results.json`` acompanharam, e o
resumo, de 2026-08-06, seguiu declarando os números da janela de 64.000:

    HuBERT Original   resumo 97,61% / EER 2,17%   real 93,92% / EER 5,93%
    WavLM Original    resumo 96,53% / EER 3,47%   real 96,09% / EER 3,62%

Como ``sync_completed_benchmark_artifacts.py`` lia as métricas do resumo, a
promoção publicaria os pesos novos com os números velhos. Aquele script passou a
ler dos ``results.json``; este fecha o outro lado, deixando o resumo
regenerável em vez de historicamente congelado.

Preserva os campos operacionais que só a execução conhece (``elapsed_s``,
``timeout_min``, ``log``, ``returncode``) copiando-os do resumo anterior quando
existe — eles não estão nos ``results.json`` e reconstruí-los seria invenção.

Uso:

    python scripts/reporting/rebuild_run_summary.py data/results/<run>
    python scripts/reporting/rebuild_run_summary.py data/results/<run> --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

#: Campos do resumo antigo que não existem em nenhum `results.json` — só a
#: execução sequencial os conhece. São copiados, nunca inventados.
_OPERATIONAL_FIELDS = (
    "elapsed_s",
    "timeout_min",
    "log",
    "returncode",
    "log_tail",
    "error",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel_to_run(path: Path, run_dir: Path) -> str:
    """Caminho como o runner o grava (``/app/...`` no container)."""
    try:
        return "/app/" + path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _arch_block(results: dict[str, Any], model: str) -> tuple[str, dict[str, Any]]:
    archs = results.get("architectures") or {}
    if model and model in archs:
        return model, archs[model]
    if len(archs) == 1:
        return next(iter(archs.items()))
    lowered = {k.lower(): (k, v) for k, v in archs.items()}
    if model.lower() in lowered:
        return lowered[model.lower()]
    raise KeyError(f"sem bloco de arquitetura para {model!r}")


def rebuild(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    previous = _load_json(summary_path) if summary_path.is_file() else {}
    # Preserva a ORDEM de execução do resumo anterior; modelos que só existem em
    # disco entram depois, em ordem alfabética.
    order = [str(m.get("model") or "") for m in previous.get("models", [])]
    prev_by_model = {str(m.get("model") or ""): m for m in previous.get("models", [])}

    found: dict[str, tuple[Path, dict[str, Any], str, dict[str, Any]]] = {}
    for results_path in sorted(run_dir.glob("*/results.json")):
        results = _load_json(results_path)
        archs = results.get("architectures") or {}
        for name, arch in archs.items():
            found[name] = (results_path, results, name, arch)

    models: list[dict[str, Any]] = []
    ordered_names = [n for n in order if n in found]
    ordered_names += sorted(n for n in found if n not in order)

    for name in ordered_names:
        results_path, results, _, arch = found[name]
        prev = prev_by_model.get(name, {})
        entry: dict[str, Any] = {
            "model": name,
            # `or "ok"` carimbava sucesso num bloco que NÃO declarou status —
            # o caso de um results.json truncado por queda durante a gravação
            # (nenhuma escrita do pipeline é atômica) ou editado à mão. O
            # `sync_completed_benchmark_artifacts.py` promove pelo status, então
            # o default otimista transformava artefato incompleto em promovido.
            # Um status desconhecido tem de PARECER desconhecido.
            "status": arch.get("status") or "unknown",
            "output_dir": _rel_to_run(results_path.parent, run_dir),
            "model_artifact": arch.get("model_artifact"),
            "clean": arch.get("clean"),
            "efficiency": arch.get("efficiency"),
            # Não estava no schema antigo e é o que teria evitado promover um
            # treino colapsado sem ninguém notar.
            "training_stability": arch.get("training_stability"),
            "test_split_sha256": (results.get("dataset") or {}).get(
                "test_split_sha256"
            ),
            "metrics_source": _rel_to_run(results_path, run_dir),
        }
        for field in _OPERATIONAL_FIELDS:
            if field in prev:
                entry[field] = prev[field]
        models.append(entry)

    payload = {k: v for k, v in previous.items() if k != "models"}
    payload["models"] = models
    payload["rebuilt_from_results_json"] = True
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Execução Sequencial de Benchmark",
        "",
        f"- Dataset: `{payload.get('dataset')}`",
        f"- Device profile: `{payload.get('device_profile')}`",
        f"- Timeout por modelo: `{payload.get('timeout_min') or 'derivado por arquitetura'}`",
        "- Resumo **regenerado** a partir dos `results.json` "
        "(`scripts/reporting/rebuild_run_summary.py`).",
        "",
        "| Modelo | Status | Estabilidade | Accuracy | AUC | EER | Latência ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    def fmt(value: Any, nd: int = 4) -> str:
        return f"{value:.{nd}f}" if isinstance(value, (int, float)) else "-"

    for m in payload.get("models", []):
        clean = m.get("clean") or {}
        eff = m.get("efficiency") or {}
        stab = (m.get("training_stability") or {}).get("status") or "-"
        lines.append(
            f"| {m.get('model')} | {m.get('status')} | {stab} | "
            f"{fmt(clean.get('accuracy'))} | {fmt(clean.get('auc_roc'))} | "
            f"{fmt(clean.get('eer'))} | {fmt(eff.get('latency_ms'), 2)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="diretório do run (ex.: data/results/<run>)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="imprime a tabela reconstruída sem gravar nada",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run não encontrado: {run_dir}")

    payload = rebuild(run_dir)
    markdown = render_markdown(payload)
    if args.dry_run:
        print(markdown)
        print(f"(simulação — {len(payload['models'])} modelos, nada gravado)")
        return 0

    for target, content in (
        ("run_summary.json", json.dumps(payload, indent=2, ensure_ascii=False)),
        ("run_summary.md", markdown),
    ):
        path = run_dir / target
        # Mesma politica do backfill: guarda o original UMA vez e nunca o
        # sobrescreve. O resumo carrega campos operacionais que so a execucao
        # conhece — perde-los numa regeneracao seria irreversivel.
        backup = path.with_suffix(path.suffix + ".pre-rebuild.bak")
        if path.is_file() and not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(content, encoding="utf-8")

    print(f"Regenerado: {run_dir / 'run_summary.json'}")
    print(f"Modelos: {len(payload['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
