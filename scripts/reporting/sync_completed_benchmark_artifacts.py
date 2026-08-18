#!/usr/bin/env python3
"""Promove modelos concluídos do benchmark para data/models.

O benchmark grava os artefatos principais em:
- data/models/bench_<modelo>.*      (usado pela inferência/Gradio)
- data/results/<execucao>/<modelo>/...  (métricas, figuras e relatórios)

Este script copia os modelos já concluídos para:
- data/models/benchmark_final/<modelo>/

Essa pasta é explicitamente incluída no Docker build pela regra de
`.dockerignore`, permitindo empacotar modelos pré-treinados sem novo treino.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("/app/"):
        return ROOT / text.removeprefix("/app/")
    path = Path(text)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PromotionRefused(RuntimeError):
    """Levantada quando o run não pode ser promovido como está."""


def _arch_result(results: dict[str, Any], model: str) -> dict[str, Any] | None:
    """Bloco da arquitetura no results.json, tolerante ao rótulo do resumo.

    O `run_summary` guarda o nome de exibição ("HuBERT Original"), que é a
    mesma chave usada em `architectures` — mas um run com uma única
    arquitetura não deixa ambiguidade, então caímos no bloco único quando o
    nome não bate exatamente.
    """
    archs = results.get("architectures") or {}
    if model in archs:
        return archs[model]
    if len(archs) == 1:
        return next(iter(archs.values()))
    lowered = {k.lower(): v for k, v in archs.items()}
    return lowered.get(model.lower())


def collect_promotable(summary_path: Path) -> list[dict[str, Any]]:
    """Lê os `results.json` do run e recusa o que não pode ser promovido.

    MOTIVAÇÃO 2026-08-09: até aqui as métricas promovidas saíam do
    `run_summary.json` (`item["clean"]`). Esse arquivo é escrito UMA vez, ao
    fim da bateria sequencial, e não é reescrito quando um subconjunto é
    reavaliado. Foi o que aconteceu com WavLM/HuBERT: os `.pt` foram refeitos
    com a janela de 48.000, os `results.json` acompanharam, e o resumo
    continuou anunciando os números da janela de 64.000 (HuBERT 97,61% /
    EER 2,17% contra os 93,92% / 5,93% reais). Promover a partir dele
    publicaria peso novo com métrica velha.

    Agora o resumo serve só de índice — quais modelos rodaram e onde. Toda
    métrica vem do `results.json` de cada um.
    """
    summary = _load_json(summary_path)
    run_dir = summary_path.parent
    items: list[dict[str, Any]] = []
    refusals: list[str] = []
    fingerprints: dict[str, str] = {}

    for item in summary.get("models", []):
        if item.get("status") != "ok":
            continue
        model = str(item.get("model") or "")
        output_dir = _project_path(item.get("output_dir")) or (run_dir / _slug(model))
        results_path = output_dir / "results.json"
        if not results_path.is_file():
            refusals.append(f"{model}: results.json ausente em {results_path}")
            continue

        results = _load_json(results_path)
        arch = _arch_result(results, model)
        if arch is None:
            refusals.append(f"{model}: sem bloco em architectures do results.json")
            continue

        sha = (results.get("dataset") or {}).get("test_split_sha256")
        if sha:
            fingerprints.setdefault(str(sha), model)

        stability = arch.get("training_stability") or {}
        if stability.get("stable") is False:
            refusals.append(
                f"{model}: training_stability.stable=false "
                f"({stability.get('status')}) — {stability.get('reason') or 'sem motivo'}"
            )
            continue

        artifact = _project_path(arch.get("model_artifact")) or _project_path(
            item.get("model_artifact")
        )
        expected = arch.get("model_artifact_fingerprint") or {}
        if artifact is None or not artifact.is_file():
            refusals.append(f"{model}: artefato ausente ({artifact})")
            continue
        integrity = expected.get("integrity")
        if integrity == "size_mismatch":
            # Veredito do backfill: o run declarou um tamanho e o arquivo no
            # caminho declarado tem outro. É o caso do SVM do
            # clean_benchmark_15k (3,61 MB no run, 47 KB em disco) — o artefato
            # do benchmark foi substituído por outro depois da execução.
            refusals.append(
                f"{model}: {expected.get('reason') or 'tamanho do artefato diverge do run'}"
            )
            continue
        if expected.get("sha256"):
            actual = _sha256(artifact)
            if actual != expected["sha256"]:
                refusals.append(
                    f"{model}: sha256 do artefato diverge do gravado no run "
                    f"(disco {str(actual)[:12]}… != run {expected['sha256'][:12]}…) "
                    "— o arquivo foi trocado depois da execução"
                )
                continue

        items.append(
            {
                "model": model,
                "artifact": artifact,
                "results_path": results_path,
                "output_dir": output_dir,
                "metrics": arch.get("clean"),
                "efficiency": arch.get("efficiency"),
                "training_stability": stability,
                "test_split_sha256": sha,
            }
        )

    if len(fingerprints) > 1:
        joined = "; ".join(f"{s[:12]}… ({m})" for s, m in fingerprints.items())
        refusals.append(
            "conjuntos de teste diferentes no mesmo run — as métricas não são "
            f"comparáveis nem promovíveis juntas: {joined}"
        )

    if refusals:
        raise PromotionRefused("promoção recusada:\n  - " + "\n  - ".join(refusals))
    return items


def sync_completed(
    summary_path: Path,
    final_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promove os modelos do run. Levanta ``PromotionRefused`` se algo não bate.

    A coleta (``collect_promotable``) roda ANTES de qualquer escrita: ou o run
    inteiro passa nas guardas, ou nada é copiado. Promover metade de um run
    deixaria `benchmark_final/` num estado que nenhum `results.json` descreve.
    """
    promotable = collect_promotable(summary_path)
    synced: list[dict[str, Any]] = []
    if not dry_run:
        final_dir.mkdir(parents=True, exist_ok=True)

    for entry in promotable:
        model = entry["model"]
        slug = _slug(model)
        target = final_dir / slug
        model_artifact = entry["artifact"]
        config = model_artifact.with_name(f"{model_artifact.stem}_config.json")

        if dry_run:
            model_copied = model_artifact.is_file()
            config_copied = config.is_file()
        else:
            target.mkdir(parents=True, exist_ok=True)
            model_copied = _copy_file(model_artifact, target / model_artifact.name)
            config_copied = _copy_file(config, target / config.name)
            legacy_results = target / "results"
            if legacy_results.exists():
                shutil.rmtree(legacy_results)

        # Impressão digital do arquivo PROMOVIDO (no dry-run, da origem, que é
        # o que seria copiado). `benchmark_final/` é a pasta que entra na imagem
        # Docker: sem o sha256 aqui, o modelo empacotado era o único elo da
        # cadeia sem identidade própria — dava para trocar o .keras dentro da
        # imagem e nada no diretório denunciaria. Com ele, o manifesto identifica
        # o arquivo que ele descreve.
        promoted = model_artifact if dry_run else target / model_artifact.name
        manifest = {
            "model": model,
            "slug": slug,
            "status": "ok",
            "source_model_artifact": str(model_artifact),
            "source_output_dir": str(entry["output_dir"]),
            # Rastreia de onde a métrica veio: o `run_summary` deixou de ser a
            # fonte justamente porque não acompanha reavaliações parciais.
            "metrics_source": str(entry["results_path"]),
            "test_split_sha256": entry["test_split_sha256"],
            "model_sha256": _sha256(promoted),
            "model_copied": model_copied,
            "config_copied": config_copied,
            "results_copied": False,
            "metrics": entry["metrics"],
            "efficiency": entry["efficiency"],
            "training_stability": entry["training_stability"],
        }
        if not dry_run:
            (target / "artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        synced.append(manifest)

    index = {
        "summary_path": str(summary_path),
        "final_dir": str(final_dir),
        "synced_count": len(synced),
        "dry_run": bool(dry_run),
        "models": synced,
    }
    if dry_run:
        return index
    registry = final_dir.parent / "registry.json"
    registry.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Benchmark Final Artifacts",
        "",
        f"- Origem: `{summary_path}`",
        f"- Modelos sincronizados: `{len(synced)}`",
        "",
        "| Modelo | Modelo | Config | Resultados | Accuracy | AUC | EER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in synced:
        metrics = item.get("metrics") or {}
        lines.append(
            f"| {item['model']} | {item['model_copied']} | "
            f"{item['config_copied']} | {item['results_copied']} | "
            f"{metrics.get('accuracy', '')} | {metrics.get('auc_roc', '')} | "
            f"{metrics.get('eer', '')} |"
        )
    (final_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        required=True,
        help=(
            "Caminho para o run_summary.json do benchmark a promover "
            "(ex.: data/results/<run>/run_summary.json). Sem default: adivinhar "
            "um 'run atual' é frágil — ver docs/development/developer-guide.md."
        ),
    )
    parser.add_argument("--final-dir", default="data/models/benchmark_final")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Roda todas as guardas e imprime o que seria promovido, sem copiar "
            "nada nem tocar em registry.json/README.md."
        ),
    )
    args = parser.parse_args()

    summary = _project_path(args.summary)
    final_dir = _project_path(args.final_dir)
    if summary is None or not summary.exists():
        raise SystemExit(f"Resumo não encontrado: {summary}")
    assert final_dir is not None
    try:
        index = sync_completed(summary, final_dir, dry_run=args.dry_run)
    except PromotionRefused as exc:
        # Recusa é o comportamento correto — sair 0 aqui faria um pipeline
        # engolir o problema e seguir para o LaTeX com artefato errado.
        print(str(exc))
        return 2
    prefix = "Promoveria" if args.dry_run else "Sincronizados"
    print(f"{prefix}: {index['synced_count']}")
    for item in index["models"]:
        metrics = item.get("metrics") or {}
        acc, eer = metrics.get("accuracy"), metrics.get("eer")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        eer_s = f"{eer:.4f}" if isinstance(eer, (int, float)) else "-"
        print(f"  - {item['model']:<24} acc={acc_s} eer={eer_s}")
    print(f"Destino: {final_dir}")
    if not args.dry_run:
        print(f"Manifesto: {final_dir.parent / 'registry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
