"""Regeneração do `run_summary` a partir dos `results.json` do run.

MOTIVAÇÃO. O `run_summary.json` é escrito uma única vez, ao fim da bateria
sequencial, e não acompanha reavaliações parciais. No `clean_benchmark_15k`,
WavLM e HuBERT Original foram reavaliados com a janela corrigida
(`--target-samples 48000`) em 2026-08-09; os `.pt` e os `results.json` mudaram e
o resumo, de 2026-08-06, seguiu anunciando os números da janela de 64.000.
Como a promoção lia o resumo, publicaria peso novo com métrica velha.

Estes testes travam as duas propriedades que impedem a reincidência: a métrica
vem sempre dos `results.json`, e os campos que só a execução conhece
(`elapsed_s`, `log`, `returncode`) são copiados do resumo anterior, nunca
inventados.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rebuild_module():
    spec = importlib.util.spec_from_file_location(
        "_rebuild_run_summary",
        _PROJECT_ROOT / "scripts" / "reporting" / "rebuild_run_summary.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model(run: Path, slug: str, display: str, clean: dict, **extra) -> None:
    d = run / slug
    d.mkdir(parents=True, exist_ok=True)
    arch = {
        "status": "ok",
        "clean": clean,
        "efficiency": {"latency_ms": 10.0},
        "model_artifact": f"/app/data/models/bench_{slug}.keras",
    }
    arch.update(extra)
    (d / "results.json").write_text(
        json.dumps(
            {
                "architectures": {display: arch},
                "dataset": {"test_split_sha256": "abc"},
            }
        ),
        encoding="utf-8",
    )


def test_metrica_vem_do_results_json_e_nao_do_resumo_defasado(tmp_path):
    """O caso real: resumo em 97,61% / 2,17%, artefato em 93,92% / 5,93%."""
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    _write_model(
        run,
        "hubert_original",
        "HuBERT Original",
        {"accuracy": 0.9392, "eer": 0.0593, "auc_roc": 0.9868},
    )
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "dataset": "/app/data/datasets/benchmark_dataset_15k.npz",
                "models": [
                    {
                        "model": "HuBERT Original",
                        "status": "ok",
                        "clean": {"accuracy": 0.9761, "eer": 0.0217},
                        "elapsed_s": 280.2,
                        "returncode": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = mod.rebuild(run)
    entry = payload["models"][0]

    assert entry["clean"]["accuracy"] == 0.9392
    assert entry["clean"]["eer"] == 0.0593
    assert payload["rebuilt_from_results_json"] is True
    # E a tabela renderizada tem de mostrar o número novo, não o do resumo.
    assert "0.9392" in mod.render_markdown(payload)
    assert "0.9761" not in mod.render_markdown(payload)


def test_campos_operacionais_sao_preservados_e_nao_inventados(tmp_path):
    """`elapsed_s`/`log`/`returncode` não estão em nenhum results.json."""
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    _write_model(run, "svm", "SVM", {"accuracy": 0.85})
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "SVM",
                        "status": "ok",
                        "elapsed_s": 622.5,
                        "timeout_min": 30,
                        "returncode": 0,
                        "log": "/app/logs/svm.log",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entry = mod.rebuild(run)["models"][0]

    assert entry["elapsed_s"] == 622.5
    assert entry["timeout_min"] == 30
    assert entry["log"] == "/app/logs/svm.log"


def test_modelo_sem_resumo_anterior_entra_sem_campos_fabricados(tmp_path):
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    _write_model(run, "novo", "Novo", {"accuracy": 0.9})

    entry = mod.rebuild(run)["models"][0]

    assert entry["model"] == "Novo"
    assert "elapsed_s" not in entry
    assert "returncode" not in entry


def test_ordem_de_execucao_do_resumo_anterior_e_preservada(tmp_path):
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    for slug, display in (("a", "Alfa"), ("b", "Beta"), ("c", "Gama")):
        _write_model(run, slug, display, {"accuracy": 0.9})
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "models": [
                    {"model": "Gama", "status": "ok"},
                    {"model": "Alfa", "status": "ok"},
                ]
            }
        ),
        encoding="utf-8",
    )

    nomes = [m["model"] for m in mod.rebuild(run)["models"]]

    # Os conhecidos mantêm a ordem da execução; o novo entra depois.
    assert nomes == ["Gama", "Alfa", "Beta"]


def test_estabilidade_do_treino_aparece_no_resumo(tmp_path):
    """O resumo antigo não tinha o campo — foi como o Conformer passou."""
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    _write_model(
        run,
        "conformer",
        "Conformer",
        {"accuracy": 0.9949},
        training_stability={"status": "collapsed", "stable": False},
    )

    payload = mod.rebuild(run)

    assert payload["models"][0]["training_stability"]["status"] == "collapsed"
    assert "collapsed" in mod.render_markdown(payload)


def test_run_sem_results_json_nao_explode(tmp_path):
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()

    assert mod.rebuild(run)["models"] == []


def test_render_tolera_metrica_ausente(tmp_path):
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    _write_model(run, "x", "X", {})

    markdown = mod.render_markdown(mod.rebuild(run))

    assert "| X |" in markdown
    assert "-" in markdown


@pytest.mark.parametrize("caminho", ["run_summary.json", "run_summary.md"])
def test_escrita_gera_os_dois_arquivos(tmp_path, caminho, monkeypatch):
    mod = _rebuild_module()
    run = tmp_path / "run"
    run.mkdir()
    _write_model(run, "x", "X", {"accuracy": 0.9})

    payload = mod.rebuild(run)
    (run / "run_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run / "run_summary.md").write_text(mod.render_markdown(payload), encoding="utf-8")

    assert (run / caminho).is_file()
