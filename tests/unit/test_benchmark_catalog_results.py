"""Catálogo, leitor de runs e artigo da aba Benchmark.

SUJEITO: `app/domain/services/benchmark/{catalog,results,article}.py`. Tudo em
pastas temporárias. O leitor de runs não pode devolver nada por amostra do
teste: há um teste só para isso.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from app.domain.services.benchmark import article, catalog, results


def _grava(caminho, dados):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados), encoding="utf-8")


# ──────── catálogo ────────


def test_as_onze_entradas_do_artigo_e_seus_tipos():
    entradas = catalog.official_entries()
    assert len(entradas) == 11
    tipos = {e.display_name: e.kind for e in entradas}
    assert tipos["SVM"] == "classical" and tipos["WavLM Original"] == "ssl"
    assert tipos["AST"] == "neural"


def test_receitas_do_artigo_por_tipo():
    por_nome = {e.benchmark_name: e for e in catalog.official_entries()}
    svm = catalog.article_hyperparameters(por_nome["SVM"])
    assert svm["candidatos"] == 15 and "C ∈" in svm["grid"]
    rf = catalog.article_hyperparameters(por_nome["RandomForest"])
    assert rf["candidatos"] == 108
    aasist = catalog.article_hyperparameters(por_nome["AASIST"])
    assert aasist["optimizer"] == "AdamW" and aasist["batch_size"] == 24
    assert aasist["epochs"] == 100
    wavlm = catalog.article_hyperparameters(por_nome["WavLM Original"])
    assert wavlm["checkpoint"] == "microsoft/wavlm-base-plus"
    assert wavlm["lote (cabeça)"] == 128


def test_plano_do_artigo_usa_o_lote_de_gpu_sem_sondar(monkeypatch):
    from benchmarks import planning

    def proibido(*_a, **_k):
        raise AssertionError("não pode sondar a GPU")

    monkeypatch.setattr(planning, "_device_snapshot", proibido)
    assert planning.article_training_config("AASIST")["batch_size"] == 24
    assert planning.article_training_config("AASIST", "cpu")["batch_size"] == 4


def test_descoberta_igual_a_da_deteccao(tmp_path):
    for nome in ("bench_a.keras", "bench_b.pkl", "bench_b_scaler.pkl", "x.txt"):
        (tmp_path / nome).write_bytes(b"x")
    final = tmp_path / "benchmark_final" / "c"
    final.mkdir(parents=True)
    (final / "bench_c.keras").write_bytes(b"x")
    assert set(catalog.discover_model_paths(tmp_path)) == {
        "bench_a",
        "bench_b",
        "bench_c",
    }


def _modelo_neural(pasta, conteudo=b"pesos"):
    artefato = pasta / "bench_aasist.keras"
    artefato.write_bytes(conteudo)
    _grava(
        pasta / "bench_aasist_config.json",
        {
            "architecture": "AASIST",
            "model_type": "tensorflow",
            "created_at": "2026-08-23",
            "metrics": {
                "final_metrics": {"accuracy": 0.95, "roc_auc": 0.99, "eer": 0.04},
                "history": {"val_eer": [0.2, 0.1, 0.05], "val_loss": [1.0, 0.5, 0.4]},
                "training_config": {"batch_size": 24, "learning_rate": 3e-4},
                "model_summary": {"total_params": 396312},
            },
            "input_contract": {
                "eer_threshold": 0.85,
                "eer_value": 0.03,
                "temperature": 6.6,
                "temperature_on_logits": True,
                "ood_threshold": 0.003,
                "band_correction": {"cutoff_hz": 7500},
                "feature_frontend": "benchmark_raw_v1",
            },
        },
    )
    return artefato


def test_modelo_neural_traz_calibracao_validacao_e_historico(tmp_path):
    _modelo_neural(tmp_path)
    (m,) = catalog.installed_models(
        catalog.discover_model_paths(tmp_path), tmp_path, "bench_aasist"
    )
    assert m.entry.benchmark_name == "AASIST" and m.is_default
    assert m.calibration["eer_threshold"] == 0.85
    assert m.calibration["band_cutoff_hz"] == 7500
    assert m.validation["eer"] == 0.03 and m.validation["accuracy"] == 0.95
    assert m.history["val_eer"] == [0.2, 0.1, 0.05]
    assert m.efficiency["params"] == 396312
    assert m.test == {}, "sem run ligado, nenhuma métrica de teste aparece"


def test_sidecar_classico_e_do_assistente_toleram_chaves_ausentes(tmp_path):
    (tmp_path / "bench_svm.pkl").write_bytes(b"x")
    _grava(
        tmp_path / "bench_svm_config.json",
        {"architecture": "SVM", "input_contract": {"eer_threshold": 0.05}},
    )
    (tmp_path / "meu.keras").write_bytes(b"x")
    _grava(tmp_path / "meu_config.json", {"trained_by": "wizard"})
    modelos = catalog.installed_models(catalog.discover_model_paths(tmp_path), tmp_path)
    assert [m.name for m in modelos] == ["bench_svm", "meu"]
    assert modelos[1].entry is None and modelos[1].history == {}


def test_metricas_de_teste_vem_do_run_que_produziu_o_artefato(tmp_path):
    artefato = _modelo_neural(tmp_path)
    digital = hashlib.sha256(artefato.read_bytes()).hexdigest()
    indice = {digital: {"run_id": "benchmark", "clean": {"eer": 0.031}}}
    (m,) = catalog.installed_models({"bench_aasist": artefato}, tmp_path, None, indice)
    assert m.test == {"eer": 0.031, "source": "benchmark"}


def test_registry_de_outro_artefato_nao_vira_metrica_de_teste(tmp_path):
    artefato = _modelo_neural(tmp_path)
    _grava(
        tmp_path / "registry.json",
        {
            "models": [
                {"slug": "aasist", "model_sha256": "outro", "metrics": {"eer": 0.5}}
            ]
        },
    )
    (m,) = catalog.installed_models({"bench_aasist": artefato}, tmp_path)
    assert m.test == {}


# ──────── runs ────────


def _run(raiz, nome="benchmark", arquitetura="AASIST", pasta="aasist", **extra):
    run = raiz / nome
    _grava(
        run / pasta / "results.json",
        {
            "architectures": {
                arquitetura: {
                    "status": "ok",
                    "clean": {"eer": 0.03, "eer_ci95_low": 0.02, "eer_ci95_high": 0.04},
                    "robustness": {
                        "10": {"eer": 0.06, "noise_condition": "matched"},
                        "5": {"eer": 0.14, "noise_condition": "unseen"},
                    },
                    "grouped_clean": {"speaker": {"worst_group_accuracy": 0.8}},
                    "efficiency": {"params": 10, "latency_ms": 5.0},
                    "training_stability": {"best_epoch": 58},
                    "history": {"val_eer": [0.1, 0.05]},
                    "scores_clean": [0.1, 0.9],
                    "scores_robustness": {"10": [0.2]},
                    "model_artifact_fingerprint": {"sha256": "abc"},
                    **extra,
                }
            },
            "dataset": {"y_test": [0, 1], "test_cluster_ids": [1, 2]},
        },
    )
    _grava(
        run / "run_summary.json",
        {
            "status": "ok",
            "models": [{"model": arquitetura, "status": "ok", "elapsed_s": 3600}],
        },
    )
    return run


def test_run_traz_so_agregados_e_nada_por_amostra_do_teste(tmp_path):
    run = _run(tmp_path)
    (run / "aasist" / "figures").mkdir()
    (run / "aasist" / "figures" / "roc.png").write_bytes(b"png")
    vista = results.load_run(run, "benchmark")
    (m,) = vista.models
    assert m.clean["eer"] == 0.03 and m.robustness["5"]["noise_condition"] == "unseen"
    assert m.worst_speaker_accuracy == 0.8 and m.stability["best_epoch"] == 58
    assert set(m.figures) == {"roc"}
    serializado = json.dumps(dataclasses.asdict(vista))
    for proibido in ("scores_clean", "scores_robustness", "y_test", "test_cluster_ids"):
        assert proibido not in serializado


def test_modelo_valido_em_superseded_completa_o_run(tmp_path):
    run = _run(tmp_path)
    _run(
        tmp_path,
        nome="benchmark/_superseded",
        arquitetura="RawNet2",
        pasta="rawnet2_d9f23c0d",
    )
    vista = results.load_run(run, "benchmark")
    assert [(m.architecture, m.superseded) for m in vista.models] == [
        ("RawNet2", True),
        ("AASIST", False),
    ]


def test_descoberta_classifica_runs_e_ignora_o_que_nao_e_run(tmp_path):
    _run(tmp_path)
    _run(tmp_path, nome="archive/pre/clean_benchmark_15k")
    _run(tmp_path, nome="tcc_reproducao_20261107-1200")
    (tmp_path / "paper" / "consolidated").mkdir(parents=True)
    tipos = {r.run_id: r.kind for r in results.discover_runs(tmp_path)}
    assert tipos == {
        "benchmark": "vigente",
        "tcc_reproducao_20261107-1200": "reproducao",
        "archive/pre/clean_benchmark_15k": "artigo",
    }


def test_run_fora_da_pasta_de_resultados_e_recusado(tmp_path):
    (tmp_path / "res").mkdir()
    with pytest.raises(ValueError):
        results.resolve_run(tmp_path / "res", "../fora")


def test_indice_de_impressao_digital(tmp_path):
    _run(tmp_path)
    indice = results.fingerprint_index(tmp_path)
    assert indice["abc"]["run_id"] == "benchmark"
    assert indice["abc"]["clean"]["eer"] == 0.03


# ──────── artigo ────────


def test_artigo_titulo_resumo_e_pdf_defasado(tmp_path):
    (tmp_path / "main.tex").write_text(
        r"""\hypersetup{pdftitle={Detectores de Clones}, pdfauthor={Fulano}}
\begin{center}\textbf{RESUMO}\end{center}
\begin{singlespace}
	\noindent
	O estudo compara \num{11} modelos em \SI{3}{\second} a \SI{16}{\kilo\hertz};
	AST obteve \num{0,14}\% de EER % comentário
	\noindent\textbf{Palavras-chave:} \textit{deepfake}; robustez.
\end{singlespace}""",
        encoding="utf-8",
    )
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF")
    import os

    os.utime(pdf, (1_000_000, 1_000_000))
    info = article.article_info(tmp_path)
    assert info.title == "Detectores de Clones" and info.author == "Fulano"
    assert info.abstract == (
        "O estudo compara 11 modelos em 3 s a 16 kHz; AST obteve 0,14% de EER"
    )
    assert info.keywords == "deepfake; robustez."
    assert info.pdf_is_stale


def test_artigo_ausente_nao_quebra(tmp_path):
    info = article.article_info(tmp_path)
    assert info.title is None and info.pdf_path is None and not info.pdf_is_stale


def test_tabela_e_significancia_do_artigo(tmp_path):
    _grava(
        tmp_path / "consolidated" / "benchmark_summary.json",
        [
            {
                "model": "AST",
                "eer": 0.0014,
                "robustness": {"10": {"eer": 0.06}},
                "path": "data\\results\\clean_benchmark_15k\\ast",
            }
        ],
    )
    _grava(
        tmp_path / "consolidated" / "benchmark_significance.json",
        {
            "pairs": [
                {
                    "model_a": "AST",
                    "model_b": "CCT",
                    "mcnemar": {"p_value_holm": 1.0},
                    "paired_bootstrap": {"observed_difference": -0.002},
                }
            ]
        },
    )
    (linha,) = results.article_summary(tmp_path)
    assert linha["eer_10db"] == 0.06 and linha["source"].endswith(
        "clean_benchmark_15k/ast"
    )
    (par,) = results.article_significance(tmp_path)
    assert par["mcnemar_p_holm"] == 1.0
