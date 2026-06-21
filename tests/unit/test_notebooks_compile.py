"""Garante que os notebooks ativos permanecem funcionais.

Trava o bug que motivou a regeneração: células de código com indentação
espúria (IndentationError) e markdown/JSON malformados. Cada célula de código
de cada notebook ATIVO (fora de `legacy/`) deve compilar; o JSON deve ser
nbformat 4 válido; e a estrutura esperada (index + features + 14 models +
4 pipeline) deve existir.

Não executa os notebooks (treino é lento) — apenas valida sintaxe e estrutura.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NB = _ROOT / "notebooks"


def _active_notebooks():
    if not _NB.exists():
        return []
    return [p for p in sorted(_NB.rglob("*.ipynb")) if "legacy" not in p.parts]


def _notebook_text(path: Path) -> str:
    nbj = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nbj.get("cells", []))


def test_expected_notebook_structure():
    names = {p.relative_to(_NB).as_posix() for p in _active_notebooks()}
    assert "00_index.ipynb" in names
    assert "features/01_feature_extraction_study.ipynb" in names
    assert sum(1 for n in names if n.startswith("models/")) == 14
    assert sum(1 for n in names if n.startswith("pipeline/")) == 4


@pytest.mark.parametrize(
    "nb_path", _active_notebooks(),
    ids=[p.relative_to(_NB).as_posix() for p in _active_notebooks()],
)
def test_notebook_code_cells_compile(nb_path):
    nbj = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nbj.get("nbformat") == 4, f"{nb_path.name}: nbformat inesperado"
    code_cells = [c for c in nbj["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{nb_path.name}: sem células de código"
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        # Compila: pega IndentationError/SyntaxError (o bug original).
        compile(src, f"{nb_path.name}#code{i}", "exec")


def test_model_notebooks_have_resolved_input_contracts():
    model_notebooks = sorted((_NB / "models").glob("*.ipynb"))
    assert len(model_notebooks) == 14
    for nb_path in model_notebooks:
        text = _notebook_text(nb_path)
        assert "Tipo de entrada (registry):** `?`" not in text
        assert "Shape preparado" in text
        assert "metrics.json" in text
        assert "confusion_matrix.png" in text
        assert "score_distribution.png" in text


def test_pipeline_notebook_documents_guarded_full_run_and_artifacts():
    text = _notebook_text(_NB / "pipeline" / "01_benchmark_tcc_full_pipeline.ipynb")
    assert "RUN_FULL_PIPELINE = False" in text
    assert "scripts\" / \"run_tcc_pipeline.py" in text
    for artifact in (
        "dataset.md",
        "dataset_manifest.json",
        "tcc_report.md",
        "figures/roc.png",
        "figures/confusion_matrices.png",
        "figures/score_distributions.png",
    ):
        assert artifact in text


def test_all_architectures_notebook_documents_full_benchmark_contract():
    text = _notebook_text(
        _NB / "pipeline" / "04_all_architectures_full_benchmark.ipynb"
    )
    assert "RUN_FULL_BENCHMARK = False" in text
    assert "TARGET_PER_CLASS = 10_000" in text
    assert "XFAKE_STORAGE_DIR" in text
    assert "--download" in text
    assert "--target-per-class" in text
    assert "--archs" in text
    for arch in (
        "WavLM",
        "HuBERT",
        "SpectrogramTransformer",
        "AASIST",
        "SVM",
        "RandomForest",
    ):
        assert arch in text
    assert '"Spectrogram Transformer",' not in text
    for artifact in (
        "dataset.md",
        "dataset_manifest.json",
        "results.json",
        "results.csv",
        "predictions_clean.csv",
        "figures/roc.png",
        "figures/confusion_matrices.png",
        "figures/score_distributions.png",
        "model_artifact",
    ):
        assert artifact in text


def test_features_notebook_covers_runtime_and_classic_features():
    text = _notebook_text(_NB / "features" / "01_feature_extraction_study.ipynb")
    for term in (
        "prepare_audio_for_model",
        "feature_frontend=\"lfcc\"",
        "feature_frontend=\"logmel\"",
        "librosa.feature.mfcc",
        "spectral_centroid",
        "spectral_bandwidth",
        "zero_crossing_rate",
        "feature_summary",
    ):
        assert term in text


@pytest.mark.parametrize(
    "rel_path",
    [
        "00_index.ipynb",
        "features/01_feature_extraction_study.ipynb",
        "models/13_svm.ipynb",
        "models/14_random_forest.ipynb",
        # Self-contained (treinam em dados sintéticos): pegam erros de runtime
        # que o compile() não detecta (ex.: APIs do TrainingService/ModelLoader/
        # factory mudarem). N1 da revisão de notebooks.
        "models/11_multiscale_cnn.ipynb",  # caminho espectrograma
        # RawNet2 cobre o caminho RAW-AUDIO (SincConv + _to_raw_audio), que de
        # outro modo só seria compile-checado — pega drift na construção do
        # modelo raw para a forma sintética preparada pelo benchmark.
        "models/03_rawnet2.ipynb",
        "pipeline/02_training_model.ipynb",
        "pipeline/03_inference.ipynb",
    ],
)
def test_lightweight_notebooks_execute_when_nbclient_is_available(rel_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    # Build-only no CI: os notebooks de modelo treinam por padrão
    # (XFAKE_RUN_EVAL=1, para o estudo ser funcional/Colab). Aqui forçamos =0 —
    # o treino é coberto por pipeline/02 (TrainingService, que sempre treina) e
    # por tests/integration/test_architectures_build (build de todas as archs).
    monkeypatch.setenv("XFAKE_RUN_EVAL", "0")

    nb_path = _NB / rel_path
    nb = nbformat.read(str(nb_path), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=600,  # treina modelos pequenos; runners de CI podem ser lentos
        kernel_name="python3",
        resources={"metadata": {"path": str(_ROOT)}},
    )
    client.execute()
