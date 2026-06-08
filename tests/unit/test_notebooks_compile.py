"""Garante que os notebooks ativos permanecem funcionais.

Trava o bug que motivou a regeneração: células de código com indentação
espúria (IndentationError) e markdown/JSON malformados. Cada célula de código
de cada notebook ATIVO (fora de `legacy/`) deve compilar; o JSON deve ser
nbformat 4 válido; e a estrutura esperada (index + features + 14 models +
3 pipeline) deve existir.

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


def _strip_ipython_magics(src: str) -> str:
    """Remove magics do IPython (`!shell`, `%magic`, `%%cellmagic`) que não são
    Python válido — permite validar só a sintaxe Python das células.

    Células iniciadas por `%%cellmagic` (corpo não-Python, ex.: `%%bash`) são
    inteiramente ignoradas.
    """
    lines = src.splitlines()
    for ln in lines:
        if ln.strip():
            if ln.lstrip().startswith("%%"):
                return ""
            break
    return "\n".join(
        ln for ln in lines if not ln.lstrip().startswith(("!", "%"))
    )


def test_expected_notebook_structure():
    names = {p.relative_to(_NB).as_posix() for p in _active_notebooks()}
    assert "00_index.ipynb" in names
    assert "features/01_feature_extraction_study.ipynb" in names
    assert sum(1 for n in names if n.startswith("models/")) == 14
    assert sum(1 for n in names if n.startswith("pipeline/")) == 3


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


def test_legacy_notebook_code_cells_compile_after_archive_notice():
    legacy_notebooks = sorted((_NB / "legacy").glob("*.ipynb"))
    assert legacy_notebooks
    for nb_path in legacy_notebooks:
        nbj = json.loads(nb_path.read_text(encoding="utf-8"))
        text = _notebook_text(nb_path)
        assert "Legacy" in text
        for i, cell in enumerate(
            [c for c in nbj["cells"] if c["cell_type"] == "code"]
        ):
            # Notebooks legacy contêm magics (`!pip install`, `%matplotlib`) —
            # remova-os antes do compile (validamos só a sintaxe Python).
            src = _strip_ipython_magics("".join(cell["source"]))
            compile(src, f"{nb_path.name}#legacy-code{i}", "exec")


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
        "models/11_multiscale_cnn.ipynb",
        "pipeline/02_training_model.ipynb",
        "pipeline/03_inference.ipynb",
    ],
)
def test_lightweight_notebooks_execute_when_nbclient_is_available(rel_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb_path = _NB / rel_path
    nb = nbformat.read(str(nb_path), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=600,  # treina modelos pequenos; runners de CI podem ser lentos
        kernel_name="python3",
        resources={"metadata": {"path": str(_ROOT)}},
    )
    client.execute()
