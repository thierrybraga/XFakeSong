"""Compatibilidade COLAB: o caminho de treino/inferência importa SEM o web layer.

Os notebooks (`models/*`, `pipeline/02`, `pipeline/03`) treinam e inferem
importando só o DOMÍNIO — `benchmarks`, `TrainingService`, `DetectionService`,
`create_model_by_name`. O `BOOTSTRAP` dos notebooks instala apenas dependências
de áudio (librosa/soundfile/PyWavelets); **NÃO** instala FastAPI/Starlette. Em um
Colab/Kaggle limpo o web layer não existe.

Havia um vazamento: `app/core/__init__.py` reexporta as exceções de domínio de
`app/core/exceptions.py`, que importava `fastapi` no topo (e `app/core/middleware`,
que importava `fastapi`/`starlette`). Assim, `from app.domain... import ...`
puxava o FastAPI e quebrava com `ModuleNotFoundError: No module named 'fastapi'`
num Colab sem o web layer.

Este teste roda num SUBPROCESSO que BLOQUEIA `fastapi`/`starlette`/`multipart`
(simulando o Colab limpo) e confirma que o caminho de treino/inferência ainda
importa e constrói um modelo. Subprocesso → isolamento total do `conftest` (que
usa o `TestClient` do FastAPI).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("tensorflow")

# Script executado num interpretador limpo, com FastAPI/Starlette "ausentes".
_COLAB_SIM = textwrap.dedent(
    """
    import os, sys, importlib.abc
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    BLOCK = {'fastapi', 'starlette', 'python_multipart', 'multipart'}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split('.')[0] in BLOCK:
                raise ModuleNotFoundError(f'simulado-ausente: {name}')
            return None

    sys.meta_path.insert(0, _Blocker())
    for _m in list(sys.modules):            # limpa o que já estiver carregado
        if _m.split('.')[0] in BLOCK:
            del sys.modules[_m]

    # Exatamente os imports dos notebooks de treino/inferência:
    from benchmarks import run_benchmark, BenchmarkConfig            # noqa: F401
    from app.domain.services.training_service import TrainingService  # noqa: F401
    from app.domain.services.detection_service import DetectionService  # noqa: F401
    from app.domain.models.architectures.factory import create_model_by_name
    # Classes de exceção de domínio também devem importar sem o web layer:
    from app.core.exceptions import AudioProcessingError, TrainingError  # noqa: F401

    assert 'fastapi' not in sys.modules, 'fastapi foi importado no caminho de domínio'
    assert 'starlette' not in sys.modules, 'starlette foi importado no caminho de domínio'

    # E o factory ainda constrói um modelo (Ensemble é leve):
    m = create_model_by_name('Ensemble', input_shape=(100, 80), num_classes=1)
    assert m.count_params() > 0
    print('CERT_OK', m.count_params())
    """
)


def test_domain_path_imports_and_builds_without_fastapi():
    """Com FastAPI/Starlette bloqueados (Colab limpo), o caminho de
    treino/inferência importa e constrói um modelo — sem `ModuleNotFoundError`."""
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", _COLAB_SIM],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        "caminho de domínio NÃO importou sem o web layer:\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
    )
    assert "CERT_OK" in proc.stdout, f"saída inesperada:\n{proc.stdout[-2000:]}"
