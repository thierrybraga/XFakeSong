"""Regressão: TODAS as interfaces leem/gravam modelos no MESMO diretório.

Antes havia divergência: o `TrainingService` (default), o wizard do Gradio, a
injeção de dependência da API e o CLI usavam `app/models`, mas o **default** do
`DetectionService` era `models` (raiz). As abas Detectar/Investigar do Gradio
instanciavam com esse default e liam um diretório VAZIO — um modelo treinado pelo
wizard (salvo em `app/models`) nunca aparecia na detecção.

Estes testes travam:
1. o default do `DetectionService` aponta para `app/models`;
2. as abas Gradio de detecção/forense reusam o MESMO singleton da API
   (`app.dependencies.get_detection_service`), para que o reload do wizard
   propague o modelo recém-treinado sem reiniciar o app.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow")


def test_detection_service_default_dir_is_app_models():
    from app.domain.services.detection_service import DetectionService

    ds = DetectionService(create_default_models=False)  # usa o default
    assert ds.models_dir == Path("app/models"), (
        f"default models_dir deveria ser app/models, veio {ds.models_dir}"
    )


def test_gradio_detection_tabs_share_api_singleton(monkeypatch):
    """As abas Detectar e Investigar devem reusar o singleton da API (app/models),
    não criar um `DetectionService()` próprio lendo o dir default antigo."""
    pytest.importorskip("gradio")

    # Criação controlada e leve: sem modelos default, dir app/models (vazio em CI).
    monkeypatch.setenv("XFAKE_CREATE_DEFAULT_MODELS", "false")
    monkeypatch.setenv("DEEPFAKE_DEVICE", "CPU")

    from app.dependencies import get_detection_service as api_get

    api_get.cache_clear()  # garante criação fresca sob o env acima

    import app.interfaces.gradio.tabs.detection as det_mod
    import app.interfaces.gradio.tabs.forensic_analysis as for_mod

    # Zera os singletons de módulo das abas (outro teste pode tê-los populado).
    monkeypatch.setattr(det_mod, "_detection_service_instance", None, raising=False)
    monkeypatch.setattr(for_mod, "_detection_service", None, raising=False)

    detect_tab_get = det_mod.get_detection_service
    forensic_get = for_mod._get_detection_service

    shared = api_get()
    assert shared is not None
    assert detect_tab_get() is shared, "aba Detectar não usa o singleton da API"
    assert forensic_get() is shared, "aba Investigar não usa o singleton da API"
    assert shared.models_dir == Path("app/models")

    api_get.cache_clear()  # não vaza o singleton para outros testes
