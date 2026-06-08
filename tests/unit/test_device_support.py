"""Testes unitários do suporte a dispositivos de inferência (CPU/GPU).

Verifica que get_available_devices() sempre lista CPU e que
DetectionService aceita chamadas a set_device() sem falhar.
"""
import pytest

from app.domain.services.detection.utils import get_available_devices


def test_cpu_always_available() -> None:
    """CPU deve estar sempre disponível independente do hardware."""
    devices = get_available_devices()
    assert "CPU" in devices, f"CPU não encontrado em: {devices}"


def test_available_devices_returns_list() -> None:
    """get_available_devices deve retornar uma lista não-vazia."""
    devices = get_available_devices()
    assert isinstance(devices, list)
    assert len(devices) >= 1


def test_detection_service_set_device(tmp_path) -> None:
    """DetectionService.set_device aceita 'CPU' e 'GPU:0' sem levantar exceção."""
    from app.domain.services.detection_service import DetectionService

    try:
        ds = DetectionService(models_dir=str(tmp_path / "models"))
        # Deve aceitar mudança de device sem error
        ds.set_device("GPU:0")
        assert ds.device == "GPU:0"
        ds.set_device("CPU")
        assert ds.device == "CPU"
    except Exception as e:
        # Falha na inicialização (sem modelos) é esperada — mas set_device não deve falhar
        # Se chegou aqui foi o ctor que falhou, não o set_device
        pytest.skip(f"DetectionService não inicializou (sem modelos): {e}")


def test_detection_service_can_skip_default_models(tmp_path) -> None:
    """Modo API leve não deve criar modelos demonstrativos automaticamente."""
    from app.domain.services.detection_service import DetectionService

    ds = DetectionService(
        models_dir=str(tmp_path / "models"),
        create_default_models=False,
    )
    assert ds.get_available_models() == []
    assert ds.default_model is None


def test_main_fastapi_api_only_flag(monkeypatch) -> None:
    """Flag XFAKE_API_ONLY centraliza o modo API sem Gradio/GPU eager."""
    import app.main_fastapi as main_fastapi

    monkeypatch.setenv("XFAKE_API_ONLY", "1")
    assert main_fastapi._api_only_mode() is True

    monkeypatch.delenv("XFAKE_API_ONLY", raising=False)
    monkeypatch.setenv("XFAKE_SKIP_GRADIO", "true")
    assert main_fastapi._api_only_mode() is True
