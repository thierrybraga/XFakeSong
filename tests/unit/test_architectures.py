import pytest
from app.domain.models.architectures.registry import ArchitectureRegistry
from app.domain.models.architectures.factory import ArchitectureFactoryRegistry


def test_registry_initialization():
    registry = ArchitectureRegistry()
    available_architectures = registry.list_architectures()

    assert "AASIST" in available_architectures
    assert "RawGAT-ST" in available_architectures


def test_get_architecture_info():
    registry = ArchitectureRegistry()
    info = registry.get_architecture("AASIST")

    assert info is not None
    assert info.name == "AASIST"
    assert "default" in info.supported_variants


def test_create_factory():
    registry = ArchitectureFactoryRegistry()
    factory = registry.get_factory("AASIST")

    assert factory is not None
    assert hasattr(factory, "create_model")


@pytest.mark.skip(
    reason=(
        "Requires TensorFlow and heavy dependencies, "
        "better for integration tests"
    )
)
def test_create_model_aasist():
    registry = ArchitectureFactoryRegistry()
    factory = registry.get_factory("AASIST")

    input_shape = (100, 80)  # Time, Feats
    model = factory.create_model(input_shape=input_shape)

    assert model is not None
