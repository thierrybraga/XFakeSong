"""Configuração pytest para smoke tests pesados.

Os smoke tests desta pasta constroem modelos TensorFlow reais e executam
pipelines completos — são lentos (minutos) e não fazem parte do run normal.

Para rodar apenas os smoke tests:
    pytest -m smoke tests/smoke/ -v

Para rodar tudo EXCETO smoke (padrão):
    pytest tests/
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: smoke tests pesados (TF real); execute com: pytest -m smoke tests/smoke/",
    )
