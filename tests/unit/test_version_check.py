"""Testes do guard de compatibilidade de versões no startup (version_check)."""

from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path

import app.core.version_check as vc

ROOT = Path(__file__).resolve().parents[2]


def _stub(monkeypatch, versions: dict):
    monkeypatch.setattr(vc, "_safe_version", lambda pkg: versions.get(pkg))


def test_flags_gradio_hfhub_incompat(monkeypatch):
    # gradio 4.x + huggingface_hub 1.x → HfFolder removido → UI não sobe
    _stub(monkeypatch, {"gradio": "4.44.1", "huggingface_hub": "1.16.1"})
    issues = vc.check_versions()
    assert any("HfFolder" in i for i in issues)
    assert any("huggingface_hub>=0.25,<1.0" in i for i in issues)


def test_healthy_hub_no_issue(monkeypatch):
    # huggingface_hub 0.36.2 (pré-1.0) é compatível com gradio 4.x
    _stub(monkeypatch, {"gradio": "4.44.1", "huggingface_hub": "0.36.2"})
    issues = vc.check_versions()
    assert not any("HfFolder" in i for i in issues)


def test_flags_gradio_starlette_incompat(monkeypatch):
    _stub(monkeypatch, {"gradio": "4.20.0", "starlette": "0.40.0"})
    issues = vc.check_versions()
    assert any("unhashable" in i for i in issues)


def test_gradio5_not_flagged_for_hub(monkeypatch):
    # gradio 5.x não usa HfFolder — não deve ser flaggado mesmo com hub 1.x
    _stub(monkeypatch, {"gradio": "5.0.0", "huggingface_hub": "1.16.1"})
    issues = vc.check_versions()
    assert not any("HfFolder" in i for i in issues)


def test_strict_raises(monkeypatch):
    _stub(monkeypatch, {"gradio": "4.44.1", "huggingface_hub": "1.16.1"})
    import pytest

    with pytest.raises(RuntimeError):
        vc.check_versions(strict=True)


def _avisos_de_datasets(monkeypatch, caplog, versao: str) -> list[str]:
    from app.domain.services.dataset import download

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(__version__=versao))
    with caplog.at_level(logging.WARNING, logger=download.logger.name):
        download.warn_if_datasets_incompatible()
    return [r.getMessage() for r in caplog.records]


def test_datasets_fixado_nao_manda_rebaixar(monkeypatch, caplog):
    # A guarda antiga disparava em qualquer major >=4 e mandava instalar <4.0.
    assert _avisos_de_datasets(monkeypatch, caplog, "5.0.1") == []


def test_datasets_4_avisa_com_a_versao_fixada(monkeypatch, caplog):
    avisos = _avisos_de_datasets(monkeypatch, caplog, "4.0.0")
    assert len(avisos) == 1 and "datasets>=5.0.1,<6.0" in avisos[0]


def test_dica_de_instalacao_do_datasets_acompanha_o_pino(monkeypatch):
    import importlib.util

    from app.domain.services.dataset import download

    monkeypatch.setattr(importlib.util, "find_spec", lambda _nome: None)
    dica = download.check_dependencies(["datasets"])[0]
    base = (ROOT / "requirements-base.txt").read_text(encoding="utf-8")
    assert re.search(rf"(?m)^{re.escape(dica)}\s*(#|$)", base)
