"""Testes do guard de compatibilidade de versões no startup (version_check)."""

from __future__ import annotations

import app.core.version_check as vc


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
