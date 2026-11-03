"""Sincronização do Hub no boot: em Docker, `data/models` é a pasta do próprio host."""

from __future__ import annotations

import huggingface_hub
import pytest

from tools.ops.sync_hf_models import sync_models


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Um snapshot do Hub em disco, sem rede, com o token recebido em `chamadas`."""
    pasta = tmp_path / "hub"
    pasta.mkdir()
    (pasta / "bench_svm.pkl").write_text("do hub", encoding="utf-8")
    (pasta / "bench_aasist.keras").write_text("do hub", encoding="utf-8")
    chamadas = []

    def snapshot_download(**kwargs):
        chamadas.append(kwargs)
        return str(pasta)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.delenv("MODEL_REPO_ID", raising=False)
    monkeypatch.delenv("XFAKE_MODEL_REPO_ID", raising=False)
    monkeypatch.delenv("DEEPFAKE_MODEL_REPO_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    return chamadas


def _sincroniza(destino, repo="usuario/modelos", force=False):
    """A lista de esperados tem 14 nomes, então o snapshot de teste fica incompleto."""
    try:
        sync_models(repo, destino, force=force)
    except RuntimeError as exc:
        assert "incompleta" in str(exc)


def test_sem_repositorio_nao_baixa_nada(hub, tmp_path):
    assert sync_models(None, tmp_path / "modelos") == 0
    assert hub == []


def test_variavel_vazia_do_compose_equivale_a_nao_definida(hub, tmp_path, monkeypatch):
    """O compose repassa `MODEL_REPO_ID` vazio quando o `.env` não define o nome."""
    monkeypatch.setenv("MODEL_REPO_ID", "")
    monkeypatch.setenv("HF_TOKEN", "")
    assert sync_models(None, tmp_path / "modelos") == 0
    assert hub == []


def test_nao_sobrescreve_arquivo_local(hub, tmp_path):
    """Um sidecar recalibrado aqui se perderia a cada boot com o repositório definido."""
    destino = tmp_path / "modelos"
    destino.mkdir()
    local = destino / "bench_svm.pkl"
    local.write_text("calibrado aqui", encoding="utf-8")

    _sincroniza(destino)

    assert local.read_text(encoding="utf-8") == "calibrado aqui"
    assert (destino / "bench_aasist.keras").read_text(encoding="utf-8") == "do hub"


def test_force_sobrescreve(hub, tmp_path):
    destino = tmp_path / "modelos"
    destino.mkdir()
    local = destino / "bench_svm.pkl"
    local.write_text("antigo", encoding="utf-8")

    _sincroniza(destino, force=True)

    assert local.read_text(encoding="utf-8") == "do hub"


def test_token_vem_do_ambiente(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-de-teste")
    _sincroniza(tmp_path / "modelos")
    assert hub[0]["token"] == "token-de-teste"
