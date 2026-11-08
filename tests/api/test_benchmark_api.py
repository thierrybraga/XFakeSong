"""API do benchmark: leitura aberta, execução com chave e as mesmas travas da aba.

SUJEITO: `app/interfaces/web/routers/benchmark.py`, com os caminhos operacionais
apontando para uma pasta temporária.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def armazenamento(tmp_path, monkeypatch):
    for nome in ("RESULTS_DIR", "DEEPFAKE_RESULTS_DIR", "XFAKE_RESULTS_DIR"):
        monkeypatch.delenv(nome, raising=False)
    for nome in ("MODELS_DIR", "DEEPFAKE_MODELS_DIR", "XFAKE_MODELS_DIR"):
        monkeypatch.setenv(nome, str(tmp_path / "models"))
    monkeypatch.setenv("XFAKE_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("XFAKE_ENABLE_BENCHMARK_RUN", raising=False)
    (tmp_path / "models").mkdir()
    resultado = tmp_path / "results" / "benchmark" / "svm"
    resultado.mkdir(parents=True)
    (resultado / "results.json").write_text(
        json.dumps(
            {
                "architectures": {
                    "SVM": {
                        "status": "ok",
                        "clean": {"eer": 0.07},
                        "scores_clean": [0.1],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_protocolo_traz_as_onze_entradas_e_as_flags(client, armazenamento):
    resposta = client.get("/api/v1/benchmark/protocol")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert len(dados["entries"]) == 11
    assert dados["flags"][:2] == ["--scope", "official"]
    assert "MODELS_DIR=/app/data/models/tcc_reproducao_" in dados["docker_example"]


def test_runs_e_detalhe_sem_dados_por_amostra(client, armazenamento):
    runs = client.get("/api/v1/benchmark/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == ["benchmark"]
    detalhe = client.get("/api/v1/benchmark/runs/benchmark")
    assert detalhe.status_code == 200
    assert detalhe.json()["models"][0]["clean"]["eer"] == 0.07
    assert "scores_clean" not in detalhe.text


def test_run_inexistente_e_caminho_fora_da_pasta(client, armazenamento):
    assert client.get("/api/v1/benchmark/runs/nao_existe").status_code == 404
    assert client.get("/api/v1/benchmark/runs/..%2F..%2Fetc").status_code in (400, 404)


def test_modelos_e_estado_sem_run(client, armazenamento):
    assert client.get("/api/v1/benchmark/models").json() == {"models": []}
    estado = client.get("/api/v1/benchmark/run/status").json()
    assert estado["active"] is False


def test_iniciar_exige_chave(client, armazenamento):
    resposta = client.post("/api/v1/benchmark/run", json={"confirm_duration": True})
    assert resposta.status_code in (401, 403)


def test_iniciar_exige_confirmacao_e_as_travas(client, armazenamento, api_key_headers):
    sem_confirmar = client.post(
        "/api/v1/benchmark/run", json={}, headers=api_key_headers
    )
    assert sem_confirmar.status_code == 400
    bloqueado = client.post(
        "/api/v1/benchmark/run",
        json={"confirm_duration": True},
        headers=api_key_headers,
    )
    assert bloqueado.status_code == 409
    assert "XFAKE_ENABLE_BENCHMARK_RUN" in bloqueado.text


def test_parar_sem_run_ativo(client, armazenamento, api_key_headers):
    resposta = client.post("/api/v1/benchmark/run/stop", headers=api_key_headers)
    assert resposta.status_code == 404
