"""Mapa de portas: cada uso na sua porta, e a trava que a porta igual fazia por acidente."""

from __future__ import annotations

import http.server
import json
import re
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from app.core.config import ports
from tools.ops import docker_release

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "infrastructure" / "docker" / "compose"
GUIA = ROOT / "docs" / "development" / "docker-environments.md"
HOST_PORTS = (
    ports.DOCKER_NVIDIA_PORT,
    ports.DOCKER_CPU_PORT,
    ports.LOCAL_PORT,
    ports.DOCS_PORT,
    *ports.RELEASE_CHECK_PORTS.values(),
)


def _texto(caminho: str) -> str:
    return (ROOT / caminho).read_text(encoding="utf-8")


def _publicadas(arquivo: Path) -> list[str]:
    servicos = yaml.safe_load(arquivo.read_text(encoding="utf-8"))["services"]
    return [p for s in servicos.values() for p in s.get("ports", [])]


def test_nenhuma_porta_do_host_se_repete():
    assert len(HOST_PORTS) == len(set(HOST_PORTS))


@pytest.mark.parametrize(
    ("arquivo", "variavel", "porta"),
    [
        ("inference.nvidia.yml", "GRADIO_PORT_NVIDIA", ports.DOCKER_NVIDIA_PORT),
        ("inference.cpu.yml", "GRADIO_PORT_CPU", ports.DOCKER_CPU_PORT),
    ],
)
def test_cada_perfil_publica_a_sua_porta(arquivo, variavel, porta):
    esperado = (
        "${GRADIO_BIND_HOST:-127.0.0.1}:"
        f"${{{variavel}:-{porta}}}:{ports.CONTAINER_PORT}"
    )
    assert _publicadas(COMPOSE / arquivo) == [esperado]


@pytest.mark.parametrize(
    "arquivo", ["train.cpu.yml", "train.nvidia.yml", "benchmark.nvidia.yml"]
)
def test_treino_e_benchmark_nao_publicam_porta(arquivo):
    assert _publicadas(COMPOSE / arquivo) == []


def test_doctor_nao_abre_o_banco_que_um_container_esta_usando():
    texto = _texto("tools/ops/doctor.py")
    assert texto.index("conflicting_docker_instances(own_port=0)") < texto.index(
        "check_database_health()"
    )


def test_docker_release_usa_as_portas_do_mapa():
    assert docker_release.PORTAS_CHECK == ports.RELEASE_CHECK_PORTS


def test_main_usa_a_porta_local_e_verifica_antes_de_carregar_a_gpu():
    texto = _texto("main.py")
    assert 'os.environ.get("PORT", LOCAL_PORT)' in texto
    assert texto.index("conflicting_docker_instances(args.port)") < texto.index(
        "setup_gpu()"
    )


@pytest.mark.parametrize(
    ("script", "trecho"),
    [
        ("Makefile", "PORT ?= {}"),
        ("start.sh", 'LOCAL_PORT="{}"'),
        ("start.bat", "--gradio-port {}"),
    ],
)
def test_scripts_sobem_a_aplicacao_local_na_porta_dela(script, trecho):
    texto = _texto(script)
    assert trecho.format(ports.LOCAL_PORT) in texto
    assert "--gradio-port 7860" not in texto


@pytest.mark.parametrize(
    ("script", "trava", "subida"),
    [
        ("Makefile", "up: check-other-instance", "up -d $(SERVICE)"),
        ("start.bat", "CALL :CHECK_OTHER_INSTANCE", "up --build -d"),
        ("start.sh", "\n    refuse_other_instance\n", "up --build -d"),
    ],
)
def test_scripts_recusam_subir_com_o_outro_perfil_rodando(script, trava, subida):
    texto = _texto(script).replace("\r\n", "\n")
    assert 0 < texto.index(trava) < texto.index(subida)


@pytest.mark.parametrize("script", ["Makefile", "start.bat", "start.sh"])
def test_scripts_mostram_o_endereco_que_o_compose_publicou(script):
    """O Makefile dizia `$(PORT)`, e o compose nunca leu essa variável."""
    assert re.search(r"port (\$\(SERVICE\)|inference-api) 7860", _texto(script))


def test_env_example_traz_as_variaveis_e_as_origens_de_cada_instancia():
    texto = _texto(".env.example")
    assert not re.search(r"^#?\s*GRADIO_PORT=", texto, re.M)
    assert "GRADIO_PORT_NVIDIA=" in texto
    assert "GRADIO_PORT_CPU=" in texto
    origens = re.search(r"^ALLOWED_ORIGINS=(.*)$", texto, re.M)[1].split(",")
    for porta in (ports.DOCKER_NVIDIA_PORT, ports.DOCKER_CPU_PORT, ports.LOCAL_PORT):
        assert f"http://127.0.0.1:{porta}" in origens
        assert f"http://localhost:{porta}" in origens


def test_guia_lista_todas_as_portas():
    linhas = GUIA.read_text(encoding="utf-8").splitlines()
    primeira_coluna = {
        linha.split("|")[1].strip() for linha in linhas if linha.startswith("| ")
    }
    for porta in HOST_PORTS:
        assert str(porta) in primeira_coluna


@pytest.mark.parametrize(
    "guia",
    [
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "docs/getting-started/installation.md",
        "docs/interfaces/gradio.md",
        "docs/interfaces/rest-api.md",
        "docs/reference/faq.md",
    ],
)
def test_guias_nao_voltam_ao_mapa_antigo(guia):
    texto = _texto(guia)
    assert not re.search(r"\bGRADIO_PORT\b", texto)
    assert "--gradio-port 7861" not in texto
    assert "python main.py --gradio --gradio-port 7860" not in texto


# ── Trava sobre o mesmo banco ──────────────────────────────────────────────


class _Resposta(http.server.BaseHTTPRequestHandler):
    corpo = b"{}"
    status = 200

    def do_GET(self):  # noqa: N802
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(self.corpo)

    def log_message(self, *args):
        pass


@pytest.fixture
def servidor() -> Iterator:
    def sobe(corpo: dict | bytes, status: int = 200) -> int:
        conteudo = corpo if isinstance(corpo, bytes) else json.dumps(corpo).encode()
        tratador = type("Tratador", (_Resposta,), {"corpo": conteudo, "status": status})
        instancia = http.server.ThreadingHTTPServer(("127.0.0.1", 0), tratador)
        threading.Thread(target=instancia.serve_forever, daemon=True).start()
        abertos.append(instancia)
        return instancia.server_address[1]

    abertos: list = []
    yield sobe
    for instancia in abertos:
        instancia.shutdown()
        instancia.server_close()


def _porta_livre() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


SAUDE = {"status": "healthy", "database": "ok", "models_loaded": 9}


@pytest.fixture
def banco_do_projeto(monkeypatch):
    # Só o caminho é comparado: nada abre este banco.
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./data/app.db")
    monkeypatch.setenv("GRADIO_PORT_NVIDIA", str(_porta_livre()))
    monkeypatch.setenv("GRADIO_PORT_CPU", str(_porta_livre()))


def test_aplicacao_reconhece_a_propria_resposta(servidor):
    assert ports.is_speechchecker(servidor(SAUDE))
    # Sem modelos, o health responde 503 e continua sendo esta aplicação.
    assert ports.is_speechchecker(servidor(SAUDE, status=503))


def test_outro_programa_na_porta_nao_conta_como_instancia(servidor):
    assert not ports.is_speechchecker(servidor({"status": "ok"}))
    assert not ports.is_speechchecker(servidor(b"<html>outro gradio</html>"))
    assert not ports.is_speechchecker(_porta_livre())


def test_instancia_docker_no_ar_e_conflito(servidor, monkeypatch, banco_do_projeto):
    porta = servidor(SAUDE)
    monkeypatch.setenv("GRADIO_PORT_CPU", str(porta))
    assert ports.conflicting_docker_instances(ports.LOCAL_PORT) == {
        "speechchecker_inference_cpu": porta
    }


def test_sem_instancia_no_ar_nao_ha_conflito(banco_do_projeto):
    assert ports.conflicting_docker_instances(ports.LOCAL_PORT) == {}


def test_a_propria_porta_nao_conta(servidor, monkeypatch, banco_do_projeto):
    porta = servidor(SAUDE)
    monkeypatch.setenv("GRADIO_PORT_NVIDIA", str(porta))
    assert ports.conflicting_docker_instances(porta) == {}


@pytest.mark.parametrize(
    "url", ["sqlite:///./data/outro.db", "sqlite:///:memory:", "postgresql://h/b"]
)
def test_outro_banco_libera_a_instancia_local(servidor, monkeypatch, url):
    monkeypatch.setenv("GRADIO_PORT_NVIDIA", str(servidor(SAUDE)))
    monkeypatch.setenv("DATABASE_URL", url)
    assert ports.conflicting_docker_instances(ports.LOCAL_PORT) == {}


def test_sem_database_url_vale_o_banco_padrao(servidor, monkeypatch):
    porta = servidor(SAUDE)
    monkeypatch.setenv("GRADIO_PORT_NVIDIA", str(porta))
    monkeypatch.setenv("GRADIO_PORT_CPU", str(_porta_livre()))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert ports.conflicting_docker_instances(ports.LOCAL_PORT) == {
        "speechchecker_inference_nvidia": porta
    }
