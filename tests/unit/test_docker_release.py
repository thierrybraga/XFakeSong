"""Caminho de volta da imagem de inferência e versões fixadas, sem precisar de Docker."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.ops import docker_release as dr

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_API = ROOT / "infrastructure" / "docker" / "environments" / "inference-api"


def _pinos(caminho: Path) -> dict[str, str]:
    pinos = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        achado = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", linha)
        if achado:
            pinos[re.sub(r"[-_.]+", "-", achado[1]).lower()] = achado[2]
    return pinos


class DockerFalso:
    """Guarda as chamadas e simula as tags. Nada aqui encosta em um Docker real."""

    def __init__(self, tags=None, rodando=False, saude="healthy"):
        self.tags = dict(tags or {})
        self.rodando = rodando
        self.saude = saude
        self.chamadas: list[tuple[str, ...]] = []
        self.compose: list[list[str]] = []

    def __call__(self, *args, check=True, env=None):
        self.chamadas.append(args)
        saida, codigo = "", 0
        if args[:2] == ("image", "inspect"):
            nome = args[-1]
            codigo = 0 if nome in self.tags else 1
            saida = self.tags.get(nome, "")
        elif args[0] == "tag":
            self.tags[args[2]] = self.tags[args[1]]
        elif args[0] == "inspect":
            if "{{.State.Health" in args[2]:
                saida = f"{str(self.rodando).lower()} {self.saude}"
            else:
                saida = str(self.rodando).lower()
        return subprocess.CompletedProcess(args, codigo, saida, "")

    def run_compose(self, cmd, **kwargs):
        self.compose.append(cmd)
        self.rodando = True
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def docker(monkeypatch):
    def fabrica(**kwargs):
        falso = DockerFalso(**kwargs)
        monkeypatch.setattr(dr, "_docker", falso)
        monkeypatch.setattr(dr.subprocess, "run", falso.run_compose)
        return falso

    return fabrica


def _dados(tmp_path: Path) -> Path:
    dados = tmp_path / "data"
    dados.mkdir()
    (dados / "app.db").write_text("banco", encoding="utf-8")
    (dados / "app.db-wal").write_text("wal", encoding="utf-8")
    return dados


def test_snapshot_guarda_a_imagem_e_copia_o_banco_com_o_wal(docker, tmp_path):
    falso = docker(tags={dr.tag("cpu"): "sha256:a"})
    dados = _dados(tmp_path)

    assert dr.snapshot("cpu", dados) == 0

    assert falso.tags[dr.tag_guardada("cpu")] == "sha256:a"
    copia = next((dados / "backups").iterdir())
    assert (copia / "app.db").read_text(encoding="utf-8") == "banco"
    assert (copia / "app.db-wal").read_text(encoding="utf-8") == "wal"
    registro = json.loads((copia / "snapshot.json").read_text(encoding="utf-8"))
    assert registro["image_id"] == "sha256:a"
    assert registro["database_files"] == ["app.db", "app.db-wal"]


def test_snapshot_recusa_copiar_banco_com_o_container_rodando(docker, tmp_path):
    falso = docker(tags={dr.tag("cpu"): "sha256:a"}, rodando=True)
    dados = _dados(tmp_path)

    with pytest.raises(SystemExit, match="está rodando"):
        dr.snapshot("cpu", dados)

    assert dr.tag_guardada("cpu") not in falso.tags
    assert not (dados / "backups").exists()


def test_snapshot_sem_banco_funciona_com_o_container_rodando(docker, tmp_path):
    falso = docker(tags={dr.tag("cpu"): "sha256:a"}, rodando=True)
    dados = _dados(tmp_path)

    assert dr.snapshot("cpu", dados, sem_banco=True) == 0

    assert falso.tags[dr.tag_guardada("cpu")] == "sha256:a"
    assert not (dados / "backups").exists()


def test_snapshot_sem_imagem_para(docker, tmp_path):
    docker()
    with pytest.raises(SystemExit, match="não existe"):
        dr.snapshot("cpu", tmp_path)


def test_rollback_troca_a_imagem_e_recria_sem_build(docker):
    falso = docker(tags={dr.tag("nvidia"): "nova", dr.tag_guardada("nvidia"): "velha"})

    assert dr.rollback("nvidia") == 0

    assert falso.tags[dr.tag("nvidia")] == "velha"
    assert falso.tags[dr.tag_revertida("nvidia")] == "nova"
    comando = falso.compose[0]
    assert comando[-4:] == ["-d", "--no-build", "--force-recreate", dr.SERVICO]
    assert "infrastructure/docker/compose/inference.nvidia.yml" in comando


def test_rollback_sem_imagem_guardada_nao_toca_em_nada(docker):
    falso = docker(tags={dr.tag("nvidia"): "nova"})

    with pytest.raises(SystemExit, match="snapshot"):
        dr.rollback("nvidia")

    assert falso.tags == {dr.tag("nvidia"): "nova"}
    assert falso.compose == []


def test_rollback_que_nao_fica_saudavel_devolve_erro(docker, capsys):
    docker(
        tags={dr.tag("cpu"): "nova", dr.tag_guardada("cpu"): "velha"},
        saude="unhealthy",
    )
    assert dr.rollback("cpu") == 1
    assert "unhealthy" in capsys.readouterr().out


def _consulta(**trocas):
    respostas = {
        "/api/v1/system/health": (
            200,
            '{"database": "ok", "models_loaded": 9, "storage_available": true}',
            {},
        ),
        "/": (307, "", {"location": "/gradio/"}),
        "/gradio/config": (401, "", {}),
        "/api/v1/system/status": (400, "", {}),
        **trocas,
    }

    def consulta(caminho, cabecalhos):
        if caminho == "/api/v1/history/":
            return (200 if cabecalhos.get("X-API-Key") == "chave" else 403), "", {}
        return respostas[caminho]

    return consulta


def test_conferir_http_aceita_imagem_saudavel():
    assert dr.conferir_http(_consulta(), "chave") == []


def test_conferir_http_aceita_503_quando_a_imagem_nasce_sem_modelos():
    corpo = '{"database": "ok", "models_loaded": 0, "storage_available": true}'
    consulta = _consulta(**{"/api/v1/system/health": (503, corpo, {})})
    assert dr.conferir_http(consulta, "chave") == []


@pytest.mark.parametrize(
    ("status", "corpo"),
    [
        (200, '{"database": "ok", "models_loaded": 0, "storage_available": true}'),
        (503, '{"database": "ok", "models_loaded": 9, "storage_available": true}'),
        (503, '{"database": "error", "models_loaded": 0, "storage_available": true}'),
        (500, "falhou"),
    ],
)
def test_conferir_http_recusa_health_incoerente(status, corpo):
    consulta = _consulta(**{"/api/v1/system/health": (status, corpo, {})})
    assert dr.conferir_http(consulta, "chave")


def test_conferir_http_acusa_rota_aberta_sem_login():
    consulta = _consulta(**{"/gradio/config": (200, "{}", {})})
    assert dr.conferir_http(consulta, "chave") == ["/gradio/config: 200, esperado 401"]


def test_versoes_da_imagem_deixa_cuda_de_fora_e_ordena():
    freeze = (
        "torch==2.5.1\n"
        "nvidia-cudnn-cu12==9.27.0.42\n"
        "nvidia-ml-py==13.615.71\n"
        "Absl-py==2.5.1\n"
        "\n"
    )
    assert dr.versoes_da_imagem(freeze) == [
        "Absl-py==2.5.1",
        "nvidia-ml-py==13.615.71",
        "torch==2.5.1",
    ]


@pytest.mark.parametrize("perfil", dr.PERFIS)
def test_arquivo_de_versoes_nao_contradiz_o_constraints_da_raiz(perfil):
    versoes = _pinos(dr.arquivo_de_versoes(perfil))
    for nome, versao in _pinos(ROOT / "constraints.txt").items():
        if nome in versoes:
            assert versoes[nome] == versao, nome


@pytest.mark.parametrize("perfil", dr.PERFIS)
def test_arquivo_de_versoes_fixa_o_essencial_e_nao_os_pacotes_cuda(perfil):
    versoes = _pinos(dr.arquivo_de_versoes(perfil))
    assert {"torch", "keras", "pip", "setuptools", "wheel"} <= set(versoes)
    assert "tensorflow-cpu" in versoes or "tensorflow" in versoes
    assert not [nome for nome in versoes if nome.endswith("-cu12")]


@pytest.mark.parametrize(
    ("dockerfile", "perfil"),
    [("Dockerfile.cpu", "cpu"), ("Dockerfile.nvidia", "nvidia")],
)
def test_dockerfile_usa_as_versoes_fixas_e_nao_deixa_cache(dockerfile, perfil):
    texto = (INFERENCE_API / dockerfile).read_text(encoding="utf-8")
    assert (
        f"PIP_CONSTRAINT=/app/{dr.arquivo_de_versoes(perfil).relative_to(ROOT).as_posix()}"
        in texto
    )
    assert f"inference-api/constraints-{perfil}.txt infrastructure/" in texto
    bloco = texto.split("RUN pip install", 1)[1].split("COPY ", 1)[0]
    assert "pip cache purge" in bloco
    for especificacao in re.findall(r"'([^']+)'", bloco):
        assert "==" in especificacao and ">" not in especificacao, especificacao


def test_ci_constroi_e_confere_as_duas_imagens_de_inferencia():
    caminho = ROOT / ".github" / "workflows" / "docker-inference.yml"
    workflow = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    job = workflow["jobs"]["imagem"]
    assert job["strategy"]["matrix"]["perfil"] == list(dr.PERFIS)
    comandos = [passo.get("run", "") for passo in job["steps"]]
    assert any(
        "docker_build.py inference-${{ matrix.perfil }} build" in c for c in comandos
    )
    assert any("docker_release.py check ${{ matrix.perfil }}" in c for c in comandos)
    # `on` vira True no YAML 1.1 do PyYAML.
    gatilhos = workflow[True]
    caminhos = gatilhos["pull_request"]["paths"]
    assert "infrastructure/docker/environments/inference-api/**" in caminhos
    assert "requirements-base.txt" in caminhos
    assert workflow["permissions"] == {"contents": "read"}
