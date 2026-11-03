"""Deploy em Docker: o que a imagem leva, de onde vem o `.env` e o que o compose publica."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "infrastructure" / "docker" / "compose"
INFERENCE_API = ROOT / "infrastructure" / "docker" / "environments" / "inference-api"
PERFIS_INFERENCIA = ("inference.cpu.yml", "inference.nvidia.yml")
DOCKERFILES_INFERENCIA = ("Dockerfile.cpu", "Dockerfile.nvidia")


def _regex(padrao: str) -> re.Pattern[str]:
    saida = ""
    i = 0
    while i < len(padrao):
        if padrao.startswith("**/", i):
            saida += "(?:.*/)?"
            i += 3
        elif padrao.startswith("**", i):
            saida += ".*"
            i += 2
        elif padrao[i] == "*":
            saida += "[^/]*"
            i += 1
        elif padrao[i] == "?":
            saida += "[^/]"
            i += 1
        elif padrao[i] == "[":
            fim = padrao.index("]", i)
            saida += padrao[i : fim + 1]
            i = fim + 1
        else:
            saida += re.escape(padrao[i])
            i += 1
    return re.compile(f"^{saida}$")


def _regras() -> list[tuple[bool, re.Pattern[str]]]:
    regras = []
    for linha in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        negado = linha.startswith("!")
        regras.append((negado, _regex(linha.lstrip("!").strip().strip("/"))))
    return regras


def _ignorado(caminho: str) -> bool:
    """Regra do BuildKit: vence o último padrão que casa o arquivo ou uma pasta acima."""
    partes = caminho.split("/")
    alvos = ["/".join(partes[: k + 1]) for k in range(len(partes))]
    resultado = False
    for negado, padrao in _regras():
        if any(padrao.match(alvo) for alvo in alvos):
            resultado = not negado
    return resultado


@pytest.mark.parametrize(
    "caminho",
    [
        ".env",
        ".env.local",
        "infrastructure/docker/.env",
        "data/app.db",
        "data/app.db-wal",
        "data/temp/gradio/abc/voz.wav",
        "data/voice_profiles/perfil.json",
        "data/uploads/audio.wav",
        ".claude/launch.json",
        "certs/servidor.pem",
        "certs/servidor.key",
        "data/models/bench_aasist_config.json.pre-temperature.bak",
        "data/backups/20261103T090000Z-nvidia/app.db",
        "app/core/__pycache__/security.cpython-311.pyc",
        "tools/ops/__pycache__/doctor.cpython-311.pyc",
    ],
)
def test_contexto_nao_leva_segredos_nem_dados(caminho):
    assert _ignorado(caminho), f"{caminho} entraria na imagem"


@pytest.mark.parametrize(
    "caminho",
    [
        "main.py",
        "constraints.txt",
        "requirements-base.txt",
        "app/core/security.py",
        "benchmarks/runner.py",
        "tools/ops/sync_hf_models.py",
        "infrastructure/docker/environments/inference-api/docker-entrypoint.sh",
        "infrastructure/docker/environments/inference-api/constraints-cpu.txt",
        "infrastructure/docker/environments/inference-api/constraints-nvidia.txt",
        "data/models/bench_aasist.keras",
        "data/models/bench_aasist_config.json",
    ],
)
def test_contexto_mantem_o_que_a_imagem_usa(caminho):
    assert not _ignorado(caminho), f"{caminho} ficaria fora da imagem"


@pytest.mark.parametrize("dockerfile", DOCKERFILES_INFERENCIA)
def test_inferencia_usa_o_ignore_da_raiz(dockerfile):
    """Um `<dockerfile>.dockerignore` substitui o da raiz, inclusive a regra do `.env`."""
    assert not (INFERENCE_API / f"{dockerfile}.dockerignore").exists()


@pytest.mark.parametrize("perfil", PERFIS_INFERENCIA)
def test_inferencia_roda_em_producao_e_so_na_maquina_local(perfil):
    texto = (COMPOSE / perfil).read_text(encoding="utf-8")
    assert re.search(r"^\s+DEEPFAKE_ENV: production$", texto, re.M)
    assert '"${GRADIO_BIND_HOST:-127.0.0.1}:${GRADIO_PORT_' in texto


@pytest.mark.parametrize("perfil", PERFIS_INFERENCIA)
def test_inferencia_define_o_fuso_das_datas_da_tela(perfil):
    """Sem TZ o container fica em UTC, e o Histórico mostra a hora adiantada."""
    texto = (COMPOSE / perfil).read_text(encoding="utf-8")
    assert re.search(r"^\s+TZ: \$\{TZ:-America/Sao_Paulo\}$", texto, re.M)


@pytest.mark.parametrize("perfil", PERFIS_INFERENCIA)
def test_inferencia_recebe_o_repositorio_de_modelos_e_o_token(perfil):
    """O entrypoint sincroniza do Hub, mas o compose não repassava estes nomes."""
    texto = (COMPOSE / perfil).read_text(encoding="utf-8")
    assert re.search(r"^\s+MODEL_REPO_ID: \$\{MODEL_REPO_ID:-\}$", texto, re.M)
    assert re.search(r"^\s+HF_TOKEN: \$\{HF_TOKEN:-", texto, re.M)
    assert re.search(r"^\s+XFAKE_SYNC_MODELS_ON_BOOT: ", texto, re.M)


@pytest.mark.parametrize("perfil", PERFIS_INFERENCIA)
def test_comentario_do_compose_nao_cita_contagem_de_modelos(perfil):
    """A contagem envelhece: já foram 13 e 11, e hoje `data/models` tem outra."""
    texto = (COMPOSE / perfil).read_text(encoding="utf-8")
    assert "benchmark_final" not in texto
    assert not re.search(r"\b\d+ modelos\b", texto)


AMBIENTES = ROOT / "infrastructure" / "docker" / "environments"


@pytest.mark.parametrize(
    "dockerfile",
    sorted(p.relative_to(AMBIENTES).as_posix() for p in AMBIENTES.glob("*/Dockerfile.nvidia")),
)
def test_imagem_nvidia_publica_as_bibliotecas_cuda_do_tensorflow(dockerfile):
    """Sem isto o TensorFlow roda em CPU sem nenhum erro, como ocorria no ssl-transformers."""
    texto = (AMBIENTES / dockerfile).read_text(encoding="utf-8")
    assert "tensorflow[and-cuda]" in texto
    assert re.search(r"^ENV LD_LIBRARY_PATH=.*cudnn/lib", texto, re.M)


@pytest.mark.parametrize(
    "dockerfile",
    sorted(p.relative_to(AMBIENTES).as_posix() for p in AMBIENTES.glob("*/Dockerfile.*") if p.suffix != ".dockerignore"),
)
def test_imagem_nao_guarda_o_cache_do_pip(dockerfile):
    """O cache chegava a 6,4 GB por imagem, sem uso em execução."""
    texto = (AMBIENTES / dockerfile).read_text(encoding="utf-8")
    assert "pip cache purge" in texto or "PIP_NO_CACHE_DIR=1" in texto


CREDENCIAIS = ("SPEECHCHECKER_API_KEY", "XFAKE_GRADIO_USERNAME", "XFAKE_GRADIO_PASSWORD")


def test_perfis_leem_as_mesmas_credenciais_sem_senha_de_reserva():
    """Mesmo login em todo perfil. Sem `.env`, o compose recusa subir."""
    linhas = [
        sorted(
            linha.strip()
            for linha in (COMPOSE / perfil).read_text(encoding="utf-8").splitlines()
            if linha.strip().startswith(CREDENCIAIS)
        )
        for perfil in PERFIS_INFERENCIA
    ]
    assert linhas[0] == linhas[1]
    assert len(linhas[0]) == len(CREDENCIAIS)
    for linha in linhas[0]:
        assert ":?Defina" in linha


@pytest.mark.parametrize(
    "dockerfile",
    [
        ROOT / "Dockerfile",
        *sorted(p for p in AMBIENTES.glob("*/Dockerfile.*") if p.suffix != ".dockerignore"),
    ],
    ids=lambda p: p.relative_to(ROOT).as_posix(),
)
def test_imagem_nao_guarda_credenciais(dockerfile):
    """O login só entra ao criar o container, e por isso sobrevive a um rebuild."""
    texto = dockerfile.read_text(encoding="utf-8")
    for nome in CREDENCIAIS:
        assert nome not in texto


@pytest.mark.parametrize("dockerfile", DOCKERFILES_INFERENCIA)
def test_healthcheck_nao_cai_na_raiz_depois_da_falha(dockerfile):
    """`/` responde mesmo sem banco ou sem modelos e escondia o 503 de `/health`."""
    texto = (INFERENCE_API / dockerfile).read_text(encoding="utf-8")
    bloco = texto.split("HEALTHCHECK", 1)[1].split("ENTRYPOINT", 1)[0]
    assert "/api/v1/system/health" in bloco
    assert not re.search(r"7860/\s", bloco)


def test_makefile_passa_o_env_em_todo_comando_de_compose():
    texto = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^COMPOSE_FILES := \$\(ENV_FLAG\) ", texto, re.M)
    receitas = re.findall(r"^\t\$\(DC\) (.*)$", texto, re.M)
    assert receitas
    for receita in receitas:
        assert receita.startswith(("$(ENV_FLAG)", "$(COMPOSE_FILES)")), receita


@pytest.mark.parametrize("script", ["Makefile", "start.bat", "start.sh"])
def test_limpeza_nao_alcanca_outros_projetos_da_maquina(script):
    linhas = (ROOT / script).read_text(encoding="utf-8").splitlines()
    texto = "\n".join(
        linha for linha in linhas if not linha.lstrip().startswith(("#", "REM "))
    )
    assert "docker builder prune" not in texto
    comandos = re.findall(r"docker image prune[^\n]*", texto)
    assert comandos
    for comando in comandos:
        assert "label=org.opencontainers.image.title=speechchecker-inference-api" in comando


@pytest.mark.parametrize(
    ("script", "condicao"),
    [
        ("start.bat", "IF NOT ERRORLEVEL 1 EXIT /B 0"),
        ("start.sh", "docker info >/dev/null 2>&1 && return 0"),
    ],
)
def test_scripts_tentam_iniciar_o_docker_desktop(script, condicao):
    texto = (ROOT / script).read_text(encoding="utf-8")
    assert texto.index(condicao) < texto.index("docker desktop start")


def test_makefile_expoe_o_caminho_de_volta():
    texto = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "tools/ops/docker_release.py snapshot $(RELEASE_PROFILE)" in texto
    assert "tools/ops/docker_release.py rollback $(RELEASE_PROFILE)" in texto


@pytest.mark.parametrize(
    ("script", "flag", "subida"),
    [
        ("start.bat", '--env-file .env"', "up --build -d"),
        ("start.sh", "COMPOSE_CMD+=(--env-file .env)", "up --build -d"),
    ],
)
def test_scripts_passam_o_env_e_validam_antes_do_build(script, flag, subida):
    texto = (ROOT / script).read_text(encoding="utf-8")
    assert flag in texto
    assert 0 < texto.index("config --quiet") < texto.index(subida)


@pytest.mark.parametrize(
    "guia",
    [
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "docs/development/docker-environments.md",
        "docs/getting-started/installation.md",
        "docs/interfaces/gradio.md",
        "docs/reference/faq.md",
    ],
)
def test_guias_nao_mandam_rodar_compose_sem_arquivo(guia):
    """Não há compose na raiz: estes comandos falham com `no configuration file`."""
    texto = (ROOT / guia).read_text(encoding="utf-8")
    assert not re.search(r"docker compose (up|build|down|logs|ps)\b", texto)
