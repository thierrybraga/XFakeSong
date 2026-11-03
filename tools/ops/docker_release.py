#!/usr/bin/env python3
"""Atualizar a imagem de inferência sem perder o caminho de volta.

Reconstruir sobrescreve a tag `speechchecker/inference-api:<perfil>`, e a imagem
anterior deixa de existir. Por isso a ordem é guardar, construir e, se der errado,
voltar. Só usa a biblioteca padrão, para rodar em qualquer shell e na CI.

Uso:
    python tools/ops/docker_release.py snapshot nvidia   # com o container parado
    python tools/ops/docker_release.py rollback nvidia   # volta e espera o health check
    python tools/ops/docker_release.py check nvidia      # confere uma imagem, como a CI
    python tools/ops/docker_release.py lock nvidia       # regrava as versões fixadas
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERFIS = ("cpu", "nvidia")
REPOSITORIO = "speechchecker/inference-api"
SERVICO = "inference-api"
ARQUIVOS_DO_BANCO = ("app.db", "app.db-wal", "app.db-shm")
# Fora das portas das instâncias, para conferir uma imagem com a aplicação no ar.
PORTAS_CHECK = {"nvidia": 17860, "cpu": 17861}

# start_period (180 s) + 5 falhas a cada 30 s: só depois disso vira `unhealthy`.
ESPERA_SAUDE_S = 420
ESPERA_RESPOSTA_S = 300

# Não entram nos arquivos de versão: tensorflow[and-cuda] e torch pedem versões
# diferentes desses pacotes, e um pino único quebraria um dos dois `pip install`.
# Com TF e torch fixos, o torch define quase todos, e o Dockerfile fixa o resto.
_CUDA = re.compile(r"^nvidia-[a-z-]+-cu12==", re.I)

# Caminhos relativos a /app que não podem entrar na imagem.
AUSENTES_NA_IMAGEM = (
    ".env",
    ".claude",
    "CLAUDE.md",
    "data/app.db",
    "data/temp",
    "data/uploads",
    "data/voice_profiles",
    "data/results",
    "data/backups",
    "tests",
    "docs",
)

Consulta = Callable[..., "tuple[int, str, dict[str, str]]"]


def tag(perfil: str) -> str:
    return f"{REPOSITORIO}:{perfil}"


def tag_guardada(perfil: str) -> str:
    return f"{REPOSITORIO}:{perfil}-previous"


def tag_revertida(perfil: str) -> str:
    return f"{REPOSITORIO}:{perfil}-reverted"


def container(perfil: str) -> str:
    return f"speechchecker_inference_{perfil}"


def arquivo_de_versoes(perfil: str) -> Path:
    return (
        ROOT
        / "infrastructure/docker/environments/inference-api"
        / f"constraints-{perfil}.txt"
    )


def _docker(
    *args: str, check: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def imagem_id(nome: str) -> str | None:
    """O id da imagem, ou None quando ela não existe."""
    r = _docker("image", "inspect", "--format", "{{.Id}}", nome, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def esta_rodando(perfil: str) -> bool:
    r = _docker(
        "inspect", "--format", "{{.State.Running}}", container(perfil), check=False
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def comando_compose(perfil: str, extras: Iterable[Path] = ()) -> list[str]:
    cmd = ["docker", "compose"]
    env_file = ROOT / ".env"
    if env_file.is_file():
        cmd += ["--env-file", str(env_file)]
    cmd += ["-f", f"infrastructure/docker/compose/inference.{perfil}.yml"]
    for extra in extras:
        cmd += ["-f", str(extra)]
    return cmd


def copiar_banco(perfil: str, dados: Path, imagem: str) -> Path:
    """Copia o banco com os arquivos do WAL, que juntos formam o estado atual."""
    momento = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destino = dados / "backups" / f"{momento}-{perfil}"
    destino.mkdir(parents=True)
    copiados = []
    for nome in ARQUIVOS_DO_BANCO:
        origem = dados / nome
        if origem.is_file():
            shutil.copy2(origem, destino / nome)
            copiados.append(nome)
    (destino / "snapshot.json").write_text(
        json.dumps(
            {"profile": perfil, "image_id": imagem, "database_files": copiados},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destino


def snapshot(perfil: str, dados: Path, sem_banco: bool = False) -> int:
    imagem = imagem_id(tag(perfil))
    if imagem is None:
        raise SystemExit(f"A imagem {tag(perfil)} não existe: não há o que guardar.")
    if not sem_banco and esta_rodando(perfil):
        raise SystemExit(
            f"{container(perfil)} está rodando, e copiar o SQLite com a aplicação "
            "escrevendo pode gerar um banco inconsistente. Pare o container ou "
            "use --sem-banco."
        )
    # Banco primeiro: se o disco encher aqui, a imagem guardada antes continua valendo.
    if not sem_banco:
        print(f"Banco copiado para {copiar_banco(perfil, dados, imagem)}")
    _docker("tag", tag(perfil), tag_guardada(perfil))
    print(f"Imagem guardada em {tag_guardada(perfil)}")
    return 0


def _saude(perfil: str) -> str:
    r = _docker(
        "inspect",
        "--format",
        "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
        container(perfil),
        check=False,
    )
    partes = r.stdout.split()
    if r.returncode != 0 or not partes or partes[0] != "true":
        return "parado"
    return partes[1] if len(partes) > 1 else "sem-healthcheck"


def esperar_saude(perfil: str, limite_s: float = ESPERA_SAUDE_S) -> str:
    fim = time.monotonic() + limite_s
    while True:
        estado = _saude(perfil)
        if estado in ("healthy", "unhealthy", "parado", "sem-healthcheck"):
            return estado
        if time.monotonic() >= fim:
            return "sem resposta"
        time.sleep(5)


def rollback(
    perfil: str, extras: Iterable[Path] = (), limite_s: float = ESPERA_SAUDE_S
) -> int:
    if imagem_id(tag_guardada(perfil)) is None:
        raise SystemExit(
            f"Não há {tag_guardada(perfil)}. Rode `snapshot` antes de reconstruir."
        )
    # Mantém o que estava em uso, para investigar a falha ou desfazer a volta.
    if imagem_id(tag(perfil)) is not None:
        _docker("tag", tag(perfil), tag_revertida(perfil))
    _docker("tag", tag_guardada(perfil), tag(perfil))
    subprocess.run(
        [
            *comando_compose(perfil, extras),
            "up",
            "-d",
            "--no-build",
            "--force-recreate",
            SERVICO,
        ],
        cwd=str(ROOT),
        check=True,
    )
    estado = esperar_saude(perfil, limite_s)
    print(f"{container(perfil)}: {estado}")
    if estado != "healthy":
        logs = _docker("logs", "--tail", "40", container(perfil), check=False)
        print(logs.stdout + logs.stderr)
        return 1
    print(f"A imagem anterior à volta continua em {tag_revertida(perfil)}.")
    return 0


def versoes_da_imagem(freeze: str) -> list[str]:
    linhas = (linha.strip() for linha in freeze.splitlines())
    fixas = [
        linha
        for linha in linhas
        if linha and not linha.startswith("#") and not _CUDA.match(linha)
    ]
    return sorted(fixas, key=str.lower)


def lock(perfil: str) -> int:
    """Grava as versões de uma imagem já validada. Rode só depois de validá-la."""
    if imagem_id(tag(perfil)) is None:
        raise SystemExit(f"A imagem {tag(perfil)} não existe.")
    saida = _docker(
        "run", "--rm", "--network", "none", "--entrypoint", "pip", tag(perfil),
        "freeze", "--all",
    ).stdout  # fmt: skip
    destino = arquivo_de_versoes(perfil)
    cabecalho = (
        f"# Versões exatas da imagem de inferência {perfil}.\n"
        f"# Gerado por `python tools/ops/docker_release.py lock {perfil}`. Não edite.\n"
        "# nvidia-*-cu12 fica de fora: ver _CUDA em docker_release.py.\n"
    )
    destino.write_text(
        cabecalho + "\n".join(versoes_da_imagem(saida)) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{destino.relative_to(ROOT)} atualizado.")
    return 0


def problemas_de_conteudo(presentes: Iterable[str]) -> list[str]:
    return [f"{item} está dentro da imagem" for item in presentes]


def conferir_conteudo(imagem: str) -> list[str]:
    script = (
        'cd /app; for p in "$@"; do [ -e "$p" ] && echo "$p"; done; '
        "find /app -name .env | sed 's#^/app/##'; "
        # `pip cache purge` deixa a pasta com um registro de poucos KB: o que importa é o peso.
        '[ "$(du -sm /root/.cache/pip 2>/dev/null | cut -f1)" -gt 10 ] 2>/dev/null '
        "&& echo /root/.cache/pip; true"
    )
    r = _docker(
        "run", "--rm", "--network", "none", "--user", "0", "--entrypoint", "sh",
        imagem, "-c", script, "sh", *AUSENTES_NA_IMAGEM,
    )  # fmt: skip
    presentes = dict.fromkeys(linha for linha in r.stdout.splitlines() if linha)
    problemas = problemas_de_conteudo(presentes)
    usuario = _docker("image", "inspect", "--format", "{{.Config.User}}", imagem)
    if usuario.stdout.strip() != "appuser":
        problemas.append(f"a imagem roda como {usuario.stdout.strip() or 'root'}")
    return problemas


def conferir_http(consulta: Consulta, chave: str) -> list[str]:
    """O que uma imagem recém-subida precisa responder, com ou sem modelos embutidos.

    Na CI a imagem nasce sem modelos, porque eles não estão no repositório, e aí o
    health check tem de responder 503: é o que mantém o container `unhealthy`.
    """
    problemas = []

    def espera(esperado: int, caminho: str, **cabecalhos: str) -> tuple[int, str, dict]:
        resposta = consulta(caminho, cabecalhos)
        if resposta[0] != esperado:
            problemas.append(f"{caminho}: {resposta[0]}, esperado {esperado}")
        return resposta

    status, corpo, _ = consulta("/api/v1/system/health", {})
    try:
        saude = json.loads(corpo)
    except ValueError:
        saude = {}
    modelos = saude.get("models_loaded")
    if status not in (200, 503) or saude.get("database") != "ok":
        problemas.append(f"health: {status} {corpo[:200]}")
    elif (status == 200) != (bool(modelos) and saude.get("storage_available") is True):
        problemas.append(f"health {status} incoerente com {corpo[:200]}")

    _, _, cabecalhos = espera(307, "/")
    if cabecalhos.get("location") != "/gradio/":
        problemas.append(f"/ redireciona para {cabecalhos.get('location')}")
    espera(401, "/gradio/config")
    espera(403, "/api/v1/history/")
    espera(200, "/api/v1/history/", **{"X-API-Key": chave})
    espera(400, "/api/v1/system/status", Host="site-malicioso.example")
    return problemas


def _consulta_em(porta: int) -> Consulta:
    def consulta(caminho: str, cabecalhos: dict[str, str]) -> tuple[int, str, dict]:
        conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=30)
        try:
            conexao.request("GET", caminho, headers=cabecalhos)
            r = conexao.getresponse()
            corpo = r.read().decode("utf-8", "replace")
            return r.status, corpo, {k.lower(): v for k, v in r.getheaders()}
        finally:
            conexao.close()

    return consulta


def conferir_execucao(perfil: str, imagem: str, porta: int) -> list[str]:
    """Sobe a imagem sem GPU e sem volumes, com credenciais que só existem aqui."""
    nome = f"speechchecker_check_{perfil}"
    chave = secrets.token_urlsafe(24)
    ambiente = {
        "DEEPFAKE_ENV": "production",
        "SPEECHCHECKER_API_KEY": chave,
        "XFAKE_GRADIO_USERNAME": "verificacao",
        "XFAKE_GRADIO_PASSWORD": secrets.token_urlsafe(24),
        "ALLOWED_HOSTS": "localhost",
        "ALLOWED_ORIGINS": "http://localhost",
        "DATABASE_URL": "sqlite:////app/data/app.db",
        "XFAKE_CREATE_DEFAULT_MODELS": "false",
        "XFAKE_SYNC_MODELS_ON_BOOT": "false",
    }
    _docker("rm", "-f", nome, check=False)
    # `-e NOME` sem valor lê do ambiente: as credenciais não vão à linha de comando.
    nomes = [item for chave_env in ambiente for item in ("-e", chave_env)]
    _docker(
        "run", "-d", "--name", nome, "-p", f"127.0.0.1:{porta}:7860", *nomes, imagem,
        env={**os.environ, **ambiente},
    )  # fmt: skip
    consulta = _consulta_em(porta)
    try:
        fim = time.monotonic() + ESPERA_RESPOSTA_S
        while True:
            try:
                consulta("/api/v1/system/health", {})
                break
            except OSError:
                pronto = _docker(
                    "inspect", "--format", "{{.State.Running}}", nome, check=False
                )
                if pronto.stdout.strip() != "true" or time.monotonic() >= fim:
                    logs = _docker("logs", "--tail", "40", nome, check=False)
                    return [f"não respondeu:\n{logs.stdout}{logs.stderr}"]
                time.sleep(3)
        problemas = conferir_http(consulta, chave)
        if problemas:
            logs = _docker("logs", "--tail", "40", nome, check=False)
            print(logs.stdout + logs.stderr)
        return problemas
    finally:
        _docker("rm", "-f", nome, check=False)


def check(perfil: str, porta: int) -> int:
    imagem = tag(perfil)
    if imagem_id(imagem) is None:
        raise SystemExit(f"A imagem {imagem} não existe.")
    problemas = conferir_conteudo(imagem) + conferir_execucao(perfil, imagem, porta)
    for problema in problemas:
        print(f"FALHOU: {problema}")
    if problemas:
        return 1
    print(f"{imagem}: conteúdo e respostas conferidos.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="acao", required=True)

    p = sub.add_parser("snapshot", help="guarda a imagem atual e copia o banco")
    p.add_argument("perfil", choices=PERFIS)
    p.add_argument("--dados", type=Path, default=ROOT / "data")
    p.add_argument("--sem-banco", action="store_true", help="guarda só a imagem")

    p = sub.add_parser("rollback", help="volta para a imagem guardada")
    p.add_argument("perfil", choices=PERFIS)
    p.add_argument(
        "--compose-extra",
        type=Path,
        action="append",
        default=[],
        help="override de compose em uso, como no `up` original",
    )

    p = sub.add_parser("check", help="confere conteúdo e respostas de uma imagem")
    p.add_argument("perfil", choices=PERFIS)
    p.add_argument("--porta", type=int, help="padrão: ver PORTAS_CHECK")

    p = sub.add_parser("lock", help="grava as versões da imagem validada")
    p.add_argument("perfil", choices=PERFIS)

    args = parser.parse_args(argv)
    if args.acao == "snapshot":
        return snapshot(args.perfil, args.dados, args.sem_banco)
    if args.acao == "rollback":
        return rollback(args.perfil, args.compose_extra)
    if args.acao == "check":
        return check(args.perfil, args.porta or PORTAS_CHECK[args.perfil])
    return lock(args.perfil)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
