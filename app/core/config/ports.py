"""Mapa único das portas do projeto.

Tabela e motivos em ``docs/development/docker-environments.md``, seção Portas.
Compose, Makefile e scripts de shell não importam este módulo e repetem os
valores; ``tests/unit/test_ports.py`` confere que continuam iguais.
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path

from app.core.config.env import PROJECT_ROOT, env_int

#: Dentro de qualquer container e no Hugging Face Spaces. Não muda por perfil.
CONTAINER_PORT = 7860

#: Publicadas no host por cada perfil de inferência.
DOCKER_NVIDIA_PORT = 7860
DOCKER_CPU_PORT = 7861

#: Aplicação local, sem Docker.
LOCAL_PORT = 7862

#: Instâncias descartáveis de ``docker_release.py check``.
RELEASE_CHECK_PORTS = {"nvidia": 17860, "cpu": 17861}

#: ``mkdocs serve``.
DOCS_PORT = 8000


def docker_instance_ports() -> dict[str, int]:
    """Porta publicada por container, com os ajustes do ambiente e do ``.env``."""
    return {
        "speechchecker_inference_nvidia": env_int(
            "GRADIO_PORT_NVIDIA", DOCKER_NVIDIA_PORT
        ),
        "speechchecker_inference_cpu": env_int("GRADIO_PORT_CPU", DOCKER_CPU_PORT),
    }


def is_speechchecker(port: int, timeout: float = 1.0) -> bool:
    """Diz se a porta responde como esta aplicação, e não só se está ocupada.

    A 7860 é o padrão do Gradio: outro projeto da máquina pode estar nela.
    """
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", "/api/v1/system/health")
        health = json.loads(connection.getresponse().read())
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()
    return isinstance(health, dict) and {"database", "models_loaded"} <= health.keys()


def _uses_project_database() -> bool:
    url = os.getenv("DATABASE_URL")
    if not url:
        return True
    if not url.startswith("sqlite:///"):
        return False
    target = Path(url.removeprefix("sqlite:///"))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve() == (PROJECT_ROOT / "data" / "app.db").resolve()


def conflicting_docker_instances(own_port: int) -> dict[str, int]:
    """Instâncias Docker no ar que usam o mesmo SQLite que este processo usaria.

    O modo WAL exige processos na mesma máquina, e aqui são duas: o Windows e a
    VM do WSL. Quando tudo usava a 7860, era a porta ocupada que impedia isso.
    """
    if not _uses_project_database():
        return {}
    return {
        name: port
        for name, port in docker_instance_ports().items()
        if port != own_port and is_speechchecker(port)
    }
