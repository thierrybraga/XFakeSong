"""Primeira execução sem artefatos e bootstrap idempotente."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.bootstrap import (
    BootstrapError,
    DatabaseState,
    OperationalPaths,
    ResourceState,
    discover_resources,
    ensure_operational_directories,
    inspect_database,
)
from app.domain.services.detection.model_loader import ModelLoader

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _paths_resolvem_a_partir_da_raiz_passada(monkeypatch):
    """Este módulo testa a RESOLUÇÃO de caminhos — o ambiente não pode interferir.

    `OperationalPaths.resolve(raiz)` consulta `MODELS_DIR`/`DEEPFAKE_MODELS_DIR`/
    `XFAKE_MODELS_DIR` (e as equivalentes de logs/resultados) antes do default
    sob a raiz. Duas fontes as definem fora daqui: o `.env` do projeto, carregado
    no import de `app.*`, e a fixture de sessão `_isolate_models_dir`, que
    protege `data/models` de escrita acidental. Com qualquer uma ativa, um teste
    que espera inventário VAZIO sob `tmp_path` enxerga o diretório de outra
    pessoa — e o resultado passa a depender da ordem de execução da suíte.
    """
    for nome in (
        "MODELS_DIR",
        "DEEPFAKE_MODELS_DIR",
        "XFAKE_MODELS_DIR",
        "RESULTS_DIR",
        "DEEPFAKE_RESULTS_DIR",
        "XFAKE_RESULTS_DIR",
        "DEEPFAKE_LOGS_DIR",
        "XFAKE_LOGS_DIR",
        "XFAKE_STORAGE_DIR",
        "DEEPFAKE_STORAGE_DIR",
    ):
        monkeypatch.delenv(nome, raising=False)


def test_empty_resource_inventory_is_valid(tmp_path):
    paths = OperationalPaths.resolve(tmp_path)
    ensure_operational_directories(paths)
    inventory = discover_resources(paths)
    assert inventory.datasets == ResourceState.EMPTY
    assert inventory.models == ResourceState.EMPTY
    assert inventory.benchmarks == ResourceState.EMPTY
    assert inventory.reports == ResourceState.EMPTY


def test_incomplete_and_complete_dataset_are_distinct(tmp_path):
    paths = OperationalPaths.resolve(tmp_path)
    ensure_operational_directories(paths)
    (paths.dataset_real / "real.wav").write_bytes(b"RIFF")
    assert discover_resources(paths).datasets == ResourceState.INCOMPLETE
    (paths.dataset_fake / "fake.wav").write_bytes(b"RIFF")
    assert discover_resources(paths).datasets == ResourceState.AVAILABLE


def test_model_loader_does_not_create_fake_models(tmp_path):
    loader = ModelLoader(tmp_path / "models")
    loader.load_available_models()
    assert loader.get_available_models() == []
    assert loader.default_model is None
    assert list((tmp_path / "models").iterdir()) == []


def test_database_states_do_not_hide_corruption(tmp_path):
    database = tmp_path / "app.db"
    assert inspect_database(database) == DatabaseState.MISSING
    database.touch()
    assert inspect_database(database) == DatabaseState.NEW
    database.write_bytes(b"not-a-sqlite-database")
    assert inspect_database(database) == DatabaseState.INVALID


def test_existing_database_data_is_preserved(tmp_path):
    database = tmp_path / "app.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE preserved (value TEXT)")
        connection.execute("INSERT INTO preserved VALUES ('keep')")
        connection.execute("PRAGMA user_version=2")
    assert inspect_database(database) == DatabaseState.COMPATIBLE
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM preserved").fetchone() == ("keep",)


def test_permission_failure_is_actionable(tmp_path, monkeypatch):
    paths = OperationalPaths.resolve(tmp_path)

    def deny(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", deny)
    with pytest.raises(BootstrapError, match="Sem permissão"):
        ensure_operational_directories(paths)


def test_main_bootstrap_twice_from_another_cwd(tmp_path):
    storage = tmp_path / "storage"
    cwd = tmp_path / "outside-project"
    cwd.mkdir()
    env = {
        **os.environ,
        "XFAKE_STORAGE_DIR": str(storage),
        "PYTHONPATH": os.pathsep.join(
            [
                str(ROOT),
                str(ROOT / ".venv" / "Lib" / "site-packages"),
            ]
        ),
    }
    # Explícito, nunca removido: sem a variável o `main.py` relê a do `.env` e o
    # bootstrap roda no `data/app.db` real — a cada `make test`, no banco do
    # usuário. O `.env` não sobrescreve o que já está no ambiente.
    env["DATABASE_URL"] = f"sqlite:///{(storage / 'app.db').as_posix()}"
    command = [sys.executable, str(ROOT / "main.py"), "--bootstrap-dirs"]
    first = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    second = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (storage / "datasets" / "real").is_dir()
    assert (storage / "datasets" / "fake").is_dir()
    assert inspect_database(storage / "app.db") == DatabaseState.COMPATIBLE


def test_importar_a_composicao_nao_cria_o_engine():
    """O bootstrap define `DATABASE_URL` antes de o `engine` nascer.

    `app.core.db.session` lê a variável no import. Se importar
    `app.dependencies` já criasse o `engine`, o `main.py --bootstrap-dirs`
    gravaria o schema no banco padrão, não em `XFAKE_STORAGE_DIR/app.db` —
    e `test_main_bootstrap_twice_from_another_cwd` só enxerga isso quando o
    `.env` local não define `DATABASE_URL`. Subprocesso: neste processo a
    sessão já foi importada pelo `conftest`.
    """
    codigo = "; ".join(
        [
            "import sys",
            "import app.dependencies",
            "carregados = [m for m in ('app.core.db.session', 'tensorflow')"
            " if m in sys.modules]",
            "assert not carregados, carregados",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_bootstrap_da_composicao_passa_a_semeadura_ao_core(monkeypatch):
    """`app.core` não importa o domínio: a semeadura entra pela composição."""
    import app.dependencies as composicao

    recebido = {}

    def _core(root, *, initialize_database, seed_database):
        recebido.update(
            root=root, initialize_database=initialize_database, seed=seed_database
        )
        return "relatorio"

    monkeypatch.setattr(composicao, "bootstrap_infrastructure", _core)
    assert composicao.bootstrap_application("raiz") == "relatorio"
    assert recebido == {
        "root": "raiz",
        "initialize_database": True,
        "seed": composicao.seed_architecture_configs,
    }


def test_semeadura_cobre_o_registry_e_e_idempotente(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import app.dependencies as composicao
    from app.core.db import models
    from app.core.db.repositories.architecture_configs import (
        ArchitectureConfigRepository,
    )
    from app.core.db.session import Base
    from app.domain.models.architectures.registry import architecture_registry

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    fabrica = sessionmaker(bind=engine)
    repositorio = ArchitectureConfigRepository(fabrica)
    monkeypatch.setattr(
        composicao, "get_architecture_config_repository", lambda: repositorio
    )

    composicao.seed_architecture_configs()
    composicao.seed_architecture_configs()

    with fabrica() as db:
        linhas = db.query(models.ArchitectureConfig).all()
    registry = architecture_registry.get_all_architectures()
    assert sorted(linha.architecture_name for linha in linhas) == sorted(registry)
    for linha in linhas:
        assert linha.variant_name == "default"
        # JSON não tem tupla: `patch_size=(16, 16)` volta como lista.
        esperado = json.loads(
            json.dumps(registry[linha.architecture_name].default_params)
        )
        assert linha.parameters == esperado
    engine.dispose()
