"""Repositórios de `app/core/db/repositories/`: as consultas das tabelas operacionais.

SUJEITO: o que routers e abas recebem. As consultas saíram de
`web/routers/{history,system}.py` e das abas Histórico, Painel e
hiperparâmetros; cada método abre e fecha a própria sessão, então as entidades
devolvidas têm de continuar legíveis depois de fechada a sessão — é assim que
a aba monta a tabela.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import models
from app.core.db.repositories.analysis_results import AnalysisResultRepository
from app.core.db.repositories.architecture_configs import (
    ArchitectureConfigRepository,
)
from app.core.db.repositories.persistence import PersistenceStatsRepository
from app.core.db.repositories.training_jobs import TrainingJobRepository
from app.core.db.session import Base


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autocommit=False, autoflush=False)
    engine.dispose()


@pytest.fixture
def analises(session_factory):
    agora = datetime(2026, 10, 18, 12, 0, 0)
    linhas = [
        ("antiga.wav", False, 0.9, "svm", agora - timedelta(days=3)),
        ("media.wav", True, 0.8, "aasist", agora - timedelta(hours=5)),
        ("recente.WAV", True, 0.7, None, agora - timedelta(minutes=1)),
    ]
    with session_factory() as db:
        for nome, fake, conf, modelo, quando in linhas:
            db.add(
                models.AnalysisResult(
                    filename=nome,
                    is_fake=fake,
                    confidence=conf,
                    model_name=modelo,
                    details={"origem": nome},
                    created_at=quando,
                )
            )
        db.commit()
    return AnalysisResultRepository(session_factory), agora


def test_pagina_do_historico_vem_ordenada_com_o_total(analises):
    repo, _ = analises
    itens, total = repo.list_page(limit=2, offset=0)
    assert total == 3
    assert [i.filename for i in itens] == ["recente.WAV", "media.wav"]

    itens, _ = repo.list_page(limit=2, offset=2)
    assert [i.filename for i in itens] == ["antiga.wav"]


def test_entidade_continua_legivel_depois_de_fechada_a_sessao(analises):
    repo, _ = analises
    item = repo.list_recent(1)[0]
    assert item.details == {"origem": "recente.WAV"}
    assert item.created_at is not None
    assert repo.get(item.id).filename == "recente.WAV"


def test_busca_combina_modelo_veredito_e_nome_sem_caixa(analises):
    repo, _ = analises
    assert [r.filename for r in repo.search(model_name="svm")] == ["antiga.wav"]
    assert [r.filename for r in repo.search(is_fake=True)] == [
        "recente.WAV",
        "media.wav",
    ]
    assert [r.filename for r in repo.search(filename_contains="wav")] == [
        "recente.WAV",
        "media.wav",
        "antiga.wav",
    ]
    assert repo.search(model_name="aasist", is_fake=False) == []
    assert len(repo.search(limit=1)) == 1


def test_nomes_de_modelo_ignoram_vazios(analises):
    repo, _ = analises
    assert sorted(repo.distinct_model_names()) == ["aasist", "svm"]


def test_contagem_desde_um_corte(analises):
    repo, agora = analises
    assert repo.count_since(agora - timedelta(hours=24)) == 2


def test_remocao_individual_e_total(analises):
    repo, _ = analises
    alvo = repo.list_recent(1)[0].id
    assert repo.delete(alvo) is True
    assert repo.delete(alvo) is False
    assert repo.get(alvo) is None
    assert repo.delete_all() == 2
    assert repo.list_page(limit=10)[1] == 0


def test_job_de_treino_criado_lido_e_atualizado(session_factory):
    repo = TrainingJobRepository(session_factory)
    criado = repo.create(
        job_id="job-1",
        status="pending",
        message="Job created",
        progress=0,
        architecture="AASIST",
        parameters={"lr": 1e-3},
    )
    assert criado.status == "pending"

    atualizado = repo.update(
        "job-1", status="completed", progress=100, metrics={"a": 1}
    )
    assert (atualizado.status, atualizado.progress) == ("completed", 100)
    assert repo.get("job-1").metrics == {"a": 1}
    assert repo.update("inexistente", status="failed") is None
    assert repo.get("inexistente") is None


def test_semeadura_so_acontece_com_a_tabela_vazia(session_factory):
    repo = ArchitectureConfigRepository(session_factory)
    padroes = {"AASIST": ("grafo", {"dropout_rate": 0.2}), "SVM": (None, {"C": 1})}
    assert repo.seed_defaults(padroes) == 2
    assert repo.seed_defaults(padroes) == 0

    linha = repo.find_default("AASIST")
    assert (linha.description, linha.parameters, linha.is_active) == (
        "grafo",
        {"dropout_rate": 0.2},
        True,
    )


def test_parametros_salvos_criam_ou_atualizam_a_linha_default(session_factory):
    repo = ArchitectureConfigRepository(session_factory)
    assert repo.find_default("Conformer") is None

    repo.save_default_parameters("Conformer", {"epochs": 5}, description="criada")
    repo.save_default_parameters("Conformer", {"epochs": 7}, description="ignorada")

    linha = repo.find_default("Conformer")
    assert linha.parameters == {"epochs": 7}
    assert linha.description == "criada"


def test_estatisticas_da_persistencia(session_factory):
    repo = PersistenceStatsRepository(session_factory)
    assert repo.schema_version() == 0
    assert repo.table_counts() == {
        "experiments": 0,
        "model_runs": 0,
        "metrics": 0,
        "configurations": 0,
        "system_snapshots": 0,
    }
