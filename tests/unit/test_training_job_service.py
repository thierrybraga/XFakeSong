"""`TrainingJobService`: ciclo de vida dos jobs de treino e K-fold da API.

SUJEITO: o estado que antes vivia no router de treino (cache TTL,
`_update_job`, tarefas de background). O router só valida e delega; o que o
cliente vê em `/training/status/{id}` e `/training/cross-validate/{id}` é
decidido aqui. O banco é um SQLite em memória — nada toca `data/app.db`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.contracts.base import ProcessingStatus
from app.core.db import models  # noqa: F401  (registra as tabelas)
from app.core.db.repositories.training_jobs import TrainingJobRepository
from app.core.db.session import Base
from app.core.exceptions import ValidationError
from app.domain.services import training_jobs as jobs_module
from app.domain.services.training_jobs import TrainingJobService, TTLCache


@pytest.fixture
def repositorio():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield TrainingJobRepository(sessionmaker(bind=engine))
    engine.dispose()


class _TreinoFalso:
    def __init__(self, resultado=None, erro=None):
        self.resultado = resultado
        self.erro = erro
        self.chamadas = []

    def train_model(self, **kwargs):
        self.chamadas.append(kwargs)
        if self.erro:
            raise self.erro
        return self.resultado

    def cross_validate_model(self, **kwargs):
        self.chamadas.append(kwargs)
        if self.erro:
            raise self.erro
        return self.resultado


def _novo_job(servico):
    return servico.create_training_job(
        architecture="AASIST",
        model_name="m",
        dataset_path="/tmp/ds.npz",
        parameters={"lr": 1e-3},
    )


def test_job_nasce_pendente_no_cache_e_no_banco(repositorio):
    servico = TrainingJobService(repositorio)
    job_id = _novo_job(servico)

    assert servico.get_status(job_id) == {
        "status": "pending",
        "message": "Job created",
        "progress": 0,
        "metrics": {},
    }
    linha = repositorio.get(job_id)
    assert (linha.architecture, linha.parameters) == ("AASIST", {"lr": 1e-3})


def test_treino_bem_sucedido_conclui_com_metricas(repositorio):
    servico = TrainingJobService(repositorio)
    job_id = _novo_job(servico)
    treino = _TreinoFalso(
        SimpleNamespace(
            status=ProcessingStatus.SUCCESS,
            data=SimpleNamespace(metrics={"accuracy": 0.95}),
        )
    )

    servico.run_training(
        job_id,
        treino,
        architecture="AASIST",
        dataset_path="/tmp/ds.npz",
        model_name="m",
        parameters={"lr": 1e-3},
        epochs=3,
        batch_size=8,
    )

    assert treino.chamadas == [
        {
            "architecture": "AASIST",
            "dataset_path": "/tmp/ds.npz",
            "config": {
                "model_name": "m",
                "parameters": {"lr": 1e-3},
                "epochs": 3,
                "batch_size": 8,
            },
        }
    ]
    esperado = {
        "status": "completed",
        "message": "Training completed successfully.",
        "progress": 100,
        "metrics": {"accuracy": 0.95},
    }
    assert servico.get_status(job_id) == esperado
    # Outra instância (cache vazio) lê o mesmo estado do banco.
    assert TrainingJobService(repositorio).get_status(job_id) == esperado


@pytest.mark.parametrize(
    ("treino", "mensagem"),
    [
        (
            _TreinoFalso(
                SimpleNamespace(status=ProcessingStatus.ERROR, errors=["a", "b"])
            ),
            "Training failed: a; b",
        ),
        (_TreinoFalso(erro=RuntimeError("boom")), "Internal error: boom"),
    ],
)
def test_treino_que_falha_termina_failed(repositorio, treino, mensagem):
    servico = TrainingJobService(repositorio)
    job_id = _novo_job(servico)
    servico.run_training(
        job_id,
        treino,
        architecture="AASIST",
        dataset_path="/tmp/ds.npz",
        model_name="m",
        parameters={},
        epochs=1,
        batch_size=1,
    )
    estado = servico.get_status(job_id)
    assert (estado["status"], estado["message"], estado["progress"]) == (
        "failed",
        mensagem,
        100,
    )


def test_status_de_job_inexistente_e_none(repositorio):
    assert TrainingJobService(repositorio).get_status("nao-existe") is None


def test_falha_de_persistencia_na_atualizacao_nao_derruba_a_tarefa(repositorio):
    servico = TrainingJobService(repositorio)
    job_id = _novo_job(servico)

    def _quebra(*_args, **_kwargs):
        raise RuntimeError("disco cheio")

    repositorio.update = _quebra
    servico._update_job(job_id, status="running")
    assert servico.get_status(job_id)["status"] == "pending"


def test_kfold_concluido_reconstroi_o_agregado(repositorio):
    servico = TrainingJobService(repositorio)
    job_id = servico.create_cross_validation_job(
        architecture="SVM", dataset_path="/tmp/ds.npz", config={"C": 1}, n_folds=3
    )
    assert repositorio.get(job_id).parameters == {"C": 1, "n_folds": 3}
    assert repositorio.get(job_id).model_name == "cv_SVM"
    with pytest.raises(ValidationError):
        servico.get_cross_validation_result(job_id)

    treino = _TreinoFalso(
        SimpleNamespace(
            status=ProcessingStatus.SUCCESS,
            data={
                "aggregated": {"eer": {"mean": 0.1, "std": 0.02}},
                "n_successful": 3,
                "n_folds": 3,
                "best_fold": 1,
                "per_fold": [{}, {}, {}],
            },
        )
    )
    servico.run_cross_validation(
        job_id,
        treino,
        architecture="SVM",
        dataset_path="/tmp/ds.npz",
        config={"C": 1},
        n_folds=3,
        save_fold_models=False,
    )

    assert servico.get_status(job_id)["message"] == (
        "CV completo: 3/3 folds bem-sucedidos. Best fold=1"
    )
    assert servico.get_cross_validation_result(job_id) == {
        "architecture": "SVM",
        "n_folds": 3,
        "n_successful": 3,
        "best_fold": 1,
        "aggregated": {"eer": {"mean": 0.1, "std": 0.02}},
        "per_fold": [],
    }


def test_resultado_de_kfold_inexistente_e_erro_de_validacao(repositorio):
    with pytest.raises(ValidationError):
        TrainingJobService(repositorio).get_cross_validation_result("nao-existe")


def test_cache_expira_antes_para_job_ativo(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr(jobs_module.time, "monotonic", lambda: agora[0])
    cache = TTLCache(ttl=30, terminal_ttl=300)
    cache.set("ativo", {"status": "running"})
    cache.set("final", {"status": "completed"})

    agora[0] += 31
    assert cache.get("ativo") is None
    assert cache.get("final") == {"status": "completed"}

    agora[0] += 300
    assert cache.get("final") is None
