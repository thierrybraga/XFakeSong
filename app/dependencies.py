"""Ponto de composição: instâncias únicas dos serviços de domínio.

Toda interface (API, Gradio, CLI) obtém os serviços por aqui, para que um
modelo treinado numa delas apareça nas outras sem reiniciar o processo. Os
diretórios vêm de :class:`~app.core.bootstrap.OperationalPaths`, que honra
``MODELS_DIR``/``DEEPFAKE_MODELS_DIR``/``XFAKE_MODELS_DIR``,
``UPLOAD_DIR``/``DEEPFAKE_UPLOADS_DIR``/``XFAKE_UPLOADS_DIR`` e
``XFAKE_STORAGE_DIR``, e resolve caminhos relativos a partir da raiz do
projeto — nunca do diretório corrente.

Também é aqui que o bootstrap das interfaces junta infraestrutura e domínio:
:func:`bootstrap_application` cria diretórios e schema (``app.core``) e semeia
``architecture_configs`` com o registry de arquiteturas (domínio), para que
``app.core`` nunca precise importar ``app.domain``.

Serviços e repositórios são importados dentro de cada provider, não no topo:
importar este módulo não pode criar o ``engine`` do SQLAlchemy (que lê
``DATABASE_URL`` no import de ``app.core.db.session``) antes de
:func:`bootstrap_application` definir essa variável — senão o
``main.py --bootstrap-dirs`` criaria o schema no banco padrão em vez de
``XFAKE_STORAGE_DIR/app.db``. Também evita carregar o TensorFlow só para
o bootstrap.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.bootstrap import (
    BootstrapReport,
    OperationalPaths,
    bootstrap_application as bootstrap_infrastructure,
)
from app.core.config.env import env_flag
from app.domain.models.architectures.registry import architecture_registry

if TYPE_CHECKING:
    from app.core.db.repositories.analysis_results import AnalysisResultRepository
    from app.core.db.repositories.architecture_configs import (
        ArchitectureConfigRepository,
    )
    from app.core.db.repositories.persistence import PersistenceStatsRepository
    from app.domain.services.detection_service import DetectionService
    from app.domain.services.feature_extraction_service import (
        AudioFeatureExtractionService,
    )
    from app.domain.services.interactive_training import (
        InteractiveTrainingService,
    )
    from app.domain.services.training_jobs import TrainingJobService
    from app.domain.services.training_service import TrainingService
    from app.domain.services.upload_service import AudioUploadService
    from app.domain.services.voice_profile_service import VoiceProfileService

logger = logging.getLogger(__name__)


@lru_cache()
def get_detection_service() -> DetectionService:
    from app.domain.services.detection_service import DetectionService

    logger.info("Inicializando DetectionService singleton...")
    # Arquitetura inicializada não equivale a modelo treinado. Modelos de
    # demonstração são opt-in para não produzir inferência fictícia.
    return DetectionService(
        models_dir=str(OperationalPaths.resolve().models),
        create_default_models=env_flag("XFAKE_CREATE_DEFAULT_MODELS", False),
    )


@lru_cache()
def get_upload_service() -> AudioUploadService:
    from app.domain.services.upload_service import AudioUploadService

    logger.info("Inicializando AudioUploadService singleton...")
    return AudioUploadService(upload_directory=str(OperationalPaths.resolve().uploads))


@lru_cache()
def get_training_service() -> TrainingService:
    from app.domain.services.training_service import TrainingService

    logger.info("Inicializando TrainingService singleton...")
    return TrainingService(models_dir=str(OperationalPaths.resolve().models))


@lru_cache()
def get_training_job_service() -> TrainingJobService:
    """Jobs de treino/K-fold da API; o cache de status é desta instância única."""
    from app.domain.services.training_jobs import TrainingJobService

    return TrainingJobService()


@lru_cache()
def get_interactive_training_service() -> InteractiveTrainingService:
    """Treino do assistente do Gradio.

    O extrator do caminho clássico é o do ``DetectionService`` (mesma
    instância da inferência, mesmo contrato), e o plano de hiperparâmetros vem
    de ``benchmarks.planning`` — o harness, que o domínio não importa.
    """
    from app.domain.services.interactive_training import InteractiveTrainingService

    def extrator_da_inferencia():
        try:
            return get_detection_service().feature_service
        except Exception:
            from app.domain.services.feature_extraction_service import (
                AudioFeatureExtractionService,
            )

            return AudioFeatureExtractionService()

    def plano_do_benchmark(arch: str):
        from benchmarks.planning import effective_hyperparameters

        return effective_hyperparameters(arch)

    return InteractiveTrainingService(
        feature_service_factory=extrator_da_inferencia,
        hyperparameter_plan=plano_do_benchmark,
    )


@lru_cache()
def get_feature_extraction_service() -> AudioFeatureExtractionService:
    from app.domain.services.feature_extraction_service import (
        AudioFeatureExtractionService,
    )

    logger.info("Inicializando AudioFeatureExtractionService singleton...")
    return AudioFeatureExtractionService()


@lru_cache()
def get_voice_profile_service() -> VoiceProfileService:
    from app.domain.services.voice_profile_service import VoiceProfileService

    logger.info("Inicializando VoiceProfileService singleton...")
    return VoiceProfileService(
        profiles_dir=str(OperationalPaths.resolve().data / "voice_profiles")
    )


# ── Repositórios (consultas fora de routers e abas) ────────────────────


@lru_cache()
def get_analysis_result_repository() -> AnalysisResultRepository:
    from app.core.db.repositories.analysis_results import AnalysisResultRepository

    return AnalysisResultRepository()


@lru_cache()
def get_architecture_config_repository() -> ArchitectureConfigRepository:
    from app.core.db.repositories.architecture_configs import (
        ArchitectureConfigRepository,
    )

    return ArchitectureConfigRepository()


@lru_cache()
def get_persistence_stats_repository() -> PersistenceStatsRepository:
    from app.core.db.repositories.persistence import PersistenceStatsRepository

    return PersistenceStatsRepository()


# ── Bootstrap (infraestrutura + dados do domínio) ──────────────────────


def seed_architecture_configs() -> None:
    """Semeia ``architecture_configs`` com o ``default_params`` do registry.

    Só age com a tabela vazia. Falhas são registradas e não interrompem o
    bootstrap.
    """
    try:
        defaults = {
            name: (info.description, info.default_params)
            for name, info in architecture_registry.get_all_architectures().items()
        }
        get_architecture_config_repository().seed_defaults(defaults)
    except Exception as e:
        logger.error(f"Erro ao semear arquiteturas: {e}")


def bootstrap_application(
    root: str | Path | None = None,
    *,
    initialize_database: bool = True,
) -> BootstrapReport:
    """Bootstrap de API, Gradio e CLI: diretórios, schema e semeadura do banco."""
    return bootstrap_infrastructure(
        root,
        initialize_database=initialize_database,
        seed_database=seed_architecture_configs,
    )
