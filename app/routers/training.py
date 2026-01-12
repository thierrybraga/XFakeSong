from fastapi import APIRouter, BackgroundTasks, Depends
from typing import Dict
import uuid
import logging
from app.schemas.api_models import TrainingRequest, TrainingResponse
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_service import TrainingService
from app.dependencies import get_detection_service, get_training_service
from app.core.auth.auth_handler import get_api_key
from app.core.interfaces.services import ProcessingStatus
from app.core.db_setup import get_flask_app
from app.extensions import db
from app.domain.models.training_job import TrainingJob

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["Training"])

# Cache leve em memória (opcional) para respostas rápidas
training_cache: Dict[str, Dict] = {}


def _update_job(job_id: str, **fields):
    try:
        flask_app = get_flask_app()
        with flask_app.app_context():
            job = TrainingJob.query.filter_by(job_id=job_id).first()
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            db.session.commit()
            # Atualizar cache
            training_cache[job_id] = {
                "status": job.status,
                "message": job.message,
                "progress": job.progress,
                "metrics": job.metrics or {}
            }
    except Exception as e:
        logger.error(f"Falha ao atualizar job {job_id}: {e}")


def run_training_task(
    job_id: str,
    request: TrainingRequest,
    service: TrainingService
):
    """
    Executa tarefa de treinamento em background usando o serviço real.
    Atualiza progresso e status no banco para persistência.
    """
    try:
        _update_job(
            job_id,
            status="running",
            message="Training started...",
            progress=5
        )

        config = {
            "model_name": request.model_name,
            "parameters": request.parameters,
            "epochs": request.epochs,
            "batch_size": request.batch_size
        }

        result = service.train_model(
            architecture=request.architecture,
            dataset_path=request.dataset_path,
            config=config
        )

        if result.status == ProcessingStatus.SUCCESS:
            if hasattr(result.data, "metrics"):
                metrics = getattr(result.data, "metrics", {})
            else:
                metrics = {}
            _update_job(
                job_id,
                status="completed",
                message="Training completed successfully.",
                progress=100,
                metrics=metrics
            )
        else:
            _update_job(
                job_id,
                status="failed",
                message=f"Training failed: {'; '.join(result.errors)}",
                progress=100
            )
    except Exception as e:
        logger.error(f"Error in background training task: {e}")
        _update_job(
            job_id,
            status="failed",
            message=f"Internal error: {str(e)}",
            progress=100
        )


@router.post(
    "/start",
    response_model=TrainingResponse,
    dependencies=[Depends(get_api_key)]
)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    service: TrainingService = Depends(get_training_service)
):
    job_id = str(uuid.uuid4())

    # Criar registro persistente
    try:
        flask_app = get_flask_app()
        with flask_app.app_context():
            job = TrainingJob(
                job_id=job_id,
                status="pending",
                message="Job created",
                progress=0,
                architecture=request.architecture,
                model_name=request.model_name,
                dataset_path=request.dataset_path,
                parameters=request.parameters
            )
            db.session.add(job)
            db.session.commit()
            training_cache[job_id] = {
                "status": job.status,
                "message": job.message,
                "progress": job.progress,
                "metrics": {}
            }
    except Exception as e:
        logger.error(f"Erro ao criar job de treinamento: {e}")
        return TrainingResponse(
            job_id=job_id, status="error", message="Falha ao criar job"
        )

    background_tasks.add_task(run_training_task, job_id, request, service)

    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Training job initiated",
        progress=0
    )


@router.get("/status/{job_id}", response_model=TrainingResponse)
async def check_training_status(job_id: str):
    # Primeiro tenta cache
    cached = training_cache.get(job_id)
    if cached:
        return TrainingResponse(
            job_id=job_id,
            status=cached["status"],
            message=cached["message"],
            progress=cached.get("progress", 0),
            metrics=cached.get("metrics", {})
        )

    # Fallback para banco
    try:
        flask_app = get_flask_app()
        with flask_app.app_context():
            job = TrainingJob.query.filter_by(job_id=job_id).first()
            if not job:
                return TrainingResponse(
                    job_id=job_id,
                    status="not_found",
                    message="Job not found"
                )
            return TrainingResponse(
                job_id=job_id,
                status=job.status,
                message=job.message,
                progress=job.progress,
                metrics=job.metrics or {}
            )
    except Exception as e:
        logger.error(f"Erro ao consultar status: {e}")
        return TrainingResponse(
            job_id=job_id, status="error", message="Erro ao consultar status"
        )


@router.get("/architectures")
async def list_architectures(
        service: DetectionService = Depends(get_detection_service)):
    """Lista todas as arquiteturas suportadas para treinamento."""
    return {
        "architectures": service.get_available_architectures()
    }
