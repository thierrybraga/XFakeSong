import logging
import time
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.auth.auth_handler import get_api_key
from app.core.database import get_background_db, get_db
from app.core.interfaces.base import ProcessingStatus
from app.dependencies import get_detection_service, get_training_service
from app.domain.models.training_job import TrainingJob
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_service import TrainingService
from app.schemas.api_models import TrainingRequest, TrainingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["Training"])


class TTLCache:
    """Cache com time-to-live para evitar divergência com o banco de dados.

    Entradas expiram após `ttl` segundos, forçando releitura do DB.
    Jobs em estado terminal (completed/failed) ficam no cache por mais tempo
    pois não mudam mais.
    """

    def __init__(self, ttl: int = 30, terminal_ttl: int = 300):
        self._data: Dict[str, Dict] = {}
        self._timestamps: Dict[str, float] = {}
        self.ttl = ttl
        self.terminal_ttl = terminal_ttl

    def get(self, key: str) -> Optional[Dict]:
        if key not in self._data:
            return None
        age = time.monotonic() - self._timestamps[key]
        entry = self._data[key]
        # Jobs terminais têm TTL mais longo
        max_ttl = self.terminal_ttl if entry.get(
            "status") in ("completed", "failed") else self.ttl
        if age > max_ttl:
            del self._data[key]
            del self._timestamps[key]
            return None
        return entry

    def set(self, key: str, value: Dict):
        self._data[key] = value
        self._timestamps[key] = time.monotonic()


# Cache com TTL de 30s para jobs ativos, 5min para terminais
training_cache = TTLCache(ttl=30, terminal_ttl=300)


def _update_job(job_id: str, **fields):
    """Atualiza job de treinamento no banco e no cache.

    Usa get_background_db() (context manager com commit/rollback automático)
    porque background tasks rodam fora do request lifecycle do FastAPI.
    """
    try:
        with get_background_db() as db:
            job = db.query(TrainingJob).filter_by(job_id=job_id).first()
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            # Capturar valores para cache antes de sair do contexto da sessão
            cache_entry = {
                "status": job.status,
                "message": job.message,
                "progress": job.progress,
                "metrics": job.metrics or {}
            }
            # commit automático ao sair do with sem exceção
        # Atualizar cache somente após commit bem-sucedido
        training_cache.set(job_id, cache_entry)
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
    service: TrainingService = Depends(get_training_service),
    db: Session = Depends(get_db)
):
    job_id = str(uuid.uuid4())

    # Criar registro persistente
    try:
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
        db.add(job)
        db.commit()
        db.refresh(job)
        training_cache.set(job_id, {
            "status": job.status,
            "message": job.message,
            "progress": job.progress,
            "metrics": {}
        })
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
async def check_training_status(job_id: str, db: Session = Depends(get_db)):
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
        job = db.query(TrainingJob).filter_by(job_id=job_id).first()
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
