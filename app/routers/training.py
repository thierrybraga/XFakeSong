import logging
import time
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy.orm import Session

from app.core.auth.auth_handler import get_api_key
from app.core.database import get_background_db, get_db
from app.core.exceptions import (
    ModelNotFoundError,
    ServiceUnavailableError,
    ValidationError,
)
from app.core.interfaces.base import ProcessingStatus
from app.core.security import limiter
from app.dependencies import get_detection_service, get_training_service
from app.domain.models.training_job import TrainingJob
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_service import TrainingService
from app.schemas.api_models import (
    CrossValidationRequest,
    CrossValidationResult,
    OnnxExportRequest,
    OnnxExportResponse,
    TrainingRequest,
    TrainingResponse,
)

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
        # API.8: era return TrainingResponse(status="error") com HTTP 200,
        # o que enganava clientes que checam apenas o status code.
        # Agora levanta exception → handler global retorna 503.
        logger.error(f"Erro ao criar job de treinamento: {e}", exc_info=True)
        raise ServiceUnavailableError("training_service")

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


# ── API.7: Cross-Validation (Sprint 4.1) ────────────────────────────────

def _run_cv_task(
    job_id: str,
    request: CrossValidationRequest,
    service: TrainingService,
):
    """Background task para K-fold cross-validation."""
    try:
        _update_job(
            job_id,
            status="running",
            message=f"K-fold CV started (k={request.n_folds})...",
            progress=5,
        )

        result = service.cross_validate_model(
            architecture=request.architecture,
            dataset_path=request.dataset_path,
            config=request.config,
            n_folds=request.n_folds,
            save_fold_models=request.save_fold_models,
        )

        if result.status == ProcessingStatus.SUCCESS:
            data = result.data or {}
            agg = data.get("aggregated", {})
            metrics = {
                f"{k}_{stat}": v
                for k, sub in agg.items() for stat, v in sub.items()
            }
            _update_job(
                job_id,
                status="completed",
                message=(
                    f"CV completo: {data.get('n_successful', 0)}/"
                    f"{request.n_folds} folds bem-sucedidos. "
                    f"Best fold={data.get('best_fold', '?')}"
                ),
                progress=100,
                metrics={
                    **metrics,
                    "n_folds": data.get("n_folds", request.n_folds),
                    "best_fold": data.get("best_fold", -1),
                    "per_fold_count": len(data.get("per_fold", [])),
                },
            )
        else:
            _update_job(
                job_id,
                status="failed",
                message=f"CV failed: {'; '.join(result.errors or ['unknown'])}",
                progress=100,
            )
    except Exception as e:
        logger.error(f"Erro em CV task {job_id}: {e}", exc_info=True)
        _update_job(
            job_id,
            status="failed",
            message=f"Internal error: {e}",
            progress=100,
        )


@router.post(
    "/cross-validate",
    response_model=TrainingResponse,
    dependencies=[Depends(get_api_key)],
    summary="Inicia K-fold Cross Validation em background (Sprint 4.1)",
)
@limiter.limit("3/minute")
async def start_cross_validation(
    request: Request,
    body: CrossValidationRequest,
    background_tasks: BackgroundTasks,
    service: TrainingService = Depends(get_training_service),
    db: Session = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    try:
        job = TrainingJob(
            job_id=job_id,
            status="pending",
            message=f"CV job created (k={body.n_folds})",
            progress=0,
            architecture=body.architecture,
            model_name=f"cv_{body.architecture}",
            dataset_path=body.dataset_path,
            parameters={**body.config, "n_folds": body.n_folds},
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        training_cache.set(job_id, {
            "status": job.status, "message": job.message,
            "progress": job.progress, "metrics": {},
        })
    except Exception as e:
        logger.error(f"Erro ao criar CV job: {e}", exc_info=True)
        raise ServiceUnavailableError("training_service")

    background_tasks.add_task(_run_cv_task, job_id, body, service)

    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Cross-validation job initiated",
        progress=0,
    )


@router.get(
    "/cross-validate/{job_id}",
    response_model=CrossValidationResult,
    summary="Resultado final de uma CV job concluída",
)
@limiter.limit("30/minute")
async def get_cv_result(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
):
    """Retorna `aggregated` + `per_fold` quando a CV está concluída.

    Para status intermediário, use `/training/status/{job_id}`.
    """
    job = db.query(TrainingJob).filter_by(job_id=job_id).first()
    if not job:
        raise ValidationError(f"Job '{job_id}' não encontrado")
    if job.status != "completed":
        raise ValidationError(
            f"Job ainda não concluído (status={job.status}). "
            f"Use /training/status/{job_id} para acompanhar."
        )

    metrics = job.metrics or {}
    # Reconstrói aggregated a partir das métricas planas que salvamos
    aggregated: Dict[str, Dict[str, float]] = {}
    for full_key, value in metrics.items():
        if "_" in full_key:
            base, stat = full_key.rsplit("_", 1)
            if stat in ("mean", "std", "min", "max"):
                aggregated.setdefault(base, {})[stat] = float(value)

    return CrossValidationResult(
        architecture=job.architecture or "unknown",
        n_folds=int(metrics.get("n_folds", 0)),
        n_successful=int(metrics.get("per_fold_count", 0)),
        best_fold=int(metrics.get("best_fold", -1)),
        aggregated=aggregated,
        per_fold=[],  # detalhes per_fold ficam apenas no log do task
    )


# ── API.7 (Sprint 3.4): Export ONNX + INT8 quantization ─────────────────


@router.post(
    "/export-onnx",
    response_model=OnnxExportResponse,
    dependencies=[Depends(get_api_key)],
    summary="Exporta um modelo carregado para ONNX (+ INT8 opcional)",
    description=(
        "Converte um modelo TensorFlow/Keras já carregado para o formato ONNX "
        "(Sprint 3.4). Com quantize_int8=True, também gera uma versão INT8 via "
        "dynamic quantization. Modelos sklearn (SVM/Random Forest) não são "
        "suportados. Requer tf2onnx + onnxruntime instalados."
    ),
)
@limiter.limit("5/minute")
async def export_model_onnx(
    request: Request,
    body: OnnxExportRequest,
    service: DetectionService = Depends(get_detection_service),
):
    from app.domain.models.inference.onnx_export import (
        export_to_onnx,
        is_onnx_available,
        quantize_int8,
    )

    # 1) Dependências disponíveis?
    if not is_onnx_available():
        raise ServiceUnavailableError(
            "onnx_export (instale: pip install tf2onnx onnxruntime)"
        )

    # 2) Modelo existe e está carregado?
    model_info = service.loaded_models.get(body.model_name)
    if model_info is None:
        raise ModelNotFoundError(body.model_name)

    # 3) Apenas modelos TensorFlow/Keras podem ser exportados via tf2onnx
    if getattr(model_info, "model_type", None) != "tensorflow":
        raise ValidationError(
            f"Export ONNX só suporta modelos TensorFlow/Keras. "
            f"'{body.model_name}' é '{model_info.model_type}'.",
            field="model_name",
        )

    out_dir = service.models_dir / "onnx"
    onnx_path = out_dir / f"{body.model_name}.onnx"

    # 4) Export FP32
    result_path = export_to_onnx(
        model_info.model,
        onnx_path,
        opset=body.opset,
        dynamic_batch=body.dynamic_batch,
    )
    if result_path is None:
        return OnnxExportResponse(
            success=False,
            message=(
                "Falha ao exportar para ONNX. Verifique os logs do servidor "
                "(arquitetura pode usar camadas não conversíveis por tf2onnx)."
            ),
        )

    size_mb = round(result_path.stat().st_size / (1024 * 1024), 3)
    int8_path = None
    size_int8_mb = None

    # 5) Quantização INT8 opcional
    if body.quantize_int8:
        int8_out = out_dir / f"{body.model_name}_int8.onnx"
        int8_result = quantize_int8(result_path, int8_out)
        if int8_result is not None:
            int8_path = str(int8_result)
            size_int8_mb = round(int8_result.stat().st_size / (1024 * 1024), 3)

    msg = f"Modelo '{body.model_name}' exportado para ONNX (opset={body.opset})."
    if body.quantize_int8 and int8_path is None:
        msg += " Quantização INT8 falhou (ver logs)."
    elif int8_path is not None:
        msg += f" INT8 gerado ({size_int8_mb} MB)."

    return OnnxExportResponse(
        success=True,
        onnx_path=str(result_path),
        onnx_int8_path=int8_path,
        size_mb=size_mb,
        size_int8_mb=size_int8_mb,
        message=msg,
    )
