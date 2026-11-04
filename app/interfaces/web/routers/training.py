import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.core.auth.auth_handler import get_api_key
from app.core.exceptions import (
    ModelNotFoundError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
)
from app.core.security import limiter
from app.dependencies import (
    get_detection_service,
    get_training_job_service,
    get_training_service,
)
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_jobs import TrainingJobService
from app.domain.services.training_service import TrainingService
from app.interfaces.web.schemas.api_models import (
    CrossValidationRequest,
    CrossValidationResult,
    OnnxExportRequest,
    OnnxExportResponse,
    TrainingRequest,
    TrainingResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["Training"])


@router.post(
    "/start", response_model=TrainingResponse, dependencies=[Depends(get_api_key)]
)
@limiter.limit("5/minute")
async def start_training(
    request: Request,
    body: TrainingRequest,
    background_tasks: BackgroundTasks,
    service: TrainingService = Depends(get_training_service),
    jobs: TrainingJobService = Depends(get_training_job_service),
):
    try:
        job_id = jobs.create_training_job(
            architecture=body.architecture,
            model_name=body.model_name,
            dataset_path=body.dataset_path,
            parameters=body.parameters,
        )
    except Exception as e:
        # API.8: era return TrainingResponse(status="error") com HTTP 200,
        # o que enganava clientes que checam apenas o status code.
        # Agora levanta exception → handler global retorna 503.
        logger.error(f"Erro ao criar job de treinamento: {e}", exc_info=True)
        raise ServiceUnavailableError("training_service")

    background_tasks.add_task(
        jobs.run_training,
        job_id,
        service,
        architecture=body.architecture,
        dataset_path=body.dataset_path,
        model_name=body.model_name,
        parameters=body.parameters,
        epochs=body.epochs,
        batch_size=body.batch_size,
    )

    return TrainingResponse(
        job_id=job_id, status="pending", message="Training job initiated", progress=0
    )


@router.get("/status/{job_id}", response_model=TrainingResponse)
@limiter.limit("60/minute")
async def check_training_status(
    request: Request,
    job_id: str,
    jobs: TrainingJobService = Depends(get_training_job_service),
):
    try:
        status = jobs.get_status(job_id)
    except Exception as e:
        logger.error(f"Erro ao consultar status: {e}")
        # API.8: era return TrainingResponse(status="error") com HTTP 200,
        # o que enganava clientes que checam apenas o status code (mesmo
        # antipadrão já corrigido em start_training — ver comentário acima).
        raise ServiceUnavailableError("training_service") from e

    if not status:
        raise NotFoundError("Training job", job_id)

    return TrainingResponse(
        job_id=job_id,
        status=status["status"],
        message=status["message"],
        progress=status.get("progress", 0),
        metrics=status.get("metrics", {}),
    )


@router.get("/architectures")
@limiter.limit("30/minute")
async def list_architectures(
    request: Request,
    service: DetectionService = Depends(get_detection_service),
):
    """Lista todas as arquiteturas suportadas para treinamento."""
    return {"architectures": service.get_available_architectures()}


# ── API.7: Cross-Validation (Sprint 4.1) ────────────────────────────────


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
    jobs: TrainingJobService = Depends(get_training_job_service),
):
    try:
        job_id = jobs.create_cross_validation_job(
            architecture=body.architecture,
            dataset_path=body.dataset_path,
            config=body.config,
            n_folds=body.n_folds,
        )
    except Exception as e:
        logger.error(f"Erro ao criar CV job: {e}", exc_info=True)
        raise ServiceUnavailableError("training_service")

    background_tasks.add_task(
        jobs.run_cross_validation,
        job_id,
        service,
        architecture=body.architecture,
        dataset_path=body.dataset_path,
        config=body.config,
        n_folds=body.n_folds,
        save_fold_models=body.save_fold_models,
    )

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
    jobs: TrainingJobService = Depends(get_training_job_service),
):
    """Retorna `aggregated` + `per_fold` quando a CV está concluída.

    Para status intermediário, use `/training/status/{job_id}`.
    """
    return CrossValidationResult(**jobs.get_cross_validation_result(job_id))


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
