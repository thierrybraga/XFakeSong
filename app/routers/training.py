from fastapi import APIRouter, BackgroundTasks, Depends
from typing import Dict
import uuid
import time
from app.schemas.api_models import TrainingRequest, TrainingResponse
from app.domain.services.detection_service import DetectionService
from app.dependencies import get_detection_service

router = APIRouter(prefix="/api/v1/training", tags=["Training"])

# Simulação de armazenamento de jobs
training_jobs: Dict[str, Dict] = {}

def run_training_task(job_id: str, request: TrainingRequest):
    """
    Simula uma tarefa de treinamento em background.
    Futuramente, integrar com ModelTrainer real.
    """
    training_jobs[job_id]["status"] = "running"
    training_jobs[job_id]["message"] = "Training started..."
    
    # Simulação de delay
    time.sleep(5)
    
    training_jobs[job_id]["status"] = "completed"
    training_jobs[job_id]["message"] = f"Training completed for {request.model_name} using {request.architecture}"


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {
        "status": "pending",
        "message": "Job created",
        "request": request
    }
    
    background_tasks.add_task(run_training_task, job_id, request)
    
    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Training job initiated"
    )

@router.get("/status/{job_id}", response_model=TrainingResponse)
async def check_training_status(job_id: str):
    if job_id not in training_jobs:
        return TrainingResponse(job_id=job_id, status="not_found", message="Job not found")
    
    job = training_jobs[job_id]
    return TrainingResponse(
        job_id=job_id,
        status=job["status"],
        message=job["message"]
    )

@router.get("/architectures")
async def list_architectures(
        service: DetectionService = Depends(get_detection_service)):
    """Lista todas as arquiteturas suportadas para treinamento."""
    return {
        "architectures": service.get_available_architectures()
    }
