"""Endpoints REST para gerenciamento de Perfis de Voz.

Usa schemas centralizados de app.schemas.api_models e exceções de domínio.
"""

import tempfile
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter, Depends, File, Request, UploadFile,
)

from app.core.exceptions import ProfileNotFoundError, TrainingError, ValidationError
from app.core.security import limiter
from app.domain.services.voice_profile_service import VoiceProfileService
from app.schemas.api_models import (
    ProfileCreate, ProfileDetectionResult, ProfileResponse,
    ProfileTrainRequest, ProfileUpdate,
)

router = APIRouter(prefix="/api/v1/profiles", tags=["Voice Profiles"])


# ── Singleton ──────────────────────────────────────────────────────────

def get_voice_profile_service() -> VoiceProfileService:
    return VoiceProfileService()


def _profile_to_response(p) -> ProfileResponse:
    """Converte VoiceProfile ORM para ProfileResponse."""
    return ProfileResponse(
        id=p.id,
        name=p.name,
        telegram_id=p.telegram_id,
        phone=p.phone,
        email=p.email,
        status=p.status,
        architecture=p.architecture,
        num_samples=p.num_samples,
        total_duration_seconds=p.total_duration_seconds,
        description=p.description,
        training_metrics=p.training_metrics,
        created_at=str(p.created_at) if p.created_at else None,
    )


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=List[ProfileResponse],
    summary="Lista todos os perfis de voz",
)
@limiter.limit("30/minute")
async def list_profiles(
    request: Request,
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    return [_profile_to_response(p) for p in svc.list_profiles()]


@router.post(
    "/",
    response_model=ProfileResponse,
    status_code=201,
    summary="Cria um novo perfil de voz",
)
@limiter.limit("10/minute")
async def create_profile(
    request: Request,
    body: ProfileCreate,
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    profile = svc.create_profile(
        name=body.name,
        telegram_id=body.telegram_id,
        phone=body.phone,
        email=body.email,
        description=body.description,
        architecture=body.architecture,
    )
    return _profile_to_response(profile)


@router.get(
    "/{profile_id}",
    response_model=ProfileResponse,
    summary="Detalhes de um perfil",
)
@limiter.limit("30/minute")
async def get_profile(
    request: Request,
    profile_id: int,
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    profile = svc.get_profile(profile_id)
    if not profile:
        raise ProfileNotFoundError(profile_id)
    return _profile_to_response(profile)


@router.put(
    "/{profile_id}",
    response_model=ProfileResponse,
    summary="Atualiza dados de um perfil",
)
@limiter.limit("10/minute")
async def update_profile(
    request: Request,
    profile_id: int,
    body: ProfileUpdate,
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    update_data = body.model_dump(exclude_unset=True)
    profile = svc.update_profile(profile_id, **update_data)
    if not profile:
        raise ProfileNotFoundError(profile_id)
    return _profile_to_response(profile)


@router.delete("/{profile_id}", summary="Remove um perfil e seus dados")
@limiter.limit("5/minute")
async def delete_profile(
    request: Request,
    profile_id: int,
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    if not svc.delete_profile(profile_id):
        raise ProfileNotFoundError(profile_id)
    return {"detail": "Perfil removido com sucesso"}


@router.post(
    "/{profile_id}/samples",
    summary="Upload de amostras de áudio para o dataset do perfil",
)
@limiter.limit("10/minute")
async def upload_samples(
    request: Request,
    profile_id: int,
    files: List[UploadFile] = File(...),
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    if not svc.get_profile(profile_id):
        raise ProfileNotFoundError(profile_id)

    audio_files = []
    for f in files:
        content = await f.read()
        audio_files.append((f.filename or "sample.wav", content))

    result = svc.add_audio_samples(profile_id, audio_files)
    if not result.get("success"):
        raise ValidationError(result.get("error", "Erro no upload"))
    return result


@router.delete(
    "/{profile_id}/samples/{filename}",
    summary="Remove uma amostra do dataset",
)
@limiter.limit("10/minute")
async def remove_sample(
    request: Request,
    profile_id: int,
    filename: str,
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    if not svc.remove_audio_sample(profile_id, filename):
        raise ValidationError(f"Amostra '{filename}' não encontrada")
    return {"detail": "Amostra removida"}


@router.post(
    "/{profile_id}/train",
    summary="Inicia treinamento do modelo do perfil",
)
@limiter.limit("3/minute")
async def train_profile(
    request: Request,
    profile_id: int,
    body: ProfileTrainRequest = ProfileTrainRequest(),
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    if not svc.get_profile(profile_id):
        raise ProfileNotFoundError(profile_id)

    result = svc.train_profile_model(
        profile_id=profile_id,
        architecture=body.architecture,
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
    )
    if not result.get("success"):
        raise TrainingError(result.get("error", "Erro no treinamento"))
    return result


@router.post(
    "/{profile_id}/detect",
    response_model=ProfileDetectionResult,
    summary="Verifica se um áudio pertence à pessoa do perfil",
)
@limiter.limit("10/minute")
async def detect_with_profile(
    request: Request,
    profile_id: int,
    file: UploadFile = File(...),
    svc: VoiceProfileService = Depends(get_voice_profile_service),
):
    if not svc.get_profile(profile_id):
        raise ProfileNotFoundError(profile_id)

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = svc.detect_with_profile(profile_id, tmp_path)
        if not result.get("success"):
            raise AudioProcessingError(result.get("error", "Erro na detecção"))
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Import necessário para o handler de detect
from app.core.exceptions import AudioProcessingError  # noqa: E402
