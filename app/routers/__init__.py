"""Routers da API REST do XFakeSong.

Expõe os módulos para uso via:
    from app.routers import detection, system, ...

E também os objetos `router` diretamente:
    from app.routers import detection_router, system_router, ...

Cada router define seu próprio prefix em `/api/v1/<area>/`.
"""

from app.routers import (
    datasets,
    detection,
    features,
    history,
    system,
    training,
    voice_profiles,
)

# Aliases convenientes para uso em FastAPI app.include_router()
datasets_router = datasets.router
detection_router = detection.router
features_router = features.router
history_router = history.router
system_router = system.router
training_router = training.router
voice_profiles_router = voice_profiles.router

ALL_ROUTERS = [
    system_router,
    detection_router,
    features_router,
    training_router,
    history_router,
    datasets_router,
    voice_profiles_router,
]

__all__ = [
    "datasets",
    "datasets_router",
    "detection",
    "detection_router",
    "features",
    "features_router",
    "history",
    "history_router",
    "system",
    "system_router",
    "training",
    "training_router",
    "voice_profiles",
    "voice_profiles_router",
    "ALL_ROUTERS",
]
