import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.core.bootstrap import OperationalPaths
from app.core.config.settings import TrainingConfig
from app.dependencies import get_detection_service, get_training_service
from app.domain.models.architectures.registry import architecture_registry
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_service import TrainingService


@dataclass
class AppContext:
    """Contexto compartilhado entre os menus.

    Diretórios e serviços vêm das mesmas fontes que a API e o Gradio usam
    (``OperationalPaths`` e ``app.dependencies``): um modelo treinado por uma
    interface aparece nas outras, e ``MODELS_DIR``/``XFAKE_STORAGE_DIR`` valem
    também para a CLI.
    """

    paths: OperationalPaths
    datasets_dir: Path
    models_dir: Path
    results_dir: Path
    detection_service: DetectionService
    training_service: TrainingService
    training_config: TrainingConfig
    available_architectures: List[str]
    logger: logging.Logger

    def __init__(self):
        self.paths = OperationalPaths.resolve()
        self.datasets_dir = self.paths.datasets
        self.models_dir = self.paths.models
        self.results_dir = self.paths.results
        for directory in (
            self.models_dir,
            self.results_dir,
            self.datasets_dir / "samples",
            self.datasets_dir / "features",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.detection_service = get_detection_service()
        self.training_service = get_training_service()
        self.training_config = TrainingConfig()
        # Fonte única: as arquiteturas que o TrainingService sabe treinar.
        self.available_architectures = architecture_registry.list_architectures()
        self.logger = logging.getLogger("SpeechCheckerCLI")
