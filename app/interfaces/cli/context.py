from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
import logging

from app.core.config.settings import TrainingConfig
from app.domain.services.feature_extraction_service import AudioFeatureExtractionService
from app.domain.services.detection_service import DetectionService
from app.domain.models.training.trainer import ModelTrainer


@dataclass
class AppContext:
    """Contexto da aplicação compartilhado entre os menus."""
    app_dir: Path
    datasets_dir: Path
    models_dir: Path
    results_dir: Path
    feature_service: AudioFeatureExtractionService
    detection_service: DetectionService
    trainer: ModelTrainer
    training_config: TrainingConfig
    available_architectures: List[str]
    logger: logging.Logger

    def __init__(self):
        self.app_dir = Path(__file__).resolve(
        ).parent.parent.parent.parent / "app"
        self.datasets_dir = self.app_dir / "datasets"
        self.models_dir = self.app_dir / "models"
        self.results_dir = self.app_dir / "results"

        # Criar diretórios se não existirem
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.datasets_dir.mkdir(exist_ok=True, parents=True)
        (self.datasets_dir / "samples").mkdir(exist_ok=True)
        (self.datasets_dir / "features").mkdir(exist_ok=True)

        # Inicializar serviços
        self.feature_service = AudioFeatureExtractionService()
        self.detection_service = DetectionService(self.models_dir)

        # Criar configuração de treinamento
        self.training_config = TrainingConfig(
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            validation_split=0.2
        )
        self.trainer = ModelTrainer(self.training_config)

        # Arquiteturas disponíveis
        self.available_architectures = [
            "aasist", "conformer", "efficientnet_lstm", "ensemble",
            "multiscale_cnn", "rawgat_st", "spectrogram_transformer",
            "hubert", "rawnet2", "wavlm", "hybrid_cnn_transformer", "sonic_sleuth"
        ]

        self.logger = logging.getLogger("XfakeSongCLI")
