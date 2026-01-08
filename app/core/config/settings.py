"""Configuração centralizada do sistema

Este módulo centraliza todas as configurações seguindo o princípio DRY
e facilita a manutenção e modificação de parâmetros do sistema.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import json
from enum import Enum


class Environment(Enum):
    """Ambientes de execução."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Níveis de log."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class PathConfig:
    """Configurações de caminhos."""
    base_dir: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("./app/datasets"))
    logs_dir: Path = field(default_factory=lambda: Path("./app/logs"))
    temp_dir: Path = field(default_factory=lambda: Path("./app/temp"))
    models_dir: Path = field(
        default_factory=lambda: Path("./app/artifacts/models"))

    # Subdiretórios de dados
    datasets_dir: Path = field(default_factory=lambda: Path("./app/datasets"))
    features_dir: Path = field(
        default_factory=lambda: Path("./app/datasets/features"))
    samples_dir: Path = field(default_factory=lambda: Path("./app/datasets"))

    # Diretórios de upload
    uploads_dir: Path = field(default_factory=lambda: Path("./app/uploads"))
    upload_training_dir: Path = field(
        default_factory=lambda: Path("./app/uploads/training"))
    upload_validation_dir: Path = field(
        default_factory=lambda: Path("./app/uploads/validation"))
    upload_test_dir: Path = field(
        default_factory=lambda: Path("./app/uploads/test"))
    upload_production_dir: Path = field(
        default_factory=lambda: Path("./app/uploads/production"))

    def __post_init__(self):
        """Ajusta caminhos relativos ao base_dir."""
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        if not self.logs_dir.is_absolute():
            self.logs_dir = self.base_dir / self.logs_dir
        if not self.temp_dir.is_absolute():
            self.temp_dir = self.base_dir / self.temp_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.base_dir / self.models_dir
        if not self.datasets_dir.is_absolute():
            self.datasets_dir = self.base_dir / self.datasets_dir
        if not self.features_dir.is_absolute():
            self.features_dir = self.base_dir / self.features_dir
        if not self.samples_dir.is_absolute():
            self.samples_dir = self.base_dir / self.samples_dir

    def create_directories(self):
        """Cria todos os diretórios necessários."""
        directories = [
            self.base_dir, self.data_dir, self.logs_dir, self.temp_dir,
            self.models_dir, self.datasets_dir, self.features_dir, self.samples_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Configurações de processamento de áudio."""
    sample_rate: int = 16000
    frame_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_fft: int = 2048
    max_duration: float = 30.0  # segundos
    min_duration: float = 1.0   # segundos
    allowed_formats: List[str] = field(
        default_factory=lambda: [
            ".wav", ".mp3", ".flac", ".m4a"])
    normalize: bool = True
    remove_silence: bool = True


@dataclass
class FeatureConfig:
    """Configurações de extração de características."""
    feature_groups: List[str] = field(
        default_factory=lambda: [
            "spectral",
            "temporal",
            "prosodic",
            "perceptual"])
    save_formats: List[str] = field(default_factory=lambda: ["json", "npy"])
    parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 32
    cache_features: bool = True

    # Configurações específicas por tipo
    spectral_config: Dict[str, Any] = field(default_factory=lambda: {
        "n_mfcc": 13,
        "n_chroma": 12,
        "n_contrast": 7
    })

    temporal_config: Dict[str, Any] = field(default_factory=lambda: {
        "window_size": 1024,
        "overlap": 0.5
    })

    prosodic_config: Dict[str, Any] = field(default_factory=lambda: {
        "f0_min": 75,
        "f0_max": 400
    })


@dataclass
class TrainingConfig:
    """Configurações de treinamento."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    # Arquiteturas disponíveis
    available_architectures: List[str] = field(default_factory=lambda: [
        "aasist", "conformer", "efficientnet_lstm", "ensemble",
        "multiscale_cnn", "rawgat_st", "spectrogram_transformer"
    ])

    # Configurações de otimização
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1"])

    # Data augmentation
    use_augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        "noise_factor": 0.1,
        "time_stretch_factor": 0.1,
        "pitch_shift_steps": 2
    })


@dataclass
class APIConfig:
    """Configurações da API."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    upload_timeout: int = 600  # 10 minutos
    enable_docs: bool = True
    api_version: str = "v1"

    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60

    # Authentication
    auth_enabled: bool = False
    jwt_secret: Optional[str] = None
    jwt_expiration: int = 3600  # 1 hora


@dataclass
class DatabaseConfig:
    """Configurações de banco de dados."""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "deepfake_detection"
    username: Optional[str] = None
    password: Optional[str] = None

    # SQLite específico
    sqlite_path: Path = field(
        default_factory=lambda: Path("./data/database.db"))

    # Pool de conexões
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30


@dataclass
class LoggingConfig:
    """Configurações de logging."""
    level: LogLevel = LogLevel.INFO
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "1 day"
    retention: str = "30 days"
    compression: str = "gz"

    # Logs específicos
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    enable_json_logging: bool = False

    # Configurações por módulo
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "upload": "INFO",
        "feature_extraction": "INFO",
        "training": "INFO",
        "api": "WARNING"
    })


@dataclass
class MonitoringConfig:
    """Configurações de monitoramento."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval: int = 30  # segundos

    # Alertas
    enable_alerts: bool = False
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None

    # Métricas específicas
    track_performance: bool = True
    track_resource_usage: bool = True
    track_errors: bool = True


@dataclass
class SecurityConfig:
    """Configurações de segurança."""
    enable_file_validation: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_mime_types: List[str] = field(default_factory=lambda: [
        "audio/wav", "audio/mpeg", "audio/flac", "audio/mp4"
    ])

    # Sanitização
    sanitize_filenames: bool = True
    quarantine_suspicious_files: bool = True

    # Criptografia
    encrypt_sensitive_data: bool = False
    encryption_key: Optional[str] = None


@dataclass
class SystemConfig:
    """Configuração principal do sistema."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    version: str = "1.0.0"

    # Configurações por módulo
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self):
        """Ajustes pós-inicialização."""
        # Ajustar debug baseado no ambiente
        if self.environment == Environment.PRODUCTION:
            self.debug = False
            self.api.debug = False
            self.logging.level = LogLevel.WARNING

        # Criar diretórios necessários
        self.paths.create_directories()

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'SystemConfig':
        """Carrega configuração de arquivo JSON."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        return cls(**config_data)

    def to_file(self, config_path: Union[str, Path]):
        """Salva configuração em arquivo JSON."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Converter para dicionário serializável
        config_dict = self._to_serializable_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário serializável."""
        def convert_value(value):
            if isinstance(value, Path):
                return str(value)
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, '__dict__'):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value

        return convert_value(self)

    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Carrega configuração de variáveis de ambiente."""
        config = cls()

        # Sobrescrever com variáveis de ambiente se existirem
        if os.getenv('DEEPFAKE_ENV'):
            config.environment = Environment(os.getenv('DEEPFAKE_ENV'))

        if os.getenv('DEEPFAKE_DEBUG'):
            config.debug = os.getenv('DEEPFAKE_DEBUG').lower() == 'true'

        if os.getenv('DEEPFAKE_API_HOST'):
            config.api.host = os.getenv('DEEPFAKE_API_HOST')

        if os.getenv('DEEPFAKE_API_PORT'):
            config.api.port = int(os.getenv('DEEPFAKE_API_PORT'))

        return config


# Instância global de configuração
_config_instance: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Retorna instância global de configuração."""
    global _config_instance
    if _config_instance is None:
        _config_instance = SystemConfig.from_env()
    return _config_instance


def set_config(config: SystemConfig):
    """Define instância global de configuração."""
    global _config_instance
    _config_instance = config


def load_config_from_file(config_path: Union[str, Path]):
    """Carrega e define configuração de arquivo."""
    config = SystemConfig.from_file(config_path)
    set_config(config)
    return config
