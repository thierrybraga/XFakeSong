"""Interfaces padronizadas para componentes do pipeline.

Este módulo define interfaces comuns para todos os componentes do pipeline,
garantindo intercambiabilidade e modularidade.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path


class ComponentType(Enum):
    """Tipos de componentes do pipeline."""
    FEATURE_EXTRACTOR = "feature_extractor"
    ARCHITECTURE = "architecture"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    VALIDATOR = "validator"
    ORCHESTRATOR = "orchestrator"


class ProcessingStage(Enum):
    """Estágios de processamento."""
    INPUT_VALIDATION = "input_validation"
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_INFERENCE = "model_inference"
    POSTPROCESSING = "postprocessing"
    OUTPUT_VALIDATION = "output_validation"


@dataclass
class ProcessingContext:
    """Contexto de processamento compartilhado entre componentes."""
    session_id: str
    input_path: Optional[Path] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None
    stage: ProcessingStage = ProcessingStage.INPUT_VALIDATION

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Resultado de processamento padronizado."""
    success: bool
    data: Any = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class IPipelineComponent(ABC):
    """Interface base para todos os componentes do pipeline."""

    @abstractmethod
    def process(self, data: Any,
                context: ProcessingContext) -> ProcessingResult:
        """Processa dados com contexto.

        Args:
            data: Dados de entrada
            context: Contexto de processamento

        Returns:
            Resultado do processamento
        """
        pass

    @abstractmethod
    def validate_input(self, data: Any, context: ProcessingContext) -> bool:
        """Valida dados de entrada.

        Args:
            data: Dados para validação
            context: Contexto de processamento

        Returns:
            True se válido
        """
        pass

    @abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        """Retorna informações do componente.

        Returns:
            Dicionário com informações
        """
        pass

    def get_requirements(self) -> Dict[str, Any]:
        """Retorna requisitos do componente.

        Returns:
            Dicionário com requisitos
        """
        return {}

    def setup(self, config: Dict[str, Any]) -> bool:
        """Configura o componente.

        Args:
            config: Configuração

        Returns:
            True se configurado com sucesso
        """
        return True

    def cleanup(self) -> None:
        """Limpa recursos do componente."""
        pass


class IFeatureExtractor(IPipelineComponent):
    """Interface para extratores de features."""

    @abstractmethod
    def extract_features(self, audio_data: np.ndarray,
                         context: ProcessingContext) -> ProcessingResult:
        """Extrai features do áudio.

        Args:
            audio_data: Dados de áudio
            context: Contexto de processamento

        Returns:
            Resultado com features extraídas
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retorna nomes das features.

        Returns:
            Lista de nomes
        """
        pass

    @abstractmethod
    def get_output_shape(
            self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calcula shape de saída.

        Args:
            input_shape: Shape de entrada

        Returns:
            Shape de saída
        """
        pass

    def process(self, data: Any,
                context: ProcessingContext) -> ProcessingResult:
        """Implementação padrão do método process."""
        if not isinstance(data, np.ndarray):
            return ProcessingResult(
                success=False,
                errors=["Dados de entrada devem ser numpy.ndarray"]
            )

        return self.extract_features(data, context)


class IArchitecture(IPipelineComponent):
    """Interface para arquiteturas de deep learning."""

    @abstractmethod
    def predict(self, features: np.ndarray,
                context: ProcessingContext) -> ProcessingResult:
        """Realiza predição.

        Args:
            features: Features de entrada
            context: Contexto de processamento

        Returns:
            Resultado da predição
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Path) -> bool:
        """Carrega modelo.

        Args:
            model_path: Caminho do modelo

        Returns:
            True se carregado com sucesso
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo.

        Returns:
            Dicionário com informações
        """
        pass

    def process(self, data: Any,
                context: ProcessingContext) -> ProcessingResult:
        """Implementação padrão do método process."""
        if not isinstance(data, np.ndarray):
            return ProcessingResult(
                success=False,
                errors=["Features devem ser numpy.ndarray"]
            )

        return self.predict(data, context)


class IPreprocessor(IPipelineComponent):
    """Interface para preprocessadores."""

    @abstractmethod
    def preprocess(self, audio_data: np.ndarray,
                   context: ProcessingContext) -> ProcessingResult:
        """Preprocessa dados de áudio.

        Args:
            audio_data: Dados de áudio
            context: Contexto de processamento

        Returns:
            Resultado do preprocessamento
        """
        pass

    def process(self, data: Any,
                context: ProcessingContext) -> ProcessingResult:
        """Implementação padrão do método process."""
        if not isinstance(data, np.ndarray):
            return ProcessingResult(
                success=False,
                errors=["Dados de entrada devem ser numpy.ndarray"]
            )

        return self.preprocess(data, context)


class IPostprocessor(IPipelineComponent):
    """Interface para postprocessadores."""

    @abstractmethod
    def postprocess(self, predictions: np.ndarray,
                    context: ProcessingContext) -> ProcessingResult:
        """Postprocessa predições.

        Args:
            predictions: Predições do modelo
            context: Contexto de processamento

        Returns:
            Resultado do postprocessamento
        """
        pass

    def process(self, data: Any,
                context: ProcessingContext) -> ProcessingResult:
        """Implementação padrão do método process."""
        return self.postprocess(data, context)


class IValidator(IPipelineComponent):
    """Interface para validadores."""

    @abstractmethod
    def validate(self, data: Any,
                 context: ProcessingContext) -> ProcessingResult:
        """Valida dados.

        Args:
            data: Dados para validação
            context: Contexto de processamento

        Returns:
            Resultado da validação
        """
        pass

    def process(self, data: Any,
                context: ProcessingContext) -> ProcessingResult:
        """Implementação padrão do método process."""
        return self.validate(data, context)


class IPipelineOrchestrator(ABC):
    """Interface para orquestradores de pipeline."""

    @abstractmethod
    def add_component(self, component: IPipelineComponent,
                      stage: ProcessingStage) -> None:
        """Adiciona componente ao pipeline.

        Args:
            component: Componente a adicionar
            stage: Estágio de processamento
        """
        pass

    @abstractmethod
    def remove_component(self, component_id: str) -> bool:
        """Remove componente do pipeline.

        Args:
            component_id: ID do componente

        Returns:
            True se removido com sucesso
        """
        pass

    @abstractmethod
    def execute_pipeline(self, input_data: Any,
                         context: ProcessingContext) -> ProcessingResult:
        """Executa pipeline completo.

        Args:
            input_data: Dados de entrada
            context: Contexto de processamento

        Returns:
            Resultado final do pipeline
        """
        pass

    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Retorna informações do pipeline.

        Returns:
            Dicionário com informações
        """
        pass

    @abstractmethod
    def validate_pipeline(self) -> List[str]:
        """Valida configuração do pipeline.

        Returns:
            Lista de erros de validação
        """
        pass


class IConfigurationManager(ABC):
    """Interface para gerenciador de configurações."""

    @abstractmethod
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Carrega configuração.

        Args:
            config_path: Caminho da configuração

        Returns:
            Dicionário de configuração
        """
        pass

    @abstractmethod
    def save_config(self, config: Dict[str, Any], config_path: Path) -> bool:
        """Salva configuração.

        Args:
            config: Configuração a salvar
            config_path: Caminho para salvar

        Returns:
            True se salvo com sucesso
        """
        pass

    @abstractmethod
    def get_component_config(self, component_type: ComponentType,
                             component_name: str) -> Dict[str, Any]:
        """Obtém configuração de componente.

        Args:
            component_type: Tipo do componente
            component_name: Nome do componente

        Returns:
            Configuração do componente
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Valida configuração.

        Args:
            config: Configuração a validar

        Returns:
            Lista de erros de validação
        """
        pass


class IPluginManager(ABC):
    """Interface para gerenciador de plugins."""

    @abstractmethod
    def load_plugin(self, plugin_path: Path) -> bool:
        """Carrega plugin.

        Args:
            plugin_path: Caminho do plugin

        Returns:
            True se carregado com sucesso
        """
        pass

    @abstractmethod
    def unload_plugin(self, plugin_name: str) -> bool:
        """Descarrega plugin.

        Args:
            plugin_name: Nome do plugin

        Returns:
            True se descarregado com sucesso
        """
        pass

    @abstractmethod
    def list_plugins(self) -> List[str]:
        """Lista plugins carregados.

        Returns:
            Lista de nomes de plugins
        """
        pass

    @abstractmethod
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Obtém informações de plugin.

        Args:
            plugin_name: Nome do plugin

        Returns:
            Informações do plugin
        """
        pass


# Protocolos para tipagem
class AudioProcessor(Protocol):
    """Protocolo para processadores de áudio."""

    def process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Processa áudio."""
        ...


class FeatureProcessor(Protocol):
    """Protocolo para processadores de features."""

    def process_features(self, features: np.ndarray) -> np.ndarray:
        """Processa features."""
        ...


class ModelPredictor(Protocol):
    """Protocolo para preditores de modelo."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Realiza predição."""
        ...


# Tipos de dados comuns
AudioData = Union[np.ndarray, List[float]]
FeatureData = Union[np.ndarray, List[List[float]]]
PredictionData = Union[np.ndarray, List[float], Dict[str, float]]
ConfigData = Dict[str, Any]
MetadataData = Dict[str, Any]


# Constantes
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = np.float32
MAX_AUDIO_LENGTH = 30.0  # segundos
MIN_AUDIO_LENGTH = 0.1   # segundos


# Exceções customizadas
class PipelineError(Exception):
    """Erro base do pipeline."""
    pass


class ComponentError(PipelineError):
    """Erro de componente."""
    pass


class ValidationError(PipelineError):
    """Erro de validação."""
    pass


class ConfigurationError(PipelineError):
    """Erro de configuração."""
    pass


class PluginError(PipelineError):
    """Erro de plugin."""
    pass
