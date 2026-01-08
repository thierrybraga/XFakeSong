"""Factory Pattern para Arquiteturas de Deep Learning

Este módulo implementa um factory pattern robusto para criação de arquiteturas,
melhorando a modularidade e intercambiabilidade do sistema.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Callable, List, Union
from dataclasses import dataclass
import logging
import importlib
import inspect
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureSpec:
    """Especificação de uma arquitetura."""
    name: str
    module_path: str
    factory_function: str
    description: str
    supported_variants: List[str]
    default_params: Dict[str, Any]
    input_requirements: Dict[str, Any]
    output_requirements: Dict[str, Any]
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class IArchitectureFactory(ABC):
    """Interface para factory de arquiteturas."""

    @abstractmethod
    def create_model(self,
                     input_shape: tuple,
                     num_classes: int = 1,
                     variant: str = "default",
                     **kwargs) -> tf.keras.Model:
        """Cria modelo da arquitetura."""
        pass

    @abstractmethod
    def get_supported_variants(self) -> List[str]:
        """Retorna variantes suportadas."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna parâmetros padrão."""
        pass

    @abstractmethod
    def validate_input_shape(self, input_shape: tuple) -> bool:
        """Valida formato de entrada."""
        pass

    @abstractmethod
    def get_input_requirements(self) -> Dict[str, Any]:
        """Retorna requisitos de entrada."""
        pass


class BaseArchitectureFactory(IArchitectureFactory):
    """Factory base para arquiteturas."""

    def __init__(self, spec: ArchitectureSpec):
        self.spec = spec
        self._factory_function = None
        self._load_factory_function()

    def _load_factory_function(self):
        """Carrega função factory do módulo."""
        try:
            module = importlib.import_module(self.spec.module_path)
            self._factory_function = getattr(
                module, self.spec.factory_function)
            logger.info(f"Factory function loaded for {self.spec.name}")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load factory for {self.spec.name}: {e}")
            raise

    def create_model(self,
                     input_shape: tuple,
                     num_classes: int = 1,
                     variant: str = "default",
                     **kwargs) -> tf.keras.Model:
        """Cria modelo da arquitetura."""
        if not self.validate_input_shape(input_shape):
            raise ValueError(
                f"Invalid input shape {input_shape} for {
                    self.spec.name}")

        if variant not in self.spec.supported_variants:
            logger.warning(f"Variant {variant} not supported, using default")
            variant = "default"

        # Combinar parâmetros padrão com os fornecidos
        params = self.spec.default_params.copy()
        params.update(kwargs)
        params['input_shape'] = input_shape
        params['num_classes'] = num_classes

        # Adicionar variant se suportado pela função
        sig = inspect.signature(self._factory_function)
        if 'variant' in sig.parameters or 'architecture' in sig.parameters:
            if 'variant' in sig.parameters:
                params['variant'] = variant
            elif 'architecture' in sig.parameters:
                params['architecture'] = variant

        try:
            model = self._factory_function(**params)
            logger.info(f"Model created: {self.spec.name} ({variant})")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {self.spec.name}: {e}")
            raise

    def get_supported_variants(self) -> List[str]:
        return self.spec.supported_variants.copy()

    def get_default_params(self) -> Dict[str, Any]:
        return self.spec.default_params.copy()

    def validate_input_shape(self, input_shape: tuple) -> bool:
        """Valida formato de entrada."""
        requirements = self.spec.input_requirements

        if 'min_sequence_length' in requirements:
            if len(
                    input_shape) >= 2 and input_shape[0] < requirements['min_sequence_length']:
                return False

        if 'feature_dim' in requirements:
            if len(
                    input_shape) >= 2 and input_shape[1] != requirements['feature_dim']:
                return False

        return True

    def get_input_requirements(self) -> Dict[str, Any]:
        return self.spec.input_requirements.copy()


class ArchitectureFactoryRegistry:
    """Registry centralizado para factories de arquiteturas."""

    def __init__(self):
        self._factories: Dict[str, IArchitectureFactory] = {}
        self._specs: Dict[str, ArchitectureSpec] = {}
        self._register_default_architectures()

    def register_factory(self, spec: ArchitectureSpec,
                         factory_class: Type[IArchitectureFactory] = None):
        """Registra uma nova factory."""
        if factory_class is None:
            factory_class = BaseArchitectureFactory

        try:
            factory = factory_class(spec)
            self._factories[spec.name] = factory
            self._specs[spec.name] = spec
            logger.info(f"Registered factory for {spec.name}")
        except Exception as e:
            logger.error(f"Failed to register factory for {spec.name}: {e}")

    def get_factory(
            self, architecture_name: str) -> Optional[IArchitectureFactory]:
        """Retorna factory para arquitetura."""
        return self._factories.get(architecture_name)

    def create_model(self,
                     architecture_name: str,
                     input_shape: tuple,
                     num_classes: int = 1,
                     variant: str = "default",
                     **kwargs) -> tf.keras.Model:
        """Cria modelo usando factory registrada."""
        factory = self.get_factory(architecture_name)
        if factory is None:
            raise ValueError(
                f"Architecture {architecture_name} not registered")

        return factory.create_model(
            input_shape, num_classes, variant, **kwargs)

    def list_architectures(self) -> List[str]:
        """Lista arquiteturas disponíveis."""
        return list(self._factories.keys())

    def get_architecture_info(
            self, architecture_name: str) -> Optional[ArchitectureSpec]:
        """Retorna informações da arquitetura."""
        return self._specs.get(architecture_name)

    def get_supported_variants(self, architecture_name: str) -> List[str]:
        """Retorna variantes suportadas."""
        factory = self.get_factory(architecture_name)
        return factory.get_supported_variants() if factory else []

    def validate_compatibility(self,
                               architecture_name: str,
                               input_shape: tuple) -> bool:
        """Valida compatibilidade entre arquitetura e entrada."""
        factory = self.get_factory(architecture_name)
        return factory.validate_input_shape(input_shape) if factory else False

    def _register_default_architectures(self):
        """Registra arquiteturas padrão do sistema."""

        # AASIST
        self.register_factory(ArchitectureSpec(
            name="AASIST",
            module_path="app.domain.models.architectures.aasist",
            factory_function="create_model",
            description="Anti-spoofing Audio Spoofing and Deepfake Detection",
            supported_variants=[
                "default",
                "cnn_baseline",
                "bidirectional_gru",
                "resnet_gru",
                "transformer",
                "aasist"],
            default_params={
                "dropout_rate": 0.2,
                "l2_reg_strength": 0.0005,
                "hidden_dim": 512,
                "num_layers": 8,
                "use_early_stopping": True,
                "use_gradient_clipping": True,
                "use_advanced_augmentation": True
            },
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # RawGAT-ST
        self.register_factory(ArchitectureSpec(
            name="RawGAT-ST",
            module_path="app.domain.models.architectures.rawgat_st",
            factory_function="create_model",
            description="Raw Graph Attention Spatio-Temporal Network",
            supported_variants=[
                "default",
                "cnn_baseline",
                "bidirectional_gru",
                "resnet_gru",
                "transformer",
                "rawgat_st"],
            default_params={
                "dropout_rate": 0.2,
                "l2_reg_strength": 0.0005,
                "attention_heads": 8,
                "hidden_dim": 512,
                "num_layers": 6,
                "use_early_stopping": True,
                "use_gradient_clipping": True,
                "use_advanced_augmentation": True
            },
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # EfficientNet-LSTM
        self.register_factory(ArchitectureSpec(
            name="EfficientNet-LSTM",
            module_path="app.domain.models.architectures.efficientnet_lstm",
            factory_function="create_model",
            description="EfficientNet with LSTM for temporal modeling",
            supported_variants=["efficientnet_lstm", "efficientnet_lstm_lite"],
            default_params={
                "lstm_units": 512,
                "attention_units": 256,
                "dropout_rate": 0.2},
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # MultiscaleCNN
        self.register_factory(ArchitectureSpec(
            name="MultiscaleCNN",
            module_path="app.domain.models.architectures.multiscale_cnn",
            factory_function="create_model",
            description="Multi-Scale Convolutional Neural Network",
            supported_variants=["multiscale_cnn", "multiscale_cnn_lite"],
            default_params={
                "architecture": "multiscale_cnn",
                "filters": [64, 128, 256, 512],
                "kernel_sizes": [3, 5, 7],
                "dropout_rate": 0.2,
                "use_early_stopping": True,
                "use_gradient_clipping": True,
                "use_advanced_augmentation": True
            },
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # SpectrogramTransformer
        self.register_factory(ArchitectureSpec(
            name="SpectrogramTransformer",
            module_path="app.domain.models.architectures.spectrogram_transformer",
            factory_function="create_model",
            description="Transformer-based model for spectrogram analysis",
            supported_variants=[
                "spectrogram_transformer",
                "spectrogram_transformer_lite"],
            default_params={
                "d_model": 512,
                "num_heads": 16,
                "num_blocks": 12,
                "dropout_rate": 0.1},
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # Conformer
        self.register_factory(ArchitectureSpec(
            name="Conformer",
            module_path="app.domain.models.architectures.conformer",
            factory_function="create_model",
            description="Conformer: Convolution-augmented Transformer",
            supported_variants=["conformer", "conformer_lite"],
            default_params={
                "d_model": 512,
                "num_blocks": 12,
                "num_heads": 16,
                "dropout_rate": 0.1},
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # Ensemble
        self.register_factory(ArchitectureSpec(
            name="Ensemble",
            module_path="app.domain.models.architectures.ensemble",
            factory_function="create_model",
            description="Ensemble of multiple architectures",
            supported_variants=[
                "ensemble",
                "ensemble_lite",
                "ensemble_feature_fusion",
                "ensemble_hybrid"],
            default_params={
                "ensemble_method": "weighted_average",
                "fusion_method": "prediction_level"},
            input_requirements={"min_sequence_length": 100, "feature_dim": 80},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # Sonic Sleuth
        self.register_factory(ArchitectureSpec(
            name="Sonic Sleuth",
            module_path="app.domain.models.architectures.sonic_sleuth",
            factory_function="create_model",
            description="CNN especializada para detecção de deepfake usando espectrogramas de mel",
            supported_variants=["sonic_sleuth"],
            default_params={
                "sample_rate": 16000, "n_fft": 2048, "hop_length": 512,
                "n_mels": 256, "dropout_rate": 0.2, "filters": [64, 128, 256, 512]
            },
            input_requirements={"max_duration": 3.0, "sample_rate": 16000},
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # RawNet2
        self.register_factory(ArchitectureSpec(
            name="RawNet2",
            module_path="app.domain.models.architectures.rawnet2",
            factory_function="create_model",
            description="Arquitetura de rede neural que opera diretamente no áudio bruto",
            supported_variants=["rawnet2", "rawnet2_lite"],
            default_params={
                "conv_filters": [128, 256, 512],
                "gru_units": 256,
                "dense_units": 128,
                "dropout_rate": 0.2
            },
            input_requirements={
                "type": "audio", "format": "raw", "sample_rate": 16000,
                "max_duration": 5.0, "preprocessing": "normalize"
            },
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # WavLM
        self.register_factory(ArchitectureSpec(
            name="WavLM",
            module_path="app.domain.models.architectures.wavlm",
            factory_function="create_model",
            description="Arquitetura de dois estágios com WavLM pré-treinado",
            supported_variants=["wavlm", "wavlm_lite"],
            default_params={
                "wavlm_model": "microsoft/wavlm-large",
                "freeze_wavlm": True,
                "classifier_units": [1024, 512, 256],
                "dropout_rate": 0.2
            },
            input_requirements={
                "type": "audio", "format": "raw", "sample_rate": 16000,
                "max_duration": 10.0, "preprocessing": "normalize"
            },
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # HuBERT
        self.register_factory(ArchitectureSpec(
            name="HuBERT",
            module_path="app.domain.models.architectures.hubert",
            factory_function="create_model",
            description="Arquitetura HuBERT padrão para detecção de deepfakes",
            supported_variants=["hubert", "hubert_lite"],
            default_params={
                "num_classes": 1, "architecture": "hubert", "hidden_size": 768,
                "num_attention_heads": 12, "num_hidden_layers": 12,
                "classifier_hidden_dim": 256, "dropout_rate": 0.3
            },
            input_requirements={
                "type": "audio", "format": "raw", "sample_rate": 16000,
                "max_duration": 10.0, "preprocessing": "normalize"
            },
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))

        # Hybrid CNN-Transformer
        self.register_factory(ArchitectureSpec(
            name="Hybrid CNN-Transformer",
            module_path="app.domain.models.architectures.hybrid_cnn_transformer",
            factory_function="create_model",
            description="Arquitetura híbrida que combina CNNs com Transformers",
            supported_variants=[
                "hybrid_cnn_transformer",
                "hybrid_cnn_transformer_lite"],
            default_params={
                "num_classes": 1, "architecture": "hybrid_cnn_transformer",
                "base_filters": 64, "num_residual_blocks": 3,
                "num_transformer_layers": 2, "attention_heads": 8, "dropout_rate": 0.1
            },
            input_requirements={
                "min_sequence_length": 100, "feature_dim": 80,
                "supports_1d_input": True, "supports_2d_input": True,
                "preprocessing": "spectrogram_or_raw"
            },
            output_requirements={
                "type": "classification",
                "activation": "sigmoid"}
        ))


# Instância global do registry
architecture_factory_registry = ArchitectureFactoryRegistry()


# Funções de conveniência
def create_model_by_name(architecture_name: str,
                         input_shape: tuple,
                         num_classes: int = 1,
                         variant: str = "default",
                         **kwargs) -> tf.keras.Model:
    """Cria modelo usando o registry global."""
    return architecture_factory_registry.create_model(
        architecture_name, input_shape, num_classes, variant, **kwargs
    )


def get_available_architectures() -> List[str]:
    """Retorna lista de arquiteturas disponíveis."""
    return architecture_factory_registry.list_architectures()


def get_architecture_info(
        architecture_name: str) -> Optional[ArchitectureSpec]:
    """Retorna informações da arquitetura."""
    return architecture_factory_registry.get_architecture_info(
        architecture_name)


def validate_architecture_compatibility(architecture_name: str,
                                        input_shape: tuple) -> bool:
    """Valida compatibilidade entre arquitetura e entrada."""
    return architecture_factory_registry.validate_compatibility(
        architecture_name, input_shape
    )
