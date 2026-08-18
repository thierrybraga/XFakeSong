"""Factory Pattern para Arquiteturas de Deep Learning

Este módulo implementa um factory pattern robusto para criação de arquiteturas,
melhorando a modularidade e intercambiabilidade do sistema.
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import tensorflow as tf

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
                     num_classes: int = 2,
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
                     # num_classes=2 alinha com ``ArchitectureRegistry.create_model``
                     # e com a convenção do projeto (real/fake = 2 classes,
                     # softmax de 2 unidades). O default 1 anterior fazia as
                     # duas portas de entrada produzirem CABEÇAS DIFERENTES
                     # para a mesma chamada — e cabeças de 1 unidade degeneram
                     # em arquiteturas com loss categórica.
                     num_classes: int = 2,
                     variant: str = "default",
                     **kwargs) -> tf.keras.Model:
        """Cria modelo da arquitetura."""
        if not self.validate_input_shape(input_shape):
            raise ValueError(
                f"Invalid input shape {input_shape} for {self.spec.name}")

        # "default" é o sentinel universal de fallback (valor padrão do argumento)
        # e sempre é aceito, mesmo quando não está listado em supported_variants —
        # só avisamos quando o chamador pede explicitamente um variant inexistente.
        if variant != "default" and variant not in self.spec.supported_variants:
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

        # Filtrar params para apenas os aceitos pela função (quando não há **kwargs)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_keyword:
            accepted = set(sig.parameters.keys())
            filtered_out = [k for k in params if k not in accepted]
            if filtered_out:
                logger.debug(f"{self.spec.name}: ignoring unsupported params: {filtered_out}")
            params = {k: v for k, v in params.items() if k in accepted}

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
        """Valida formato de entrada.

        Notas:
        - `input_type` define a família esperada do tensor:
            * "raw_audio"     → (T, 1) ou (T,) — T >= min_sequence_length
            * "spectrogram"   → (T, F) ou (T, F, 1) — feature_dim valida F
            * "any" (default) → não valida shape rigorosamente
        - Constraints legadas (`feature_dim`, `min_sequence_length`)
          continuam funcionando para compat.
        """
        requirements = self.spec.input_requirements
        input_type = requirements.get("input_type", "any")

        # --- Validações por input_type ---
        if input_type == "raw_audio":
            # Aceita (T, 1) ou (T,) — só checa o min length se especificado
            if len(input_shape) == 0:
                return False
            T = input_shape[0]
            if "min_sequence_length" in requirements:
                if T < requirements["min_sequence_length"]:
                    return False
            # Se for (T, C), C deve ser 1
            if len(input_shape) >= 2 and input_shape[-1] != 1:
                # Permitir (T, 1) ou (T,), rejeitar (T, K>1) para raw audio
                if len(input_shape) == 2:
                    return False
            return True

        if input_type == "spectrogram":
            # Aceita (T, F) ou (T, F, 1)
            if len(input_shape) < 2:
                return False
            T = input_shape[0]
            F = input_shape[1]
            if "min_sequence_length" in requirements:
                if T < requirements["min_sequence_length"]:
                    return False
            if "feature_dim" in requirements:
                if F != requirements["feature_dim"]:
                    return False
            # (T, F, 1) — terceira dim deve ser 1
            if len(input_shape) == 3 and input_shape[2] != 1:
                return False
            return True

        # --- Caminho legado (compat com specs sem input_type) ---
        if "min_sequence_length" in requirements:
            if (
                len(input_shape) >= 2
                and input_shape[0] < requirements["min_sequence_length"]
            ):
                return False
        if "feature_dim" in requirements:
            if (
                len(input_shape) >= 2
                and input_shape[1] != requirements["feature_dim"]
            ):
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
                     num_classes: int = 2,
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

    # Saída esperada por arquitetura — única informação que NÃO vive no
    # ArchitectureRegistry; todo o resto é derivado dele (fonte única).
    #
    # CORREÇÃO: os valores estavam desatualizados. A convenção do projeto é
    # `num_classes == 1 → sigmoid de 1 unidade`, `num_classes >= 2 → softmax de
    # N unidades` — ou seja, a ativação depende de num_classes para quase todas
    # as arquiteturas, e o benchmark usa num_classes=2 (softmax). Só AASIST e
    # RawGAT-ST fogem disso: emitem LOGITS (a loss aplica o softmax).
    # "sigmoid_or_softmax" deixa explícito que a ativação é condicional em vez
    # de afirmar "sigmoid" para modelos que na prática emitem softmax.
    _OUTPUT_REQUIREMENTS = {
        "AASIST": {"type": "classification", "activation": "logits"},
        "RawGAT-ST": {"type": "classification", "activation": "logits"},
        "EfficientNet-LSTM": {
            "type": "classification", "activation": "sigmoid_or_softmax"
        },
        "MultiscaleCNN": {
            "type": "classification", "activation": "sigmoid_or_softmax"
        },
        "SpectrogramTransformer": {
            "type": "classification", "activation": "sigmoid_or_softmax"
        },
        # Conformer promove num_classes<2 para 2 → sempre softmax.
        "Conformer": {"type": "classification", "activation": "softmax"},
        "Ensemble": {"type": "classification", "activation": "sigmoid_or_softmax"},
        "Sonic Sleuth": {
            "type": "classification", "activation": "sigmoid_or_softmax"
        },
        "RawNet2": {"type": "classification", "activation": "sigmoid_or_softmax"},
        "WavLM": {"type": "classification", "activation": "sigmoid_or_softmax"},
        "HuBERT": {"type": "classification", "activation": "sigmoid_or_softmax"},
        "Hybrid CNN-Transformer": {
            "type": "classification", "activation": "sigmoid_or_softmax"
        },
    }

    def _register_default_architectures(self):
        """Deriva as specs do ``ArchitectureRegistry`` (fonte única de verdade).

        Historicamente este módulo DUPLICAVA nome/variantes/default_params/
        input_requirements por arquitetura, e as duas cópias divergiram
        silenciosamente (ex.: l2 do AASIST 5e-4 aqui vs 2e-4 no registry;
        EfficientNet-LSTM "spectrogram" aqui vs "raw_audio" lá). Agora a
        factory materializa ``ArchitectureSpec`` a partir do registry e só
        acrescenta ``output_requirements`` (campo que o registry não modela).

        Nota: ``default_params`` do registry incluem chaves de TREINO
        (patience, gradient_clip, augmentation_strength) que pertencem ao
        pipeline, não ao ``create_model``. Elas são EXCLUÍDAS aqui: o filtro
        por assinatura em ``create_model`` não protege funções com
        ``**kwargs`` (ex.: multiscale_cnn encaminha kwargs para o builder
        interno, que rejeitaria ``patience`` com TypeError).
        """
        from app.domain.models.architectures.registry import (
            _TRAINING_ONLY_PARAM_KEYS as training_only_keys,
            architecture_registry,
        )

        for info in architecture_registry.get_all_architectures().values():
            model_params = {
                key: value
                for key, value in info.default_params.items()
                if key not in training_only_keys
            }
            self.register_factory(ArchitectureSpec(
                name=info.name,
                module_path=info.module_path,
                factory_function=info.function_name,
                description=info.description,
                supported_variants=list(info.supported_variants),
                default_params=model_params,
                input_requirements=dict(info.input_requirements),
                output_requirements=dict(self._OUTPUT_REQUIREMENTS.get(
                    info.name,
                    {"type": "classification", "activation": "sigmoid"},
                )),
            ))


# Instância global do registry
architecture_factory_registry = ArchitectureFactoryRegistry()


# Funções de conveniência
def create_model_by_name(architecture_name: str,
                         input_shape: tuple,
                         num_classes: int = 2,
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
