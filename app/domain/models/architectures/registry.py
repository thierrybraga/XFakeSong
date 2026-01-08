"""Registry de Arquiteturas Disponíveis

Este módulo centraliza o registro de todas as arquiteturas de deep learning
disponíveis no sistema, facilitando a integração com o pipeline de detecção.
"""

from typing import Dict, List, Callable, Any, Tuple
from dataclasses import dataclass
import logging
from .architecture_patcher import patch_architecture_for_safety, validate_model_safety

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureInfo:
    """Informações sobre uma arquitetura."""
    name: str
    module_path: str
    function_name: str
    description: str
    supported_variants: List[str]
    default_params: Dict[str, Any]
    input_requirements: Dict[str, Any]


class ArchitectureRegistry:
    """Registry centralizado de arquiteturas."""

    def __init__(self):
        self._architectures: Dict[str, ArchitectureInfo] = {}
        self._register_default_architectures()

    def _register_default_architectures(self):
        """Registra as arquiteturas padrão do sistema."""

        # AASIST
        self.register(
            ArchitectureInfo(
                name="AASIST",
                module_path="app.domain.models.architectures.aasist",
                function_name="create_model",
                description="Anti-spoofing Audio Spoofing and Deepfake Detection - configuração otimizada para reduzir overfitting",
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
                    "use_advanced_augmentation": True,
                    # Training params
                    "patience": 20,
                    "lr_patience": 10,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.25
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # RawGAT-ST
        self.register(
            ArchitectureInfo(
                name="RawGAT-ST",
                module_path="app.domain.models.architectures.rawgat_st",
                function_name="create_model",
                description="Raw Graph Attention Spatio-Temporal Network - configuração otimizada para reduzir overfitting",
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
                    "use_advanced_augmentation": True,
                    # Training params
                    "patience": 18,
                    "lr_patience": 9,
                    "gradient_clip": 0.8,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # EfficientNet-LSTM
        self.register(
            ArchitectureInfo(
                name="EfficientNet-LSTM",
                module_path="app.domain.models.architectures.efficientnet_lstm",
                function_name="create_model",
                description="EfficientNet with LSTM for temporal modeling - configuração padrão para máxima acurácia",
                supported_variants=[
                    "efficientnet_lstm",
                    "efficientnet_lstm_lite"],
                default_params={
                    "lstm_units": 512,
                    "attention_units": 256,
                    "dropout_rate": 0.2,
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # MultiscaleCNN
        self.register(
            ArchitectureInfo(
                name="MultiscaleCNN",
                module_path="app.domain.models.architectures.multiscale_cnn",
                function_name="create_model",
                description="Multi-Scale Convolutional Neural Network - configuração otimizada para reduzir overfitting",
                supported_variants=["multiscale_cnn", "multiscale_cnn_lite"],
                default_params={
                    "architecture": "multiscale_cnn",
                    "filters": [64, 128, 256, 512],
                    "kernel_sizes": [3, 5, 7],
                    "dropout_rate": 0.2,
                    "use_early_stopping": True,
                    "use_gradient_clipping": True,
                    "use_advanced_augmentation": True,
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.2,
                    "augmentation_strength": 0.35
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # SpectrogramTransformer
        self.register(
            ArchitectureInfo(
                name="SpectrogramTransformer",
                module_path="app.domain.models.architectures.spectrogram_transformer",
                function_name="create_model",
                description="Transformer-based model for spectrogram analysis - configuração padrão para máxima acurácia",
                supported_variants=[
                    "spectrogram_transformer",
                    "spectrogram_transformer_lite"],
                default_params={
                    "d_model": 512,
                    "num_heads": 16,
                    "num_blocks": 12,
                    "dropout_rate": 0.1,
                    # Training params
                    "patience": 25,
                    "lr_patience": 12,
                    "gradient_clip": 0.5,
                    "augmentation_strength": 0.2
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # Conformer
        self.register(
            ArchitectureInfo(
                name="Conformer",
                module_path="app.domain.models.architectures.conformer",
                function_name="create_model",
                description="Conformer: Convolution-augmented Transformer - configuração padrão para máxima acurácia",
                supported_variants=["conformer", "conformer_lite"],
                default_params={
                    "d_model": 512,
                    "num_blocks": 12,
                    "num_heads": 16,
                    "dropout_rate": 0.1,
                    # Training params
                    "patience": 22,
                    "lr_patience": 11,
                    "gradient_clip": 0.6,
                    "augmentation_strength": 0.25
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # Ensemble
        self.register(
            ArchitectureInfo(
                name="Ensemble",
                module_path="app.domain.models.architectures.ensemble",
                function_name="create_model",
                description="Ensemble of multiple architectures",
                supported_variants=[
                    "ensemble",
                    "ensemble_lite",
                    "ensemble_feature_fusion",
                    "ensemble_hybrid"],
                default_params={
                    "ensemble_method": "weighted_average",
                    "fusion_method": "prediction_level",
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80}
            )
        )

        # Sonic Sleuth
        self.register(
            ArchitectureInfo(
                name="Sonic Sleuth",
                module_path="app.domain.models.architectures.sonic_sleuth",
                function_name="create_model",
                description="CNN especializada para detecção de deepfake usando espectrogramas de mel - configuração padrão para máxima acurácia",
                supported_variants=["sonic_sleuth"],
                default_params={
                    "sample_rate": 16000,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "n_mels": 256,
                    "dropout_rate": 0.2,
                    "filters": [
                        64,
                        128,
                        256,
                        512],
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "max_duration": 3.0,
                    "sample_rate": 16000}
            )
        )

        # RawNet2
        self.register(
            ArchitectureInfo(
                name="RawNet2",
                module_path="app.domain.models.architectures.rawnet2",
                function_name="create_model",
                description="Arquitetura de rede neural que opera diretamente no áudio bruto para detecção de deepfake - configuração padrão para máxima acurácia",
                supported_variants=["rawnet2", "rawnet2_lite"],
                default_params={
                    "conv_filters": [128, 256, 512],
                    "gru_units": 256,
                    "dense_units": 128,
                    "dropout_rate": 0.2,
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 5.0,
                    "preprocessing": "normalize"
                }
            )
        )

        # WavLM
        self.register(
            ArchitectureInfo(
                name="WavLM",
                module_path="app.domain.models.architectures.wavlm",
                function_name="create_model",
                description="Arquitetura de dois estágios com WavLM pré-treinado como extrator de características e classificador MLP - configuração padrão para máxima acurácia",
                supported_variants=["wavlm", "wavlm_lite"],
                default_params={
                    "wavlm_model": "microsoft/wavlm-large",
                    "freeze_wavlm": True,
                    "classifier_units": [1024, 512, 256],
                    "dropout_rate": 0.2,
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 5.0,
                    "preprocessing": "normalize"
                }
            )
        )

        # SVM
        self.register(
            ArchitectureInfo(
                name="SVM",
                module_path="app.domain.models.architectures.svm",
                function_name="create_model",
                description="Support Vector Machine para classificação de deepfake com kernel RBF - modelo clássico de machine learning",
                supported_variants=[
                    "svm", "svm_linear", "svm_poly", "svm_rbf"],
                default_params={
                    "kernel": "rbf",
                    "C": 1.0,
                    "gamma": "scale",
                    "probability": True,
                    "random_state": 42,
                    # Training params (unused for SVM but kept for consistency)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "features",
                    "format": "tabular",
                    "preprocessing": "standardize"
                }
            )
        )

        # Random Forest
        self.register(
            ArchitectureInfo(
                name="RandomForest",
                module_path="app.domain.models.architectures.random_forest",
                function_name="create_model",
                description="Random Forest para classificação de deepfake com ensemble de árvores de decisão - modelo clássico de machine learning",
                supported_variants=[
                    "random_forest",
                    "random_forest_balanced",
                    "random_forest_entropy"],
                default_params={
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                    "n_jobs": -1
                },
                input_requirements={
                    "type": "features",
                    "format": "tabular",
                    "preprocessing": "normalize"
                }
            )
        )

        # HuBERT
        self.register(
            ArchitectureInfo(
                name="HuBERT",
                module_path="app.domain.models.architectures.hubert",
                function_name="create_model",
                description="Arquitetura HuBERT padrão para detecção de deepfakes com extrator de características e classificador MLP em duas etapas com foco em acurácia",
                supported_variants=["hubert", "hubert_lite"],
                default_params={
                    "num_classes": 1,
                    "architecture": "hubert",
                    "hidden_size": 768,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "classifier_hidden_dim": 256,
                    "dropout_rate": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 10.0,
                    "preprocessing": "normalize"
                }
            )
        )

        # Hybrid CNN-Transformer
        self.register(
            ArchitectureInfo(
                name="Hybrid CNN-Transformer",
                module_path="app.domain.models.architectures.hybrid_cnn_transformer",
                function_name="create_model",
                description="Arquitetura híbrida que combina CNNs para extração de features espaciais com Transformers para modelagem temporal, otimizada para detecção de deepfakes em áudio",
                supported_variants=[
                    "hybrid_cnn_transformer",
                    "hybrid_cnn_transformer_lite"],
                default_params={
                    "num_classes": 1,
                    "architecture": "hybrid_cnn_transformer",
                    "base_filters": 64,
                    "num_residual_blocks": 3,
                    "num_transformer_layers": 2,
                    "attention_heads": 8,
                    "dropout_rate": 0.1,
                    # Training params
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "features",
                    "format": "spectrogram",
                    "min_sequence_length": 100,
                    "feature_dim": 80,
                    "supports_1d_input": True,
                    "supports_2d_input": True,
                    "preprocessing": "spectrogram_or_raw"
                }
            )
        )

    def register(self, architecture_info: ArchitectureInfo):
        """Registra uma nova arquitetura."""
        self._architectures[architecture_info.name] = architecture_info
        logger.info(
            f"Arquitetura {
                architecture_info.name} registrada com sucesso")

    def get_architecture(self, name: str) -> ArchitectureInfo:
        """Obtém informações de uma arquitetura."""
        if name not in self._architectures:
            raise ValueError(
                f"Arquitetura '{name}' não encontrada. Disponíveis: {
                    list(
                        self._architectures.keys())}")
        return self._architectures[name]

    def list_architectures(self) -> List[str]:
        """Lista todas as arquiteturas disponíveis."""
        return list(self._architectures.keys())

    def get_all_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Retorna todas as arquiteturas registradas."""
        return self._architectures.copy()

    def get_active_config(self, architecture_name: str,
                          variant: str = "default") -> Dict[str, Any]:
        """Obtém a configuração ativa do banco de dados (ou default se falhar)."""
        try:
            from app.domain.models import ArchitectureConfig
            # from app.core.db_setup import get_flask_app  <-- Removido para evitar ciclo

            # Precisamos de um contexto de aplicação se não estivermos em um
            # Mas cuidado com criação excessiva de apps.
            # Assumimos que quem chama já está em contexto ou podemos criar um leve.
            # Se falhar import do db ou contexto, fallback para default_params
            # do registry.

            # Tentar buscar no DB
            config = ArchitectureConfig.query.filter_by(
                architecture_name=architecture_name,
                variant_name=variant,
                is_active=True
            ).first()

            if config:
                return config.parameters

            # Fallback se não encontrar variante específica: tentar default
            if variant != "default":
                config = ArchitectureConfig.query.filter_by(
                    architecture_name=architecture_name,
                    variant_name="default",
                    is_active=True
                ).first()
                if config:
                    return config.parameters

        except Exception as e:
            logger.warning(
                f"Não foi possível carregar config do DB para {architecture_name}: {e}. Usando hardcoded.")

        # Fallback final: Hardcoded
        return self.get_architecture(architecture_name).default_params

    def create_model(self, architecture_name: str, input_shape: Tuple[int, ...],
                     num_classes: int = 2, variant: str = None, safe_mode: bool = True, **kwargs):
        """Cria um modelo usando a arquitetura especificada.

        Args:
            architecture_name: Nome da arquitetura
            input_shape: Forma do input
            num_classes: Número de classes
            variant: Variante da arquitetura
            safe_mode: Se True, aplica correções para prevenir data leakage
            **kwargs: Parâmetros adicionais
        """
        arch_info = self.get_architecture(architecture_name)

        # Importar dinamicamente o módulo
        module = __import__(
            arch_info.module_path, fromlist=[
                arch_info.function_name])
        create_model_func = getattr(module, arch_info.function_name)

        # Preparar parâmetros básicos
        params = {}

        # Adicionar variant se especificado
        if variant:
            if variant not in arch_info.supported_variants:
                raise ValueError(f"Variant '{variant}' não suportada para {architecture_name}. "
                                 f"Disponíveis: {arch_info.supported_variants}")
            params['architecture'] = variant
        else:
            # Usar primeira variante como padrão
            if arch_info.supported_variants:
                params['architecture'] = arch_info.supported_variants[0]

        # Adicionar kwargs do usuário
        params.update(kwargs)

        # Criar modelo
        model = create_model_func(input_shape, num_classes, **params)

        # Aplicar correções de segurança se solicitado
        if safe_mode:
            is_safe, issues = validate_model_safety(model)
            if not is_safe:
                logger.warning(
                    f"Modelo {architecture_name} possui problemas de data leakage: {issues}")
                logger.info("Aplicando correções automáticas...")
                model = patch_architecture_for_safety(
                    model, normalization_type='layer')
                logger.info("Correções aplicadas com sucesso")
            else:
                logger.info(f"Modelo {architecture_name} já está seguro")

        return model

    def validate_input_shape(self, architecture_name: str,
                             input_shape: Tuple[int, ...]) -> bool:
        """Valida se o input_shape é compatível com a arquitetura."""
        arch_info = self.get_architecture(architecture_name)
        requirements = arch_info.input_requirements

        # Validações básicas
        if len(input_shape) < 2:
            return False

        sequence_length, feature_dim = input_shape[0], input_shape[1]

        if sequence_length < requirements.get("min_sequence_length", 0):
            return False

        if feature_dim < requirements.get("min_feature_dim", 0):
            return False

        return True

    def sync_defaults_to_db(self):
        """Sincroniza os parâmetros padrão do registry para o banco de dados."""
        try:
            from app.domain.models import ArchitectureConfig
            from app.extensions import db

            # Iterar sobre todas as arquiteturas registradas
            for name, info in self._architectures.items():
                # Verificar se já existe configuração default
                config = ArchitectureConfig.query.filter_by(
                    architecture_name=name,
                    variant_name="default"
                ).first()

                if not config:
                    logger.info(f"Criando configuração default no DB para {name}")
                    new_config = ArchitectureConfig(
                        architecture_name=name,
                        variant_name="default",
                        description=info.description,
                        parameters=info.default_params,
                        is_active=True
                    )
                    db.session.add(new_config)
                else:
                    # Opcional: Atualizar se necessário, mas cuidado para não sobrescrever customizações
                    # Por enquanto, mantemos o que está no banco se já existir
                    pass
            
            db.session.commit()
            logger.info("Sincronização de configurações padrão concluída.")
        except Exception as e:
            logger.error(f"Erro ao sincronizar defaults para o DB: {e}")
            db.session.rollback()


# Instância global do registry
architecture_registry = ArchitectureRegistry()

# Funções de conveniência


def get_available_architectures() -> List[str]:
    """Retorna lista de arquiteturas disponíveis."""
    return architecture_registry.list_architectures()


def create_model_by_name(architecture_name: str, input_shape: Tuple[int, ...],
                         num_classes: int = 2, variant: str = None, safe_mode: bool = True, **kwargs):
    """Cria um modelo pela nome da arquitetura.

    Args:
        architecture_name: Nome da arquitetura
        input_shape: Forma do input
        num_classes: Número de classes
        variant: Variante da arquitetura
        safe_mode: Se True, aplica correções para prevenir data leakage
        **kwargs: Parâmetros adicionais
    """
    return architecture_registry.create_model(
        architecture_name, input_shape, num_classes, variant, safe_mode, **kwargs)


def create_safe_model_by_name(architecture_name: str, input_shape: Tuple[int, ...],
                              num_classes: int = 2, variant: str = None, **kwargs):
    """Cria um modelo seguro (sem data leakage) pela nome da arquitetura."""
    return create_model_by_name(
        architecture_name, input_shape, num_classes, variant, safe_mode=True, **kwargs)


def get_architecture_info(architecture_name: str) -> ArchitectureInfo:
    """Obtém informações sobre uma arquitetura."""
    return architecture_registry.get_architecture(architecture_name)


def validate_architecture_input(
        architecture_name: str, input_shape: Tuple[int, ...]) -> bool:
    """Valida input para uma arquitetura."""
    return architecture_registry.validate_input_shape(
        architecture_name, input_shape)


def load_hyperparameters_json(
        architecture_name: str, results_dir: str) -> Dict[str, Any]:
    """Carrega hiperparâmetros recomendados de um arquivo JSON.

    Args:
        architecture_name: Nome da arquitetura
        results_dir: Diretório onde estão os resultados/configs

    Returns:
        Dict com hiperparâmetros ou dict vazio se não encontrar
    """
    import os
    import json

    # Normalizar nome para busca de arquivo (ex: "Random Forest" ->
    # "random_forest")
    safe_name = architecture_name.lower().replace(" ", "_")

    # Tentar variações de nomes de arquivo
    possible_files = [
        f"{safe_name}_hyperparameters.json",
        f"{safe_name}_params.json",
        f"{safe_name}.json"
    ]

    for filename in possible_files:
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    f"Erro ao ler hiperparâmetros de {file_path}: {e}")
                return {}

    return {}


# Exportar principais classes e funções
__all__ = [
    "ArchitectureInfo",
    "ArchitectureRegistry",
    "architecture_registry",
    "get_available_architectures",
    "create_model_by_name",
    "create_safe_model_by_name",
    "get_architecture_info",
    "validate_architecture_input",
    "load_hyperparameters_json"
]
