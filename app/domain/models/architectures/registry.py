"""Registry de Arquiteturas Disponíveis

Este módulo centraliza o registro de todas as arquiteturas de deep learning
disponíveis no sistema, facilitando a integração com o pipeline de detecção.
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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

    # Mapeamento bidirecional: display_name ↔ snake_case
    _DISPLAY_TO_SNAKE: Dict[str, str] = {
        "AASIST": "aasist",
        "RawGAT-ST": "rawgat_st",
        "EfficientNet-LSTM": "efficientnet_lstm",
        "MultiscaleCNN": "multiscale_cnn",
        "SpectrogramTransformer": "spectrogram_transformer",
        "Conformer": "conformer",
        "Ensemble": "ensemble",
        "Sonic Sleuth": "sonic_sleuth",
        "RawNet2": "rawnet2",
        "WavLM": "wavlm",
        "HuBERT": "hubert",
        "Hybrid CNN-Transformer": "hybrid_cnn_transformer",
    }
    _SNAKE_TO_DISPLAY: Dict[str, str] = {v: k for k, v in _DISPLAY_TO_SNAKE.items()}

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
                    # create_model params: dropout_rate, l2_reg_strength, hidden_dim, num_layers
                    "dropout_rate": 0.2,
                    "l2_reg_strength": 0.0005,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 20,
                    "lr_patience": 10,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.25
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 4.0,
                    "preprocessing": "normalize"}
            )
        )

        # RawGAT-ST
        self.register(
            ArchitectureInfo(
                name="RawGAT-ST",
                module_path="app.domain.models.architectures.rawgat_st",
                function_name="create_model",
                description="Raw Graph Attention Spatio-Temporal Network - otimizado com HS-GAL 3 camadas e focal loss",
                supported_variants=[
                    "default",
                    "cnn_baseline",
                    "bidirectional_gru",
                    "resnet_gru",
                    "transformer",
                    "rawgat_st"],
                default_params={
                    # create_model params: dropout_rate, l2_reg_strength, attention_heads, hidden_dim, num_layers
                    "dropout_rate": 0.2,
                    "l2_reg_strength": 0.0005,
                    "attention_heads": 8,
                    "hidden_dim": 512,
                    "num_layers": 6,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 18,
                    "lr_patience": 9,
                    "gradient_clip": 0.8,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 4.0,
                    "preprocessing": "normalize"}
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
                    # create_model params: lstm_units (int), dropout_rate
                    "lstm_units": 256,
                    "dropout_rate": 0.3,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 4.0,
                    "preprocessing": "normalize",
                    "supports_2d_input": True}
            )
        )

        # MultiscaleCNN (Res2Net — Gao et al., TPAMI 2021)
        self.register(
            ArchitectureInfo(
                name="MultiscaleCNN",
                module_path="app.domain.models.architectures.multiscale_cnn",
                function_name="create_model",
                description="Res2Net: Multi-scale backbone with hierarchical residual connections (Gao et al., TPAMI 2021). Res2Net-50 config: scale=4, baseWidth=26, [3,4,6,3] blocks.",
                supported_variants=["multiscale_cnn", "multiscale_cnn_lite"],
                default_params={
                    # create_model params: base_width, scale, layer_config, dropout_rate
                    "base_width": 26,
                    "scale": 8,
                    "dropout_rate": 0.2,
                    # Training params (used by pipeline, not by create_model)
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
                description="Audio Spectrogram Transformer (AST) - ViT-Base with overlapping patches for audio deepfake detection",
                supported_variants=[
                    "spectrogram_transformer",
                    "spectrogram_transformer_lite"],
                default_params={
                    # create_model params: architecture only (no model hyperparams accepted)
                    # Training params (used by pipeline, not by create_model)
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
                    # create_model params: architecture only (no model hyperparams accepted)
                    # Training params (used by pipeline, not by create_model)
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
                description="Multi-spectrogram ensemble (Mel+LFCC+CQT) with MLP fusion — Pham et al. 2024",
                supported_variants=[
                    "ensemble",
                    "ensemble_score",
                    "ensemble_lite"],
                default_params={
                    "dropout_rate": 0.3,
                    "use_mfcc_branch": True,
                    "use_cross_attention": True,
                    "use_gated_fusion": True,
                    "use_se_blocks": True,
                    "aux_loss_weight": 0.3,
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
                    "max_duration": 4.0,
                    "preprocessing": "normalize",
                    "supports_2d_input": True}
            )
        )

        # Sonic Sleuth (Alshehri et al., MDPI Computers 2024)
        self.register(
            ArchitectureInfo(
                name="Sonic Sleuth",
                module_path="app.domain.models.architectures.sonic_sleuth",
                function_name="create_model",
                description="Sonic Sleuth (Alshehri et al., 2024): LFCC/MFCC/CQT feature extraction + 3×Conv2D(32→64→128) + Dense(256→128) + Dropout(0.1). Best: LFCC 98.27% accuracy.",
                supported_variants=["sonic_sleuth", "sonic_sleuth_mfcc", "sonic_sleuth_cqt", "sonic_sleuth_lfcc_cqt"],
                default_params={
                    "sample_rate": 16000,
                    "use_batch_norm": True,
                    "num_conv_blocks": 5,
                    "use_residual": True,
                    "use_se_blocks": True,
                    "use_gap_gmp": True,
                    "dropout_rate": 0.3,
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
                    "sample_rate": 16000,
                    "preprocessing": "normalize"}
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
                    # create_model params: sinc_filters, sinc_kernel_size, res_filters, gru_units, dense_units, dropout_rate
                    "sinc_filters": 128,
                    "sinc_kernel_size": 1024,
                    "res_filters": [128, 128, 256, 256, 256, 256],
                    "gru_units": 512,
                    "dense_units": 512,
                    "dropout_rate": 0.3,
                    # Training params (used by pipeline, not by create_model)
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
                    # create_model params: wavlm_model, freeze_wavlm, classifier_units, dropout_rate
                    "wavlm_model": "microsoft/wavlm-large",
                    "freeze_wavlm": True,
                    "classifier_units": [1024, 512, 256],
                    "dropout_rate": 0.2,
                    # Training params (used by pipeline, not by create_model)
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

        # HuBERT
        self.register(
            ArchitectureInfo(
                name="HuBERT",
                module_path="app.domain.models.architectures.hubert",
                function_name="create_model",
                description="Arquitetura baseada em HuBERT (Hidden-Unit BERT) para detecção de deepfakes em áudio bruto - fidelidade ao paper",
                supported_variants=["hubert", "hubert_lite"],
                default_params={
                    # create_model params: model_name, freeze_hubert, classifier_hidden_dim, dropout_rate
                    "model_name": "facebook/hubert-base-ls960",
                    "freeze_hubert": True,
                    "classifier_hidden_dim": 256,
                    "dropout_rate": 0.3,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
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
                description="CCT (Compact Convolutional Transformer) — Hassani et al. 2021, adapted for audio deepfake per Bartusiak & Delp 2022",
                supported_variants=[
                    "hybrid_cnn_transformer",
                    "hybrid_cnn_transformer_lite"],
                default_params={
                    # create_model params: projection_dim, num_heads, transformer_layers,
                    #   conv_channels, dropout_rate, stochastic_depth_rate, use_positional_emb
                    "projection_dim": 256,
                    "num_heads": 4,
                    "transformer_layers": 4,
                    "conv_channels": [64, 128],
                    "dropout_rate": 0.1,
                    "stochastic_depth_rate": 0.1,
                    "use_positional_emb": True,
                    # Training params (used by pipeline, not by create_model)
                    "patience": 15,
                    "lr_patience": 8,
                    "gradient_clip": 1.0,
                    "augmentation_strength": 0.3
                },
                input_requirements={
                    "type": "audio",
                    "format": "raw",
                    "sample_rate": 16000,
                    "max_duration": 4.0,
                    "preprocessing": "normalize",
                    "supports_2d_input": True
                }
            )
        )

    def register(self, architecture_info: ArchitectureInfo):
        """Registra uma nova arquitetura."""
        self._architectures[architecture_info.name] = architecture_info
        logger.info(
            f"Arquitetura {architecture_info.name} registrada com sucesso")

    def get_architecture(self, name: str) -> ArchitectureInfo:
        """Obtém informações de uma arquitetura."""
        if name not in self._architectures:
            raise ValueError(
                f"Arquitetura '{name}' não encontrada. Disponíveis: {list(self._architectures.keys())}")
        return self._architectures[name]

    def list_architectures(self) -> List[str]:
        """Lista todas as arquiteturas disponíveis."""
        return list(self._architectures.keys())

    def get_all_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Retorna todas as arquiteturas registradas."""
        return self._architectures.copy()

    # ── Name Mapping Layer ──────────────────────────────────────────────

    @classmethod
    def to_snake_case(cls, name: str) -> str:
        """Converte display name → snake_case. Se já for snake, retorna como está."""
        if name in cls._DISPLAY_TO_SNAKE:
            return cls._DISPLAY_TO_SNAKE[name]
        # Já é snake_case ou desconhecido
        return name.lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def to_display_name(cls, snake: str) -> str:
        """Converte snake_case → display name para UI."""
        if snake in cls._SNAKE_TO_DISPLAY:
            return cls._SNAKE_TO_DISPLAY[snake]
        return snake

    @classmethod
    def normalize_architecture_name(cls, name: str) -> str:
        """Normaliza qualquer formato de nome para o display name do registry.

        Aceita tanto 'sonic_sleuth' quanto 'Sonic Sleuth' e retorna
        o display name canônico registrado no registry.
        """
        # Se já é um display name válido
        if name in cls._DISPLAY_TO_SNAKE:
            return name
        # Tentar converter de snake_case
        if name in cls._SNAKE_TO_DISPLAY:
            return cls._SNAKE_TO_DISPLAY[name]
        # Fallback: tentar case-insensitive
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        for snake, display in cls._SNAKE_TO_DISPLAY.items():
            if snake == name_lower:
                return display
        raise ValueError(
            f"Arquitetura '{name}' não encontrada. "
            f"Disponíveis (snake): {list(cls._SNAKE_TO_DISPLAY.keys())}"
        )

    def get_architecture_by_any_name(self, name: str) -> ArchitectureInfo:
        """Busca arquitetura aceitando display name OU snake_case."""
        try:
            return self.get_architecture(name)
        except ValueError:
            display = self.normalize_architecture_name(name)
            return self.get_architecture(display)

    def list_architectures_snake(self) -> List[str]:
        """Lista todas as arquiteturas em snake_case (para dropdowns/API)."""
        return [self.to_snake_case(name) for name in self._architectures.keys()]

    def list_architecture_choices(self) -> List[Tuple[str, str]]:
        """Retorna pares (display_label, snake_case) para dropdowns Gradio."""
        choices = []
        for display_name in self._architectures:
            snake = self.to_snake_case(display_name)
            info = self._architectures[display_name]
            input_type = info.input_requirements.get("type", "unknown")
            label = f"{display_name} ({input_type})"
            choices.append((label, snake))
        return choices

    def get_active_config(self, architecture_name: str,
                          variant: str = "default") -> Dict[str, Any]:
        """Obtém a configuração ativa do banco de dados (ou default se falhar)."""
        try:
            from app.core.database import SessionLocal
            from app.domain.models.architecture_config import ArchitectureConfig

            db_session = SessionLocal()
            try:
                # Tentar buscar no DB usando SQLAlchemy nativo
                config = db_session.query(ArchitectureConfig).filter_by(
                    architecture_name=architecture_name,
                    variant_name=variant,
                    is_active=True
                ).first()

                if config:
                    return config.parameters

                # Fallback se não encontrar variante específica: tentar default
                if variant != "default":
                    config = db_session.query(ArchitectureConfig).filter_by(
                        architecture_name=architecture_name,
                        variant_name="default",
                        is_active=True
                    ).first()
                    if config:
                        return config.parameters
            finally:
                db_session.close()

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

        # Filtrar params para apenas os aceitos pela função (quando não há **kwargs)
        sig = inspect.signature(create_model_func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_keyword:
            accepted = set(sig.parameters.keys())
            filtered_out = [k for k in params if k not in accepted]
            if filtered_out:
                logger.debug(f"{architecture_name}: ignoring unsupported params: {filtered_out}")
            params = {k: v for k, v in params.items() if k in accepted}

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
            from app.core.database import SessionLocal
            from app.domain.models.architecture_config import ArchitectureConfig

            db_session = SessionLocal()
            try:
                # Iterar sobre todas as arquiteturas registradas
                for name, info in self._architectures.items():
                    # Verificar se já existe configuração default
                    config = db_session.query(ArchitectureConfig).filter_by(
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
                        db_session.add(new_config)

                db_session.commit()
                logger.info("Sincronização de configurações padrão concluída.")
            finally:
                db_session.close()
        except Exception as e:
            logger.error(f"Erro ao sincronizar defaults para o DB: {e}")


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
    import json
    import os

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


def get_architecture_choices() -> List[Tuple[str, str]]:
    """Retorna pares (display_label, snake_case) para dropdowns."""
    return architecture_registry.list_architecture_choices()


def get_valid_snake_names() -> set:
    """Retorna set de snake_case names válidos (para validação de schemas)."""
    return set(architecture_registry.list_architectures_snake())


def normalize_arch_name(name: str) -> str:
    """Normaliza qualquer formato para display name canônico."""
    return ArchitectureRegistry.normalize_architecture_name(name)


def to_snake(name: str) -> str:
    """Converte qualquer formato para snake_case."""
    return ArchitectureRegistry.to_snake_case(name)


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
    "load_hyperparameters_json",
    "get_architecture_choices",
    "get_valid_snake_names",
    "normalize_arch_name",
    "to_snake",
]
