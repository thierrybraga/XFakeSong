import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

from app.domain.models.architectures.registry import (
    get_available_architectures,
    create_model_by_name,
    get_architecture_info
)
# Importar custom layers para carregar modelos
from app.domain.models.architectures.rawnet2 import (
    AudioResamplingLayer,
    AudioNormalizationLayer,
    MultiScaleConv1DBlock
)
from app.domain.models.architectures.layers import (
    AudioFeatureNormalization,
    AttentionLayer,
    GraphAttentionLayer,
    SliceLayer,
    apply_gru_block,
    flatten_features_for_gru,
    apply_reshape_for_cnn,
    residual_block
)
from app.domain.models.architectures.safe_normalization import SafeInstanceNormalization

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Informações sobre um modelo carregado."""
    name: str
    architecture: str
    model: Any
    scaler: Optional[StandardScaler]
    input_shape: tuple
    model_type: str  # 'tensorflow' ou 'sklearn'


class ModelLoader:
    """Responsável por carregar e gerenciar modelos."""

    def __init__(self, models_dir: Union[str, Path]):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.default_model = None

    def load_available_models(self):
        """Carrega todos os modelos disponíveis."""
        logger.info("Carregando modelos disponíveis...")

        # Tentar carregar modelos salvos
        model_files = list(self.models_dir.glob("*.h5")) + \
            list(self.models_dir.glob("*.pkl"))

        for model_file in model_files:
            try:
                self._load_single_model(model_file)
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo {model_file}: {e}")

        # Se não há modelos carregados, criar modelos padrão
        if not self.loaded_models:
            logger.info(
                "Nenhum modelo salvo encontrado. Criando modelos padrão...")
            self._create_default_models()

        # Definir modelo padrão
        if self.loaded_models:
            self.default_model = list(self.loaded_models.keys())[0]
            logger.info(f"Modelo padrão definido: {self.default_model}")

    def _load_single_model(self, model_path: Path):
        """Carrega um único modelo."""
        model_name = model_path.stem

        try:
            if model_path.suffix == '.h5':
                # Modelo TensorFlow
                # Definir custom objects para arquiteturas específicas (ex:
                # RawNet2)
                custom_objects = {
                    'AudioResamplingLayer': AudioResamplingLayer,
                    'AudioNormalizationLayer': AudioNormalizationLayer,
                    'MultiScaleConv1DBlock': MultiScaleConv1DBlock,
                    'AudioFeatureNormalization': AudioFeatureNormalization,
                    'AttentionLayer': AttentionLayer,
                    'GraphAttentionLayer': GraphAttentionLayer,
                    'SliceLayer': SliceLayer,
                    'SafeInstanceNormalization': SafeInstanceNormalization
                }

                try:
                    model = tf.keras.models.load_model(
                        str(model_path), custom_objects=custom_objects)
                except TypeError:
                    # Fallback sem custom objects se não forem necessários
                    model = tf.keras.models.load_model(str(model_path))

                model_type = 'tensorflow'

                # Tentar carregar scaler correspondente
                scaler_path = model_path.parent / f"{model_name}_scaler.pkl"
                scaler = None
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)

                # Inferir input_shape do modelo
                input_shape = model.input_shape[1:]  # Remove batch dimension

            elif model_path.suffix == '.pkl':
                # Modelo sklearn
                if '_scaler' in model_name:
                    return  # Skip scaler files

                model = joblib.load(model_path)
                model_type = 'sklearn'

                # Carregar scaler correspondente
                scaler_path = model_path.parent / f"{model_name}_scaler.pkl"
                scaler = None
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)

                # Para modelos sklearn, input_shape será definido durante a
                # predição
                input_shape = None

            else:
                logger.warning(
                    f"Formato de arquivo não suportado: {model_path}")
                return

            # Determinar arquitetura: via metadados ou inferência
            if 'architecture' in metadata:
                architecture = metadata['architecture']
            else:
                architecture = self._infer_architecture_from_name(model_name)
            
            # Sobrescrever input_shape se definido nos metadados
            if 'input_shape' in metadata:
                input_shape = tuple(metadata['input_shape'])

            model_info = ModelInfo(
                name=model_name,
                architecture=architecture,
                model=model,
                scaler=scaler,
                input_shape=input_shape,
                model_type=model_type
            )

            self.loaded_models[model_name] = model_info
            logger.info(
                f"Modelo {model_name} carregado com sucesso ({model_type})")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_path}: {e}")
            raise

    def _infer_architecture_from_name(self, model_name: str) -> str:
        """Infere a arquitetura baseada no nome do modelo."""
        name_lower = model_name.lower()

        if 'aasist' in name_lower:
            return 'AASIST'
        elif 'rawgat' in name_lower:
            return 'RawGAT-ST'
        elif 'efficientnet' in name_lower:
            return 'EfficientNet-LSTM'
        elif 'multiscale' in name_lower:
            return 'MultiscaleCNN'
        elif 'spectrogram' in name_lower or 'transformer' in name_lower:
            return 'SpectrogramTransformer'
        elif 'conformer' in name_lower:
            return 'Conformer'
        elif 'ensemble' in name_lower:
            return 'Ensemble'
        elif 'rawnet2' in name_lower:
            return 'RawNet2'
        elif 'wavlm' in name_lower:
            return 'WavLM'
        elif 'hubert' in name_lower:
            return 'HuBERT'
        elif 'sonic' in name_lower or 'sleuth' in name_lower:
            return 'Sonic Sleuth'
        elif 'svm' in name_lower:
            return 'SVM'
        elif 'random_forest' in name_lower:
            return 'RandomForest'
        elif 'neural_network' in name_lower:
            return 'SimpleNN'
        else:
            return 'Unknown'

    def _create_default_models(self):
        """Cria modelos padrão para demonstração."""
        logger.info("Criando modelos padrão...")

        input_shape = (100, 80)  # Formato padrão

        # Criar alguns modelos leves para demonstração
        lightweight_architectures = ['MultiscaleCNN', 'EfficientNet-LSTM']

        for arch_name in lightweight_architectures:
            try:
                logger.info(f"Criando modelo {arch_name}...")

                # Criar modelo usando variant lite se disponível
                arch_info = get_architecture_info(arch_name)
                variant = None
                if 'lite' in [
                        v for v in arch_info.supported_variants if 'lite' in v]:
                    variant = [
                        v for v in arch_info.supported_variants if 'lite' in v][0]

                model = create_model_by_name(
                    arch_name,
                    input_shape,
                    num_classes=2,
                    variant=variant
                )

                model_info = ModelInfo(
                    name=f"{arch_name.lower()}_default",
                    architecture=arch_name,
                    model=model,
                    scaler=StandardScaler(),
                    input_shape=input_shape,
                    model_type='tensorflow'
                )

                self.loaded_models[model_info.name] = model_info
                logger.info(f"Modelo {arch_name} criado com sucesso")

            except Exception as e:
                logger.warning(f"Erro ao criar modelo {arch_name}: {e}")

    def get_available_models(self) -> List[str]:
        """Retorna lista de modelos disponíveis."""
        return list(self.loaded_models.keys())

    def get_available_architectures(self) -> List[str]:
        """Retorna lista de arquiteturas disponíveis."""
        return get_available_architectures()

    def find_model(self, architecture: str,
                   variant: str = None) -> Optional[str]:
        """Encontra um modelo carregado que corresponda à arquitetura e variante."""
        arch_lower = architecture.lower()
        variant_lower = variant.lower() if variant else None

        for name, info in self.loaded_models.items():
            # Verificar arquitetura
            if info.architecture.lower() == arch_lower:
                # Se variante for especificada, verificar se está no nome do modelo
                # Esta é uma heurística simples, já que variant não é
                # explicitamente salvo no ModelInfo
                if variant_lower:
                    if variant_lower in name.lower():
                        return name
                else:
                    # Se não especificou variante, retorna o primeiro da
                    # arquitetura
                    return name
        return None
