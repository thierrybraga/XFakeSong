"""Architecture Patcher for Data Leakage Prevention"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any, List, Tuple
import logging
from .safe_normalization import (
    SafeInstanceNormalization,
    SafeLayerNormalization,
    SafeGroupNormalization
)

logger = logging.getLogger(__name__)


class ArchitecturePatcher:
    """
    Classe para corrigir arquiteturas existentes removendo data leakage.
    Substitui camadas problemáticas por versões seguras.
    """

    def __init__(self):
        self.replacements_made = []

    def patch_model(self, model: models.Model,
                    normalization_type: str = 'layer') -> models.Model:
        """
        Corrige um modelo existente substituindo camadas problemáticas.

        Args:
            model: Modelo a ser corrigido
            normalization_type: Tipo de normalização segura ('layer', 'instance', 'group')

        Returns:
            Modelo corrigido
        """
        logger.info(f"Iniciando correção do modelo {model.name}")
        self.replacements_made = []

        def clone_function(layer):
            # Verifica BatchNormalization
            if isinstance(layer, layers.BatchNormalization):
                original_name = layer.name

                if normalization_type == 'layer':
                    new_layer = SafeLayerNormalization(
                        name=f"safe_layer_norm_{original_name}",
                        axis=-1,
                        epsilon=1e-6
                    )
                    replacement_name = "SafeLayerNormalization"
                elif normalization_type == 'instance':
                    new_layer = SafeInstanceNormalization(
                        name=f"safe_instance_norm_{original_name}",
                        axis=-1,
                        epsilon=1e-6
                    )
                    replacement_name = "SafeInstanceNormalization"
                elif normalization_type == 'group':
                    new_layer = SafeGroupNormalization(
                        name=f"safe_group_norm_{original_name}",
                        groups=32,
                        axis=-1,
                        epsilon=1e-6
                    )
                    replacement_name = "SafeGroupNormalization"
                else:
                    new_layer = layers.LayerNormalization(
                        name=f"layer_norm_{original_name}",
                        axis=-1,
                        epsilon=1e-6
                    )
                    replacement_name = "LayerNormalization"

                self.replacements_made.append(
                    f"BatchNormalization '{original_name}' -> {replacement_name}")
                return new_layer

            # Verifica Lambda
            if isinstance(layer, layers.Lambda):
                # Mantém Lambda se não for obviamente perigosa
                pass

            # Verifica camadas customizadas problemáticas pelo nome da classe
            if layer.__class__.__name__ in [
                    'AudioFeatureNormalization', 'CustomNormalization']:
                original_name = layer.name
                new_layer = SafeInstanceNormalization(
                    name=f"safe_norm_{original_name}",
                    axis=-1,
                    epsilon=1e-6
                )
                self.replacements_made.append(
                    f"Custom normalization '{original_name}' -> SafeInstanceNormalization")
                return new_layer

            return layer

        try:
            # Usa clone_model para reconstruir o modelo substituindo camadas
            patched_model = models.clone_model(
                model, clone_function=clone_function)

            if not self.replacements_made:
                logger.info("Nenhuma correção necessária encontrada.")
                return model

            # Copia pesos quando possível
            self._copy_compatible_weights(model, patched_model)

            logger.info(
                f"Modelo corrigido com sucesso. Substituições: {len(self.replacements_made)}")
            for replacement in self.replacements_made:
                logger.info(f"  - {replacement}")

            return patched_model

        except Exception as e:
            logger.error(f"Erro ao corrigir modelo: {e}")
            logger.warning("Retornando modelo original")
            return model

    def _patch_config(self, config: Dict[str, Any],
                      normalization_type: str) -> Dict[str, Any]:
        """
        Corrige a configuração do modelo substituindo camadas problemáticas.
        """
        if 'layers' in config:
            for i, layer_config in enumerate(config['layers']):
                config['layers'][i] = self._patch_layer_config(
                    layer_config, normalization_type
                )

        return config

    def _patch_layer_config(self, layer_config: Dict[str, Any],
                            normalization_type: str) -> Dict[str, Any]:
        """
        Corrige a configuração de uma camada específica.
        """
        class_name = layer_config.get('class_name', '')

        # Substitui BatchNormalization
        if class_name == 'BatchNormalization':
            layer_config = self._replace_batch_normalization(
                layer_config, normalization_type
            )

        # Remove camadas de pré-processamento problemáticas
        elif class_name == 'Lambda':
            layer_config = self._check_lambda_layer(layer_config)

        # Verifica camadas customizadas problemáticas
        elif class_name in ['AudioFeatureNormalization']:
            layer_config = self._replace_custom_normalization(
                layer_config, normalization_type
            )

        return layer_config

    def _replace_batch_normalization(self, layer_config: Dict[str, Any],
                                     normalization_type: str) -> Dict[str, Any]:
        """
        Substitui BatchNormalization por normalização segura.
        """
        # Copia a configuração original para preservar inbound_nodes e outros
        # metadados
        new_layer_config = layer_config.copy()

        original_name = layer_config.get('config', {}).get('name', 'unnamed')

        if normalization_type == 'layer':
            new_class_name = 'DeepFake>SafeLayerNormalization'
            new_inner_config = {
                'name': f"safe_layer_norm_{original_name}",
                'axis': -1,
                'epsilon': 1e-6
            }
        elif normalization_type == 'instance':
            new_class_name = 'DeepFake>SafeInstanceNormalization'
            new_inner_config = {
                'name': f"safe_instance_norm_{original_name}",
                'axis': -1,
                'epsilon': 1e-6
            }
        elif normalization_type == 'group':
            new_class_name = 'DeepFake>SafeGroupNormalization'
            new_inner_config = {
                'name': f"safe_group_norm_{original_name}",
                'groups': 32,
                'axis': -1,
                'epsilon': 1e-6
            }
        else:
            # Fallback para LayerNormalization
            new_class_name = 'LayerNormalization'
            new_inner_config = {
                'name': f"layer_norm_{original_name}",
                'axis': -1,
                'epsilon': 1e-6
            }

        # Atualizar class_name e config, preservando o resto
        new_layer_config['class_name'] = new_class_name

        # Preservar chaves extras do config original que possam ser relevantes, se necessário
        # Mas para normalização, geralmente os parametros mudam completamente.
        # Importante: atualizar o nome também no nível superior se existir
        if 'name' in new_layer_config:
            new_layer_config['name'] = new_inner_config['name']

        new_layer_config['config'] = new_inner_config

        self.replacements_made.append(
            f"BatchNormalization '{original_name}' -> {new_class_name}"
        )

        return new_layer_config

    def _replace_custom_normalization(self, layer_config: Dict[str, Any],
                                      normalization_type: str) -> Dict[str, Any]:
        """
        Substitui camadas de normalização customizadas problemáticas.
        """
        new_layer_config = layer_config.copy()
        original_name = layer_config.get('config', {}).get('name', 'unnamed')

        new_class_name = 'DeepFake>SafeInstanceNormalization'
        new_inner_config = {
            'name': f"safe_norm_{original_name}",
            'axis': -1,
            'epsilon': 1e-6
        }

        new_layer_config['class_name'] = new_class_name
        if 'name' in new_layer_config:
            new_layer_config['name'] = new_inner_config['name']
        new_layer_config['config'] = new_inner_config

        self.replacements_made.append(
            f"Custom normalization '{original_name}' -> SafeInstanceNormalization"
        )

        return new_layer_config

    def _check_lambda_layer(
            self, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica se camadas Lambda contêm pré-processamento problemático.
        """
        layer_name = layer_config.get('config', {}).get('name', '')

        # Lista de nomes de camadas Lambda problemáticas
        problematic_names = [
            'multiscale_preprocessing',
            'spectrogram_preprocessing',
            'efficientnet_preprocessing'
        ]

        if any(name in layer_name for name in problematic_names):
            # Substitui por camada de identidade
            new_config = {
                'class_name': 'Lambda',
                'config': {
                    'name': f"safe_{layer_name}",
                    'function': 'lambda x: x'  # Função identidade
                }
            }

            self.replacements_made.append(
                f"Problematic Lambda '{layer_name}' -> Identity function"
            )

            return new_config

        return layer_config

    def _copy_compatible_weights(self, source_model: models.Model,
                                 target_model: models.Model) -> None:
        """
        Copia pesos compatíveis entre modelos.
        """
        try:
            source_weights = source_model.get_weights()
            target_weights = target_model.get_weights()

            # Copia pesos quando as dimensões são compatíveis
            copied_count = 0
            for i, (src_weight, tgt_weight) in enumerate(
                    zip(source_weights, target_weights)):
                if src_weight.shape == tgt_weight.shape:
                    target_weights[i] = src_weight
                    copied_count += 1

            target_model.set_weights(target_weights)
            logger.info(f"Copiados {copied_count}/{len(source_weights)} pesos")

        except Exception as e:
            logger.warning(f"Não foi possível copiar pesos: {e}")


def patch_architecture_for_safety(model: models.Model,
                                  normalization_type: str = 'layer') -> models.Model:
    """
    Função de conveniência para corrigir uma arquitetura.

    Args:
        model: Modelo a ser corrigido
        normalization_type: Tipo de normalização ('layer', 'instance', 'group')

    Returns:
        Modelo corrigido
    """
    patcher = ArchitecturePatcher()
    return patcher.patch_model(model, normalization_type)


def validate_model_safety(model: models.Model) -> Tuple[bool, List[str]]:
    """
    Valida se um modelo está livre de data leakage.

    Args:
        model: Modelo a ser validado

    Returns:
        Tuple com (is_safe, list_of_issues)
    """
    issues = []

    for layer in model.layers:
        layer_class = layer.__class__.__name__

        # Verifica BatchNormalization
        if layer_class == 'BatchNormalization':
            issues.append(f"BatchNormalization encontrada: {layer.name}")

        # Verifica camadas customizadas problemáticas
        elif layer_class == 'AudioFeatureNormalization':
            issues.append(
                f"AudioFeatureNormalization problemática: {
                    layer.name}")

        # Verifica camadas Lambda suspeitas
        elif layer_class == 'Lambda':
            if any(name in layer.name for name in [
                   'preprocessing', 'normalize']):
                issues.append(f"Lambda suspeita: {layer.name}")

    is_safe = len(issues) == 0
    return is_safe, issues
