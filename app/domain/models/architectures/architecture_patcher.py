"""Validação de segurança de arquiteturas (data leakage real).

Histórico: este módulo já tratou **BatchNormalization como data leakage** e a
substituía automaticamente por LayerNormalization em ``patch_model``. Isso
estava conceitualmente errado — BN usa estatísticas do batch apenas em
treino e médias móveis congeladas na inferência; é o padrão dos papers de
referência (RawNet2, AASIST, Res2Net). A troca silenciosa desviava as
implementações dos papers e ainda era cega a BN aninhada em camadas
customizadas (ex.: encoder do Conformer).

O que sobra como preocupação REAL de leakage em modelos Keras deste projeto:

- camadas ``Lambda`` de pré-processamento com estatísticas globais do
  dataset embutidas (legado);
- a antiga ``AudioFeatureNormalization`` com ``adapt()`` sobre o dataset
  inteiro — a classe atual herda de ``SafeInstanceNormalization`` (por
  amostra) e é segura; modelos salvos antigos podem conter a versão leaky.

``validate_model_safety`` sinaliza apenas esses casos e ``patch_model`` é
hoje conservador: **não** reescreve normalizações; loga os achados para
decisão humana (retreinar é o caminho correto, não cirurgia de grafo).
As camadas ``Safe*`` continuam exportadas para desserializar modelos já
patchados no passado.
"""

import logging
from typing import List, Tuple

from tensorflow.keras import layers, models

from .safe_normalization import (  # noqa: F401 — mantidos p/ desserialização
    SafeGroupNormalization,
    SafeInstanceNormalization,
    SafeLayerNormalization,
)

logger = logging.getLogger(__name__)

#: Substrings de nomes de Lambda historicamente associadas a pré-processamento
#: com estatísticas globais (leakage real quando ajustadas no dataset inteiro).
_SUSPICIOUS_LAMBDA_NAMES = (
    "multiscale_preprocessing",
    "spectrogram_preprocessing",
    "efficientnet_preprocessing",
)


def validate_model_safety(model: models.Model) -> Tuple[bool, List[str]]:
    """Valida se um modelo está livre de data leakage REAL.

    BatchNormalization **não** é sinalizada (não é leakage; ver docstring do
    módulo). São sinalizadas apenas camadas Lambda de pré-processamento
    suspeitas, cujo estado pode embutir estatísticas do dataset.

    Returns:
        ``(is_safe, issues)``.
    """
    issues: List[str] = []
    for layer in model.layers:
        if isinstance(layer, layers.Lambda) and any(
            name in layer.name for name in _SUSPICIOUS_LAMBDA_NAMES
        ):
            issues.append(f"Lambda de pré-processamento suspeita: {layer.name}")
    return len(issues) == 0, issues


def patch_architecture_for_safety(
    model: models.Model, normalization_type: str = "layer"
) -> models.Model:
    """Compatibilidade: loga achados e retorna o modelo INALTERADO.

    A reescrita automática de BatchNorm→LayerNorm foi removida (era um
    desvio silencioso dos papers). Se ``validate_model_safety`` apontar
    Lambdas suspeitas, o caminho correto é corrigir a arquitetura na fonte e
    retreinar — não operar o grafo salvo.

    Args:
        model: modelo a inspecionar.
        normalization_type: ignorado (mantido por compatibilidade de API).

    Returns:
        O próprio ``model``, sem modificações.
    """
    del normalization_type
    is_safe, issues = validate_model_safety(model)
    if not is_safe:
        logger.warning(
            "Modelo '%s' contém camadas potencialmente leaky (%s). "
            "Nenhum patch automático foi aplicado — corrija a arquitetura e "
            "retreine.",
            model.name,
            "; ".join(issues),
        )
    return model


class ArchitecturePatcher:
    """Compatibilidade de API com o patcher antigo (agora não-mutante)."""

    def __init__(self):
        self.replacements_made: List[str] = []

    def patch_model(
        self, model: models.Model, normalization_type: str = "layer"
    ) -> models.Model:
        """Ver :func:`patch_architecture_for_safety`."""
        self.replacements_made = []
        return patch_architecture_for_safety(model, normalization_type)
